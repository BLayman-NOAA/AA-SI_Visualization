"""Core echogram plotting functions for the AA-SI visualization package.

Provides a pipeline-based architecture for rendering Sv, MVBS, ML-feature,
and cluster-result echograms with configurable axes, colormaps, and overlay
lines.  The public API consists of:

- :func:`plot_sv_echogram`: regular Sv or MVBS data
- :func:`plot_flattened_data_echogram`: ML-processed data
- :func:`plot_cluster_echogram`: clustering results
- :func:`plot_processed_echogram_main`: low-level orchestrator used by the above
"""

import logging
import numpy as np
from aa_si_utils import utils
from .echogram_handlers import create_handler
from . import _plotting_utils as putils

logger = logging.getLogger(__name__)


def _setup_parameters(ds_Sv, frequency_nominal, max_depth, min_depth, ping_min, ping_max, 
                     sv_vmin, sv_vmax, sv_cmap, ds_Sv_original, use_corrected_Sv, 
                     x_axis_units, y_axis_units, meters_per_second, y_to_x_aspect_ratio_override,
                     ml_vmin, ml_vmax, echodata, ml_dataset_name, ml_specific_data_name):
    """
    Setup and validate all parameters for echogram plotting.
    
    Args:
        All parameters from plot_processed_echogram_main()
        
    Returns:
        dict: Consolidated parameters with defaults applied and ML variable name constructed
    """
    # Set default ping range if not provided
    if ping_max is None:
        ping_max = len(ds_Sv['ping_time']) - 1
    
    # Get frequency labels if frequency_nominal provided
    freq_labels = None
    n_channels = None
    if frequency_nominal is not None:
        # Check if frequency_nominal is array-like or needs conversion
        if isinstance(frequency_nominal, (list, tuple)):
            freq_labels = [f"{int(f/1000)} kHz" if isinstance(f, (int, float)) else str(f) for f in frequency_nominal]
        elif hasattr(frequency_nominal, '__iter__') and not isinstance(frequency_nominal, str):
            # NumPy array or similar
            freq_labels = [f"{int(f/1000)} kHz" for f in frequency_nominal]
        else:
            # Single value or string - wrap in list
            freq_labels = [str(frequency_nominal)]
        n_channels = len(freq_labels)
    
    # Construct ML variable name
    ml_data_variable = None
    if ml_dataset_name is not None:
        if ml_specific_data_name is not None and ml_specific_data_name != "" and ml_specific_data_name != ml_dataset_name:
            ml_data_variable = f"{ml_dataset_name}_{ml_specific_data_name}"
        else:
            ml_data_variable = f"{ml_dataset_name}"
    
    return {
        'ping_min': ping_min,
        'ping_max': ping_max,
        'freq_labels': freq_labels,
        'n_channels': n_channels,
        'ml_data_variable': ml_data_variable,
        'max_depth': max_depth,
        'min_depth': min_depth,
        'sv_vmin': sv_vmin,
        'sv_vmax': sv_vmax,
        'sv_cmap': sv_cmap,
        'use_corrected_Sv': use_corrected_Sv,
        'x_axis_units': x_axis_units,
        'y_axis_units': y_axis_units,
        'meters_per_second': meters_per_second,
        'y_to_x_aspect_ratio_override': y_to_x_aspect_ratio_override,
        'ml_vmin': ml_vmin,
        'ml_vmax': ml_vmax,
        'ml_dataset_name': ml_dataset_name,
        'ml_specific_data_name': ml_specific_data_name
    }


def _prepare_ml_data(ds_Sv, params):
    """
    Prepare ML data by regridding if needed, or return regular Sv variable name.
    Relies on handler for cluster detection.
    
    Args:
        ds_Sv: Dataset containing the data
        params: Parameter dict from _setup_parameters()
        
    Returns:
        tuple: (ds_Sv_modified, ml_info_dict, sv_variable_name, cluster_info_dict)
            - ds_Sv_modified: Dataset with regridded ML data added (if applicable)
            - ml_info_dict: Dict with ML metadata (None for regular Sv)
            - sv_variable_name: Name of variable to plot
            - cluster_info_dict: Dict with cluster metadata (None for non-cluster data)
    """
    ml_data_variable = params['ml_data_variable']
    
    # Check if ML data processing needed
    if ml_data_variable is None:
        # Regular Sv/MVBS data
        sv_variable_name = 'Sv' if not params['use_corrected_Sv'] else 'Sv_corrected'
        return (ds_Sv, None, sv_variable_name, None)
    
    # ML data processing
    if ml_data_variable not in ds_Sv:
        raise ValueError(f"ML data variable '{ml_data_variable}' not found in dataset")
    
    logger.info("Plotting ML data from variable: %s", ml_data_variable)
    
    # Check if already gridded
    ml_data = ds_Sv[ml_data_variable]
    data_dims = set(ml_data.dims)
    is_already_gridded = (
        'ping_time' in data_dims and 
        len(data_dims) == 2 and
        ('range_sample' in data_dims or 'echo_range' in data_dims)
    )
    
    if is_already_gridded:
        logger.info("Detected already-gridded single-channel data")
        sv_variable_name = ml_data_variable
        
        # Analyze cluster data
        cluster_info = _analyze_cluster_data(ml_data)
        
        # Set parameters for single-channel display
        params['n_channels'] = 1
        params['freq_labels'] = ['Cluster Results']
        
        return (ds_Sv, None, sv_variable_name, cluster_info)
    
    # Regular ML data processing (multi-feature)
    from aa_si_ml import ml
    regridded_data = ml.extract_ml_data_gridded(
        ds_Sv,
        specific_data_name=params['ml_specific_data_name'],
        dataset_name=params['ml_dataset_name'],
        fill_value=np.nan,
        store_in_dataset=False
    )
    
    # Create variable name and add to dataset
    sv_variable_name = 'regridded_' + ml_data_variable
    ds_Sv[sv_variable_name] = regridded_data
    
    logger.debug("Regridded ML data shape: %s", regridded_data.shape)
    
    # Get grid coordinates from regridded data
    grid_coords = ml.get_grid_coordinates(ds_Sv, sv_variable_name)
    
    # Discover feature dimension
    feature_dim_name = [dim for dim in regridded_data.dims 
                       if dim not in grid_coords][0]
    n_features = regridded_data.sizes[feature_dim_name]
    
    # Create feature labels based on actual feature coordinates
    freq_labels = None
    if hasattr(regridded_data.coords[feature_dim_name], 'values'):
        feature_coords = regridded_data.coords[feature_dim_name].values
        if feature_dim_name == 'channel':
            # Regular frequency channels
            freq_labels = [str(coord) for coord in feature_coords]
        elif feature_dim_name == 'feature':
            # Feature-based (e.g., baseline + differences)
            freq_labels = [str(coord) for coord in feature_coords]
        else:
            # Generic feature names
            freq_labels = [f"Feature {i}" for i in range(n_features)]
    else:
        freq_labels = [f"Feature {i}" for i in range(n_features)]
    
    logger.debug("ML data features: %s", freq_labels)
    
    # Determine color scale for ML data
    ml_vmin = params['ml_vmin']
    ml_vmax = params['ml_vmax']
    
    if ml_vmin is None or ml_vmax is None:
        ml_data_values = regridded_data.values
        valid_data = ml_data_values[~np.isnan(ml_data_values)]
        if len(valid_data) > 0:
            if ml_vmin is None:
                ml_vmin = np.percentile(valid_data, 1)
            if ml_vmax is None:
                ml_vmax = np.percentile(valid_data, 99)
        else:
            ml_vmin, ml_vmax = 0, 1  # Fallback
    
    # Update params with ML-specific values
    params['freq_labels'] = freq_labels
    params['n_channels'] = n_features
    params['sv_vmin'] = ml_vmin
    params['sv_vmax'] = ml_vmax
    
    # Create ML info dict
    ml_info = {
        'regridded_data': regridded_data,
        'feature_labels': freq_labels,
        'n_features': n_features,
        'vmin': ml_vmin,
        'vmax': ml_vmax,
        'grid_coords': grid_coords,
        'feature_dim': feature_dim_name
    }
    
    return (ds_Sv, ml_info, sv_variable_name, None)


def _analyze_cluster_data(cluster_data):
    """
    Extract cluster-specific metadata needed for categorical coloring.
    
    Args:
        cluster_data: xarray.DataArray containing cluster assignments
        
    Returns:
        dict: Cluster analysis information containing:
            - unique_labels: Array of unique cluster IDs
            - min_label: Minimum cluster ID (may be -1 for DBSCAN noise)
            - max_label: Maximum cluster ID
            - num_clusters: Number of unique clusters
            - label_counts: Dict mapping cluster ID to sample count
            - label_percentages: Dict mapping cluster ID to percentage
    """
    # Get cluster values, excluding NaN fill values
    cluster_values = cluster_data.values
    valid_clusters = cluster_values[~np.isnan(cluster_values)]
    
    if len(valid_clusters) == 0:
        logger.warning("No valid cluster data found")
        return {
            'unique_labels': np.array([0]),
            'min_label': 0,
            'max_label': 0,
            'num_clusters': 1,
            'label_counts': {0: 0},
            'label_percentages': {0: 0.0}
        }
    # Analyze cluster labels
    unique_labels = np.unique(valid_clusters)
    num_clusters = len(unique_labels)
    min_label = int(min(unique_labels))
    max_label = int(max(unique_labels))
    
    # Calculate counts and percentages
    label_counts = {}
    label_percentages = {}
    
    for label in unique_labels:
        count = np.sum(cluster_values == label)
        percentage = count / len(valid_clusters) * 100
        label_counts[int(label)] = int(count)
        label_percentages[int(label)] = percentage
    
    logger.info("Cluster analysis:")
    logger.info("  Cluster range: %d to %d", min_label, max_label)
    logger.info("  Number of clusters: %d", num_clusters)
    for label in unique_labels:
        count = label_counts[int(label)]
        percentage = label_percentages[int(label)]
        label_name = 'Noise' if label == -1 else f'Cluster {int(label)}'
        logger.info("  %s: %s samples (%.1f%%)", label_name, f"{count:,}", percentage)
    
    return {
        'unique_labels': unique_labels,
        'min_label': min_label,
        'max_label': max_label,
        'num_clusters': num_clusters,
        'label_counts': label_counts,
        'label_percentages': label_percentages
    }


def _filter_nan_frequencies(ds_Sv, sv_variable_name, params, cluster_info):
    """
    Filter out frequency channels that contain only NaN data.
    
    Args:
        ds_Sv: Dataset containing the data
        sv_variable_name: Name of variable to check
        params: Parameter dict from _setup_parameters()
        cluster_info: Cluster info dict (None for non-cluster data)
        
    Returns:
        dict: Updated params with filtered frequencies
    """
    # Skip filtering for single-channel data (cluster or single frequency)
    if cluster_info or not params['n_channels'] or params['n_channels'] <= 1:
        if params['n_channels']:
            params['valid_channel_indices'] = list(range(params['n_channels']))
        return params
    
    logger.info("Checking for all-NaN frequencies...")
    valid_channels = []
    valid_freq_labels = []
    
    for freq_idx in range(params['n_channels']):
        # Get a sample slice to check for data
        try:
            if sv_variable_name in ds_Sv:
                if 'channel' in ds_Sv[sv_variable_name].dims:
                    sample_data = ds_Sv[sv_variable_name].isel(channel=freq_idx)
                    has_valid_data = not np.all(np.isnan(sample_data.values))
                else:
                    # Single channel data
                    has_valid_data = True
            else:
                has_valid_data = True
                
            if has_valid_data:
                valid_channels.append(freq_idx)
                if params['freq_labels']:
                    valid_freq_labels.append(params['freq_labels'][freq_idx])
            else:
                freq_label = params['freq_labels'][freq_idx] if params['freq_labels'] else f"Channel {freq_idx}"
                logger.info("  Skipping %s: all data is NaN", freq_label)
        except Exception as e:
            logger.warning("  Error checking frequency %s: %s", freq_idx, e)
            valid_channels.append(freq_idx)  # Include on error to avoid breaking
            if params['freq_labels']:
                valid_freq_labels.append(params['freq_labels'][freq_idx])
    
    if len(valid_channels) == 0:
        raise ValueError("All frequencies contain only NaN data - nothing to plot")
    
    if len(valid_channels) < params['n_channels']:
        logger.info("  Reduced from %s to %s valid frequencies", params['n_channels'], len(valid_channels))
        params['n_channels'] = len(valid_channels)
        params['freq_labels'] = valid_freq_labels
        params['valid_channel_indices'] = valid_channels
    else:
        logger.info("  All %s frequencies contain valid data", params['n_channels'])
        params['valid_channel_indices'] = list(range(params['n_channels']))
    
    return params


def _create_cluster_colormap(cluster_analysis, base_colors=None):
    """
    Generate categorical colormap for cluster visualization.
    
    Args:
        cluster_analysis: Dict from _analyze_cluster_data() with cluster metadata
        base_colors: Optional list of base colors (hex strings)
        
    Returns:
        tuple: (cmap, norm, tick_positions, tick_labels)
            - cmap: matplotlib ListedColormap for clusters
            - norm: matplotlib BoundaryNorm for discrete boundaries
            - tick_positions: List of positions for colorbar ticks
            - tick_labels: List of labels for colorbar ticks
    """
    import matplotlib.colors as mcolors
    
    unique_labels = cluster_analysis['unique_labels']
    min_label = cluster_analysis['min_label']
    max_label = cluster_analysis['max_label']
    num_clusters = cluster_analysis['num_clusters']
    
    # Define default base colors (bright colors that show well on black background)
    if base_colors is None:
        base_colors = [
            "#00F3FC", "#35E200", "#0400FF", "#F943FF", "#F30101", 
            "#EDFF4D", "#4E9200", "#970021", "#5600C7", "#017685FF", "#FFA600FF"
        ]
    
    # Generate additional colors if needed using golden ratio spacing
    if num_clusters > len(base_colors):
        hue_offset = 0.3
        additional_colors = utils.generate_colors(hue_offset, num_clusters - len(base_colors))
        
        cluster_colors = base_colors + additional_colors
    else:
        cluster_colors = base_colors[:num_clusters]
    
    logger.info("Using %s distinct colors for clusters", len(cluster_colors))
    
    # Create colormap based on whether noise points exist (DBSCAN vs K-means)
    no_data_color = "#2E2E2E"  # Gray for NaN/unassigned
    
    if min_label == -1:
        # DBSCAN case: includes noise points at label -1
        noise_color = "#000000"  # Black for noise
        colors_list = [noise_color] + cluster_colors[:num_clusters-1]
        
        # Create boundaries: [-1.5, -0.5, 0.5, 1.5, 2.5, ...]
        bounds = [-1.5, -0.5] + [i + 0.5 for i in range(num_clusters - 1)]
        
        # Tick positions and labels
        tick_positions = list(unique_labels)
        tick_labels = ['Noise'] + [f'Cluster {int(i)}' for i in unique_labels if i != -1]
        
    else:
        # K-means case: no noise points
        colors_list = cluster_colors[:num_clusters]
        
        # Create boundaries: [-0.5, 0.5, 1.5, 2.5, ...]
        bounds = [i - 0.5 for i in range(num_clusters)] + [num_clusters - 0.5]
        
        # Tick positions and labels
        tick_positions = list(unique_labels)
        tick_labels = [f'Cluster {int(i)}' for i in unique_labels]
    
    # Create ListedColormap and BoundaryNorm
    cmap = mcolors.ListedColormap(colors_list)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Set color for NaN values
    cmap.set_bad(color=no_data_color)
    
    return (cmap, norm, tick_positions, tick_labels)


def _calculate_ranges(handler, params, ds_Sv_original):
    """
    Calculate depth and ping ranges using the handler.
    
    Args:
        handler: EchogramDataHandler instance
        params: Parameter dict from _setup_parameters()
        ds_Sv_original: Original Sv dataset (for MVBS conversion)
        
    Returns:
        dict: Range information with depth and ping details
    """
    min_depth = params['min_depth']
    max_depth = params['max_depth']
    ping_min = params['ping_min']
    ping_max = params['ping_max']
    
    # Depth range setup - auto-detect if needed
    if min_depth is None or max_depth is None:
        logger.info("Auto-detecting depth range from data...")
        depth_ref_dataset = ds_Sv_original if ds_Sv_original else handler.dataset
        
        # Check data structure type from handler
        data_type = handler.detect_structure()['type']
        
        if data_type.startswith('Cluster'):
            # Cluster data - use handler's MVBS flag to determine approach
            if handler.is_mvbs_structured():
                # MVBS-derived cluster - use echo_range values directly
                auto_min_depth = float(handler.echo_range_values[0])
                auto_max_depth = float(handler.echo_range_values[-1])
            else:
                # Sv-derived cluster - use utils helper without channel indexing
                # Get a representative ping for depth range
                try:
                    echo_range_data = depth_ref_dataset['echo_range']
                    # For Sv-derived data, echo_range is 2D (channel, ping_time) or 3D
                    # Just use first available depth profile
                    if 'channel' in echo_range_data.dims:
                        depths_sample = echo_range_data.isel(
                            channel=0, 
                            ping_time=min(ping_min, len(depth_ref_dataset['ping_time'])-1)
                        )
                    else:
                        depths_sample = echo_range_data.isel(
                            ping_time=min(ping_min, len(depth_ref_dataset['ping_time'])-1)
                        )
                    valid_depths = depths_sample.values[~np.isnan(depths_sample.values)]
                    if len(valid_depths) > 0:
                        auto_min_depth = float(np.min(valid_depths))
                        auto_max_depth = float(np.max(valid_depths))
                    else:
                        auto_min_depth, auto_max_depth = 0.0, 250.0  # Fallback
                except Exception as e:
                    logger.warning("Could not auto-detect depth range: %s", e)
                    auto_min_depth, auto_max_depth = 0.0, 250.0  # Fallback
        else:
            # Regular multi-channel data
            auto_min_depth, auto_max_depth = utils.find_data_depth_range(
                depth_ref_dataset, ping_min, ping_max, channel=0
            )
        
        if min_depth is None:
            min_depth = auto_min_depth
        if max_depth is None:
            max_depth = auto_max_depth
        logger.info("Using depth range: %.1fm to %.1fm", min_depth, max_depth)
    
    # Call handler methods to calculate indices and extents
    min_depth_idx, max_depth_idx = handler.calculate_depth_indices(min_depth, max_depth)
    ping_min_actual, ping_max_actual = handler.calculate_ping_range(ping_min, ping_max, ds_Sv_original)
    min_depth_shown, max_depth_shown = handler.get_depth_extent(min_depth_idx, max_depth_idx)
    
    # Return consolidated range information
    return {
        'depth': {
            'min_depth': min_depth,
            'max_depth': max_depth,
            'min_index': min_depth_idx,
            'max_index': max_depth_idx,
            'min_shown': min_depth_shown,
            'max_shown': max_depth_shown
        },
        'ping': {
            'min': ping_min_actual,
            'max': ping_max_actual
        }
    }





def _calculate_axes(handler, ranges, params, echodata):
    """
    Calculate x-axis and y-axis extents and plot dimensions.
    
    Args:
        handler: EchogramDataHandler instance
        ranges: Range dict from _calculate_ranges()
        params: Parameter dict from _setup_parameters()
        echodata: Original echodata for GPS calculations
        
    Returns:
        dict: Axis configuration with extents, labels, and plot dimensions
    """
    # Get ping times from handler
    ping_times = handler.get_ping_times()
    
    # Calculate x-axis extent
    x_min, x_max, x_label = putils.calculate_x_axis_extent(
        ping_times,
        ranges['ping']['min'], ranges['ping']['max'],
        params['x_axis_units'],
        meters_per_second=params['meters_per_second'],
        echodata=echodata,
        handler=handler,
    )
    
    # Calculate y-axis extent
    data_type = handler.detect_structure()['type']
    y_min, y_max, y_label = putils.calculate_y_axis_extent(
        ranges['depth']['min_shown'],
        ranges['depth']['max_shown'],
        ranges['depth']['min_index'],
        ranges['depth']['max_index'],
        params['y_axis_units'],
        data_type,
    )
    
    # Calculate plot dimensions and aspect ratio
    extent, aspect_ratio, width_mult, height_mult = putils.calculate_plot_dimensions(
        x_min, x_max, y_min, y_max, params['y_to_x_aspect_ratio_override']
    )
    
    # Log diagnostics
    logger.info("%s Echogram dimensions:", data_type)
    
    if handler.is_mvbs_structured():
        logger.info("  Original ping range requested: %s to %s", params['ping_min'], params['ping_max'])
        logger.info("  MVBS ping indices used: %s to %s", ranges['ping']['min'], ranges['ping']['max'])
        logger.info("  MVBS time range: %s to %s", ping_times[ranges['ping']['min']], ping_times[ranges['ping']['max']])
        logger.info("  MVBS depth range: %s to %s", ranges['depth']['min_shown'], ranges['depth']['max_shown'])
    elif data_type.startswith("ML"):
        logger.info("  ML data regridded from flattened format")
        logger.info("  Ping range: %s to %s", ranges['ping']['min'], ranges['ping']['max'])
    
    logger.info("  X-axis (%s): %.1f to %.1f", params['x_axis_units'], x_min, x_max)
    logger.info("  Y-axis (%s): %.1f to %.1f", params['y_axis_units'], y_min, y_max)
    logger.info("  Features: %s", params['n_channels'])
    logger.info("  Depth Range: %.1fm to %.1fm", ranges['depth']['min_shown'], ranges['depth']['max_shown'])
    logger.info("  Aspect ratio (1:1 in specified units): %.3f", aspect_ratio)
    
    # Return consolidated axis configuration
    return {
        'x': {'min': x_min, 'max': x_max, 'label': x_label},
        'y': {'min': y_min, 'max': y_max, 'label': y_label},
        'extent': extent,
        'aspect_ratio': aspect_ratio,
        'width_multiplier': width_mult,
        'height_multiplier': height_mult
    }


def _add_overlay_line(ax, ds, line_spec, ping_min, ping_max, axes_config):
    """
    Add line overlay to echogram subplot.
    
    Supports dataset variable reference with automatic coordinate transformation.
    
    Args:
        ax (matplotlib.axes.Axes): Subplot to add line to.
        ds (xarray.Dataset): Dataset containing overlay variables.
        line_spec (dict): Line specification: ``{'var': 'variable_name', 'style': {...}}``.
        ping_min (int): Start ping index for current plot window.
        ping_max (int): End ping index for current plot window.
        axes_config (dict): Axis configuration from ``_calculate_axes()`` containing
            x-axis extent info.
        
    Note:
        - Y-axis: Variable values in meters (must match echogram depth units)
        - X-axis: Automatically transformed to match echogram's x-axis units
        - Automatically removes NaN values for clean line display
    """
    var_name = line_spec['var']
    
    if var_name not in ds:
        logger.warning("Overlay variable '%s' not found in dataset", var_name)
        return
    
    # Get variable data and slice to plot window
    y_coords = ds[var_name].values[ping_min:ping_max+1]
    
    # Transform x-coordinates to match echogram's x-axis scale
    x_min = axes_config['x']['min']
    x_max = axes_config['x']['max']
    num_points = len(y_coords)
    
    # Create x-coordinates that span the full echogram x-range
    x_coords = np.linspace(x_min, x_max, num_points)
    
    # Remove NaN points for clean line
    valid_mask = ~np.isnan(y_coords)
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    
    if len(x_coords) == 0:
        logger.warning("No valid data for overlay variable '%s' in plot range", var_name)
        return
    
    # Get style from line_spec
    plot_kwargs = line_spec.get('style', {})
    
    # Apply default styling for visibility on black background
    plot_kwargs.setdefault('color', 'cyan')
    plot_kwargs.setdefault('linewidth', 2.5)
    plot_kwargs.setdefault('alpha', 0.9)
    plot_kwargs.setdefault('zorder', 10)
    
    ax.plot(x_coords, y_coords, **plot_kwargs)


def _create_cluster_plot(fig, handler, axes_config, ranges, cluster_info, cluster_colors):
    """Render a single-panel cluster classification echogram.

    Args:
        fig: matplotlib Figure.
        handler: EchogramDataHandler instance.
        axes_config: Axis configuration from _calculate_axes().
        ranges: Range dict from _calculate_ranges().
        cluster_info: Cluster info dict from _prepare_ml_data().
        cluster_colors: Optional list of hex color strings.

    Returns:
        list: Axes objects created (length 1).
    """
    import matplotlib.pyplot as plt

    logger.info("Creating cluster plot...")
    ax = plt.subplot(1, 1, 1)
    ax.set_facecolor('black')

    ping_range = (ranges['ping']['min'], ranges['ping']['max'])
    depth_range = (ranges['depth']['min_index'], ranges['depth']['max_index'])
    cluster_data = handler.slice_data_for_frequency(0, ping_range, depth_range)

    cmap, norm, tick_positions, tick_labels = _create_cluster_colormap(
        cluster_info, base_colors=cluster_colors
    )

    im = ax.imshow(
        cluster_data.T,
        aspect=axes_config['aspect_ratio'],
        cmap=cmap,
        norm=norm,
        extent=axes_config['extent'],
        interpolation='nearest',
    )

    ax.set_title('Cluster Analysis - Multi-frequency Classification',
                 fontsize=14, fontweight='bold', color='white', pad=20)
    ax.set_ylabel(axes_config['y']['label'], fontsize=12, color='white')
    ax.set_xlabel(axes_config['x']['label'], fontsize=10, color='white')
    ax.tick_params(colors='white')

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(right=0.85, top=0.92)

    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar_ax.set_facecolor('black')
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Cluster ID', fontsize=12, color='white')
    cbar.ax.tick_params(colors='white')
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(tick_labels)

    plt.suptitle('Cluster Analysis - Multi-frequency Acoustic Backscatter Classification',
                 fontsize=16, fontweight='bold', color='white', y=0.96)

    return [ax]


def _create_multi_frequency_plot(fig, handler, axes_config, ranges, params):
    """Render a multi-panel continuous-colormap echogram (one row per channel).

    Args:
        fig: matplotlib Figure.
        handler: EchogramDataHandler instance.
        axes_config: Axis configuration from _calculate_axes().
        ranges: Range dict from _calculate_ranges().
        params: Parameter dict from _setup_parameters().

    Returns:
        list: Axes objects created (one per frequency).
    """
    import matplotlib.pyplot as plt

    logger.info("Creating multi-frequency plot...")
    data_type = handler.detect_structure()['type']

    sv_image = None
    freq_labels = params['freq_labels']
    n_channels = params['n_channels']
    axes_list = []

    for plot_idx, freq_idx in enumerate(
        params.get('valid_channel_indices', range(n_channels))
    ):
        ax = plt.subplot(n_channels, 1, plot_idx + 1)
        axes_list.append(ax)
        ax.set_facecolor('black')

        freq_label = freq_labels[plot_idx] if freq_labels else f"Feature {plot_idx}"

        ping_range = (ranges['ping']['min'], ranges['ping']['max'])
        depth_range = (ranges['depth']['min_index'], ranges['depth']['max_index'])
        sv_data = handler.slice_data_for_frequency(freq_idx, ping_range, depth_range)

        im = ax.imshow(
            sv_data.T,
            aspect=axes_config['aspect_ratio'],
            vmin=params['sv_vmin'],
            vmax=params['sv_vmax'],
            cmap=params['sv_cmap'],
            extent=axes_config['extent'],
        )

        if plot_idx == 0:
            sv_image = im

        title_suffix = "ML Data" if data_type.startswith("ML") else f'{data_type} Echogram'
        ax.set_title(f'{freq_label} - {title_suffix}',
                     fontsize=12, fontweight='bold', color='white', pad=20)
        ax.set_ylabel(axes_config['y']['label'], fontsize=10, color='white')
        ax.tick_params(colors='white')

        if plot_idx == n_channels - 1:
            ax.set_xlabel(axes_config['x']['label'], fontsize=10, color='white')

    plt.tight_layout(pad=3.0, h_pad=5.0, w_pad=2.0)
    fig.subplots_adjust(bottom=0.12, top=0.92)

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar_ax.set_facecolor('black')
    cbar = fig.colorbar(sv_image, cax=cbar_ax, orientation='horizontal')

    if data_type.startswith("ML"):
        cbar_label = 'ML Feature Value'
        plot_title = 'ML Data Echogram'
    else:
        cbar_label = 'Sv (dB)'
        plot_title = f'{data_type} Echogram - {"Gridded " if data_type == "MVBS" else ""}Calibrated Data'

    cbar.set_label(cbar_label, fontsize=10, color='white')
    cbar.ax.tick_params(colors='white')
    plt.suptitle(plot_title, fontsize=16, fontweight='bold', y=0.96, color='white')

    return axes_list


def _create_plot(handler, axes_config, ranges, params, ml_info=None, cluster_info=None, cluster_colors=None, overlay_lines=None):
    """Create the final echogram plot using matplotlib.

    Dispatches to :func:`_create_cluster_plot` for cluster data or
    :func:`_create_multi_frequency_plot` for continuous Sv / ML data.

    Args:
        handler: EchogramDataHandler instance.
        axes_config: Axis configuration from _calculate_axes().
        ranges: Range dict from _calculate_ranges().
        params: Parameter dict from _setup_parameters().
        ml_info: ML info dict from _prepare_ml_data() (None for regular Sv).
        cluster_info: Cluster info dict (None for non-cluster data).
        cluster_colors: Optional list of hex color strings.
        overlay_lines: Optional list of line overlay specifications.
    """
    import matplotlib.pyplot as plt

    is_cluster_mode = (cluster_info is not None)

    plt.style.use('dark_background')

    fig_width = 24 * axes_config['width_multiplier']
    if is_cluster_mode:
        fig_height = 12 * axes_config['height_multiplier']
    else:
        fig_height = 12 * params['n_channels'] * axes_config['height_multiplier']

    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('black')

    if is_cluster_mode:
        axes_list = _create_cluster_plot(
            fig, handler, axes_config, ranges, cluster_info, cluster_colors
        )
    else:
        axes_list = _create_multi_frequency_plot(
            fig, handler, axes_config, ranges, params
        )

    if overlay_lines is not None:
        ping_min = ranges['ping']['min']
        ping_max = ranges['ping']['max']
        for ax in axes_list:
            for line_spec in overlay_lines:
                _add_overlay_line(ax, handler.dataset, line_spec, ping_min, ping_max, axes_config)

    plt.show()
    plt.style.use('default')


def plot_processed_echogram_main(ds_Sv, frequency_nominal, max_depth=None, min_depth=None, 
                                ping_min=0, ping_max=None, sv_vmin=-80, sv_vmax=-20, 
                                sv_cmap='viridis', ds_Sv_original=None, use_corrected_Sv=False, 
                                x_axis_units='seconds', y_axis_units='meters', 
                                meters_per_second=None, y_to_x_aspect_ratio_override=None, 
                                ml_vmin=None, ml_vmax=None, echodata=None, 
                                ml_dataset_name=None, ml_specific_data_name=None, cluster_colors=None, 
                                overlay_lines=None):
    """
    Create an echogram plot for MVBS (gridded), regular Sv data, or ML/normalized data.
    
    Args:
        ds_Sv (xarray.Dataset): Either MVBS dataset (gridded), regular Sv dataset,
            or ML-ready dataset to plot.
        frequency_nominal (array-like): Nominal frequencies for each channel.
        max_depth (float, optional): Maximum depth to display in meters.
            Defaults to auto-detect from data.
        min_depth (float, optional): Minimum depth to display in meters.
            Defaults to auto-detect from data.
        ping_min (int, optional): Start ping index in the ORIGINAL Sv data.
            For MVBS data, converted to appropriate MVBS indices. Defaults to 0.
        ping_max (int, optional): End ping index in the ORIGINAL Sv data.
            Defaults to last ping.
        sv_vmin (float): Minimum color scale limit in dB. Defaults to -80.
        sv_vmax (float): Maximum color scale limit in dB. Defaults to -20.
        sv_cmap (str): Colormap for Sv data. Defaults to ``'viridis'``.
        ds_Sv_original (xarray.Dataset, optional): Original Sv dataset (needed
            for MVBS data to convert ping indices properly).
        use_corrected_Sv (bool): Whether to use ``'Sv_corrected'`` instead of
            ``'Sv'``. Defaults to ``False``.
        x_axis_units (str): X-axis units: ``'seconds'`` (default), ``'pings'``,
            ``'bins'`` (MVBS only), or ``'meters'``.
        y_axis_units (str): Y-axis units: ``'meters'`` (default),
            ``'range_sample'``, or ``'bins'`` (MVBS only).
        meters_per_second (float, optional): Speed in m/s for converting time
            to distance (required for ``x_axis_units='meters'``).
        y_to_x_aspect_ratio_override (float, optional): Override aspect ratio
            for plot.
        ml_vmin (float, optional): Minimum color scale limit for ML data.
            Auto-determined from data if ``None``.
        ml_vmax (float, optional): Maximum color scale limit for ML data.
            Auto-determined from data if ``None``.
        echodata (xarray.Dataset, optional): Original echodata for GPS
            calculations when ``meters_per_second`` is not provided.
        ml_dataset_name (str, optional): Name of ML dataset to plot.
        ml_specific_data_name (str, optional): Specific ML data name within
            the dataset.
        cluster_colors (list, optional): List of hex color strings for cluster
            visualization.
        overlay_lines (list, optional): List of line specification dicts for
            overlay lines.
        
    Examples:
        >>> # Plot regular Sv data
        >>> plot_processed_echogram_main(ds_Sv, frequency_nominal)
        
        >>> # Plot MVBS data with original Sv for ping conversion
        >>> plot_processed_echogram_main(ds_Sv_mvbs, frequency_nominal,
        ...                             ds_Sv_original=ds_Sv,
        ...                             ping_min=100, ping_max=800)
        
        >>> # Plot ML data
        >>> plot_processed_echogram_main(ds_ml_ready, frequency_nominal,
        ...                             ml_dataset_name='ml_data_clean',
        ...                             ml_specific_data_name='normalized')
    """
    
    logger.info("Setting up parameters...")
    params = _setup_parameters(
        ds_Sv, frequency_nominal, max_depth, min_depth, ping_min, ping_max,
        sv_vmin, sv_vmax, sv_cmap, ds_Sv_original, use_corrected_Sv,
        x_axis_units, y_axis_units, meters_per_second, y_to_x_aspect_ratio_override,
        ml_vmin, ml_vmax, echodata, ml_dataset_name, ml_specific_data_name
    )
    logger.info("  Ping range: %s to %s", params['ping_min'], params['ping_max'])
    if params['ml_data_variable']:
        logger.info("  ML variable: %s", params['ml_data_variable'])
    
    logger.info("Preparing data...")
    ds_Sv_plot, ml_info, sv_variable_name, cluster_info = _prepare_ml_data(ds_Sv, params)
    if cluster_info:
        logger.info("  Cluster data detected: %s clusters", cluster_info['num_clusters'])
    elif ml_info:
        logger.info("  ML data regridded: %s features", ml_info['n_features'])
        logger.info("  Color scale: %.2f to %.2f", params['sv_vmin'], params['sv_vmax'])
    else:
        logger.info("  Using variable: %s", sv_variable_name)
    
    params = _filter_nan_frequencies(ds_Sv_plot, sv_variable_name, params, cluster_info)

    logger.info("Creating data handler...")
    handler = create_handler(ds_Sv_plot, sv_variable_name, params['ml_data_variable'])
    data_type = handler.detect_structure()['type']
    logger.info("  Handler created: %s", handler.__class__.__name__)
    logger.info("  Data type: %s", data_type)
    
    logger.info("Calculating ranges...")
    ranges = _calculate_ranges(handler, params, ds_Sv_original)
    logger.info("  Depth range: %.1fm to %.1fm", ranges['depth']['min_shown'], ranges['depth']['max_shown'])
    logger.info("  Ping range: %s to %s", ranges['ping']['min'], ranges['ping']['max'])
    
    logger.info("Calculating axes...")
    axes_config = _calculate_axes(handler, ranges, params, echodata)
    
    logger.info("Creating plot...")
    _create_plot(handler, axes_config, ranges, params, ml_info, cluster_info, cluster_colors=cluster_colors, overlay_lines=overlay_lines)


def plot_cluster_echogram(ds_ml_ready, dataset_name, specific_data_name, 
                         max_depth=None, min_depth=None, ping_min=None, ping_max=None,
                         x_axis_units='seconds', y_axis_units='meters', 
                         meters_per_second=None, echodata=None,
                         y_to_x_aspect_ratio_override=None, gridded_data=None,
                         ds_Sv_original=None, cluster_colors=None, overlay_lines=None):
    """
    Plot cluster analysis results as echogram visualization.
    
    Simplified interface for plotting clustering results (K-means, DBSCAN, etc.).
    Automatically handles both Sv-derived and MVBS-derived cluster data.
    Uses categorical colormap with distinct colors for each cluster.
    
    Args:
        ds_ml_ready (xarray.Dataset): Dataset containing cluster results.
        dataset_name (str): Base dataset name (e.g., ``'ml_data_clean'``).
        specific_data_name (str): Name of cluster result to plot
            (e.g., ``'kmeans_clusters'``, ``'dbscan_clusters'``).
        max_depth (float, optional): Maximum depth in meters.
            Auto-detected from data if ``None``.
        min_depth (float, optional): Minimum depth in meters.
            Auto-detected from data if ``None``.
        ping_min (int, optional): Start ping index. Defaults to 100.
            For MVBS-derived data, converted to MVBS bin indices.
        ping_max (int, optional): End ping index. Defaults to 800.
        x_axis_units (str): X-axis units: ``'seconds'`` (default), ``'pings'``,
            ``'bins'`` (MVBS only), or ``'meters'``.
        y_axis_units (str): Y-axis units: ``'meters'`` (default),
            ``'range_sample'``, or ``'bins'`` (MVBS only).
        meters_per_second (float, optional): Speed for ``'meters'`` x-axis
            (auto-calculated from GPS if ``echodata`` provided).
        echodata (xarray.Dataset, optional): Original echodata for GPS-based
            calculations.
        y_to_x_aspect_ratio_override (float, optional): Override aspect ratio.
        gridded_data (xarray.DataArray, optional): Pre-gridded cluster data.
            If ``None``, will regrid from flattened format.
        ds_Sv_original (xarray.Dataset, optional): Original Sv dataset (needed
            for MVBS-derived data to convert ping indices).
        cluster_colors (list, optional): List of hex color strings for cluster
            visualization.
        overlay_lines (list, optional): List of line specification dicts for
            overlay lines.
        
    Examples:
        >>> # Basic usage with K-means results
        >>> plot_cluster_echogram(ds_ml, 'ml_data_clean', 'kmeans_clusters')
        
        >>> # With DBSCAN results and custom range
        >>> plot_cluster_echogram(ds_ml, 'ml_data_clean', 'dbscan_clusters',
        ...                      ping_min=200, ping_max=600)
        
        >>> # MVBS-derived clusters with bin axes
        >>> plot_cluster_echogram(ds_ml_mvbs, 'ml_data_clean', 'kmeans_clusters',
        ...                      ds_Sv_original=ds_Sv,
        ...                      x_axis_units='bins', y_axis_units='bins')
        
        >>> # With custom depth range
        >>> plot_cluster_echogram(ds_ml, 'ml_data_clean', 'kmeans_clusters',
        ...                      min_depth=10, max_depth=100)
    """
    # Set default ping range if not provided
    if ping_min is None:
        ping_min = 100
    if ping_max is None:
        ping_max = 800
    
    full_result_name = f"{dataset_name}_{specific_data_name}"
    grid_result_name = f"{full_result_name}_grid"
    
    if gridded_data is not None:
        logger.info("Using provided gridded cluster data")
        ds_ml_ready[grid_result_name] = gridded_data
    
    if grid_result_name not in ds_ml_ready:
        if full_result_name in ds_ml_ready:
            logger.info("Regridding %s for visualization...", full_result_name)
            from aa_si_ml import ml
            gridded_results = ml.extract_ml_data_gridded(
                ds_ml_ready, 
                specific_data_name=specific_data_name,
                dataset_name=dataset_name,
                fill_value=np.nan,
                store_in_dataset=True  # This creates the _grid variable
            )
            logger.info("  Regridded to shape: %s", gridded_results.shape)
        else:
            raise ValueError(f"Neither {grid_result_name} nor {full_result_name} found in dataset")
    else:
        logger.info("Using existing gridded results: %s", grid_result_name)
    
    # Print data statistics
    cluster_data_var = ds_ml_ready[grid_result_name]
    total_values = cluster_data_var.size
    nan_count = np.sum(np.isnan(cluster_data_var.values))
    valid_count = total_values - nan_count
    
    logger.info("Gridded cluster data analysis:")
    logger.info("  Total values: %s", f"{total_values:,}")
    logger.info("  NaN values: %s (%.1f%%)", f"{nan_count:,}", nan_count/total_values*100)
    logger.info("  Valid values: %s (%.1f%%)", f"{valid_count:,}", valid_count/total_values*100)
    
    # Detect MVBS structure
    is_mvbs_derived = putils.is_mvbs_dataset(ds_ml_ready)
    
    if is_mvbs_derived:
        logger.info("Detected MVBS-derived cluster data")
        if ds_Sv_original is None:
            logger.warning("ds_Sv_original not provided for MVBS data. Ping range conversion may not be accurate.")
    else:
        logger.info("Detected regular Sv-derived cluster data")
    
    return plot_processed_echogram_main(
        ds_Sv=ds_ml_ready,
        frequency_nominal=None,
        max_depth=max_depth,
        min_depth=min_depth,
        ping_min=ping_min,
        ping_max=ping_max,
        sv_vmin=None,
        sv_vmax=None,
        sv_cmap='viridis',
        ds_Sv_original=ds_Sv_original,
        use_corrected_Sv=False,
        x_axis_units=x_axis_units,
        y_axis_units=y_axis_units,
        meters_per_second=meters_per_second,
        y_to_x_aspect_ratio_override=y_to_x_aspect_ratio_override,
        ml_vmin=None,
        ml_vmax=None,
        echodata=echodata,
        ml_dataset_name=dataset_name,
        ml_specific_data_name=f"{specific_data_name}_grid",
        cluster_colors=cluster_colors,
        overlay_lines=overlay_lines
    )


def plot_sv_echogram(ds_Sv, ds_Sv_original=None, frequency_nominal=None, min_depth=None, max_depth=None,
                     ping_min=0, ping_max=None, sv_vmin=-80, sv_vmax=-20, 
                     sv_cmap='viridis', use_corrected_Sv=False,
                     x_axis_units='seconds', y_axis_units='meters',
                     meters_per_second=None, echodata=None,
                     y_to_x_aspect_ratio_override=None, overlay_lines=None
                     ):
    """
    Plot Sv echogram data (regular or MVBS).
    
    Automatically detects MVBS format and handles both regular Sv and MVBS datasets.
    For MVBS data, ``ds_Sv_original`` is required for proper ping range conversion.
    
    Args:
        ds_Sv (xarray.Dataset): Sv dataset to plot (regular or MVBS).
        ds_Sv_original (xarray.Dataset, optional): Original Sv dataset
            (required for MVBS data to convert ping indices).
        frequency_nominal (array-like, optional): Nominal frequencies for each
            channel. Auto-detected if ``None``.
        min_depth (float, optional): Minimum depth in meters.
            Auto-detected from data if ``None``.
        max_depth (float, optional): Maximum depth in meters.
            Auto-detected from data if ``None``.
        ping_min (int, optional): Start ping index. Defaults to 0.
            For MVBS data: in ORIGINAL Sv indices (converted to MVBS bins).
        ping_max (int, optional): End ping index. Defaults to last ping.
        sv_vmin (float): Minimum color scale limit in dB. Defaults to -80.
        sv_vmax (float): Maximum color scale limit in dB. Defaults to -20.
        sv_cmap (str): Colormap name. Defaults to ``'viridis'``.
        use_corrected_Sv (bool): Use ``'Sv_corrected'`` instead of ``'Sv'``.
            Defaults to ``False``.
        x_axis_units (str): X-axis units: ``'seconds'`` (default), ``'pings'``,
            ``'bins'`` (MVBS only), or ``'meters'``.
        y_axis_units (str): Y-axis units: ``'meters'``, ``'range_sample'``, or
            ``'bins'`` (MVBS only).
        meters_per_second (float, optional): Speed for ``'meters'`` x-axis
            (auto-calculated from GPS if ``echodata`` provided).
        echodata (xarray.Dataset, optional): Original echodata for GPS-based
            calculations.
        y_to_x_aspect_ratio_override (float, optional): Override aspect ratio.
        overlay_lines (list, optional): List of line specification dicts for
            overlay lines.
        
    Examples:
        >>> # Basic usage - regular Sv
        >>> plot_sv_echogram(ds_Sv)
        
        >>> # Basic usage - MVBS
        >>> plot_sv_echogram(ds_Sv_mvbs, ds_Sv_original=ds_Sv)
        
        >>> # With custom depth and color range
        >>> plot_sv_echogram(ds_Sv, min_depth=10, max_depth=100,
        ...                  sv_vmin=-70, sv_vmax=-30)
        
        >>> # MVBS with custom ping range (in original Sv indices)
        >>> plot_sv_echogram(ds_Sv_mvbs, ds_Sv_original=ds_Sv,
        ...                  ping_min=100, ping_max=800)
        
        >>> # With distance on x-axis
        >>> plot_sv_echogram(ds_Sv, x_axis_units='meters', echodata=ed)
        
        >>> # MVBS with bin axes
        >>> plot_sv_echogram(ds_Sv_mvbs, ds_Sv_original=ds_Sv,
        ...                  x_axis_units='bins', y_axis_units='bins')
    """
    # Detect MVBS format
    is_mvbs = putils.is_mvbs_dataset(ds_Sv)
    
    # Validation for MVBS data
    if is_mvbs:
        if ds_Sv_original is None:
            raise ValueError(
                "Detected MVBS data format. The 'ds_Sv_original' parameter is required "
                "for MVBS plotting to properly convert ping indices. Please provide the "
                "original Sv dataset."
            )
        logger.info("Detected MVBS data format")
        logger.info("  Using ds_Sv_original for ping range conversion")
        
        # Validate axis units for MVBS
        if x_axis_units not in ['seconds', 'pings', 'bins', 'meters']:
            raise ValueError(f"Invalid x_axis_units '{x_axis_units}' for MVBS data. "
                           f"Valid options: ['seconds', 'pings', 'bins', 'meters']")
        if y_axis_units not in ['meters', 'range_sample', 'bins']:
            raise ValueError(f"Invalid y_axis_units '{y_axis_units}' for MVBS data. "
                           f"Valid options: ['meters', 'range_sample', 'bins']")
    else:
        logger.info("Detected regular Sv data format")
        
        # Validation for regular Sv - bins not allowed
        if x_axis_units == 'bins':
            raise ValueError("x_axis_units='bins' is only valid for MVBS data")
        if y_axis_units == 'bins':
            raise ValueError("y_axis_units='bins' is only valid for MVBS data")
        
        # ds_Sv_original not needed for regular Sv, but warn if provided
        if ds_Sv_original is not None:
            logger.info("  Note: ds_Sv_original provided but not needed for regular Sv data")
    
    # Auto-detect frequency_nominal
    if frequency_nominal is None:
        if 'frequency_nominal' in ds_Sv:
            frequency_nominal = ds_Sv['frequency_nominal'].values
        elif is_mvbs and ds_Sv_original is not None and 'frequency_nominal' in ds_Sv_original.get("Environment", {}):
            frequency_nominal = ds_Sv_original["Environment"]['frequency_nominal'].values
        else:
            raise ValueError("frequency_nominal not found in dataset and not provided")
    
    # Call main processing function
    return plot_processed_echogram_main(
        ds_Sv=ds_Sv,
        frequency_nominal=frequency_nominal,
        max_depth=max_depth,
        min_depth=min_depth,
        ping_min=ping_min,
        ping_max=ping_max,
        sv_vmin=sv_vmin,
        sv_vmax=sv_vmax,
        sv_cmap=sv_cmap,
        ds_Sv_original=ds_Sv_original if is_mvbs else None,
        use_corrected_Sv=use_corrected_Sv,
        x_axis_units=x_axis_units,
        y_axis_units=y_axis_units,
        meters_per_second=meters_per_second,
        y_to_x_aspect_ratio_override=y_to_x_aspect_ratio_override,
        ml_vmin=None,
        ml_vmax=None,
        echodata=echodata,
        ml_dataset_name=None,
        ml_specific_data_name=None,
        overlay_lines=overlay_lines
    )


def plot_flattened_data_echogram(ds_ml, ml_dataset_name, ds_Sv_original=None, frequency_nominal=None,
                     ml_specific_data_name=None, min_depth=None, max_depth=None,
                     ping_min=0, ping_max=None, ml_vmin=None, ml_vmax=None,
                     sv_cmap='viridis', x_axis_units='seconds', y_axis_units='meters',
                     meters_per_second=None, echodata=None,
                     y_to_x_aspect_ratio_override=None, overlay_lines=None
                     ):
    """
    Plot ML-processed echogram data (regular Sv-derived or MVBS-derived).
    
    Automatically detects MVBS format and handles both regular Sv and MVBS-based ML data.
    For MVBS-derived ML data, ``ds_Sv_original`` is required for proper ping range
    conversion.
    
    Args:
        ds_ml (xarray.Dataset): Dataset containing ML data (must have flattened
            ML features).
        ml_dataset_name (str): Name of the ML dataset variable
            (e.g., ``'ml_data_clean'``).
        ds_Sv_original (xarray.Dataset, optional): Original Sv dataset (required
            for MVBS-derived ML data to convert ping indices).
        frequency_nominal (array-like, optional): Nominal frequencies.
            Auto-detected from ML features if ``None``.
        ml_specific_data_name (str, optional): Specific ML data name within
            dataset (e.g., ``'normalized'``).
        min_depth (float, optional): Minimum depth in meters.
            Auto-detected from data if ``None``.
        max_depth (float, optional): Maximum depth in meters.
            Auto-detected from data if ``None``.
        ping_min (int, optional): Start ping index. Defaults to 0.
            For MVBS-derived: in ORIGINAL Sv indices (converted to MVBS bins).
        ping_max (int, optional): End ping index. Defaults to last ping.
        ml_vmin (float, optional): Minimum color scale limit.
            Auto-detected from data if ``None``.
        ml_vmax (float, optional): Maximum color scale limit.
            Auto-detected from data if ``None``.
        sv_cmap (str): Colormap name. Defaults to ``'viridis'``.
        x_axis_units (str): X-axis units: ``'seconds'`` (default), ``'pings'``,
            ``'bins'`` (MVBS only), or ``'meters'``.
        y_axis_units (str): Y-axis units: ``'meters'``, ``'range_sample'``, or
            ``'bins'`` (MVBS only).
        meters_per_second (float, optional): Speed for ``'meters'`` x-axis
            (auto-calculated from GPS if ``echodata`` provided).
        echodata (xarray.Dataset, optional): Original echodata for GPS-based
            calculations.
        y_to_x_aspect_ratio_override (float, optional): Override aspect ratio.
        overlay_lines (list, optional): List of line specification dicts for
            overlay lines.
        
    Examples:
        >>> # Basic usage - regular Sv-derived ML
        >>> plot_flattened_data_echogram(ds_ml, 'ml_data_clean')
        
        >>> # Basic usage - MVBS-derived ML
        >>> plot_flattened_data_echogram(ds_ml_mvbs, 'ml_data_clean',
        ...                             ds_Sv_original=ds_Sv)
        
        >>> # With specific ML data and custom color range
        >>> plot_flattened_data_echogram(ds_ml, 'ml_data_clean',
        ...                             ml_specific_data_name='normalized',
        ...                             ml_vmin=-2, ml_vmax=2)
        
        >>> # MVBS-derived with custom ping range (in original Sv indices)
        >>> plot_flattened_data_echogram(ds_ml_mvbs, 'ml_data_clean',
        ...                             ds_Sv_original=ds_Sv,
        ...                             ping_min=100, ping_max=800)
        
        >>> # MVBS-derived with bin axes
        >>> plot_flattened_data_echogram(ds_ml_mvbs, 'ml_data_clean',
        ...                             ds_Sv_original=ds_Sv,
        ...                             x_axis_units='bins', y_axis_units='bins')
        
        >>> # With custom depth range
        >>> plot_flattened_data_echogram(ds_ml, 'features',
        ...                             min_depth=10, max_depth=100)
    """
    # Validation
    if ml_dataset_name is None:
        raise ValueError("ml_dataset_name is required for ML data plotting")
    
    # Construct ML variable name
    if ml_specific_data_name and ml_specific_data_name != ml_dataset_name:
        ml_var = f"{ml_dataset_name}_{ml_specific_data_name}"
    else:
        ml_var = ml_dataset_name
    
    if ml_var not in ds_ml:
        raise ValueError(f"ML variable '{ml_var}' not found in dataset. Available: {list(ds_ml.data_vars)}")
    
    # Detect MVBS format
    is_mvbs = putils.is_mvbs_dataset(ds_ml)
    
    if is_mvbs:
        if ds_Sv_original is None:
            raise ValueError(
                "Detected MVBS-derived ML data format. The 'ds_Sv_original' parameter is required "
                "for MVBS-based ML plotting to properly convert ping indices. Please provide the "
                "original Sv dataset."
            )
        logger.info("Plotting ML echogram from MVBS structure...")
        logger.info("  ML variable: %s", ml_var)
        logger.info("  Using original Sv dataset for ping range conversion")
        
        if x_axis_units not in ['seconds', 'pings', 'bins', 'meters']:
            raise ValueError(f"Invalid x_axis_units '{x_axis_units}' for MVBS-derived ML data. "
                           f"Valid options: ['seconds', 'pings', 'bins', 'meters']")
        if y_axis_units not in ['meters', 'range_sample', 'bins']:
            raise ValueError(f"Invalid y_axis_units '{y_axis_units}' for MVBS-derived ML data. "
                           f"Valid options: ['meters', 'range_sample', 'bins']")
    else:
        logger.info("Plotting ML echogram from regular Sv structure...")
        logger.info("  ML variable: %s", ml_var)
        
        if x_axis_units == 'bins':
            raise ValueError("x_axis_units='bins' is only valid for MVBS-derived ML data")
        if y_axis_units == 'bins':
            raise ValueError("y_axis_units='bins' is only valid for MVBS-derived ML data")
        
        if ds_Sv_original is not None:
            logger.info("  Note: ds_Sv_original provided but not needed for regular Sv-derived ML data")
    
    if frequency_nominal is None:
        if 'frequency_nominal' in ds_ml.coords:
            frequency_nominal = ds_ml['frequency_nominal'].values
            logger.info("  Auto-detected frequencies: %s", [f'{int(f/1000)} kHz' for f in frequency_nominal])
        elif is_mvbs and ds_Sv_original is not None and 'frequency_nominal' in ds_Sv_original.get("Environment", {}):
            frequency_nominal = ds_Sv_original["Environment"]['frequency_nominal'].values
            logger.info("  Auto-detected frequencies from original: %s", [f'{int(f/1000)} kHz' for f in frequency_nominal])
        else:
            logger.info("  Frequency labels will be auto-detected from ML features")
    
    return plot_processed_echogram_main(
        ds_Sv=ds_ml,
        frequency_nominal=frequency_nominal,
        max_depth=max_depth,
        min_depth=min_depth,
        ping_min=ping_min,
        ping_max=ping_max,
        sv_vmin=ml_vmin,
        sv_vmax=ml_vmax,
        sv_cmap=sv_cmap,
        ds_Sv_original=ds_Sv_original if is_mvbs else None,
        use_corrected_Sv=False,
        x_axis_units=x_axis_units,
        y_axis_units=y_axis_units,
        meters_per_second=meters_per_second,
        y_to_x_aspect_ratio_override=y_to_x_aspect_ratio_override,
        ml_vmin=ml_vmin,
        ml_vmax=ml_vmax,
        echodata=echodata,
        ml_dataset_name=ml_dataset_name,
        ml_specific_data_name=ml_specific_data_name,
        overlay_lines=overlay_lines
    )





