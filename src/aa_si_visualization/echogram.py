"""Core echogram plotting functions for the AA-SI visualization package.

Provides a pipeline-based architecture for rendering Sv, MVBS, ML-feature,
and cluster-result echograms with configurable axes, colormaps, and overlay
lines.  The public API consists of:

- :func:`plot_sv_echogram`: regular Sv or MVBS data
- :func:`plot_flattened_data_echogram`: ML-processed data
- :func:`plot_cluster_echogram`: clustering results
- :func:`plot_processed_echogram_main`: low-level orchestrator used by the above
"""

import numpy as np
from aa_si_utils import utils
from .echogram_handlers import create_handler


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
    
    print(f"Plotting ML data from variable: {ml_data_variable}")
    
    # Check if already gridded
    ml_data = ds_Sv[ml_data_variable]
    data_dims = set(ml_data.dims)
    is_already_gridded = (
        'ping_time' in data_dims and 
        len(data_dims) == 2 and
        ('range_sample' in data_dims or 'echo_range' in data_dims)
    )
    
    if is_already_gridded:
        print("  Detected already-gridded single-channel data")
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
    
    print(f"  Regridded ML data shape: {regridded_data.shape}")
    
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
    
    print(f"  ML data features: {freq_labels}")
    
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
        print("WARNING: No valid cluster data found")
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
    
    # Print analysis
    print(f"Cluster analysis:")
    print(f"  Cluster range: {min_label} to {max_label}")
    print(f"  Number of clusters: {num_clusters}")
    for label in unique_labels:
        count = label_counts[int(label)]
        percentage = label_percentages[int(label)]
        label_name = 'Noise' if label == -1 else f'Cluster {int(label)}'
        print(f"  {label_name}: {count:,} samples ({percentage:.1f}%)")
    
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
    
    print("Checking for all-NaN frequencies...")
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
                print(f"  Skipping {freq_label}: all data is NaN")
        except Exception as e:
            print(f"  Error checking frequency {freq_idx}: {e}")
            valid_channels.append(freq_idx)  # Include on error to avoid breaking
            if params['freq_labels']:
                valid_freq_labels.append(params['freq_labels'][freq_idx])
    
    if len(valid_channels) == 0:
        raise ValueError("All frequencies contain only NaN data - nothing to plot")
    
    if len(valid_channels) < params['n_channels']:
        print(f"  Reduced from {params['n_channels']} to {len(valid_channels)} valid frequencies")
        params['n_channels'] = len(valid_channels)
        params['freq_labels'] = valid_freq_labels
        params['valid_channel_indices'] = valid_channels
    else:
        print(f"  All {params['n_channels']} frequencies contain valid data")
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
    
    print(f"Using {len(cluster_colors)} distinct colors for clusters")
    
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
        print("Auto-detecting depth range from data...")
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
                    print(f"Warning: Could not auto-detect depth range: {e}")
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
        print(f"Using depth range: {min_depth:.1f}m to {max_depth:.1f}m")
    
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


def _calculate_x_axis_seconds(handler, ping_times, ping_range, is_mvbs):
    """
    Calculate x-axis extent in seconds from start.
    
    Args:
        handler: EchogramDataHandler instance
        ping_times: Array of ping times
        ping_range: Tuple of (ping_min, ping_max)
        is_mvbs: Boolean indicating MVBS structure
        
    Returns:
        tuple: (x_min, x_max, x_label)
    """
    ping_min, ping_max = ping_range
    
    # Calculate seconds from start for both MVBS and regular Sv
    start_time_seconds = (ping_times[ping_min] - ping_times[0]) / np.timedelta64(1, 's')
    end_time_seconds = (ping_times[ping_max] - ping_times[0]) / np.timedelta64(1, 's')
    
    return (start_time_seconds, end_time_seconds, 'Time (seconds from start)')


def _calculate_x_axis_pings(handler, ping_times, ping_range, is_mvbs):
    """
    Calculate x-axis extent in ping numbers or bin indices.
    
    Args:
        handler: EchogramDataHandler instance
        ping_times: Array of ping times
        ping_range: Tuple of (ping_min, ping_max)
        is_mvbs: Boolean indicating MVBS structure
        
    Returns:
        tuple: (x_min, x_max, x_label)
    """
    ping_min, ping_max = ping_range
    
    if is_mvbs:
        original_ping_min = handler.ping_min if hasattr(handler, 'ping_min') else ping_min
        original_ping_max = handler.ping_max if hasattr(handler, 'ping_max') else ping_max
        x_label = f'MVBS Bin (pings {original_ping_min} to {original_ping_max})'
    else:
        x_label = 'Ping Number'
    
    return (ping_min, ping_max, x_label)


def _calculate_x_axis_bins(handler, ping_times, ping_range, is_mvbs):
    """
    Calculate x-axis extent in MVBS time bins.
    
    Args:
        handler: EchogramDataHandler instance
        ping_times: Array of ping times
        ping_range: Tuple of (ping_min, ping_max)
        is_mvbs: Boolean indicating MVBS structure
        
    Returns:
        tuple: (x_min, x_max, x_label)
        
    Raises:
        ValueError: If data is not MVBS structured
    """
    if not is_mvbs:
        raise ValueError("x_axis_units='bins' is only valid for MVBS data")
    
    ping_min, ping_max = ping_range
    return (ping_min, ping_max, 'MVBS Time Bins')


def _calculate_x_axis_meters(handler, ping_times, ping_range, is_mvbs, meters_per_second, echodata):
    """
    Calculate x-axis extent in meters (distance traveled).
    
    Args:
        handler: EchogramDataHandler instance
        ping_times: Array of ping times
        ping_range: Tuple of (ping_min, ping_max)
        is_mvbs: Boolean indicating MVBS structure
        meters_per_second: Speed in m/s (optional, will calculate from GPS if None)
        echodata: Original echodata for GPS calculations (required if meters_per_second is None)
        
    Returns:
        tuple: (x_min, x_max, x_label)
        
    Raises:
        ValueError: If meters_per_second is None and echodata is not provided
    """
    ping_min, ping_max = ping_range
    
    # Calculate time in seconds first
    start_time_seconds = (ping_times[ping_min] - ping_times[0]) / np.timedelta64(1, 's')
    end_time_seconds = (ping_times[ping_max] - ping_times[0]) / np.timedelta64(1, 's')
    
    # Calculate or get meters_per_second
    if meters_per_second is None:
        if echodata is None:
            raise ValueError("echodata parameter is required when meters_per_second is not provided and x_axis_units='meters'")
        
        original_ping_min = handler.ping_min if hasattr(handler, 'ping_min') else ping_min
        original_ping_max = handler.ping_max if hasattr(handler, 'ping_max') else ping_max
        
        print("Using GPS calculation for meters_per_second...")
        # TODO: Verify index alignment between ping_time and Platform lat/lon timestamps
        start_lat = echodata["Platform"]["latitude"][original_ping_min]
        start_lon = echodata["Platform"]["longitude"][original_ping_min]
        end_lat = echodata["Platform"]["latitude"][original_ping_max]
        end_lon = echodata["Platform"]["longitude"][original_ping_max]
        
        # Calculate distance using haversine formula
        distance_meters = utils.haversine_distance(start_lat, start_lon, end_lat, end_lon)
        
        duration_seconds = end_time_seconds - start_time_seconds
        
        if duration_seconds > 0:
            meters_per_second = distance_meters / duration_seconds
        else:
            print("Warning: Zero duration detected, using default speed")
            meters_per_second = 5.0  # Default 5 m/s for marine vessels
        
        print(f"GPS calculation details:")
        print(f"  Start: lat={start_lat:.6f}, lon={start_lon:.6f}")
        print(f"  End: lat={end_lat:.6f}, lon={end_lon:.6f}")
        print(f"  Distance: {distance_meters:.0f} m")
        print(f"  Duration: {duration_seconds:.1f} seconds ({duration_seconds/3600:.2f} hours)")
        print(f"  Calculated speed: {meters_per_second:.2f} m/s ({meters_per_second*3.6:.1f} km/h)")
    
    x_min = start_time_seconds * meters_per_second
    x_max = end_time_seconds * meters_per_second
    
    return (x_min, x_max, 'Distance (meters)')


def _calculate_x_axis_with_handler(handler, ping_times, ping_range, x_axis_units, meters_per_second, echodata):
    """
    Dispatcher function to calculate x-axis extent based on units.
    Uses strategy pattern to delegate to appropriate calculator function.
    
    Args:
        handler: EchogramDataHandler instance
        ping_times: Array of ping times
        ping_range: Tuple of (ping_min, ping_max)
        x_axis_units: Units for x-axis ('seconds', 'pings', 'bins', 'meters')
        meters_per_second: Speed for meters calculation (optional)
        echodata: Original echodata for GPS calculation (optional)
        
    Returns:
        tuple: (x_min, x_max, x_label)
        
    Raises:
        ValueError: If x_axis_units is invalid for the data type
    """
    # Create lookup dict for calculators
    calculators = {
        'seconds': _calculate_x_axis_seconds,
        'pings': _calculate_x_axis_pings,
        'bins': _calculate_x_axis_bins,
        'meters': _calculate_x_axis_meters
    }
    
    # Validate unit
    if x_axis_units not in calculators:
        valid_units = ['seconds', 'pings', 'meters']
        if handler.is_mvbs_structured():
            valid_units.append('bins')
        raise ValueError(f"Invalid x_axis_units '{x_axis_units}'. Valid options: {valid_units}")
    
    # Get handler state
    is_mvbs = handler.is_mvbs_structured()
    
    # Call appropriate calculator
    calculator = calculators[x_axis_units]
    
    if x_axis_units == 'meters':
        return calculator(handler, ping_times, ping_range, is_mvbs, meters_per_second, echodata)
    else:
        return calculator(handler, ping_times, ping_range, is_mvbs)


def _calculate_y_axis_extent_local(min_depth_shown, max_depth_shown, min_depth_index, max_depth_index, y_axis_units, data_type=None):
    """Calculate y-axis extent and label string based on the requested units.

    Args:
        min_depth_shown: Minimum depth value in meters.
        max_depth_shown: Maximum depth value in meters.
        min_depth_index: Minimum depth index.
        max_depth_index: Maximum depth index
        y_axis_units: Units for y-axis ('meters', 'range_sample', 'bins')
        data_type: Type of data (for validation, optional)
        
    Returns:
        tuple: (y_extent_min, y_extent_max, y_label)
        
    Raises:
        ValueError: If y_axis_units is invalid for the data type
    """
    if y_axis_units == 'meters':
        y_extent_min = min_depth_shown
        y_extent_max = max_depth_shown
        y_label = 'Depth (m)'
    elif y_axis_units == 'range_sample':
        y_extent_min = min_depth_index
        y_extent_max = max_depth_index
        y_label = 'Range Sample Index'
    elif y_axis_units == 'bins':
        # Allow 'bins' for MVBS and Cluster-MVBS data
        if data_type and data_type not in ['MVBS', 'ML-MVBS', 'Cluster-MVBS']:
            raise ValueError("y_axis_units='bins' is only valid for MVBS data")
        y_extent_min = min_depth_index
        y_extent_max = max_depth_index
        y_label = 'MVBS Depth Bins'
    else:
        valid_y_units = ['meters', 'range_sample']
        if data_type and data_type in ['MVBS', 'ML-MVBS', 'Cluster-MVBS']:
            valid_y_units.append('bins')
        raise ValueError(f"Invalid y_axis_units '{y_axis_units}'. Use {valid_y_units}")
    
    return y_extent_min, y_extent_max, y_label


def _calculate_plot_dimensions_and_aspect_local(x_extent_min, x_extent_max, y_extent_min, y_extent_max, y_to_x_aspect_ratio_override=None):
    """Calculate plot extent, aspect ratio, and figure-size multipliers.

    Args:
        x_extent_min: Minimum x-axis extent
        x_extent_max: Maximum x-axis extent
        y_extent_min: Minimum y-axis extent
        y_extent_max: Maximum y-axis extent
        y_to_x_aspect_ratio_override: Optional override for aspect ratio
        
    Returns:
        tuple: (extent, aspect_ratio, width_multiplier, height_multiplier)
    """
    # imshow expects [left, right, bottom, top]
    extent = [x_extent_min, x_extent_max, y_extent_max, y_extent_min]
    
    # Calculate 1:1 aspect ratio in actual units
    x_range = abs(x_extent_max - x_extent_min)
    y_range = abs(y_extent_max - y_extent_min)
    aspect_ratio = y_range / x_range
    
    # Override aspect ratio if specified
    if y_to_x_aspect_ratio_override is not None:
        aspect_ratio = (1 / y_to_x_aspect_ratio_override * (1 / aspect_ratio))

    # Calculate figure size multipliers
    width_multiplier = 1
    height_multiplier = 1

    if aspect_ratio < 1:
        width_multiplier = min(10, 1 / aspect_ratio)
    else:
        height_multiplier = min(3, aspect_ratio)
    
    return extent, aspect_ratio, width_multiplier, height_multiplier


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
    
    # Calculate x-axis extent using handler-aware calculator
    x_min, x_max, x_label = _calculate_x_axis_with_handler(
        handler, 
        ping_times, 
        (ranges['ping']['min'], ranges['ping']['max']),
        params['x_axis_units'],
        params['meters_per_second'],
        echodata
    )
    
    # Calculate y-axis extent using local helper
    y_min, y_max, y_label = _calculate_y_axis_extent_local(
        ranges['depth']['min_shown'],
        ranges['depth']['max_shown'],
        ranges['depth']['min_index'],
        ranges['depth']['max_index'],
        params['y_axis_units'],
        handler.detect_structure()['type']  # Pass data type for validation
    )
    
    # Calculate plot dimensions and aspect ratio using local helper
    extent, aspect_ratio, width_mult, height_mult = _calculate_plot_dimensions_and_aspect_local(
        x_min, x_max, y_min, y_max, params['y_to_x_aspect_ratio_override']
    )
    
    # Print diagnostics
    data_type = handler.detect_structure()['type']
    print(f"{data_type} Echogram dimensions:")
    
    if handler.is_mvbs_structured():
        print(f"  Original ping range requested: {params['ping_min']} to {params['ping_max']}")
        print(f"  MVBS ping indices used: {ranges['ping']['min']} to {ranges['ping']['max']}")
        print(f"  MVBS time range: {ping_times[ranges['ping']['min']]} to {ping_times[ranges['ping']['max']]}")
        print(f"  MVBS depth range: {ranges['depth']['min_shown']} to {ranges['depth']['max_shown']}")
    elif data_type.startswith("ML"):
        print(f"  ML data regridded from flattened format")
        print(f"  Ping range: {ranges['ping']['min']} to {ranges['ping']['max']}")
    
    print(f"  X-axis ({params['x_axis_units']}): {x_min:.1f} to {x_max:.1f}")
    print(f"  Y-axis ({params['y_axis_units']}): {y_min:.1f} to {y_max:.1f}")
    print(f"  Features: {params['n_channels']}")
    print(f"  Depth Range: {ranges['depth']['min_shown']:.1f}m to {ranges['depth']['max_shown']:.1f}m")
    print(f"  Aspect ratio (1:1 in specified units): {aspect_ratio:.3f}")
    
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
        print(f"WARNING: Overlay variable '{var_name}' not found in dataset")
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
        print(f"WARNING: No valid data for overlay variable '{var_name}' in plot range")
        return
    
    # Get style from line_spec
    plot_kwargs = line_spec.get('style', {})
    
    # Apply default styling for visibility on black background
    plot_kwargs.setdefault('color', 'cyan')
    plot_kwargs.setdefault('linewidth', 2.5)
    plot_kwargs.setdefault('alpha', 0.9)
    plot_kwargs.setdefault('zorder', 10)
    
    ax.plot(x_coords, y_coords, **plot_kwargs)


def _create_plot(handler, axes_config, ranges, params, ml_info=None, cluster_info=None, cluster_colors=None, overlay_lines=None):
    """
    Create the final echogram plot using matplotlib.
    Handles both multi-frequency continuous data and single-channel categorical cluster data.
    
    Args:
        handler: EchogramDataHandler instance
        axes_config: Axis configuration from _calculate_axes()
        ranges: Range dict from _calculate_ranges()
        params: Parameter dict from _setup_parameters()
        ml_info: ML info dict from _prepare_ml_data() (None for regular Sv)
        cluster_info: Cluster info dict from _prepare_ml_data() (None for non-cluster data)
    """
    import matplotlib.pyplot as plt
    
    # Cluster mode detection
    is_cluster_mode = (cluster_info is not None)
    
    # Set black background style and create figure
    plt.style.use('dark_background')
    
    if is_cluster_mode:
        # Single panel for cluster data
        fig_width = 24 * axes_config['width_multiplier']
        fig_height = 12 * axes_config['height_multiplier']
    else:
        # Multi-panel for frequency data
        fig_width = 24 * axes_config['width_multiplier']
        fig_height = 12 * params['n_channels'] * axes_config['height_multiplier']
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('black')
    
    # Determine data type for title
    data_type = handler.detect_structure()['type']

    axes_list = [] 

    if is_cluster_mode:
        # Cluster mode
        print("Creating cluster plot...")
        
        ax = plt.subplot(1, 1, 1)
        ax.set_facecolor('black')
        axes_list.append(ax)
        # Get cluster data slice
        ping_range = (ranges['ping']['min'], ranges['ping']['max'])
        depth_range = (ranges['depth']['min_index'], ranges['depth']['max_index'])
        cluster_data = handler.slice_data_for_frequency(0, ping_range, depth_range)
        
        # Create cluster colormap
        cmap, norm, tick_positions, tick_labels = _create_cluster_colormap(cluster_info, base_colors=cluster_colors)
        
        im = ax.imshow(
            cluster_data.T,  # Transpose to have range on y-axis
            aspect=axes_config['aspect_ratio'],
            cmap=cmap,
            norm=norm,
            extent=axes_config['extent'],
            interpolation='nearest'  # Sharp boundaries for clusters
        )
        
        # Set title and labels
        title = f'Cluster Analysis - Multi-frequency Classification'
        ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
        ax.set_ylabel(axes_config['y']['label'], fontsize=12, color='white')
        ax.set_xlabel(axes_config['x']['label'], fontsize=10, color='white')
        ax.tick_params(colors='white')
        
        # Adjust layout with space for vertical colorbar
        plt.tight_layout(pad=3.0)
        fig.subplots_adjust(right=0.85, top=0.92)
        
        # Create vertical colorbar on right side
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        cbar_ax.set_facecolor('black')
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Cluster ID', fontsize=12, color='white')
        cbar.ax.tick_params(colors='white')
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
        
        plt.suptitle('Cluster Analysis - Multi-frequency Acoustic Backscatter Classification',
                    fontsize=16, fontweight='bold', color='white', y=0.96)
        
    else:
        # multi frequency mode: Multiple panels with continuous colormap
        print("Creating multi-frequency plot...")
        
        # Store the image for shared colorbar
        sv_image = None
        
        # Get frequency labels
        freq_labels = params['freq_labels']
        n_channels = params['n_channels']

        # Loop through frequencies and create subplots
        for plot_idx, freq_idx in enumerate(params.get('valid_channel_indices', range(params['n_channels']))):
            # Set black background for each subplot
            ax = plt.subplot(n_channels, 1, plot_idx + 1)
            axes_list.append(ax)
            ax.set_facecolor('black')
            
            # Get frequency label
            freq_label = freq_labels[plot_idx] if freq_labels else f"Feature {plot_idx}"
            
            # Get data slice from handler
            ping_range = (ranges['ping']['min'], ranges['ping']['max'])
            depth_range = (ranges['depth']['min_index'], ranges['depth']['max_index'])
            sv_data = handler.slice_data_for_frequency(freq_idx, ping_range, depth_range)
    
            # Create the echogram using imshow
            im = ax.imshow(
                sv_data.T,  # Transpose to have range on y-axis
                aspect=axes_config['aspect_ratio'],
                vmin=params['sv_vmin'],
                vmax=params['sv_vmax'],
                cmap=params['sv_cmap'],
                extent=axes_config['extent']
            )
            
            # Store first image for shared colorbar
            if freq_idx == 0:
                sv_image = im
            
            # Customize the plot with white text for black background
            if data_type.startswith("ML"):
                title_suffix = "ML Data"
            else:
                title_suffix = f'{data_type} Echogram'
            
            ax.set_title(f'{freq_label} - {title_suffix}', 
                        fontsize=12, fontweight='bold', color='white', pad=20)
            ax.set_ylabel(axes_config['y']['label'], fontsize=10, color='white')
            ax.tick_params(colors='white')
            
            # Set x-label only on the bottom subplot
            if freq_idx == n_channels - 1:
                ax.set_xlabel(axes_config['x']['label'], fontsize=10, color='white')
        

                    
        # Adjust layout with more spacing for titles and between subplots
        plt.tight_layout(pad=3.0, h_pad=5.0, w_pad=2.0)
        
        # Create space for colorbars at the bottom and more space below title
        fig.subplots_adjust(bottom=0.12, top=0.92)
        
        # Add shared colorbar at the bottom
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
        cbar_ax.set_facecolor('black')
        cbar = fig.colorbar(sv_image, cax=cbar_ax, orientation='horizontal')
        
        # Set colorbar label based on data type
        if data_type.startswith("ML"):
            cbar_label = 'ML Feature Value'
        else:
            cbar_label = 'Sv (dB)'
        
        cbar.set_label(cbar_label, fontsize=10, color='white')
        cbar.ax.tick_params(colors='white')
        
        # Set plot title based on data type
        if data_type.startswith("ML"):
            plot_title = 'ML Data Echogram'
        else:
            plot_title = f'{data_type} Echogram - {"Gridded " if data_type == "MVBS" else ""}Calibrated Data'
        
        plt.suptitle(plot_title, fontsize=16, fontweight='bold', y=0.96, color='white')
    
    if overlay_lines is not None:
        ping_min = ranges['ping']['min']
        ping_max = ranges['ping']['max']
        
        for ax in axes_list:
            for line_spec in overlay_lines:
                _add_overlay_line(ax, handler.dataset, line_spec, ping_min, ping_max, axes_config)


    plt.show()
    
    # Reset style to default after plotting
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
    
    print("Setting up parameters...")
    params = _setup_parameters(
        ds_Sv, frequency_nominal, max_depth, min_depth, ping_min, ping_max,
        sv_vmin, sv_vmax, sv_cmap, ds_Sv_original, use_corrected_Sv,
        x_axis_units, y_axis_units, meters_per_second, y_to_x_aspect_ratio_override,
        ml_vmin, ml_vmax, echodata, ml_dataset_name, ml_specific_data_name
    )
    print(f"  Ping range: {params['ping_min']} to {params['ping_max']}")
    if params['ml_data_variable']:
        print(f"  ML variable: {params['ml_data_variable']}")
    
    print("Preparing data...")
    ds_Sv_plot, ml_info, sv_variable_name, cluster_info = _prepare_ml_data(ds_Sv, params)
    if cluster_info:
        print(f"  Cluster data detected: {cluster_info['num_clusters']} clusters")
    elif ml_info:
        print(f"  ML data regridded: {ml_info['n_features']} features")
        print(f"  Color scale: {params['sv_vmin']:.2f} to {params['sv_vmax']:.2f}")
    else:
        print(f"  Using variable: {sv_variable_name}")
    
    params = _filter_nan_frequencies(ds_Sv_plot, sv_variable_name, params, cluster_info)

    print("Creating data handler...")
    handler = create_handler(ds_Sv_plot, sv_variable_name, params['ml_data_variable'])
    data_type = handler.detect_structure()['type']
    print(f"  Handler created: {handler.__class__.__name__}")
    print(f"  Data type: {data_type}")
    
    print("Calculating ranges...")
    ranges = _calculate_ranges(handler, params, ds_Sv_original)
    print(f"  Depth range: {ranges['depth']['min_shown']:.1f}m to {ranges['depth']['max_shown']:.1f}m")
    print(f"  Ping range: {ranges['ping']['min']} to {ranges['ping']['max']}")
    
    print("Calculating axes...")
    axes_config = _calculate_axes(handler, ranges, params, echodata)
    
    print("Creating plot...")
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
        print("Using provided gridded cluster data")
        ds_ml_ready[grid_result_name] = gridded_data
    
    if grid_result_name not in ds_ml_ready:
        if full_result_name in ds_ml_ready:
            print(f"Regridding {full_result_name} for visualization...")
            from aa_si_ml import ml
            gridded_results = ml.extract_ml_data_gridded(
                ds_ml_ready, 
                specific_data_name=specific_data_name,
                dataset_name=dataset_name,
                fill_value=np.nan,
                store_in_dataset=True  # This creates the _grid variable
            )
            print(f"  Regridded to shape: {gridded_results.shape}")
        else:
            raise ValueError(f"Neither {grid_result_name} nor {full_result_name} found in dataset")
    else:
        print(f"Using existing gridded results: {grid_result_name}")
    
    # Print data statistics
    cluster_data_var = ds_ml_ready[grid_result_name]
    total_values = cluster_data_var.size
    nan_count = np.sum(np.isnan(cluster_data_var.values))
    valid_count = total_values - nan_count
    
    print(f"Gridded cluster data analysis:")
    print(f"  Total values: {total_values:,}")
    print(f"  NaN values: {nan_count:,} ({nan_count/total_values*100:.1f}%)")
    print(f"  Valid values: {valid_count:,} ({valid_count/total_values*100:.1f}%)")
    
    # Detect MVBS structure
    is_mvbs_derived = False
    if 'echo_range' in ds_ml_ready.coords:
        echo_range_dims = ds_ml_ready['echo_range'].dims
        # MVBS has 1D echo_range, regular Sv has 2D or 3D
        is_mvbs_derived = len(echo_range_dims) == 1
    
    if is_mvbs_derived:
        print("Detected MVBS-derived cluster data")
        if ds_Sv_original is None:
            print("WARNING: ds_Sv_original not provided for MVBS data. Ping range conversion may not be accurate.")
    else:
        print("Detected regular Sv-derived cluster data")
    
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
    is_mvbs = False
    if 'echo_range' in ds_Sv.coords:
        echo_range_dims = ds_Sv['echo_range'].dims
        # MVBS has 1D echo_range, regular Sv has 2D or 3D
        is_mvbs = len(echo_range_dims) == 1
    
    # Validation for MVBS data
    if is_mvbs:
        if ds_Sv_original is None:
            raise ValueError(
                "Detected MVBS data format. The 'ds_Sv_original' parameter is required "
                "for MVBS plotting to properly convert ping indices. Please provide the "
                "original Sv dataset."
            )
        print("Detected MVBS data format")
        print("  Using ds_Sv_original for ping range conversion")
        
        # Validate axis units for MVBS
        if x_axis_units not in ['seconds', 'pings', 'bins', 'meters']:
            raise ValueError(f"Invalid x_axis_units '{x_axis_units}' for MVBS data. "
                           f"Valid options: ['seconds', 'pings', 'bins', 'meters']")
        if y_axis_units not in ['meters', 'range_sample', 'bins']:
            raise ValueError(f"Invalid y_axis_units '{y_axis_units}' for MVBS data. "
                           f"Valid options: ['meters', 'range_sample', 'bins']")
    else:
        print("Detected regular Sv data format")
        
        # Validation for regular Sv - bins not allowed
        if x_axis_units == 'bins':
            raise ValueError("x_axis_units='bins' is only valid for MVBS data")
        if y_axis_units == 'bins':
            raise ValueError("y_axis_units='bins' is only valid for MVBS data")
        
        # ds_Sv_original not needed for regular Sv, but warn if provided
        if ds_Sv_original is not None:
            print("  Note: ds_Sv_original provided but not needed for regular Sv data")
    
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
    is_mvbs = False
    if 'echo_range' in ds_ml.coords:
        echo_range_dims = ds_ml['echo_range'].dims
        # MVBS has 1D echo_range, regular Sv has 2D or 3D
        is_mvbs = len(echo_range_dims) == 1
    
    if is_mvbs:
        if ds_Sv_original is None:
            raise ValueError(
                "Detected MVBS-derived ML data format. The 'ds_Sv_original' parameter is required "
                "for MVBS-based ML plotting to properly convert ping indices. Please provide the "
                "original Sv dataset."
            )
        print(f"Plotting ML echogram from MVBS structure...")
        print(f"  ML variable: {ml_var}")
        print(f"  Using original Sv dataset for ping range conversion")
        
        if x_axis_units not in ['seconds', 'pings', 'bins', 'meters']:
            raise ValueError(f"Invalid x_axis_units '{x_axis_units}' for MVBS-derived ML data. "
                           f"Valid options: ['seconds', 'pings', 'bins', 'meters']")
        if y_axis_units not in ['meters', 'range_sample', 'bins']:
            raise ValueError(f"Invalid y_axis_units '{y_axis_units}' for MVBS-derived ML data. "
                           f"Valid options: ['meters', 'range_sample', 'bins']")
    else:
        print(f"Plotting ML echogram from regular Sv structure...")
        print(f"  ML variable: {ml_var}")
        
        if x_axis_units == 'bins':
            raise ValueError("x_axis_units='bins' is only valid for MVBS-derived ML data")
        if y_axis_units == 'bins':
            raise ValueError("y_axis_units='bins' is only valid for MVBS-derived ML data")
        
        if ds_Sv_original is not None:
            print("  Note: ds_Sv_original provided but not needed for regular Sv-derived ML data")
    
    if frequency_nominal is None:
        if 'frequency_nominal' in ds_ml.coords:
            frequency_nominal = ds_ml['frequency_nominal'].values
            print(f"  Auto-detected frequencies: {[f'{int(f/1000)} kHz' for f in frequency_nominal]}")
        elif is_mvbs and ds_Sv_original is not None and 'frequency_nominal' in ds_Sv_original.get("Environment", {}):
            frequency_nominal = ds_Sv_original["Environment"]['frequency_nominal'].values
            print(f"  Auto-detected frequencies from original: {[f'{int(f/1000)} kHz' for f in frequency_nominal]}")
        else:
            print(f"  Frequency labels will be auto-detected from ML features")
    
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





