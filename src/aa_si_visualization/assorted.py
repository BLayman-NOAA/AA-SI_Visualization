import echopype as ep 
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from aa_si_utils import utils


def sv_differences_echograms(ds_Sv_baseline, ds_Sv_calibrated, frequencies, max_depth=None, min_depth=None, ping_min=None, ping_max=None, sv_scale_min=-80, sv_scale_max=-20, sv_scale_diff_min=-15, sv_scale_diff_max=15, sv_cmap='viridis', diff_cmap='RdBu_r', x_axis_units='pings', y_axis_units='meters', meters_per_second=None, y_to_x_aspect_ratio_override=None):
    """
    
    Generates a multi-panel figure showing baseline Sv, calibrated Sv, and difference
    echograms for each frequency. Provides visual comparison of calibration effects
    across frequencies, depths, and ping times.
    
    Args:
        ds_Sv_baseline: Baseline Sv dataset
        ds_Sv_calibrated: Calibrated Sv dataset with applied parameters
        frequencies: Array of nominal frequencies for labeling
        max_depth (float, optional): Maximum depth for plotting in meters (default: auto-detect from data)
        min_depth (float, optional): Minimum depth for plotting in meters (default: auto-detect from data)
        ping_min (int, optional): Starting ping index for plotting (default: 0)
        ping_max (int, optional): Ending ping index for plotting (default: last ping)
        sv_scale_min (float): Minimum Sv value for color scale (default: -80)
        sv_scale_max (float): Maximum Sv value for color scale (default: -20)
        sv_scale_diff_min (float): Minimum difference value for color scale (default: -15)
        sv_scale_diff_max (float): Maximum difference value for color scale (default: 15)
        sv_cmap (str): Colormap for Sv data (default: 'viridis')
        diff_cmap (str): Colormap for difference data (default: 'RdBu_r')
        x_axis_units (str): X-axis units: 'pings' (default), 'seconds', or 'meters'
        y_axis_units (str): Y-axis units: 'meters' (default) or 'range_sample'
        meters_per_second (float, optional): Speed for converting time to distance (required for x_axis_units='meters')
    """
    # Set default ping range if not provided
    if ping_min is None:
        ping_min = 0
    if ping_max is None:
        ping_max = len(ds_Sv_baseline['ping_time']) - 1
    
    # Check for frequencies with all NaN data
    valid_freq_indices = []
    valid_frequencies = []
    for freq_idx, freq in enumerate(frequencies):
        baseline_data = ds_Sv_baseline['Sv'].isel(channel=freq_idx, ping_time=slice(ping_min, ping_max))
        if not np.all(np.isnan(baseline_data.values)):
            valid_freq_indices.append(freq_idx)
            valid_frequencies.append(freq)
        else:
            print(f"Warning: {int(freq/1000)} kHz has all NaN values - excluding from plot")

    if len(valid_freq_indices) == 0:
        print("Error: All frequencies have NaN data. Cannot create plot.")
        return

    # Update frequencies to only include valid ones
    frequencies = valid_frequencies

    # Set default depth range and calculate indices using helper function
    min_depth, max_depth, min_depth_index, max_depth_index, min_depth_shown, max_depth_shown = _setup_depth_range_and_indices(
        ds_Sv_baseline, min_depth, max_depth, ping_min, ping_max
    )

    # Get data dimensions and create range/time axes for plotting
    freq_labels = [f"{int(f/1000)} kHz" for f in frequencies] 

    # Calculate x-axis extent using helper function
    ping_times = ds_Sv_baseline['ping_time'].values
    x_extent_min, x_extent_max, x_label = _calculate_x_axis_extent(
        ping_times, ping_min, ping_max, x_axis_units, meters_per_second, None
    )
    
    # Calculate y-axis extent using helper function
    y_extent_min, y_extent_max, y_label = _calculate_y_axis_extent(
        min_depth_shown, max_depth_shown, min_depth_index, max_depth_index, y_axis_units
    )
    
    # Calculate plot dimensions and aspect ratio using helper function
    extent, aspect_ratio, width_multiplier, height_multiplier = _calculate_plot_dimensions_and_aspect(
        x_extent_min, x_extent_max, y_extent_min, y_extent_max, y_to_x_aspect_ratio_override
    )
    
    print(f"  X-axis ({x_axis_units}): {x_extent_min:.1f} to {x_extent_max:.1f}")
    print(f"  Y-axis ({y_axis_units}): {y_extent_min:.1f} to {y_extent_max:.1f}")
    print(f"  Depth Range: {min_depth_shown:.1f}m to {max_depth_shown:.1f}m")
    print(f"  Aspect ratio (1:1 in specified units): {aspect_ratio:.3f}")

    # Create figure with adjusted spacing for colorbar
    fig_width = 24 * width_multiplier
    fig_height = 24 * height_multiplier

    print(f"Figure size: {fig_width} x {fig_height}")   

    # Calculate differences
    sv_diff_data = ds_Sv_calibrated['Sv'] - ds_Sv_baseline['Sv']

    plt.style.use('dark_background')
    # Set up figure with multiple subplots
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Store images for shared colorbars
    sv_image = None
    diff_image = None

    num_freqs = len(freq_labels)    

    # Subplot layout: 3 columns (baseline, CAL file, difference) x  rows (frequencies)
    for plot_idx, (freq_idx, freq_label) in enumerate(zip(valid_freq_indices, freq_labels)):
    
        # Get data for this frequency
        baseline_data = ds_Sv_baseline['Sv'].isel(channel=freq_idx, ping_time=slice(ping_min, ping_max), range_sample=slice(min_depth_index, max_depth_index))
        cal_data = ds_Sv_calibrated['Sv'].isel(channel=freq_idx, ping_time=slice(ping_min, ping_max), range_sample=slice(min_depth_index, max_depth_index))
        diff_data = sv_diff_data.isel(channel=freq_idx, ping_time=slice(ping_min, ping_max), range_sample=slice(min_depth_index, max_depth_index))
        
        
        # Get data for this frequency
        baseline_data = ds_Sv_baseline['Sv'].isel(channel=freq_idx, ping_time=slice(ping_min, ping_max), range_sample=slice(min_depth_index, max_depth_index))
        cal_data = ds_Sv_calibrated['Sv'].isel(channel=freq_idx, ping_time=slice(ping_min, ping_max), range_sample=slice(min_depth_index, max_depth_index))
        diff_data = sv_diff_data.isel(channel=freq_idx, ping_time=slice(ping_min, ping_max), range_sample=slice(min_depth_index, max_depth_index))
        
        
        # Baseline echogram
        ax1 = plt.subplot(num_freqs, 3, plot_idx*3 + 1)
        im1 = ax1.imshow(baseline_data.T, aspect=aspect_ratio, 
                        vmin=sv_scale_min, vmax=sv_scale_max, cmap=sv_cmap,
                        extent=extent)
        ax1.set_title(f'{freq_label} - Baseline', fontsize=12, fontweight='bold', pad=20)
        ax1.set_ylabel(y_label, fontsize=10)
        ax1.set_xlabel(x_label, fontsize=10)
        
        # Store first Sv image for shared colorbar
        if plot_idx == 0:
            sv_image = im1

        # CAL file echogram
        ax2 = plt.subplot(num_freqs, 3, plot_idx*3 + 2)
        ax2.imshow(cal_data.T, aspect=aspect_ratio,
                        vmin=sv_scale_min, vmax=sv_scale_max, cmap=sv_cmap,
                        extent=extent)
        ax2.set_title(f'{freq_label} - CAL Report Calibration', fontsize=12, fontweight='bold', pad=20)
        # ax2.set_xlabel(x_label, fontsize=10)
        
        # Difference echogram
        ax3 = plt.subplot(num_freqs, 3, plot_idx*3 + 3)
        im3 = ax3.imshow(diff_data.T, aspect=aspect_ratio,
                        vmin=sv_scale_diff_min, vmax=sv_scale_diff_max, cmap=diff_cmap,
                        extent=extent)
        ax3.set_title(f'{freq_label} - Difference (CAL - Baseline)', fontsize=12, fontweight='bold', pad=20)
        # ax3.set_xlabel(x_label, fontsize=10)
        
        # Store first difference image for shared colorbar
        if plot_idx == 0:
            diff_image = im3

    # Adjust layout with more spacing for titles and between subplots
    plt.tight_layout(pad=3.0, h_pad=5.0, w_pad=2.0)
    
    # Create space for colorbars at the bottom and more space below title
    fig.subplots_adjust(bottom=0.12, top=0.92)
    
    # Add shared colorbars at the bottom
    # Sv colorbar (left side)
    cbar_ax1 = fig.add_axes([0.08, 0.05, 0.35, 0.02])  # [left, bottom, width, height]
    cbar1 = fig.colorbar(sv_image, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('Sv (dB)', fontsize=10)
    
    # Difference colorbar (right side)
    cbar_ax2 = fig.add_axes([0.57, 0.05, 0.35, 0.02])  # [left, bottom, width, height]
    cbar2 = fig.colorbar(diff_image, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Δ Sv (dB)', fontsize=10)
    
    plt.suptitle('Calibration Comparison: EchoPype Defaults vs CAL Report Parameters', 
                fontsize=16, fontweight='bold', y=0.96)
    plt.show()

# Helper functions

def _setup_depth_range_and_indices(dataset, min_depth, max_depth, ping_min, ping_max):
    """
    Helper function to set up depth range and calculate depth indices.
    
    Returns:
        tuple: (min_depth, max_depth, min_depth_index, max_depth_index, min_depth_shown, max_depth_shown)
    """
    # Set default depth range if not provided - find actual data range
    if min_depth is None or max_depth is None:
        print("Auto-detecting depth range from data...")
        auto_min_depth, auto_max_depth = utils.find_data_depth_range(
            dataset, ping_min, ping_max, channel=0
        )
        if min_depth is None:
            min_depth = auto_min_depth
        if max_depth is None:
            max_depth = auto_max_depth
        print(f"Using depth range: {min_depth:.1f}m to {max_depth:.1f}m")

    # Calculate depth indices
    min_depth_index = utils.get_closest_index_for_depth(dataset, min_depth)
    max_depth_index = utils.get_closest_index_for_depth(dataset, max_depth)

    # Get actual depths from echo_range
    actual_depths = dataset.echo_range.isel(channel=0, ping_time=0).values
    
    # Use actual depth range
    max_depth_shown = actual_depths[min(max_depth_index, len(actual_depths)-1)]
    min_depth_shown = actual_depths[min_depth_index]
    
    return min_depth, max_depth, min_depth_index, max_depth_index, min_depth_shown, max_depth_shown

def _calculate_x_axis_extent(ping_times, ping_min, ping_max, x_axis_units, meters_per_second=None, echodata=None):
    """
    Helper function to calculate x-axis extent and labels based on units.
    
    Returns:
        tuple: (x_extent_min, x_extent_max, x_label)
    """
    if x_axis_units == 'seconds':
        start_time_seconds = (ping_times[ping_min] - ping_times[0]) / np.timedelta64(1, 's')
        end_time_seconds = (ping_times[ping_max] - ping_times[0]) / np.timedelta64(1, 's')
        x_extent_min = start_time_seconds
        x_extent_max = end_time_seconds
        x_label = 'Time (seconds from start)'
    elif x_axis_units == 'pings':
        x_extent_min = ping_min
        x_extent_max = ping_max
        x_label = 'Ping Number'
    elif x_axis_units == 'meters':
        start_time_seconds = (ping_times[ping_min] - ping_times[0]) / np.timedelta64(1, 's')
        end_time_seconds = (ping_times[ping_max] - ping_times[0]) / np.timedelta64(1, 's')
        
        if meters_per_second is None:
            if echodata is None:
                raise ValueError("echodata parameter is required when meters_per_second is not provided")
            print("using gps calculation of meters_per_second")
            start_lat = echodata["Platform"]["latitude"][ping_min]
            start_lon = echodata["Platform"]["longitude"][ping_min]
            end_lat = echodata["Platform"]["latitude"][ping_max]
            end_lon = echodata["Platform"]["longitude"][ping_max]
            distance_meters = utils.haversine_distance(start_lat, start_lon, end_lat, end_lon)

            duration_seconds = (end_time_seconds - start_time_seconds).astype('timedelta64[s]').astype(float)

            if duration_seconds > 0:
                meters_per_second = distance_meters / duration_seconds
            else:
                print("Warning: Zero duration detected, using default speed")
                meters_per_second = 5.0  # Default 5 m/s for marine vessels
            
            print(f"GPS calculation details:")
            print(f"  Start: lat={start_lat:.6f}, lon={start_lon:.6f}")
            print(f"  End: lat={end_lat:.6f}, lon={end_lon:.6f}")
            print(f"  Distance: ({distance_meters:.0f} m)")
            print(f"  Duration: {duration_seconds:.1f} seconds ({duration_seconds/3600:.2f} hours)")
            print(f"  Calculated speed: {meters_per_second:.2f} m/s ({meters_per_second*3.6:.1f} km/h)")
        
        x_extent_min = start_time_seconds * meters_per_second
        x_extent_max = end_time_seconds * meters_per_second
        x_label = 'Distance (meters)'
    else:
        raise ValueError(f"Invalid x_axis_units '{x_axis_units}'. Use 'pings', 'seconds', or 'meters'")
    
    return x_extent_min, x_extent_max, x_label

def _calculate_y_axis_extent(min_depth_shown, max_depth_shown, min_depth_index, max_depth_index, y_axis_units, data_type=None):
    """
    Helper function to calculate y-axis extent and labels based on units.
    
    Returns:
        tuple: (y_extent_min, y_extent_max, y_label)
    """
    if y_axis_units == 'meters':
        y_extent_min = min_depth_shown
        y_extent_max = max_depth_shown
        y_label = 'Depth (m)'
    elif y_axis_units == 'range_sample':
        y_extent_min = min_depth_index
        y_extent_max = max_depth_index
        y_label = 'Range Sample Index'
    elif y_axis_units == 'bins' and data_type == "MVBS":
        y_extent_min = min_depth_index
        y_extent_max = max_depth_index
        y_label = 'MVBS Depth Bins'
    else:
        valid_y_units = ['meters', 'range_sample']
        if data_type == "MVBS":
            valid_y_units.append('bins')
        raise ValueError(f"Invalid y_axis_units '{y_axis_units}'. Use {valid_y_units}")
    
    return y_extent_min, y_extent_max, y_label

def _calculate_plot_dimensions_and_aspect(x_extent_min, x_extent_max, y_extent_min, y_extent_max, y_to_x_aspect_ratio_override=None):
    """
    Helper function to calculate plot dimensions and aspect ratio.
    
    Returns:
        tuple: (extent, aspect_ratio, width_multiplier, height_multiplier)
    """
    # Set up extent for imshow plots - note that imshow expects [left, right, bottom, top]
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

