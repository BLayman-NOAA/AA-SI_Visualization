"""Assorted visualization utilities for calibration comparison panels."""

import logging
import numpy as np
import matplotlib.pyplot as plt
from aa_si_utils import utils
from . import _plotting_utils as putils

logger = logging.getLogger(__name__)


def sv_differences_echograms(ds_Sv_baseline, ds_Sv_calibrated, frequencies, max_depth=None, min_depth=None, ping_min=None, ping_max=None, sv_scale_min=-80, sv_scale_max=-20, sv_scale_diff_min=-15, sv_scale_diff_max=15, sv_cmap='viridis', diff_cmap='RdBu_r', x_axis_units='pings', y_axis_units='meters', meters_per_second=None, y_to_x_aspect_ratio_override=None):
    """Generate multi-panel baseline / calibrated / difference echograms.

    Creates a figure with three columns (baseline Sv, calibrated Sv, and their
    difference) for each valid frequency.  Frequencies that contain only NaN
    data are automatically excluded.
    
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
        y_to_x_aspect_ratio_override (float, optional): Override for the y-to-x aspect ratio
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
            logger.warning("%s kHz has all NaN values, excluding from plot", int(freq/1000))

    if len(valid_freq_indices) == 0:
        logger.error("All frequencies have NaN data. Cannot create plot.")
        return

    # Update frequencies to only include valid ones
    frequencies = valid_frequencies

    # Set default depth range and calculate indices using shared utility
    min_depth, max_depth, min_depth_index, max_depth_index, min_depth_shown, max_depth_shown = putils.setup_depth_range(
        ds_Sv_baseline, min_depth, max_depth, ping_min, ping_max
    )

    # Get data dimensions and create range/time axes for plotting
    freq_labels = [f"{int(f/1000)} kHz" for f in frequencies] 

    # Calculate x-axis extent
    ping_times = ds_Sv_baseline['ping_time'].values
    x_extent_min, x_extent_max, x_label = putils.calculate_x_axis_extent(
        ping_times, ping_min, ping_max, x_axis_units,
        meters_per_second=meters_per_second, echodata=None,
    )
    
    # Calculate y-axis extent
    y_extent_min, y_extent_max, y_label = putils.calculate_y_axis_extent(
        min_depth_shown, max_depth_shown, min_depth_index, max_depth_index, y_axis_units
    )
    
    # Calculate plot dimensions and aspect ratio
    extent, aspect_ratio, width_multiplier, height_multiplier = putils.calculate_plot_dimensions(
        x_extent_min, x_extent_max, y_extent_min, y_extent_max, y_to_x_aspect_ratio_override
    )
    
    logger.info("  X-axis (%s): %.1f to %.1f", x_axis_units, x_extent_min, x_extent_max)
    logger.info("  Y-axis (%s): %.1f to %.1f", y_axis_units, y_extent_min, y_extent_max)
    logger.info("  Depth Range: %.1fm to %.1fm", min_depth_shown, max_depth_shown)
    logger.info("  Aspect ratio (1:1 in specified units): %.3f", aspect_ratio)

    # Create figure with adjusted spacing for colorbar
    fig_width = 24 * width_multiplier
    fig_height = 24 * height_multiplier

    logger.info("Figure size: %s x %s", fig_width, fig_height)   

    # Calculate differences
    sv_diff_data = ds_Sv_calibrated['Sv'] - ds_Sv_baseline['Sv']

    plt.style.use('dark_background')
    # Set up figure with multiple subplots
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Store images for shared colorbars
    sv_image = None
    diff_image = None

    num_freqs = len(freq_labels)    

    # Subplot layout: 3 columns (baseline, calibrated, difference) x N rows (frequencies)
    for plot_idx, (freq_idx, freq_label) in enumerate(zip(valid_freq_indices, freq_labels)):
    
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

        # Calibrated echogram
        ax2 = plt.subplot(num_freqs, 3, plot_idx*3 + 2)
        ax2.imshow(cal_data.T, aspect=aspect_ratio,
                        vmin=sv_scale_min, vmax=sv_scale_max, cmap=sv_cmap,
                        extent=extent)
        ax2.set_title(f'{freq_label} - CAL Report Calibration', fontsize=12, fontweight='bold', pad=20)
        
        # Difference echogram
        ax3 = plt.subplot(num_freqs, 3, plot_idx*3 + 3)
        im3 = ax3.imshow(diff_data.T, aspect=aspect_ratio,
                        vmin=sv_scale_diff_min, vmax=sv_scale_diff_max, cmap=diff_cmap,
                        extent=extent)
        ax3.set_title(f'{freq_label} - Difference (CAL - Baseline)', fontsize=12, fontweight='bold', pad=20)
        
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



