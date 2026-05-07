"""Shared plotting utilities for axis calculation, layout, and data detection."""

import logging
import numpy as np
from aa_si_utils import utils

logger = logging.getLogger(__name__)


def is_mvbs_dataset(ds):
    """Check whether a dataset uses the MVBS (gridded) structure.

    MVBS datasets have a 1D ``echo_range`` coordinate, while regular Sv
    datasets have a multi-dimensional one.

    Args:
        ds: xarray.Dataset to check.

    Returns:
        bool: True if the dataset has MVBS structure.
    """
    if 'echo_range' not in ds.coords:
        return False
    return len(ds['echo_range'].dims) == 1


def calculate_y_axis_extent(min_depth_shown, max_depth_shown, min_depth_index,
                            max_depth_index, y_axis_units, data_type=None):
    """Calculate y-axis extent and label string based on the requested units.

    Args:
        min_depth_shown: Minimum depth value in meters.
        max_depth_shown: Maximum depth value in meters.
        min_depth_index: Minimum depth index.
        max_depth_index: Maximum depth index.
        y_axis_units: Units for the y-axis ('meters', 'range_sample', or
            'bins').
        data_type: Optional data type string used for validation (e.g.
            'MVBS', 'ML-MVBS', 'Cluster-MVBS').

    Returns:
        tuple: (y_extent_min, y_extent_max, y_label)

    Raises:
        ValueError: If y_axis_units is not valid for the given data type.
    """
    if y_axis_units == 'meters':
        return min_depth_shown, max_depth_shown, 'Depth (m)'

    if y_axis_units == 'range_sample':
        return min_depth_index, max_depth_index, 'Range Sample Index'

    mvbs_types = {'MVBS', 'ML-MVBS', 'Cluster-MVBS'}
    if y_axis_units == 'bins':
        if data_type and data_type not in mvbs_types:
            raise ValueError("y_axis_units='bins' is only valid for MVBS data")
        return min_depth_index, max_depth_index, 'MVBS Depth Bins'

    valid = ['meters', 'range_sample']
    if data_type and data_type in mvbs_types:
        valid.append('bins')
    raise ValueError(f"Invalid y_axis_units '{y_axis_units}'. Use {valid}")


def calculate_plot_dimensions(x_extent_min, x_extent_max, y_extent_min,
                              y_extent_max, y_to_x_aspect_ratio_override=None):
    """Calculate imshow extent, aspect ratio, and figure size multipliers.

    Args:
        x_extent_min: Minimum x-axis extent.
        x_extent_max: Maximum x-axis extent.
        y_extent_min: Minimum y-axis extent.
        y_extent_max: Maximum y-axis extent.
        y_to_x_aspect_ratio_override: Optional manual aspect ratio override.

    Returns:
        tuple: (extent, aspect_ratio, width_multiplier, height_multiplier)
            where *extent* is ``[left, right, bottom, top]`` for imshow.
    """
    extent = [x_extent_min, x_extent_max, y_extent_max, y_extent_min]

    x_range = abs(x_extent_max - x_extent_min)
    y_range = abs(y_extent_max - y_extent_min)
    aspect_ratio = y_range / x_range

    if y_to_x_aspect_ratio_override is not None:
        aspect_ratio = (1 / y_to_x_aspect_ratio_override * (1 / aspect_ratio))

    width_multiplier = 1
    height_multiplier = 1
    if aspect_ratio < 1:
        width_multiplier = min(10, 1 / aspect_ratio)
    else:
        height_multiplier = min(3, aspect_ratio)

    return extent, aspect_ratio, width_multiplier, height_multiplier


def calculate_x_axis_extent(ping_times, ping_min, ping_max, x_axis_units,
                            meters_per_second=None, echodata=None,
                            handler=None):
    """Calculate x-axis extent and label for the requested unit system.

    Supports 'seconds', 'pings', 'bins' (MVBS only), and 'meters'.

    Args:
        ping_times: Array of ping time values.
        ping_min: Minimum ping index (already converted for MVBS if needed).
        ping_max: Maximum ping index.
        x_axis_units: One of 'seconds', 'pings', 'bins', or 'meters'.
        meters_per_second: Speed in m/s for distance conversion. If None and
            x_axis_units is 'meters', will attempt GPS calculation.
        echodata: Original echodata object for GPS speed calculation. Required
            when meters_per_second is None and x_axis_units is 'meters'.
        handler: Optional EchogramDataHandler, used for MVBS ping labels and
            GPS index lookup.

    Returns:
        tuple: (x_extent_min, x_extent_max, x_label)

    Raises:
        ValueError: If x_axis_units is invalid or required parameters are
            missing for the chosen unit.
    """
    is_mvbs = handler.is_mvbs_structured() if handler else False

    if x_axis_units == 'seconds':
        start = (ping_times[ping_min] - ping_times[0]) / np.timedelta64(1, 's')
        end = (ping_times[ping_max] - ping_times[0]) / np.timedelta64(1, 's')
        return start, end, 'Time (seconds from start)'

    if x_axis_units == 'pings':
        if is_mvbs:
            orig_min = getattr(handler, 'ping_min', ping_min)
            orig_max = getattr(handler, 'ping_max', ping_max)
            label = f'MVBS Bin (pings {orig_min} to {orig_max})'
        else:
            label = 'Ping Number'
        return ping_min, ping_max, label

    if x_axis_units == 'bins':
        if not is_mvbs:
            raise ValueError("x_axis_units='bins' is only valid for MVBS data")
        return ping_min, ping_max, 'MVBS Time Bins'

    if x_axis_units == 'meters':
        start = (ping_times[ping_min] - ping_times[0]) / np.timedelta64(1, 's')
        end = (ping_times[ping_max] - ping_times[0]) / np.timedelta64(1, 's')

        if meters_per_second is None:
            meters_per_second = _calculate_speed_from_gps(
                handler, echodata, ping_min, ping_max, start, end
            )

        return start * meters_per_second, end * meters_per_second, 'Distance (meters)'

    valid = ['seconds', 'pings', 'meters']
    if is_mvbs:
        valid.append('bins')
    raise ValueError(f"Invalid x_axis_units '{x_axis_units}'. Valid options: {valid}")


def _calculate_speed_from_gps(handler, echodata, ping_min, ping_max,
                              start_seconds, end_seconds):
    """Derive vessel speed from GPS coordinates in echodata.

    Args:
        handler: EchogramDataHandler (used to get original ping indices for
            MVBS data).
        echodata: Echodata object with Platform lat/lon.
        ping_min: Current ping min index.
        ping_max: Current ping max index.
        start_seconds: Start time in seconds from first ping.
        end_seconds: End time in seconds from first ping.

    Returns:
        float: Calculated speed in meters per second.

    Raises:
        ValueError: If echodata is not provided.
    """
    if echodata is None:
        raise ValueError(
            "echodata parameter is required when meters_per_second is not "
            "provided and x_axis_units='meters'"
        )

    orig_min = getattr(handler, 'ping_min', ping_min) if handler else ping_min
    orig_max = getattr(handler, 'ping_max', ping_max) if handler else ping_max

    logger.info("Using GPS calculation for meters_per_second...")
    start_lat = echodata["Platform"]["latitude"][orig_min]
    start_lon = echodata["Platform"]["longitude"][orig_min]
    end_lat = echodata["Platform"]["latitude"][orig_max]
    end_lon = echodata["Platform"]["longitude"][orig_max]

    distance_meters = utils.haversine_distance(
        start_lat, start_lon, end_lat, end_lon
    )
    duration_seconds = end_seconds - start_seconds

    if duration_seconds > 0:
        speed = distance_meters / duration_seconds
    else:
        logger.warning("Zero duration detected, using default speed of 5 m/s")
        speed = 5.0

    logger.debug("GPS calculation details:")
    logger.debug("  Start: lat=%.6f, lon=%.6f", start_lat, start_lon)
    logger.debug("  End: lat=%.6f, lon=%.6f", end_lat, end_lon)
    logger.debug("  Distance: %.0f m", distance_meters)
    logger.debug("  Duration: %.1f s (%.2f hours)",
                 duration_seconds, duration_seconds / 3600)
    logger.debug("  Calculated speed: %.2f m/s (%.1f km/h)",
                 speed, speed * 3.6)

    return speed


def setup_depth_range(dataset, min_depth, max_depth, ping_min, ping_max):
    """Auto-detect and resolve depth range, returning indices and shown values.

    If either min_depth or max_depth is None the range is auto-detected from
    the dataset's echo_range coordinate.

    Args:
        dataset: xarray.Dataset with an ``echo_range`` variable.
        min_depth: Requested minimum depth in meters, or None.
        max_depth: Requested maximum depth in meters, or None.
        ping_min: Start ping index used for auto-detection.
        ping_max: End ping index used for auto-detection.

    Returns:
        tuple: (min_depth, max_depth, min_depth_index, max_depth_index,
            min_depth_shown, max_depth_shown)
    """
    if min_depth is None or max_depth is None:
        logger.info("Auto-detecting depth range from data...")
        auto_min, auto_max = utils.find_data_depth_range(
            dataset, ping_min, ping_max, channel=0
        )
        if min_depth is None:
            min_depth = auto_min
        if max_depth is None:
            max_depth = auto_max
        logger.info("Using depth range: %.1fm to %.1fm", min_depth, max_depth)

    min_idx = utils.get_closest_index_for_depth(dataset, min_depth)
    max_idx = utils.get_closest_index_for_depth(dataset, max_depth)

    actual_depths = dataset.echo_range.isel(channel=0, ping_time=0).values
    min_depth_shown = actual_depths[min_idx]
    max_depth_shown = actual_depths[min(max_idx, len(actual_depths) - 1)]

    return min_depth, max_depth, min_idx, max_idx, min_depth_shown, max_depth_shown
