"""Shared plotting utilities for axis calculation, layout, and data detection."""

import logging
import numpy as np
import pandas as pd
from aa_si_utils import utils

logger = logging.getLogger(__name__)

# Drawn size of a single echogram panel, in inches. A panel is the imshow box
# for one frequency; the figure stacks one per channel plus fixed chrome bands.
BASE_PANEL_WIDTH_IN = 24.0
MAX_PANEL_WIDTH_IN = 60.0
MIN_PANEL_ASPECT = 1 / 8
MAX_PANEL_ASPECT = 3.0

# Chrome reserved around the panels, in inches.
PANEL_GAP_IN = 1.1
TITLE_BAND_IN = 1.2
COLORBAR_BAND_IN = 1.6
YLABEL_BAND_IN = 1.0
RIGHT_MARGIN_IN = 0.4
CLUSTER_COLORBAR_BAND_IN = 1.8


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

    New code should use :func:`calculate_panel_geometry`, which sizes the panel
    directly instead of relying on imshow to shrink the axes box.

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


def calculate_panel_geometry(x_extent_min, x_extent_max, y_extent_min,
                             y_extent_max, y_to_x_aspect_ratio_override=None,
                             x_range=None):
    """Calculate imshow extent and the drawn size of a single echogram panel.

    The panel shape follows the square of the data aspect, which is what the
    older imshow-driven sizing produced, but clamped so that very wide, shallow
    windows do not collapse the panel into a sliver.

    Args:
        x_extent_min: Minimum x-axis extent.
        x_extent_max: Maximum x-axis extent.
        y_extent_min: Minimum y-axis extent.
        y_extent_max: Maximum y-axis extent.
        y_to_x_aspect_ratio_override: Optional panel width-to-height ratio.
            When given, the panel is exactly this many times wider than tall
            and the automatic clamp is skipped.
        x_range: Optional x span to shape the panel with, for axes whose extent
            is not in a unit comparable to depth. A datetime axis passes its
            span in seconds so it draws like the 'seconds' axis rather than
            being shaped by date numbers, which count days.

    Returns:
        dict: With keys ``extent`` (``[left, right, bottom, top]`` for imshow),
            ``data_aspect``, ``panel_aspect`` (height / width of the drawn
            panel), ``panel_width``, ``panel_height`` (both in inches), and
            ``clamped`` (whether the automatic panel aspect hit a limit).
    """
    extent = [x_extent_min, x_extent_max, y_extent_max, y_extent_min]

    if x_range is None:
        x_range = abs(x_extent_max - x_extent_min)
    y_range = abs(y_extent_max - y_extent_min)
    data_aspect = y_range / x_range

    if y_to_x_aspect_ratio_override is not None:
        panel_aspect = 1 / y_to_x_aspect_ratio_override
        clamped = False
    else:
        panel_aspect = min(max(data_aspect ** 2, MIN_PANEL_ASPECT), MAX_PANEL_ASPECT)
        clamped = panel_aspect != data_aspect ** 2

    panel_width = BASE_PANEL_WIDTH_IN
    if data_aspect < 1:
        panel_width = min(MAX_PANEL_WIDTH_IN, BASE_PANEL_WIDTH_IN / data_aspect)

    return {
        'extent': extent,
        'data_aspect': data_aspect,
        'panel_aspect': panel_aspect,
        'panel_width': panel_width,
        'panel_height': panel_width * panel_aspect,
        'clamped': clamped,
    }


def figure_height_for_panels(panel_height, n_panels,
                             colorbar_band=COLORBAR_BAND_IN):
    """Calculate figure height in inches for a vertical stack of panels.

    Args:
        panel_height: Drawn height of one panel in inches.
        n_panels: Number of stacked panels.
        colorbar_band: Inches reserved below the panels for the colorbar and
            x-axis label.

    Returns:
        float: Figure height in inches.
    """
    return (
        TITLE_BAND_IN
        + n_panels * panel_height
        + (n_panels - 1) * PANEL_GAP_IN
        + colorbar_band
    )


def to_datetime64(value):
    """Convert a timestamp-like value to a timezone-naive UTC datetime64.

    Ping times are stored as naive UTC, so an offset-aware input is converted
    to UTC before the offset is dropped.

    Args:
        value: Anything pandas can read as a timestamp: an ISO string, a
            ``datetime``, a ``numpy.datetime64``, or a ``pandas.Timestamp``.

    Returns:
        numpy.datetime64: The value as naive UTC.

    Raises:
        ValueError: If the value cannot be parsed as a timestamp.
    """
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"could not read {value!r} as a timestamp") from exc
    if stamp.tz is not None:
        stamp = stamp.tz_convert('UTC').tz_localize(None)
    return stamp.to_datetime64()


def _nearest_ping_index(ping_times, value, label):
    """Find the ping closest to a requested time, warning if it falls outside."""
    target = to_datetime64(value)
    if target < ping_times[0] or target > ping_times[-1]:
        logger.warning(
            "%s=%s is outside the data (%s to %s); using the closest ping",
            label, target, ping_times[0], ping_times[-1],
        )
    index = int(np.argmin(np.abs(ping_times - target)))
    logger.info("  %s=%s resolved to ping %s (%s)",
                label, target, index, ping_times[index])
    return index


def resolve_ping_bounds(ping_times, ping_min, ping_max,
                        time_min=None, time_max=None):
    """Resolve the displayed ping window, applying defaults and time bounds.

    A time bound takes precedence over the matching ping index and is resolved
    to the closest ping. A ``None`` bound means the full extent.

    Args:
        ping_times: Array of ping times for the dataset the indices refer to.
        ping_min: Start ping index, or None for the first ping.
        ping_max: End ping index, or None for the last ping.
        time_min: Optional start timestamp, overriding ping_min.
        time_max: Optional end timestamp, overriding ping_max.

    Returns:
        tuple: (ping_min, ping_max) as integer indices.

    Raises:
        ValueError: If the resolved window is empty or inverted.
    """
    if time_min is not None:
        ping_min = _nearest_ping_index(ping_times, time_min, 'time_min')
    elif ping_min is None:
        ping_min = 0

    if time_max is not None:
        ping_max = _nearest_ping_index(ping_times, time_max, 'time_max')
    elif ping_max is None:
        ping_max = len(ping_times) - 1

    if ping_min >= ping_max:
        raise ValueError(
            f"empty ping window: start ping {ping_min} is not before end ping "
            f"{ping_max}"
        )
    return ping_min, ping_max


def apply_datetime_x_axis(ax):
    """Label an x-axis whose extent is in matplotlib date numbers.

    Ticks land on round clock times and carry only the part that changes, with
    the shared date shown once as the axis offset.

    Args:
        ax: matplotlib Axes to format.
    """
    import matplotlib.dates as mdates

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.xaxis.get_offset_text().set_color('white')


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
        handler: Optional EchogramDataHandler, used for MVBS ping labels.

    Returns:
        tuple: (x_extent_min, x_extent_max, x_label)

    Raises:
        ValueError: If x_axis_units is invalid or required parameters are
            missing for the chosen unit.
    """
    is_mvbs = handler.is_mvbs_structured() if handler else False

    if x_axis_units == 'datetime':
        import matplotlib.dates as mdates

        return (mdates.date2num(ping_times[ping_min]),
                mdates.date2num(ping_times[ping_max]),
                'Time (UTC)')

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
                echodata, ping_times[ping_min], ping_times[ping_max],
                start, end
            )

        return start * meters_per_second, end * meters_per_second, 'Distance (meters)'

    valid = ['seconds', 'datetime', 'pings', 'meters']
    if is_mvbs:
        valid.append('bins')
    raise ValueError(f"Invalid x_axis_units '{x_axis_units}'. Valid options: {valid}")


def _calculate_speed_from_gps(echodata, start_time, end_time,
                              start_seconds, end_seconds):
    """Derive vessel speed from GPS coordinates in echodata.

    Platform latitude/longitude are dimensioned by ``time1`` (NMEA datagram
    timestamps), not by ping, so positions are looked up by nearest time
    rather than by ping index.

    Args:
        echodata: Echodata object with Platform lat/lon.
        start_time: Timestamp (datetime64) of the first ping shown.
        end_time: Timestamp (datetime64) of the last ping shown.
        start_seconds: Start time in seconds from first ping.
        end_seconds: End time in seconds from first ping.

    Returns:
        float: Calculated speed in meters per second.

    Raises:
        ValueError: If echodata is not provided or contains no valid GPS
            fixes.
    """
    if echodata is None:
        raise ValueError(
            "echodata parameter is required when meters_per_second is not "
            "provided and x_axis_units='meters'"
        )

    logger.info("Using GPS calculation for meters_per_second...")
    lat = echodata["Platform"]["latitude"]
    lon = echodata["Platform"]["longitude"]

    valid = np.isfinite(lat.values) & np.isfinite(lon.values)
    if not valid.any():
        raise ValueError(
            "No valid GPS fixes in echodata Platform group; pass "
            "meters_per_second explicitly"
        )
    lat = lat.isel(time1=np.flatnonzero(valid))
    lon = lon.isel(time1=np.flatnonzero(valid))

    start_lat = lat.sel(time1=start_time, method="nearest")
    start_lon = lon.sel(time1=start_time, method="nearest")
    end_lat = lat.sel(time1=end_time, method="nearest")
    end_lon = lon.sel(time1=end_time, method="nearest")

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
