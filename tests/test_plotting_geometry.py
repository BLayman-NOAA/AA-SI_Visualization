# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for echogram panel sizing and time-based ping windows."""

import datetime as dt

import numpy as np
import pytest

from aa_si_visualization._plotting_utils import (
    BASE_PANEL_WIDTH_IN,
    MAX_PANEL_ASPECT,
    MAX_PANEL_WIDTH_IN,
    MIN_PANEL_ASPECT,
    PANEL_GAP_IN,
    calculate_panel_geometry,
    calculate_x_axis_extent,
    figure_height_for_panels,
    resolve_ping_bounds,
    to_datetime64,
)

# 2 channels over 1750 m of a 4150 s transect. Representative of the echograms
# that already looked right before the panel sizing was reworked.
DEEP_WINDOW = (0, 4150, 0, 1800)

# 5 channels over 25 m of a multi-hour transect. This is the window that used
# to collapse each panel into a sliver.
SHALLOW_WINDOW = (0, 20000, 0, 25)


def test_deep_window_keeps_squared_data_aspect():
    geometry = calculate_panel_geometry(*DEEP_WINDOW)

    assert geometry['panel_aspect'] == pytest.approx(geometry['data_aspect'] ** 2)
    assert not geometry['clamped']


def test_shallow_window_clamps_panel_aspect_and_width():
    geometry = calculate_panel_geometry(*SHALLOW_WINDOW)

    assert geometry['panel_aspect'] == MIN_PANEL_ASPECT
    assert geometry['panel_width'] == MAX_PANEL_WIDTH_IN
    assert geometry['panel_height'] == pytest.approx(
        MAX_PANEL_WIDTH_IN * MIN_PANEL_ASPECT
    )
    assert geometry['clamped']


def test_tall_window_clamps_at_max_aspect():
    geometry = calculate_panel_geometry(0, 100, 0, 1000)

    assert geometry['panel_aspect'] == MAX_PANEL_ASPECT
    assert geometry['panel_width'] == BASE_PANEL_WIDTH_IN
    assert geometry['clamped']


def test_override_bypasses_the_clamp():
    geometry = calculate_panel_geometry(*SHALLOW_WINDOW,
                                        y_to_x_aspect_ratio_override=20)

    assert geometry['panel_aspect'] == pytest.approx(0.05)
    assert not geometry['clamped']


def test_override_applies_where_the_clamp_would_not():
    geometry = calculate_panel_geometry(*DEEP_WINDOW,
                                        y_to_x_aspect_ratio_override=2)

    assert geometry['panel_aspect'] == pytest.approx(0.5)


def test_extent_is_ordered_for_imshow_with_y_inverted():
    geometry = calculate_panel_geometry(*DEEP_WINDOW)

    assert geometry['extent'] == [0, 4150, 1800, 0]


def test_figure_height_counts_one_gap_between_panels():
    single = figure_height_for_panels(7.5, 1)
    triple = figure_height_for_panels(7.5, 3)

    assert triple - single == pytest.approx(2 * 7.5 + 2 * PANEL_GAP_IN)


def test_explicit_x_range_shapes_the_panel():
    # A datetime axis is laid out in date numbers (days), so it passes its span
    # in seconds; the panel must match what the 'seconds' axis would give.
    seconds = calculate_panel_geometry(0, 4150, 0, 1800)
    days = calculate_panel_geometry(0, 4150 / 86400, 0, 1800, x_range=4150)

    assert days['panel_aspect'] == pytest.approx(seconds['panel_aspect'])
    assert days['panel_width'] == pytest.approx(seconds['panel_width'])


# 1000 pings, one per second, starting at midnight UTC.
PING_TIMES = np.datetime64("2024-10-15T00:00:00") + np.arange(1000).astype(
    "timedelta64[s]"
)


@pytest.mark.parametrize("value", [
    "2024-10-15T00:05:00",
    "2024-10-15 00:05:00",
    np.datetime64("2024-10-15T00:05:00"),
    dt.datetime(2024, 10, 15, 0, 5),
])
def test_to_datetime64_reads_common_timestamp_forms(value):
    assert to_datetime64(value) == np.datetime64("2024-10-15T00:05:00")


def test_to_datetime64_converts_an_offset_aware_value_to_utc():
    assert to_datetime64("2024-10-15T02:05:00+02:00") == np.datetime64(
        "2024-10-15T00:05:00"
    )


def test_to_datetime64_rejects_a_non_timestamp():
    with pytest.raises(ValueError, match="could not read"):
        to_datetime64("not a time")


def test_ping_bounds_default_to_the_full_extent():
    assert resolve_ping_bounds(PING_TIMES, None, None) == (0, 999)


def test_ping_bounds_pass_through_indices_untouched():
    assert resolve_ping_bounds(PING_TIMES, 100, 400) == (100, 400)


def test_time_bounds_resolve_to_the_closest_ping():
    bounds = resolve_ping_bounds(
        PING_TIMES, 0, 999,
        time_min="2024-10-15T00:01:40", time_max="2024-10-15T00:05:00",
    )

    assert bounds == (100, 300)


def test_a_time_bound_wins_over_the_matching_ping_index():
    # time_min replaces ping_min; ping_max is left alone.
    assert resolve_ping_bounds(
        PING_TIMES, 700, 900, time_min="2024-10-15T00:01:40"
    ) == (100, 900)


def test_time_bound_outside_the_data_clamps_and_warns(caplog):
    with caplog.at_level("WARNING"):
        bounds = resolve_ping_bounds(PING_TIMES, None, None, time_min="2020-01-01")

    assert bounds == (0, 999)
    assert "outside the data" in caplog.text


def test_inverted_window_is_rejected():
    with pytest.raises(ValueError, match="empty ping window"):
        resolve_ping_bounds(
            PING_TIMES, None, None,
            time_min="2024-10-15T00:05:00", time_max="2024-10-15T00:01:40",
        )


def test_datetime_axis_extent_is_in_date_numbers():
    import matplotlib.dates as mdates

    x_min, x_max, label = calculate_x_axis_extent(
        PING_TIMES, 100, 400, 'datetime'
    )

    assert label == 'Time (UTC)'
    assert mdates.num2date(x_min).replace(tzinfo=None) == dt.datetime(
        2024, 10, 15, 0, 1, 40
    )
    assert x_max - x_min == pytest.approx(300 / 86400)


def test_unknown_x_axis_units_lists_datetime_as_an_option():
    with pytest.raises(ValueError, match="datetime"):
        calculate_x_axis_extent(PING_TIMES, 0, 999, 'clock')
