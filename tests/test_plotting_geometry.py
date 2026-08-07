# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for echogram panel sizing."""

import pytest

from aa_si_visualization._plotting_utils import (
    BASE_PANEL_WIDTH_IN,
    MAX_PANEL_ASPECT,
    MAX_PANEL_WIDTH_IN,
    MIN_PANEL_ASPECT,
    PANEL_GAP_IN,
    calculate_panel_geometry,
    figure_height_for_panels,
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
