# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for auto-detected echogram depth bounds."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aa_si_visualization.echogram import _calculate_ranges, _setup_parameters
from aa_si_visualization.echogram_handlers import create_handler

N_PINGS = 400
N_SAMPLES = 100
DEPTH_STEP = 0.5  # the range axis spans 0 to 49.5 m

# Only this band carries finite Sv; everything shallower or deeper is NaN, as it
# would be after surface and seafloor masking.
FIRST_VALID = 4  # 2.0 m
LAST_VALID = 60  # 30.0 m


@pytest.fixture
def masked_sv():
    ping_time = pd.date_range("2024-10-15T13:38", periods=N_PINGS, freq="1s").values
    echo_range = np.tile(
        np.arange(N_SAMPLES) * DEPTH_STEP, (1, N_PINGS, 1)
    ).astype("float32")
    sv = np.full((1, N_PINGS, N_SAMPLES), np.nan, dtype="float32")
    sv[:, :, FIRST_VALID:LAST_VALID + 1] = -70.0

    return xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "range_sample"), sv),
            "echo_range": (("channel", "ping_time", "range_sample"), echo_range),
            "frequency_nominal": (("channel",), np.array([38000.0], dtype="float32")),
        },
        coords={
            "channel": ["ch1"],
            "ping_time": ping_time,
            "range_sample": np.arange(N_SAMPLES),
        },
    )


def _depth_range(ds, min_depth, max_depth):
    params = _setup_parameters(
        ds, None, max_depth, min_depth, None, None, -80, -20, "viridis",
        None, False, "seconds", "meters", None, None,
        None, None, None, None, None,
    )
    return _calculate_ranges(create_handler(ds, "Sv"), params, None)["depth"]


def test_none_max_depth_uses_deepest_data_not_the_range_axis(masked_sv):
    depth = _depth_range(masked_sv, None, None)

    assert depth["max_depth"] == LAST_VALID * DEPTH_STEP
    assert depth["min_depth"] == FIRST_VALID * DEPTH_STEP


def test_max_depth_alone_auto_detects(masked_sv):
    depth = _depth_range(masked_sv, 0, None)

    assert depth["min_depth"] == 0
    assert depth["max_depth"] == LAST_VALID * DEPTH_STEP


def test_explicit_bounds_are_left_alone(masked_sv):
    depth = _depth_range(masked_sv, 0, 25)

    assert (depth["min_depth"], depth["max_depth"]) == (0, 25)
