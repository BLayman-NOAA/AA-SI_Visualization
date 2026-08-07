# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for full-extent ping bounds."""

import numpy as np
import pandas as pd
import xarray as xr

from aa_si_visualization.echogram import _setup_parameters

N_ORIGINAL = 5000
N_MVBS = 250


def _dataset(n_pings):
    ping_time = pd.date_range("2024-10-15T11:42", periods=n_pings, freq="1s").values
    return xr.Dataset(
        {"Sv": (("channel", "ping_time", "range_sample"),
                np.zeros((1, n_pings, 10), dtype="float32"))},
        coords={
            "channel": ["ch1"],
            "ping_time": ping_time,
            "range_sample": np.arange(10),
        },
    )


def _ping_window(ds_Sv, ds_Sv_original, ping_min=None, ping_max=None):
    params = _setup_parameters(
        ds_Sv, None, None, None, ping_min, ping_max, -80, -20, "viridis",
        ds_Sv_original, False, "seconds", "meters", None, None,
        None, None, None, None, None,
    )
    return params["ping_min"], params["ping_max"]


def test_none_bounds_span_the_whole_sv_dataset():
    ds = _dataset(N_ORIGINAL)

    assert _ping_window(ds, None) == (0, N_ORIGINAL - 1)


def test_none_bounds_resolve_against_the_original_ping_axis():
    """MVBS bounds index the original Sv axis, not the coarser MVBS grid.

    MVBSHandler.calculate_ping_range converts them via original_ping_times,
    so defaulting off the MVBS grid would cover only the start of the data.
    """
    ds_mvbs = _dataset(N_MVBS)
    ds_original = _dataset(N_ORIGINAL)

    assert _ping_window(ds_mvbs, ds_original) == (0, N_ORIGINAL - 1)


def test_explicit_bounds_are_left_alone():
    ds = _dataset(N_ORIGINAL)

    assert _ping_window(ds, None, ping_min=100, ping_max=500) == (100, 500)
