# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for plotting datasets gridded on depth rather than echo_range.

``compute_MVBS`` names its range coordinate after the variable it binned
along, so an MVBS built with ``range_var="depth"`` carries no ``echo_range``
at all. These tests pin the handlers to reading whichever axis the dataset
actually has.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aa_si_visualization._plotting_utils import is_mvbs_dataset, resolve_range_var
from aa_si_visualization.echogram_handlers import (
    ClusterDataHandler,
    MvbsDataHandler,
    SvDataHandler,
    create_handler,
)


def _mvbs(range_var="echo_range", n_range=8, n_ping=5, n_chan=2):
    """Gridded MVBS with its vertical axis named *range_var*."""
    return xr.Dataset(
        {"Sv": (("channel", "ping_time", range_var),
                np.arange(n_chan * n_ping * n_range, dtype=float).reshape(
                    n_chan, n_ping, n_range))},
        coords={
            "channel": [f"ch{i}" for i in range(n_chan)],
            "ping_time": pd.date_range("2016-07-25", periods=n_ping, freq="20s"),
            range_var: np.arange(n_range, dtype=float) * 20.0,
        },
    )


def _sv(n_range=8, n_ping=5, n_chan=2, with_depth=True):
    """Ungridded Sv carrying echo_range, and optionally depth alongside it."""
    er = np.broadcast_to(np.arange(n_range, dtype=float),
                         (n_chan, n_ping, n_range)).copy()
    data = {
        "Sv": (("channel", "ping_time", "range_sample"), np.zeros_like(er)),
        "echo_range": (("channel", "ping_time", "range_sample"), er),
    }
    if with_depth:
        data["depth"] = (("channel", "ping_time", "range_sample"), er + 9.0)
    return xr.Dataset(
        data,
        coords={
            "channel": [f"ch{i}" for i in range(n_chan)],
            "ping_time": pd.date_range("2016-07-25", periods=n_ping, freq="1s"),
            "range_sample": np.arange(n_range),
        },
    )


def _cluster(range_var="echo_range", n_range=8, n_ping=5):
    """Single-channel cluster labels on a gridded vertical axis."""
    return xr.Dataset(
        {"labels": (("ping_time", range_var),
                    np.zeros((n_ping, n_range), dtype=float))},
        coords={
            "ping_time": pd.date_range("2016-07-25", periods=n_ping, freq="20s"),
            range_var: np.arange(n_range, dtype=float) * 20.0,
        },
    )


@pytest.mark.parametrize("range_var", ["echo_range", "depth"])
def test_resolve_range_var_reads_the_gridded_axis(range_var):
    assert resolve_range_var(_mvbs(range_var), "Sv") == range_var


def test_resolve_range_var_prefers_echo_range_for_ungridded_sv():
    # Sv routed through ep_add_depth carries both; the range axis is still
    # range_sample, and echo_range stays the meter mapping for it.
    assert resolve_range_var(_sv(with_depth=True), "Sv") == "echo_range"


def test_resolve_range_var_falls_back_to_depth():
    ds = _sv(with_depth=True).drop_vars("echo_range")
    assert resolve_range_var(ds, "Sv") == "depth"


def test_resolve_range_var_returns_none_when_absent():
    ds = _sv(with_depth=False).drop_vars("echo_range")
    assert resolve_range_var(ds, "Sv") is None


@pytest.mark.parametrize("range_var", ["echo_range", "depth"])
def test_is_mvbs_dataset_on_either_axis(range_var):
    assert is_mvbs_dataset(_mvbs(range_var)) is True


def test_is_mvbs_dataset_false_for_ungridded_sv():
    assert is_mvbs_dataset(_sv()) is False


@pytest.mark.parametrize("range_var", ["echo_range", "depth"])
def test_create_handler_picks_mvbs_on_either_axis(range_var):
    handler = create_handler(_mvbs(range_var), "Sv")
    assert isinstance(handler, MvbsDataHandler)
    assert handler.range_var == range_var
    assert handler.detect_structure()["dimensions"][-1] == range_var


def test_create_handler_picks_sv_when_depth_rides_along():
    handler = create_handler(_sv(with_depth=True), "Sv")
    assert isinstance(handler, SvDataHandler)
    assert handler.range_var == "echo_range"


@pytest.mark.parametrize("range_var", ["echo_range", "depth"])
def test_mvbs_slicing_and_extent_on_either_axis(range_var):
    handler = create_handler(_mvbs(range_var), "Sv")
    handler.calculate_depth_indices(20.0, 80.0)
    assert (handler.min_depth_index, handler.max_depth_index) == (1, 4)
    assert handler.get_depth_extent(1, 4) == (20.0, 80.0)

    sliced = handler.slice_data_for_frequency(0, (0, 3), (1, 4))
    assert sliced.sizes["ping_time"] == 3
    assert sliced.sizes[range_var] == 3


@pytest.mark.parametrize("range_var", ["echo_range", "depth"])
def test_cluster_handler_slices_on_either_axis(range_var):
    handler = ClusterDataHandler(_cluster(range_var), "labels")
    info = handler.detect_structure()
    assert info["type"] == "Cluster-MVBS"
    assert info["dimensions"] == ["ping_time", range_var]

    sliced = handler.slice_data_for_frequency(0, (0, 3), (1, 4))
    assert sliced.sizes[range_var] == 3


@pytest.mark.parametrize("range_var", ["echo_range", "depth"])
def test_create_handler_detects_cluster_on_either_axis(range_var):
    handler = create_handler(_cluster(range_var), "labels")
    assert isinstance(handler, ClusterDataHandler)
