# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Tests for echogram data-handler helpers."""

from __future__ import annotations

import numpy as np
import pytest

from aa_si_visualization.echogram_handlers import _closest_index_in_1d


def test_closest_index_in_1d_basic():
    values = np.array([5.0, 15.0, 25.0, 35.0])
    assert _closest_index_in_1d(values, 24.0) == 2
    assert _closest_index_in_1d(values, 0.0) == 0


def test_closest_index_in_1d_ignores_nan_padding():
    # MVBS echo_range NaN-padded at depth: plain argmin would return index 2
    # (the first NaN); nanargmin must return the closest valid depth instead.
    values = np.array([5.0, 15.0, np.nan, np.nan])
    assert _closest_index_in_1d(values, 5.0) == 0
    assert _closest_index_in_1d(values, 500.0) == 1


def test_closest_index_in_1d_all_nan_raises():
    values = np.array([np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match="entirely NaN"):
        _closest_index_in_1d(values, 10.0)
