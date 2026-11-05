"""Bateman solver checked against closed-form solutions."""

import numpy as np
import pytest

from nucmed.decay import (F18, MO99, MO99_TO_TC99M_BRANCH, TC99M, activities, generator_elution,
                          parent_daughter_analytic, populations, time_of_max_daughter_activity)


def test_single_nuclide_half_life():
    a = activities([F18], [1e6], [0.0, F18.half_life_h, 2 * F18.half_life_h])[:, 0]
    np.testing.assert_allclose(a / a[0], [1.0, 0.5, 0.25], rtol=1e-12)


def test_matrix_exponential_matches_analytic_parent_daughter():
    t = np.linspace(0, 200, 81)
    num = populations([MO99, TC99M], [1e9, 0.0], t, [MO99_TO_TC99M_BRANCH])[:, 1]
    ana = parent_daughter_analytic(MO99.lam, TC99M.lam, 1e9, t, MO99_TO_TC99M_BRANCH)
    np.testing.assert_allclose(num, ana, rtol=1e-9, atol=1e-6)


def test_time_of_maximum_daughter_activity():
    t = np.linspace(0, 60, 60001)
    a_tc = activities([MO99, TC99M], [1.0, 0.0], t, [MO99_TO_TC99M_BRANCH])[:, 1]
    t_max = time_of_max_daughter_activity(MO99.lam, TC99M.lam)
    assert t[np.argmax(a_tc)] == pytest.approx(t_max, abs=1e-3)
    assert 22.0 < t_max < 24.0  # the textbook "about 23 h" for Mo/Tc generators


def test_transient_equilibrium_ratio():
    """At late times A_d / A_p -> b * ld / (ld - lp)."""
    a = activities([MO99, TC99M], [1.0, 0.0], [240.0], [MO99_TO_TC99M_BRANCH])[0]
    expected = MO99_TO_TC99M_BRANCH * TC99M.lam / (TC99M.lam - MO99.lam)
    assert a[1] / a[0] == pytest.approx(expected, rel=1e-6)


def test_generator_regrowth_after_elution():
    """After a full elution at t=24 h the column regrows from zero, so the
    next elution 24 h later yields less than the first (parent has decayed)."""
    _, tc, eluted = generator_elution([24.0, 48.0], a_mo0=100.0, t_end_h=50.0, dt_h=0.01)
    assert len(eluted) == 2
    assert eluted[1] < eluted[0]
    ratio_expected = np.exp(-MO99.lam * 24.0)  # same ingrowth time, parent smaller
    assert eluted[1] / eluted[0] == pytest.approx(ratio_expected, rel=1e-3)
