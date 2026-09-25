"""The buffered decile rule (prism.portfolio.construct.buffered_decile_row; spec rule
``decile_long_short_buffered``): Novy-Marx and Velikov (2016)'s buy/hold spread on both legs."""

import numpy as np
import pandas as pd
import pytest

from prism.portfolio.construct import _decile_row, buffered_decile_row
from prism.spec import SpecError, apply_to_weights, resolve_to_weights_params


def test_with_no_previous_legs_it_is_the_plain_decile_row():
    scores = np.random.default_rng(1).normal(size=200)
    scores[[3, 17]] = np.nan
    np.testing.assert_array_equal(buffered_decile_row(scores, np.zeros(200), 0.10, 0.20), _decile_row(scores, 0.10))


def test_a_held_name_stays_inside_the_hold_band_and_leaves_outside_it():
    scores = np.arange(100, dtype=float)  # name i ranks i-th from the bottom
    previous = np.zeros(100)
    previous[[85, 70]] = 1.0  # long last time: 85 is in the top 20%, 70 is not
    previous[[12, 30]] = -1.0  # short last time: 12 is in the bottom 20%, 30 is not
    row = buffered_decile_row(scores, previous, 0.10, 0.20)
    assert row[85] > 0 and row[70] == 0.0
    assert row[12] < 0 and row[30] == 0.0
    assert (row[90:] > 0).all() and (row[:10] < 0).all()  # the entry deciles always enter
    assert row[row > 0].sum() == pytest.approx(0.5) and row[row < 0].sum() == pytest.approx(-0.5)
    assert np.allclose(row[row > 0], 0.5 / 11)  # ten entrants plus one holder, equal weight


def test_a_held_name_without_a_score_is_flat_and_a_thin_cross_section_is_empty():
    scores = np.arange(50, dtype=float)
    scores[45] = np.nan
    previous = np.zeros(50)
    previous[45] = 1.0
    assert buffered_decile_row(scores, previous, 0.10, 0.20)[45] == 0.0
    assert not buffered_decile_row(np.array([1.0, np.nan, 2.0]), np.zeros(3), 0.10, 0.20).any()


def test_the_buffer_trades_less_on_a_persistent_noisy_score():
    rng = np.random.default_rng(3)
    latent = rng.normal(size=300)
    plain_prev, buffered_prev = np.zeros(300), np.zeros(300)
    plain_turnover = buffered_turnover = 0.0
    for _ in range(120):
        latent = 0.97 * latent + 0.25 * rng.normal(size=300)
        plain = _decile_row(latent, 0.10)
        buffered = buffered_decile_row(latent, np.sign(buffered_prev), 0.10, 0.20)
        plain_turnover += np.abs(plain - plain_prev).sum()
        buffered_turnover += np.abs(buffered - buffered_prev).sum()
        plain_prev, buffered_prev = plain, buffered
    assert buffered_turnover < 0.8 * plain_turnover


def test_the_spec_rule_reads_the_components_own_previous_output_and_validates_its_parameters():
    index = [f"S{i}" for i in range(100)]
    scores = pd.Series(np.arange(100, dtype=float), index=index)
    params = {"decile": 0.10, "hold_quantile": 0.20, "max_gross": 1.0, "max_symbol_abs_weight": 0.35}
    fresh = apply_to_weights("decile_long_short_buffered", scores, params)
    plain = apply_to_weights("decile_long_short", scores, {"decile": 0.10, "max_gross": 1.0,
                                                           "max_symbol_abs_weight": 0.35})
    pd.testing.assert_series_equal(fresh, plain)
    held = apply_to_weights("decile_long_short_buffered", scores, params, previous={"S85": 0.05, "S70": 0.05})
    assert held["S85"] > 0 and held["S70"] == 0.0
    with pytest.raises(SpecError):
        resolve_to_weights_params("decile_long_short_buffered", {"decile": 0.2, "hold_quantile": 0.1})
    with pytest.raises(SpecError):
        resolve_to_weights_params("decile_long_short_buffered", {"decile": 0.1})
