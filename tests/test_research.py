"""Regression tests for btc-cross-asset-regime.

These tests pin the execution-fairness properties the README claims:
Friday signal -> next-session execution (no same-close fills), dividend
total returns, turnover cost accounting, and drawdown computation.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research import (
    _total_return,
    _weekly_hold,
    build_weights,
    max_drawdown,
    strategy_returns,
)


def trading_index(start: str = "2024-01-01", periods: int = 40) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


class WeeklyHoldTest(unittest.TestCase):
    def test_no_same_close_execution(self):
        """A signal first true on Friday must not earn Friday's return.

        This is the single most important fairness property: the weight
        applied to Friday's close must reflect information available at
        Thursday's close (i.e., the previous week's state).
        """
        index = trading_index()
        # Signal flips on at 2024-01-05 (the first Friday in the index).
        desired = pd.Series(0.0, index=index)
        desired.loc["2024-01-05":] = 1.0
        weight = _weekly_hold(desired, index)
        # On the Friday the signal appears, the applied weight must still be 0.
        self.assertEqual(float(weight.loc["2024-01-05"]), 0.0)
        # It becomes 1 only from the next trading session.
        self.assertEqual(float(weight.loc["2024-01-08"]), 1.0)

    def test_weight_always_bounded_zero_one(self):
        index = trading_index(periods=120)
        rng = np.random.default_rng(5)
        desired = pd.Series(rng.uniform(-1, 2, len(index)), index=index)
        weight = _weekly_hold(desired, index)
        self.assertTrue(((weight >= 0.0) & (weight <= 1.0)).all())

    def test_pre_signal_weight_is_zero(self):
        index = trading_index(periods=40)
        desired = pd.Series(0.0, index=index)
        weight = _weekly_hold(desired, index)
        self.assertTrue((weight == 0.0).all())


class TotalReturnTest(unittest.TestCase):
    def test_dividend_adds_to_return(self):
        close = pd.Series([100.0, 100.0, 100.0], index=trading_index(periods=3))
        dividend = pd.Series([0.0, 2.0, 0.0], index=close.index)
        ret = _total_return(close, dividend)
        # Price flat, 2.0 dividend on a 100 base -> +2% on the ex-date session.
        self.assertAlmostEqual(float(ret.iloc[1]), 0.02, places=12)

    def test_no_dividend_matches_price_return(self):
        close = pd.Series([100.0, 105.0, 99.0], index=trading_index(periods=3))
        pd.testing.assert_series_equal(
            _total_return(close, None),
            _total_return(close, pd.Series(0.0, index=close.index)),
        )


class CostAccountingTest(unittest.TestCase):
    def test_turnover_cost_reduces_return(self):
        """A flip 0 -> 1 -> 0 costs 2x one-way on each flip and must be charged."""
        index = trading_index(periods=30)
        weights = pd.DataFrame(
            {
                "flip": pd.Series(0.0, index=index),
            }
        )
        weights.loc[index[5]:index[14], "flip"] = 1.0  # hold one week-ish, then out
        prices = pd.DataFrame(
            {
                "QQQ": pd.Series(np.full(30, 100.0), index=index),
                "SHY": pd.Series(np.full(30, 100.0), index=index),
                "QQQ_dividend": 0.0,
                "SHY_dividend": 0.0,
            },
            index=index,
        )
        gross_zero = strategy_returns(prices, weights, one_way_cost_bps=0.0)
        gross_five = strategy_returns(prices, weights, one_way_cost_bps=5.0)
        # Flat prices: every difference between the two runs is pure cost.
        total_cost = (gross_five["flip"] - gross_zero["flip"]).sum()
        # Two flips (in, out) x 2 sides x 5bp = 20bp total drag.
        self.assertAlmostEqual(-total_cost, 0.0020, places=10)


class DrawdownTest(unittest.TestCase):
    def test_known_drawdown(self):
        returns = pd.Series([0.10, -0.50, 0.10, 0.10])
        # Equity: 1.1 -> 0.55 -> 0.605 -> 0.6655; peak 1.1, trough 0.55.
        self.assertAlmostEqual(max_drawdown(returns), -0.50, places=12)

    def test_monotonic_gains_have_zero_drawdown(self):
        self.assertEqual(max_drawdown(pd.Series([0.01, 0.02, 0.03])), 0.0)


class GateLogicTest(unittest.TestCase):
    def test_btc_gate_requires_both_conditions(self):
        index = trading_index(periods=200)
        rng = np.random.default_rng(9)
        features = pd.DataFrame(
            {
                "btc_above_sma50": rng.random(200) > 0.5,
                "btc_roc20_pos": rng.random(200) > 0.5,
                "qqq_above_sma100": rng.random(200) > 0.5,
                "qqq_roc20_pos": rng.random(200) > 0.5,
                "qqq_roc20": rng.normal(0, 0.05, 200),
                "qqq_roc5": rng.normal(0, 0.02, 200),
                "btc_roc20": rng.normal(0, 0.1, 200),
                "btc_roc5": rng.normal(0, 0.03, 200),
                "gold_roc20": rng.normal(0, 0.04, 200),
                "gold_roc5": rng.normal(0, 0.02, 200),
                "dollar_roc20": rng.normal(0, 0.03, 200),
                "dollar_roc5": rng.normal(0, 0.01, 200),
                "long_bond_roc20": rng.normal(0, 0.04, 200),
            },
            index=index,
        )
        weights = build_weights(features)
        gate_weight = weights["BTC gate"]
        # Weekly holding means the applied weight is the previous week's gate;
        # wherever it is 1, the raw gate must have been true the prior week.
        shifted_gate = (
            (features["btc_above_sma50"] & features["btc_roc20_pos"])
            .astype(float)
            .resample("W-FRI")
            .last()
            .reindex(index, method="ffill")
            .shift(1)
            .fillna(0.0)
        )
        mask = gate_weight.gt(0)
        self.assertTrue(
            (gate_weight[mask] == shifted_gate[mask]).all(),
            "applied weight must equal the lagged raw gate",
        )


if __name__ == "__main__":
    unittest.main()
