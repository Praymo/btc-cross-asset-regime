"""End-to-end check for the close-only execution convention."""

from __future__ import annotations

import unittest

import pandas as pd

from research import _weekly_hold, strategy_returns


class ExecutionTimingTest(unittest.TestCase):
    def test_friday_signal_trades_monday_close_and_earns_from_tuesday(self):
        index = pd.bdate_range("2024-01-01", periods=10)
        desired = pd.Series(0.0, index=index)
        desired.loc["2024-01-05":] = 1.0
        weight = _weekly_hold(desired, index)

        qqq = pd.Series(100.0, index=index)
        qqq.loc["2024-01-08"] = 110.0
        qqq.loc["2024-01-09":] = 121.0
        prices = pd.DataFrame(
            {
                "QQQ": qqq,
                "SHY": 100.0,
                "QQQ_dividend": 0.0,
                "SHY_dividend": 0.0,
            },
            index=index,
        )

        returns = strategy_returns(
            prices,
            pd.DataFrame({"gate": weight}),
            one_way_cost_bps=0.0,
        )

        self.assertEqual(float(weight.loc["2024-01-08"]), 1.0)
        self.assertEqual(float(returns.loc["2024-01-08", "gate"]), 0.0)
        self.assertAlmostEqual(float(returns.loc["2024-01-09", "gate"]), 0.1)


if __name__ == "__main__":
    unittest.main()
