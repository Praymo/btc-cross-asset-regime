"""Render static figures from the cached BTC and ETF data."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT))

from research import run_research  # noqa: E402


COLORS = {
    "ink": "#1F2933",
    "qqq": "#7A8793",
    "btc": "#C96A1B",
    "confirm": "#8B6FB3",
    "regime": "#245B8A",
    "anti_fiat": "#A85D78",
    "grid": "#D7DEE5",
}


def _finish(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(colors=COLORS["ink"], labelsize=9)


def render_equity_curves(returns: pd.DataFrame) -> None:
    equity = (1 + returns.fillna(0.0)).cumprod()
    order = [
        "QQQ buy & hold",
        "BTC gate",
        "BTC + QQQ confirm",
        "Regime score",
        "Regime score + experimental anti-fiat cap",
    ]
    color_order = [COLORS["qqq"], COLORS["btc"], COLORS["confirm"], COLORS["regime"], COLORS["anti_fiat"]]

    fig, ax = plt.subplots(figsize=(11, 5.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for name, color in zip(order, color_order):
        style = "--" if "experimental" in name else "-"
        width = 1.5 if name == "QQQ buy & hold" else 1.9
        ax.plot(equity.index, equity[name], label=name, color=color, linewidth=width, linestyle=style)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1, log scale")
    ax.set_title("BTC cross-asset regime candidates: cumulative wealth")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8.5)
    _finish(ax)
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.text(0.01, 0.01, "2016-08-25 to 2026-08-24; QQQ/SHY total-return legs include Nasdaq cash distributions; 5 bp one-way cost.", fontsize=8, color=COLORS["ink"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "strategy_equity_curves.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def render_period_metrics(by_period: pd.DataFrame) -> None:
    periods = ["Development 2016-2019", "Validation 2020-2022", "Holdout 2023-2026"]
    strategies = ["QQQ buy & hold", "BTC gate", "Regime score"]
    labels = ["Development", "Validation", "Holdout"]
    colors = [COLORS["qqq"], COLORS["btc"], COLORS["regime"]]
    width = 0.24
    x = np.arange(len(periods))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    for offset, strategy, color in zip((-width, 0, width), strategies, colors):
        subset = by_period.set_index(["Period", "Strategy"]).loc[(periods, strategy), :].reset_index()
        axes[0].bar(x + offset, subset["Sharpe_rf0"], width=width, color=color, label=strategy, edgecolor=COLORS["ink"], linewidth=0.35)
        axes[1].bar(x + offset, subset["Max_drawdown"] * 100, width=width, color=color, edgecolor=COLORS["ink"], linewidth=0.35)

    axes[0].set_title("Sharpe by period")
    axes[0].set_ylabel("Sharpe (risk-free rate = 0)")
    axes[1].set_title("Maximum drawdown by period")
    axes[1].set_ylabel("Maximum drawdown (%)")
    for ax in axes:
        ax.set_xticks(x, labels)
        _finish(ax)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.text(0.01, 0.01, "Three main comparators shown; the full metrics table retains the QQQ-confirm and anti-fiat-cap ablations.", fontsize=8, color=COLORS["ink"])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "period_sharpe_drawdown.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_differences(returns: pd.DataFrame, block_size: int = 20, simulations: int = 2000, seed: int = 42) -> tuple[np.ndarray, float]:
    paired = returns[["Regime score", "BTC gate"]].dropna()
    values = paired.to_numpy()
    n = len(values)
    rng = np.random.default_rng(seed)
    differences = np.empty(simulations)
    max_start = max(1, n - block_size + 1)
    for simulation in range(simulations):
        starts = rng.integers(0, max_start, size=int(np.ceil(n / block_size)))
        sample = np.concatenate([values[start : start + block_size] for start in starts], axis=0)[:n]
        sharpes = [sample[:, column].mean() / sample[:, column].std(ddof=1) * np.sqrt(252) for column in range(2)]
        differences[simulation] = sharpes[0] - sharpes[1]
    observed = paired["Regime score"].mean() / paired["Regime score"].std(ddof=1) * np.sqrt(252) - paired["BTC gate"].mean() / paired["BTC gate"].std(ddof=1) * np.sqrt(252)
    return differences, float(observed)


def render_bootstrap(returns: pd.DataFrame) -> None:
    holdout = returns.loc["2023-01-01":"2026-08-24"]
    differences, observed = _bootstrap_differences(holdout)
    p05, p95 = np.quantile(differences, [0.05, 0.95])
    probability = float((differences > 0).mean())
    reported = pd.read_csv(OUTPUTS / "holdout_sharpe_bootstrap.csv", index_col=0)["value"]
    assert np.isclose(observed, reported["observed_diff"])
    assert np.isclose(p05, reported["p05"])
    assert np.isclose(p95, reported["p95"])
    assert np.isclose(probability, reported["probability_diff_gt_0"])

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.hist(differences, bins=36, color="#B7C9D8", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=COLORS["ink"], linewidth=1.0, label="Zero difference")
    ax.axvline(observed, color=COLORS["btc"], linewidth=2.0, linestyle="--", label=f"Observed: {observed:+.2f}")
    ax.axvspan(p05, p95, color="#E8EEF3", alpha=0.65, zorder=0)
    ax.text(0.98, 0.95, f"90% interval: [{p05:+.2f}, {p95:+.2f}]\nP(diff > 0): {probability:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=COLORS["ink"])
    ax.set_xlabel("Regime score Sharpe − BTC gate Sharpe")
    ax.set_ylabel("Bootstrap draws")
    ax.set_title("Holdout Sharpe difference under a 20-session moving block bootstrap")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    _finish(ax)
    fig.text(0.01, 0.01, "2023-01-01 to 2026-08-24; 2,000 paired draws; seed=42. The interval crossing zero limits the claim.", fontsize=8, color=COLORS["ink"])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES / "holdout_sharpe_bootstrap.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    _, _, _, _, returns, by_period, _ = run_research(force_download=False)
    render_equity_curves(returns)
    render_period_metrics(by_period)
    render_bootstrap(returns)
    print("Rendered 3 BTC regime research figures in figures/")


if __name__ == "__main__":
    main()
