from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_PATH = DATA_DIR / "cross_asset_daily.csv"
META_PATH = DATA_DIR / "data_metadata.json"

START = "2016-08-25"
END = "2026-08-24"
NASDAQ_TICKERS = ("QQQ", "SHY", "GLD", "UUP", "TLT")


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return session


def _fetch_nasdaq_window(
    session: requests.Session, ticker: str, start: str, end: str
) -> pd.DataFrame:
    url = f"https://api.nasdaq.com/api/quote/{ticker}/historical"
    params = {
        "assetclass": "etf",
        "fromdate": start,
        "todate": end,
        "limit": 5000,
    }
    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", {}).get("tradesTable", {}).get("rows") or []
    if not rows:
        raise RuntimeError(f"Nasdaq returned no price rows for {ticker}: {start} to {end}")
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y")
    frame["close"] = pd.to_numeric(frame["close"].astype(str).str.replace("$", "", regex=False))
    return frame[["date", "close"]]


def fetch_nasdaq_prices(session: requests.Session, ticker: str) -> pd.Series:
    # Nasdaq exposes a rolling ten-year window and rejects fully historical
    # sub-windows. Request the whole supported range and validate the first date.
    windows = ((START, END),)
    frames = [_fetch_nasdaq_window(session, ticker, start, end) for start, end in windows]
    prices = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last")
    prices = prices.sort_values("date").set_index("date")["close"]
    prices.name = ticker
    return prices


def fetch_nasdaq_dividends(session: requests.Session, ticker: str) -> pd.Series:
    url = f"https://api.nasdaq.com/api/quote/{ticker}/dividends"
    response = session.get(
        url, params={"assetclass": "etf", "limit": 500}, timeout=60
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", {}).get("dividends", {}).get("rows") or []
    if not rows:
        return pd.Series(dtype=float, name=f"{ticker}_dividend")
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["exOrEffDate"], format="%m/%d/%Y")
    frame["amount"] = pd.to_numeric(
        frame["amount"].astype(str).str.replace("$", "", regex=False), errors="coerce"
    )
    dividends = frame.groupby("date")["amount"].sum().sort_index()
    dividends.name = f"{ticker}_dividend"
    return dividends


def fetch_coinmetrics_btc(session: requests.Session) -> pd.Series:
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params = {
        "assets": "btc",
        "metrics": "PriceUSD",
        "frequency": "1d",
        "start_time": START,
        "end_time": END,
        "page_size": 10000,
    }
    response = session.get(url, params=params, timeout=90)
    response.raise_for_status()
    rows = response.json().get("data", [])
    if not rows:
        raise RuntimeError("Coin Metrics returned no BTC data")
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None).dt.normalize()
    frame["BTC"] = pd.to_numeric(frame["PriceUSD"])
    return frame.drop_duplicates("date").set_index("date")["BTC"].sort_index()


def download_data(force: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    session = _session()
    prices = {ticker: fetch_nasdaq_prices(session, ticker) for ticker in NASDAQ_TICKERS}
    btc = fetch_coinmetrics_btc(session)
    data = pd.concat([btc, *prices.values()], axis=1).sort_index()

    for ticker in ("QQQ", "SHY"):
        dividends = fetch_nasdaq_dividends(session, ticker)
        data[f"{ticker}_dividend"] = dividends.reindex(data.index).fillna(0.0)

    data = data.loc[START:END]
    data.to_csv(DATA_PATH, index_label="date")
    metadata = {
        "as_of": END,
        "start": str(data.index.min().date()),
        "end": str(data.index.max().date()),
        "rows": int(len(data)),
        "sources": {
            "BTC": "Coin Metrics Community API PriceUSD, daily",
            "ETFs": "Nasdaq historical close API",
            "dividends": "Nasdaq dividend API for QQQ and SHY",
        },
        "important_caveat": "ETF total returns add cash distributions on ex-dates; prices are not vendor-adjusted closes.",
    }
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def validate_input_data(data: pd.DataFrame) -> pd.DataFrame:
    checks = []
    required = ["BTC", "QQQ", "SHY", "GLD", "UUP", "TLT"]
    for column in required:
        series = data[column]
        checks.append(
            {
                "check": f"{column} coverage",
                "value": f"{series.first_valid_index().date()} to {series.last_valid_index().date()}",
                "status": "PASS" if series.notna().sum() > 2300 else "REVIEW",
            }
        )
        checks.append(
            {
                "check": f"{column} positive prices",
                "value": int((series.dropna() <= 0).sum()),
                "status": "PASS" if (series.dropna() > 0).all() else "FAIL",
            }
        )
    return pd.DataFrame(checks)


def _total_return(close: pd.Series, dividend: pd.Series | None = None) -> pd.Series:
    aligned_dividend = 0.0 if dividend is None else dividend.reindex(close.index).fillna(0.0)
    return (close + aligned_dividend).div(close.shift(1)).sub(1.0)


def _weekly_hold(desired_weight: pd.Series, trading_index: pd.DatetimeIndex) -> pd.Series:
    # Friday/last-session information sets next week's position. The one-row shift
    # prevents same-close execution and eliminates the most common look-ahead error.
    weekly_decision = desired_weight.reindex(trading_index).resample("W-FRI").last()
    weight = weekly_decision.reindex(trading_index, method="ffill").shift(1)
    return weight.fillna(0.0).clip(0.0, 1.0)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    trading = data.loc[data["QQQ"].notna()].copy()
    # BTC trades every day. Compute on its native calendar, then align to US sessions.
    btc_native = data["BTC"].dropna()
    btc_features = pd.DataFrame(index=btc_native.index)
    btc_features["btc_above_sma50"] = btc_native.gt(btc_native.rolling(50).mean())
    btc_features["btc_roc20_pos"] = btc_native.pct_change(20).gt(0)
    btc_features["btc_roc20"] = btc_native.pct_change(20)
    btc_features["btc_roc5"] = btc_native.pct_change(5)
    features = btc_features.reindex(trading.index, method="ffill")

    features["qqq_above_sma100"] = trading["QQQ"].gt(trading["QQQ"].rolling(100).mean())
    features["qqq_roc20"] = trading["QQQ"].pct_change(20)
    features["qqq_roc5"] = trading["QQQ"].pct_change(5)
    features["qqq_roc20_pos"] = features["qqq_roc20"].gt(0)
    features["gold_roc20"] = trading["GLD"].pct_change(20)
    features["gold_roc5"] = trading["GLD"].pct_change(5)
    features["dollar_roc20"] = trading["UUP"].pct_change(20)
    features["dollar_roc5"] = trading["UUP"].pct_change(5)
    features["long_bond_roc20"] = trading["TLT"].pct_change(20)

    binary_columns = [
        "btc_above_sma50",
        "btc_roc20_pos",
        "qqq_above_sma100",
        "qqq_roc20_pos",
    ]
    features[binary_columns] = features[binary_columns].fillna(False).astype(bool)
    return features


def build_weights(features: pd.DataFrame) -> pd.DataFrame:
    weights = pd.DataFrame(index=features.index)
    btc_gate = features["btc_above_sma50"] & features["btc_roc20_pos"]
    weights["BTC gate"] = _weekly_hold(btc_gate.astype(float), features.index)

    qqq_confirm = btc_gate & features["qqq_above_sma100"]
    weights["BTC + QQQ confirm"] = _weekly_hold(qqq_confirm.astype(float), features.index)

    score = (
        features["btc_above_sma50"].astype(int)
        + features["btc_roc20_pos"].astype(int)
        + features["qqq_above_sma100"].astype(int)
        + features["qqq_roc20_pos"].astype(int)
        + features["dollar_roc20"].lt(0).astype(int)
        + features["long_bond_roc20"].gt(0).astype(int)
    )
    scaled = pd.Series(np.select([score >= 5, score >= 3], [1.0, 0.5], default=0.0), index=score.index)
    weights["Regime score"] = _weekly_hold(scaled, features.index)

    anti_fiat = (
        btc_gate
        & features["gold_roc20"].gt(0)
        & features["dollar_roc20"].lt(0)
        & features["qqq_roc20"].lt(0)
    )
    scaled = scaled.mask(anti_fiat, scaled.clip(upper=0.25))
    weights["Regime score + experimental anti-fiat cap"] = _weekly_hold(scaled, features.index)
    return weights


def strategy_returns(
    data: pd.DataFrame, weights: pd.DataFrame, one_way_cost_bps: float = 5.0
) -> pd.DataFrame:
    trading = data.loc[weights.index]
    qqq_ret = _total_return(trading["QQQ"], trading["QQQ_dividend"])
    shy_ret = _total_return(trading["SHY"], trading["SHY_dividend"])
    returns = pd.DataFrame(index=weights.index)
    returns["QQQ buy & hold"] = qqq_ret
    cost_rate = one_way_cost_bps / 10_000.0
    for name in weights:
        qqq_weight = weights[name]
        gross = qqq_weight.shift(1).fillna(0.0) * qqq_ret + (1 - qqq_weight.shift(1).fillna(0.0)) * shy_ret
        two_asset_turnover = 2.0 * qqq_weight.diff().abs().fillna(qqq_weight.abs())
        returns[name] = gross - cost_rate * two_asset_turnover
    return returns.dropna(how="all")


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.fillna(0.0)).cumprod()
    return float((equity.div(equity.cummax()).sub(1)).min())


def performance_metrics(returns: pd.Series, exposure: pd.Series | None = None) -> dict[str, float]:
    clean = returns.dropna()
    years = len(clean) / 252.0
    total = float((1 + clean).prod())
    cagr = total ** (1 / years) - 1 if years > 0 and total > 0 else np.nan
    volatility = float(clean.std(ddof=1) * np.sqrt(252))
    sharpe = float(clean.mean() / clean.std(ddof=1) * np.sqrt(252)) if clean.std(ddof=1) > 0 else np.nan
    drawdown = max_drawdown(clean)
    annual_turnover = 0.0
    if exposure is not None and years > 0:
        annual_turnover = float(2.0 * exposure.reindex(clean.index).diff().abs().sum() / years)
    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe_rf0": sharpe,
        "Max_drawdown": drawdown,
        "Calmar": cagr / abs(drawdown) if drawdown < 0 else np.nan,
        "QQQ_exposure": float(exposure.reindex(clean.index).mean()) if exposure is not None else 1.0,
        "Annual_turnover": annual_turnover,
    }


def moving_block_bootstrap_sharpe_diff(
    candidate: pd.Series,
    baseline: pd.Series,
    block_size: int = 20,
    simulations: int = 2000,
    seed: int = 42,
) -> pd.Series:
    paired = pd.concat([candidate, baseline], axis=1).dropna()
    paired.columns = ["candidate", "baseline"]
    values = paired.to_numpy()
    n = len(values)
    rng = np.random.default_rng(seed)
    differences = np.empty(simulations)
    max_start = max(1, n - block_size + 1)
    for simulation in range(simulations):
        starts = rng.integers(0, max_start, size=int(np.ceil(n / block_size)))
        sample = np.concatenate(
            [values[start : start + block_size] for start in starts], axis=0
        )[:n]
        sharpes = []
        for column in range(2):
            series = sample[:, column]
            sharpes.append(series.mean() / series.std(ddof=1) * np.sqrt(252))
        differences[simulation] = sharpes[0] - sharpes[1]
    return pd.Series(
        {
            "observed_diff": performance_metrics(paired["candidate"])["Sharpe_rf0"]
            - performance_metrics(paired["baseline"])["Sharpe_rf0"],
            "bootstrap_median": float(np.median(differences)),
            "p05": float(np.quantile(differences, 0.05)),
            "p95": float(np.quantile(differences, 0.95)),
            "probability_diff_gt_0": float((differences > 0).mean()),
        }
    )


def summarize(
    returns: pd.DataFrame, weights: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = {
        "Development 2016-2019": (START, "2019-12-31"),
        "Validation 2020-2022": ("2020-01-01", "2022-12-31"),
        "Holdout 2023-2026": ("2023-01-01", END),
        "Full sample": (START, END),
    }
    rows = []
    for period, (start, end) in periods.items():
        for strategy in returns:
            exposure = None if strategy == "QQQ buy & hold" else weights[strategy]
            metrics = performance_metrics(returns.loc[start:end, strategy], exposure)
            rows.append({"Period": period, "Strategy": strategy, **metrics})
    by_period = pd.DataFrame(rows)
    full_sample = by_period.loc[by_period["Period"].eq("Full sample")].drop(columns="Period")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_period.to_csv(OUTPUT_DIR / "metrics_by_period.csv", index=False)
    full_sample.to_csv(OUTPUT_DIR / "full_sample_metrics.csv", index=False)
    return by_period, full_sample


def run_research(force_download: bool = False):
    data = download_data(force=force_download)
    checks = validate_input_data(data)
    features = build_features(data)
    weights = build_weights(features)
    returns = strategy_returns(data, weights)
    by_period, full_sample = summarize(returns, weights)
    holdout = returns.loc["2023-01-01":END]
    holdout_bootstrap = moving_block_bootstrap_sharpe_diff(
        holdout["Regime score"],
        holdout["BTC gate"],
        block_size=20,
        simulations=2000,
        seed=42,
    )
    holdout_bootstrap.to_csv(OUTPUT_DIR / "holdout_sharpe_bootstrap.csv", header=["value"])
    return data, checks, features, weights, returns, by_period, full_sample


if __name__ == "__main__":
    _, checks, _, _, _, _, full_sample = run_research(force_download=False)
    print(checks.to_string(index=False))
    print("\nFull-sample metrics")
    print(full_sample.round(3).to_string(index=False))
    print("\nHoldout Sharpe bootstrap")
    print(pd.read_csv(OUTPUT_DIR / "holdout_sharpe_bootstrap.csv", index_col=0).to_string(header=False))
