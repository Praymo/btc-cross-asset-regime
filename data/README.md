# Public data inputs / 公共数据输入

`cross_asset_daily.csv` is a compact public snapshot used to generate the included outputs. It contains no private records. Its provenance is recorded in `data_metadata.json`.

`cross_asset_daily.csv` 是生成随附输出的规模有限公共数据快照，不包含私人记录；来源信息见 `data_metadata.json`。

- BTC `PriceUSD`: Coin Metrics Community API, <https://community-api.coinmetrics.io/v4/timeseries/asset-metrics>
- ETF closes: Nasdaq historical API, for example <https://api.nasdaq.com/api/quote/QQQ/historical>
- QQQ and SHY dividends: Nasdaq dividend API, for example <https://api.nasdaq.com/api/quote/QQQ/dividends>

Run `python scripts/download_data.py` from the project root to refresh the cache. The API may revise history or reject requests; the included result tables correspond to the included snapshot, as of 2026-08-24.
