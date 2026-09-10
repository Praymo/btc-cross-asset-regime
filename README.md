# BTC Cross-Asset Regime Research / BTC 跨资产状态研究

This release tests whether BTC momentum can be used as a stock risk switch after conditioning on QQQ trend, gold, the dollar, and long bonds. It preserves the original BTC → QQQ/SHY rule as a baseline and treats the cross-asset score as an experimental candidate.

这个发布版检验：在加入 QQQ 趋势、黄金、美元和长债之后，BTC 动量能否更可靠地作为股票风险开关。原始 BTC → QQQ/SHY 规则保留为基线，跨资产评分模型仍是待验证候选。

## Hypothesis and contribution / 假设与贡献

- **Hypothesis:** BTC can behave as a liquidity risk-on sensor, but BTC, gold, the dollar, and equities can also rise or fall for different reasons. A cross-asset confirmation rule may reduce false equity risk-on signals.
- **Implementation:** Signals are formed at the Friday/last-session close and held from the following session. The model allocates between QQQ and SHY, allows 0%, 50%, or 100% QQQ exposure, and charges 5 bp per one-way trade in the reported returns.
- **Contribution:** The notebook separates development (2016–2019), validation (2020–2022), and holdout (2023–2026) periods, reports exposure, turnover, drawdown and costs, and uses a moving-block bootstrap for the holdout Sharpe difference.

- **假设：** BTC 可以充当流动性 risk-on 传感器，但 BTC、黄金、美元和股票也可能因为不同原因同涨同跌。跨资产确认规则可能减少错误的股票 risk-on 信号。
- **实现：** 信号在周五或当周最后一个交易日收盘形成，从下一交易时段执行。模型在 QQQ 与 SHY 之间配置，QQQ 仓位允许 0%、50% 或 100%，报告收益按单边 5bp 交易成本扣除。
- **贡献：** 笔记本区分开发期（2016–2019）、验证期（2020–2022）和留出期（2023–2026），报告风险暴露、换手、回撤和成本，并用移动区块 bootstrap 检验留出样本的夏普差异。

## Results from the included snapshot / 随附快照结果

The public cache runs from 2016-08-25 to 2026-08-24. Full-sample Sharpe is **1.29** for the original BTC gate and **1.34** for the regime score. The holdout Sharpe is **1.55** versus **1.90**, with maximum drawdown **-12.6%** versus **-8.6%**. The validation-period Sharpe is **1.04** versus **0.63**, so the improvement is state-dependent and not a stable claim of superiority. The included outputs also report that the holdout bootstrap probability of a positive Sharpe difference is about 89%, while the 90% interval crosses zero.

The short-term pattern “BTC and gold up, dollar and QQQ down” identifies a recent anti-fiat divergence, but its historical next-20-session QQQ return is not reliably worse. It remains a diagnostic label rather than a forced short or liquidation rule.

公共缓存区间是 2016-08-25 至 2026-08-24。全样本中，原始 BTC 开关夏普为 **1.29**，跨资产评分为 **1.34**。留出期夏普分别为 **1.55** 与 **1.90**，最大回撤分别为 **-12.6%** 与 **-8.6%**。验证期夏普分别为 **1.04** 与 **0.63**，因此改善具有状态依赖，不能写成稳定胜出。随附输出还显示，留出期 bootstrap 得到评分模型夏普差为正的概率约 89%，但 90% 区间跨过零。

“BTC 与黄金上涨、美元与 QQQ 下跌”的短期组合能识别近期反法币背离，但历史上其后 20 个交易日 QQQ 收益并没有稳定变差。因此它只保留为诊断标签，不自动变成做空或清仓规则。

## Reproduce / 复现

The release includes a modest public cache. Use Python 3.10 or newer.

发布版包含规模有限的公共数据缓存。建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Recompute the included result tables from the cached data
python research.py

# Refresh public data first, if needed
python scripts/download_data.py
python research.py

# Execute the notebook to a separate file; the distributed notebook has no old outputs
jupyter nbconvert --to notebook --execute btc_cross_asset_regime_research.ipynb \
  --output /tmp/btc-regime-executed.ipynb
```

The default research entry point uses the included cache and writes `outputs/holdout_sharpe_bootstrap.csv` with the fixed-seed holdout comparison used in the notebook. `scripts/download_data.py` explicitly requests current public data from Coin Metrics and Nasdaq; source revisions, missing rows, and API limits can change results.

默认研究入口使用随附缓存。`scripts/download_data.py` 会明确从 Coin Metrics 和 Nasdaq 请求公共数据；源数据修订、缺失值和 API 限流都可能改变结果。

## Data / 数据

The cache contains daily BTC `PriceUSD` from the Coin Metrics Community API, Nasdaq historical closes for QQQ/SHY/GLD/UUP/TLT, and Nasdaq dividend records for QQQ and SHY. The ETF total-return calculation adds cash distributions on ex-dates; the other assets are signals only. No private or machine-specific files are included.

缓存包含 Coin Metrics Community API 的 BTC `PriceUSD` 日数据、Nasdaq 的 QQQ/SHY/GLD/UUP/TLT 历史收盘价，以及 QQQ/SHY 的 Nasdaq 分红记录。ETF 总回报在除息日加入现金分红；其他资产只用于信号。发布版不包含私人数据或机器路径。

The metadata and source endpoints are recorded in `data/data_metadata.json` and in `research.py`:

- <https://community-api.coinmetrics.io/v4/timeseries/asset-metrics>
- <https://api.nasdaq.com/api/quote/QQQ/historical>
- <https://api.nasdaq.com/api/quote/QQQ/dividends>

## Limitations / 限制

The sample is historical and relatively short for regime inference. Nasdaq's public history boundary determines the start date; ETF closes are not vendor-adjusted total-return series; BTC trades continuously while ETFs trade on U.S. sessions; the strategy has high turnover; the zero-risk-free Sharpe is only a common comparison metric; and the cross-asset score was not selected through a live walk-forward process. This is research code, not investment advice.

样本是历史数据，且对状态切换推断来说仍然偏短。起始日期受 Nasdaq 公开历史边界影响；ETF 收盘价不是供应商调整后的总回报序列；BTC 全天交易而 ETF 按美国交易时段交易；策略换手较高；零无风险利率夏普只用于同口径比较；跨资产评分也没有经过实时滚动训练。因此这是研究代码，不构成投资建议。

## License and assistance / 许可证与辅助工具

Original code and project-authored documentation are released under the MIT License. Source datasets retain their respective providers’ terms; the code license does not grant additional rights over those datasets. Development and release preparation used Codex assistance; results are historical research outputs, not live performance.

原创代码与项目文档使用 MIT License。源数据遵循各提供方条款，代码许可证不额外授予数据权利。开发与发布整理使用了 Codex 辅助；结果属于历史研究输出，不是实盘业绩。
