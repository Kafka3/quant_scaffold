# Phase 4F — 2026 Q1 True Out-of-Sample Validation
> 生成时间: 2026-04-28 21:46
> 候选: tpe_trial_490
> 配置: 1% risk, 10x cap, isolated, max_open=1

---
## 1. 数据

- 文件: `data/raw/BTCUSDT_5m_2026Q1.csv`
- 周期: 2026-01-01 00:00:00+00:00 ~ 2026-04-01 00:00:00+00:00
- K 线数: 25,921
- 缺失: 无
- 信号: long=14, short=13

## 2. 结果

| 指标 | 值 |
|------|----|
| total_return | 9.61% |
| total_trades | 26 |
| win_rate | 34.6% |
| profit_factor | 1.51 |
| sharpe_ratio | 1.95 |
| max_drawdown_pct | -8.51% |
| expectancy_r | 0.38 |
| long_trades | 13 |
| short_trades | 13 |
| long_pnl | 2572.05 |
| short_pnl | 7039.06 |
| avg_actual_risk_pct | 0.0099% |
| max_actual_risk_pct | 0.0100% |
| avg_leverage | 3.59x |
| max_leverage_observed | 10.00x |
| p95_leverage | 9.60x |
| skipped_trades | 0 |
| skip_rate | 0.0% |
| skip_stop_too_tight | 0 |
| skip_leverage_exceeds_cap | 0 |
| total_fees | 260.00 |
| total_slippage | 1233.40 |
| cost_as_pct_of_gross_profit | 5.3% |

## 3. 验收标准

- ✅ total_trades >= 10
- ✅ total_return > 0
- ✅ profit_factor >= 1.3
- ✅ sharpe_ratio > 0
- ✅ max_drawdown_pct <= 10%
- ✅ max_leverage_observed <= 10
- ✅ skip_rate < 50% (不异常高)
- ✅ long/short 不极端失衡

**结论: ✅ Phase 4F 通过**

## 4. 建议

✅ 2026-Q1 样本外验证通过。

**建议: 进入 Phase 5A — Freqtrade Adapter + Backtest Parity。**

---

> 固定配置: tpe_trial_490 | 1% actual risk | 10x hard cap | isolated | max_open=1 | min_stop=0.1%
