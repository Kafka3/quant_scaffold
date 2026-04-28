# Phase 6A — Offline Top-4 Dynamic Controller Summary
> 生成时间: 2026-04-28 17:32
> 候选: tpe_trial_490
> 动态参数: right_bars, rr_target, min_separation, min_close_ratio
> Policy: always_execute / simple_rule_filter / learned_policy (RandomForest)

---
## 动作空间

| Action | right_bars | rr_target | min_separation | min_close_ratio |
|--------|:----------:|:---------:|:--------------:|:---------------:|
| Skip Trade                | — | — | — | — |
| Base tpe_trial_490        |  5 |  3.14 | 15 | 0.5124 |
| Faster Confirmation       |  3 |   2.4 |  8 |  0.85 |
| Balanced                  |  4 |   2.6 | 10 |  0.85 |
| Quality                   |  4 |   2.8 | 12 |   0.9 |
| High Quality Low Freq     |  5 |   3.0 | 14 |   0.9 |

---
## 数据集

- 总信号行数: 410
- 特征数: 16
- 时间段: <StringArray>
['train', 'val', 'test']
Length: 3, dtype: str
- 各 action 信号分布:

  - base_tpe490: 187 条
  - faster_confirm: 78 条
  - balanced: 67 条
  - quality: 41 条
  - high_quality_low_freq: 37 条

---
## Policy 对比

### train

| Policy | 执行数 | 跳过数 | 执行率 | 总收益 | PF | Sharpe | DD | avg_r | WinRate |
|-------|:------:|:------:|:-----:|:-----:|:--:|:------:|:--:|:----:|:-------:|
| always_execute       | 185 |   0 | 100% | 296.50% | 2.46 | 4.25 | -8.28% | 0.87 | 43.8% |
| simple_rule_filter   | 185 |   0 | 100% | 296.50% | 2.46 | 4.25 | -8.28% | 0.87 | 43.8% |
| learned_policy       | 126 |   0 | 100% | 492.90% | 14.87 | 16.32 | -1.07% | 2.29 | 84.1% |

### val

| Policy | 执行数 | 跳过数 | 执行率 | 总收益 | PF | Sharpe | DD | avg_r | WinRate |
|-------|:------:|:------:|:-----:|:-----:|:--:|:------:|:--:|:----:|:-------:|
| always_execute       |  26 |   0 | 100% |  -0.47% | 0.96 | -0.06 | -8.07% | -0.01 | 15.4% |
| simple_rule_filter   |  26 |   0 | 100% |  -0.47% | 0.96 | -0.06 | -8.07% | -0.01 | 15.4% |
| learned_policy       |  27 |   0 | 100% |   0.07% | 1.00 | -0.00 | -9.43% | -0.00 | 18.5% |

### test

| Policy | 执行数 | 跳过数 | 执行率 | 总收益 | PF | Sharpe | DD | avg_r | WinRate |
|-------|:------:|:------:|:-----:|:-----:|:--:|:------:|:--:|:----:|:-------:|
| always_execute       |  34 |   0 | 100% |   0.19% | 1.01 | 0.09 | -9.07% | 0.02 | 20.6% |
| simple_rule_filter   |  34 |   0 | 100% |   0.19% | 1.01 | 0.09 | -9.07% | 0.02 | 20.6% |
| learned_policy       |  25 |   0 | 100% | -10.53% | 0.45 | -2.99 | -12.44% | -0.40 | 12.0% |

### 2024-2025-Full

| Policy | 执行数 | 跳过数 | 执行率 | 总收益 | PF | Sharpe | DD | avg_r | WinRate |
|-------|:------:|:------:|:-----:|:-----:|:--:|:------:|:--:|:----:|:-------:|
| always_execute       | 245 |   0 | 100% | 296.22% | 2.25 | 3.32 | -8.28% | 0.66 | 37.6% |
| simple_rule_filter   | 245 |   0 | 100% | 296.22% | 2.25 | 3.32 | -8.28% | 0.66 | 37.6% |
| learned_policy       | 178 |   0 | 100% | 482.44% | 7.93 | 8.52 | -3.94% | 1.56 | 64.0% |

---
## Top-10 特征重要性

| 特征 | 重要性 |
|------|:------:|
| stoch_slope | 0.1429 |
| hour_of_day | 0.1399 |
| atr_pct | 0.1094 |
| channel_width | 0.1083 |
| above_ratio | 0.0926 |
| price_position_vs_channel | 0.0765 |
| day_of_week | 0.0731 |
| volume_zscore | 0.0720 |
| ema_slope | 0.0709 |
| stoch_d | 0.0471 |

---
## Test 结果 (2025-Q4, 严格样本外)

| 指标 | always_execute | simple_rule_filter | learned_policy |
|------|:--------------:|:------------------:|:--------------:|
| total_return | 0.19 | 0.19 | -10.53 |
| profit_factor | 1.01 | 1.01 | 0.45 |
| sharpe_ratio | 0.09 | 0.09 | -2.99 |
| max_drawdown_pct | -9.07 | -9.07 | -12.44 |
| expectancy_r | 0.02 | 0.02 | -0.40 |
| win_rate | 0.21 | 0.21 | 0.12 |
| executed_trades | 34 | 34 | 25 |
| execution_rate | 1.00 | 1.00 | 1.00 |

### 通过检查: ❌ 未全部通过
- ❌ PF更高
- ❌ ExpectR更高
- ❌ DD不恶化
- ✅ 执行数>=50%

**结论: ❌ Phase 6A 未完全通过 — 需要进一步调优**

---
### 下一阶段

- Phase 6B: 扩展动作空间 / 引入 PPO
- Phase 5A: Freqtrade Adapter + Backtest Parity
