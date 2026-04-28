# TPE Phase 1 — Core 12-Parameter Search Summary
- **Study name**: `tpe_core_12p`
- **Total trials**: 500
- **Completed trials**: 196
- **Pruned trials**: 304
- **Failed trials**: 0

## Best Trial (ID 9)
- **Score**: 0.900749
- **Parameters**:
  - `k_period`: 11
  - `d_period`: 2
  - `smooth`: 1
  - `oversold`: 12
  - `overbought`: 88
  - `left_bars`: 6
  - `right_bars`: 4
  - `min_separation`: 14
  - `max_separation`: 49
  - `ema_period`: 33
  - `min_close_ratio`: 0.90165154932049
  - `rr_target`: 2.578684483831301
- **Key Metrics**:
  - Total trades (train): 39
  - Profit factor (train): 5.000
  - Win rate (train): 0.692
  - Expectancy R (train): 1.2628
  - Max drawdown % (train): -1.02
  - Total return % (train): 11.05
  - Quarterly positive ratio: 1.00
  - Val return %: 0.80
  - No-cost return %: 11.83

## Top-10 Results
| Rank | Trial | Score | PF | WinRate | ExpectR | Trades | Q+Ratio | ValRet | NoCostRet |
|------|-------|-------|-----|---------|---------|--------|---------|--------|-----------|
| 1 | 9 | 0.9007 | 5.00 | 0.69 | 1.263 | 39 | 1.00 | 0.80 | 11.83 |
| 2 | 421 | 0.7826 | 3.31 | 0.53 | 1.091 | 121 | 1.00 | 1.31 | 35.39 |
| 3 | 251 | 0.7758 | 3.26 | 0.51 | 1.081 | 114 | 1.00 | 2.23 | 33.41 |
| 4 | 199 | 0.7655 | 3.46 | 0.51 | 1.010 | 116 | 1.00 | 0.99 | 36.34 |
| 5 | 302 | 0.7637 | 3.10 | 0.53 | 1.068 | 138 | 1.00 | 1.13 | 35.80 |
| 6 | 267 | 0.7630 | 3.21 | 0.52 | 1.042 | 114 | 1.00 | 1.26 | 31.77 |
| 7 | 400 | 0.7625 | 3.13 | 0.54 | 1.053 | 125 | 1.00 | 0.93 | 34.05 |
| 8 | 292 | 0.7539 | 3.31 | 0.52 | 0.993 | 109 | 1.00 | 0.65 | 32.04 |
| 9 | 262 | 0.7445 | 2.89 | 0.50 | 1.034 | 90 | 1.00 | 1.61 | 23.08 |
| 10 | 477 | 0.7374 | 2.98 | 0.52 | 1.003 | 147 | 1.00 | 0.92 | 36.76 |

## Parameter Importance
| Parameter | Importance |
|-----------|------------|
| right_bars | 0.7022 |
| rr_target | 0.1158 |
| min_separation | 0.0517 |
| min_close_ratio | 0.0477 |
| oversold | 0.0319 |
| left_bars | 0.0227 |
| ema_period | 0.0118 |
| d_period | 0.0082 |
| k_period | 0.0072 |
| smooth | 0.0010 |

## Convergence Curve
- Initial best (trial 0): -9999.0000
- Final best (trial 499): 0.9007
- 90% of improvement reached by trial ~1
