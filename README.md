# Quant Research Scaffold — Strategy Confirmation Phase

**Current phase: Strategy Confirmation**

This project validates a divergence + trend continuation strategy on BTCUSDT 5m data.
Freqtrade migration is **paused** until all confirmation checks pass.

---

## Strategy (baseline)

**Config:** `configs/baseline_ema55_stoch143_2r.yaml`

| Parameter | Value |
|---|---|
| EMA channel | 55 (High/Low) |
| Stochastic | 14/1/3, oversold 20, overbought 80 |
| Pivot | left=3, right=3, min_sep=5, max_sep=35, strict |
| Setup max bars | 12 |
| RR target | 2.0 |
| Stop | Pivot2 structure high/low |
| Trigger entry | High/p2 (long), Low@p2 (short) |
| Structure invalidation | Stop anchor break cancels setup |

### Rules

**Bullish continuation:**
1. Prior uptrend (≥60% of last 12 bars close above EMA high, shift(1))
2. Pivot1 low ≤ oversold (stoch ≤ 20), close inside/below EMA high
3. Pivot2 lower low, close below EMA low (deep pullback)
4. Oscillator higher at pivot2 than pivot1 (divergence)
5. Signal on confirmation bar (pivot2 + right_bars=3)
6. Entry: next bar where High > trigger_price (High@p2)
7. Stop: pivot2 Low; Target: entry + 2R

**Bearish continuation:** symmetric (pivot highs, prior downtrend, overbought ≥80)

---

## Current status

- [x] Baseline config locked
- [x] Pivot rolling-window bug fixed (`shift(-right)` → `shift(-1)`)
- [x] Sample data replaced with real BTCUSDT 2026Q1
- [x] Full-chain synthetic tests (11/11 pass)
- [x] Trade audit (29/29 valid)
- [x] Segment report by quarter
- [ ] Plateau analysis (full grid — estimated 17 min)
- [ ] Freqtrade migration (paused)

---

## Core modules

| Module | Purpose |
|---|---|
| `features/indicators.py` | Stochastic, ATR, EMA |
| `features/divergence.py` | Pivot detection + divergence logic |
| `features/trend_filter.py` | EMA channel + prior-trend state |
| `strategy/signal_builder.py` | Unify all conditions into signals / pending setups |
| `strategy/risk_model.py` | Position sizing (Phase 4+) |
| `backtest/event_engine.py` | Event-driven backtest (entry, stop/target, equity) |
| `backtest/cost_model.py` | Slippage + fee model (Phase 4+) |
| `data/loaders/csv_loader.py` | OHLCV CSV loading |

## Scripts

| Script | Purpose |
|---|---|
| `scripts/audit_trades.py` | Per-trade breakdown with pivot, stoch, channel state |
| `scripts/segment_report.py` | Per-quarter performance report |
| `scripts/plateau_check.py` | Small-range parameter grid (no TPE) |

## Tests

| File | Purpose |
|---|---|
| `tests/test_smoke.py` | Smoke tests (config, build, backtest, timing) |
| `tests/test_synthetic_pivot.py` | Pivot rolling-window semantics |
| `tests/test_full_chain_synthetic.py` | Full-chain: divergence→setup→entry→exit→stop/target |

## Data

- `data/raw/BTCUSDT_5m_2024_2025.csv` — Full training set (210k bars)
- `data/raw/BTCUSDT_5m_2024.csv` — 2024 only
- `data/raw/BTCUSDT_5m_2025.csv` — 2025 only
- `data/raw/BTCUSDT_5m_2026Q1.csv` — Out-of-sample (26k bars)
- `example_data/sample_ohlcv.csv` — Small BTC sample for tests

## Quick start

```bash
# Run all tests
python -m pytest tests/

# Audit trades
python scripts/audit_trades.py

# Quarter-by-quarter report
python scripts/segment_report.py

# Parameter plateau check (50 combos smoke test)
python scripts/plateau_check.py --quick

# Full plateau grid (1296 combos, ~17 min)
python scripts/plateau_check.py
```

## Design principles

- Same signal definition across all stages
- Same cost model across all stages
- Consistent objective function between optimization and validation
- Favor stable parameter plateaus over sharp peaks
- No future functions: shift(1) on all lookback, pivot right-bar confirmation
