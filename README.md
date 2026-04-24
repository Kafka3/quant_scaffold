# Quant Research Scaffold

A starter research scaffold for a divergence + trend strategy pipeline:

- Grid Search (coarse search)
- Optuna + TPE (fine tuning)
- Walk-forward validation
- Parameter plateau analysis
- HMM / regime filtering
- RL for dynamic parameter/risk switching (last stage)

## Suggested flow

1. Put your market data into `data/raw/`.
2. Implement feature logic in `features/`.
3. Implement signal and risk logic in `strategy/`.
4. Run a single backtest from `main.py`.
5. Run coarse search from `optimize/grid_search.py`.
6. Run fine search from `optimize/optuna_search.py`.
7. Run rolling validation from `optimize/walk_forward.py`.
8. Add regime labels from `regime/`.
9. Only after the above is stable, test RL in `rl/`.

## Design principles

- Reuse the same signal definition across all stages.
- Reuse the same cost model across all stages.
- Keep the objective function consistent between optimization and validation.
- Favor stable parameter plateaus over sharp peaks.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

## Core modules

- `features/indicators.py`: indicators such as stochastic, ATR, EMA
- `features/divergence.py`: pivot and divergence detection
- `features/trend_filter.py`: trend / EMA channel filters
- `strategy/signal_builder.py`: unify all conditions into signals
- `strategy/risk_model.py`: stop, target, position sizing
- `backtest/vectorbt_engine.py`: vectorbt-based backtest entry point
- `optimize/grid_search.py`: coarse parameter sweep
- `optimize/optuna_search.py`: fine tuning with Optuna
- `optimize/walk_forward.py`: rolling train/test validation
- `regime/hmm_model.py`: HMM training and prediction
- `rl/env.py`: Gymnasium environment for dynamic parameter switching

## Recommended next step

Start by replacing the placeholder divergence logic with your exact stochastic D + pivot structure rules.
