from pathlib import Path
import copy
import optuna

from configs.settings import load_settings
from data.loaders.csv_loader import load_ohlcv_csv
from strategy.signal_builder import build_signals
from backtest.vectorbt_engine import run_backtest
from backtest.metrics import summarize_portfolio
from backtest.validation import simple_objective


def objective(trial: optuna.Trial, settings: dict, df):
    cfg = copy.deepcopy(settings)
    cfg["strategy"]["trend"]["ema_period"] = trial.suggest_int("ema_period", 21, 89)
    cfg["strategy"]["risk"]["rr_target"] = trial.suggest_float("rr_target", 1.0, 3.0)
    cfg["strategy"]["pivots"]["left_bars"] = trial.suggest_int("left_bars", 2, 5)
    cfg["strategy"]["pivots"]["right_bars"] = cfg["strategy"]["pivots"]["left_bars"]
    cfg["strategy"]["pivots"]["max_separation"] = trial.suggest_int("max_separation", 8, 30)

    bundle = build_signals(df, cfg["strategy"])
    pf = run_backtest(df, bundle, cfg["backtest"])
    summary = summarize_portfolio(pf)
    return simple_objective(summary)


def main() -> None:
    settings = load_settings(Path("configs/base.yaml"))
    df = load_ohlcv_csv(settings["data"]["path"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(lambda t: objective(t, settings, df), n_trials=100)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
