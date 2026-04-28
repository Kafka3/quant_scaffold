#!/usr/bin/env python
"""
TPE Round 2 — Local fine-tuning search with narrowed parameter ranges.

Evaluates on 6 training quarters (2024-Q1 to 2025-Q2), uses the composite
scoring function from the round-2 spec, and runs independent studies per seed.

Outputs
-------
- reports/tpe_round2/tpe_round2_seed{seed}.db      (Optuna SQLite storage)
- reports/tpe_round2/tpe_round2_trials_seed{seed}.csv (all trial records)
- reports/tpe_round2/tpe_round2_summary_seed{seed}.md  (per-seed summary)
- reports/tpe_round2/tpe_round2_best_params_seed{seed}.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# ------------------------------------------------------------------
# Ensure project root is on PYTHONPATH
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.loaders.csv_loader import load_ohlcv_csv
from optimize.utils import safe_profit_factor, safe_win_rate, slice_dataframe
from strategy.signal_builder import build_signals
from backtest.event_engine import run_backtest

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
TRAIN_PERIODS = [
    ("2024-Q1", "2024-01-01", "2024-04-01"),
    ("2024-Q2", "2024-04-01", "2024-07-01"),
    ("2024-Q3", "2024-07-01", "2024-10-01"),
    ("2024-Q4", "2024-10-01", "2025-01-01"),
    ("2025-Q1", "2025-01-01", "2025-04-01"),
    ("2025-Q2", "2025-04-01", "2025-07-01"),
]

BT_CFG = {
    "initial_cash": 100000.0,
    "fee_per_trade": 5.0,
    "slippage": 5.0,
    "allow_short": True,
}

DEFAULT_N_TRIALS = 500
REPORT_DIR = PROJECT_ROOT / "reports" / "tpe_round2"

# ------------------------------------------------------------------
# Config builder
# ------------------------------------------------------------------

def sample_config(trial: optuna.Trial) -> dict:
    """Sample the narrowed parameter set for round-2 fine-tuning."""
    ema_period = trial.suggest_int("ema_period", 30, 60)
    min_close_ratio = trial.suggest_categorical("min_close_ratio", [0.80, 0.85, 0.90, 0.95])
    k_period = trial.suggest_int("k_period", 10, 14)
    d_period = trial.suggest_categorical("d_period", [2, 3])
    oversold = trial.suggest_int("oversold", 10, 15)
    overbought = trial.suggest_int("overbought", oversold + 65, 90)
    left_bars = trial.suggest_categorical("left_bars", [5, 6, 7])
    right_bars = trial.suggest_categorical("right_bars", [3, 4, 5])
    min_separation = trial.suggest_int("min_separation", 8, 18)
    max_separation = trial.suggest_int("max_separation", min_separation + 20, 60)
    rr_target = trial.suggest_float("rr_target", 2.20, 2.80, step=0.02)

    return {
        "stochastic": {
            "k_period": k_period,
            "d_period": d_period,
            "smooth": 1,               # fixed
            "oversold": oversold,
            "overbought": overbought,
        },
        "pivots": {
            "left_bars": left_bars,
            "right_bars": right_bars,
            "min_separation": min_separation,
            "max_separation": max_separation,
            "strict": True,             # fixed
        },
        "trend": {
            "ema_period": ema_period,
            "lookback_bars": 24,        # fixed
            "min_close_ratio": min_close_ratio,
        },
        "risk": {
            "atr_period": 14,           # fixed
            "stop_buffer": 0.0,         # fixed
            "rr_target": rr_target,
        },
        "setup": {
            "setup_max_bars": 12,                       # fixed
            "replace_same_side_setup": True,            # fixed
            "invalidate_on_stop_anchor_break": True,    # fixed
        },
    }


# ------------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------------

def _compute_pf(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    gp = float(wins["pnl"].sum()) if len(wins) > 0 else 0.0
    gl = float(abs(losses["pnl"].sum())) if len(losses) > 0 else 0.0
    if gl > 0:
        return gp / gl
    return float("inf") if gp > 0 else 0.0


def _compute_wr(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    wins = trades[trades["pnl"] > 0]
    return len(wins) / len(trades)


def _compute_max_dd(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    safe_peak = peak.where(peak > 0, pd.NA)
    dd = (equity - safe_peak) / safe_peak
    if dd.isna().all():
        return 0.0
    return float(dd.min() * 100)


def _compute_return(equity: pd.Series) -> float:
    if equity.empty or equity.iloc[0] <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0] - 1) * 100


# ------------------------------------------------------------------
# Quarterly metrics
# ------------------------------------------------------------------

def derive_quarterly_metrics(result, periods):
    """Derive per-period metrics from a continuous backtest result."""
    trades = result.trades
    equity = result.equity
    metrics = []
    for name, start, end in periods:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")

        eq_slice = equity.loc[start_ts:end_ts]
        if len(eq_slice) == 0:
            metrics.append({
                "name": name,
                "total_trades": 0,
                "profit_factor": 0.0,
                "win_rate": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
            })
            continue

        # Trades with entry_time inside the period
        mask = (trades["entry_time"] >= start_ts) & (trades["entry_time"] < end_ts)
        period_trades = trades.loc[mask]

        total_trades = len(period_trades)
        pf = _compute_pf(period_trades)
        wr = _compute_wr(period_trades)
        total_return = _compute_return(eq_slice)
        max_dd = _compute_max_dd(eq_slice)

        metrics.append({
            "name": name,
            "total_trades": total_trades,
            "profit_factor": pf,
            "win_rate": wr,
            "total_return": total_return,
            "max_drawdown": max_dd,
        })
    return metrics


# ------------------------------------------------------------------
# Round-2 scoring function
# ------------------------------------------------------------------

def compute_score_r2(full_summary: dict, quarterly_metrics: list) -> float:
    """
    Round-2 composite scoring with hard filters.

    Parameters (all from full_summary):
        full_total_trades
        min_trades_per_period    — fewest trades in any single quarter
        positive_ratio           — fraction of quarters with positive return
        avg_profit_factor        — mean of per-quarter profit factors
        min_profit_factor        — minimum per-quarter profit factor
        avg_quarter_return       — mean of per-quarter total_return (%)
        max_quarter_drawdown_worst — worst (most negative) per-quarter max_drawdown
    """
    full_total_trades = full_summary.get("full_total_trades", 0)
    min_trades_per_period = full_summary.get("min_trades_per_period", 0)
    positive_ratio = full_summary.get("positive_ratio", 0.0)
    min_profit_factor = full_summary.get("min_profit_factor", 0.0)
    avg_profit_factor = full_summary.get("avg_profit_factor", 0.0)
    avg_quarter_return = full_summary.get("avg_quarter_return", 0.0)  # already in %
    max_quarter_dd_worst = full_summary.get("max_quarter_drawdown_worst", 0.0)

    # Hard filters
    if full_total_trades < 100:
        return -9999.0
    if min_trades_per_period < 8:
        return -9999.0
    if positive_ratio < 0.75:
        return -9999.0
    if min_profit_factor < 1.0:
        return -9999.0

    score = (
        0.25 * min(avg_profit_factor, 5.0) / 5.0
        + 0.20 * positive_ratio
        + 0.20 * min(min_profit_factor, 5.0) / 5.0
        + 0.15 * max(-1.0, min(avg_quarter_return / 5.0, 1.0))
        + 0.10 * min(min_trades_per_period / 20.0, 1.0)
        - 0.10 * abs(max_quarter_dd_worst) / 20.0
    )
    return score


# ------------------------------------------------------------------
# Per-trial evaluation
# ------------------------------------------------------------------

def evaluate_trial(cfg: dict, df_full: pd.DataFrame):
    """Run full backtest over the entire training period (no slicing)."""
    train_start = TRAIN_PERIODS[0][1]
    train_end = TRAIN_PERIODS[-1][2]

    df_train = slice_dataframe(df_full, train_start, train_end)
    if len(df_train) < 200:
        return None

    # Build signals once on the full training set
    bundle = build_signals(df_train, cfg)
    result = run_backtest(df_train, bundle, BT_CFG)

    # Derive quarterly metrics from the continuous run
    quarterly = derive_quarterly_metrics(result, TRAIN_PERIODS)

    return {
        "result": result,
        "quarterly": quarterly,
    }


def objective(trial: optuna.Trial, df_full: pd.DataFrame):
    cfg = sample_config(trial)

    try:
        res = evaluate_trial(cfg, df_full)
    except Exception as exc:
        trial.set_user_attr("error", str(exc))
        return -9999.0

    if res is None:
        return -9999.0

    result = res["result"]
    quarterly = res["quarterly"]
    trades = result.trades
    equity = result.equity

    # --- Aggregate quarterly stats into a flat summary dict ----------
    q_trades = [m["total_trades"] for m in quarterly]
    q_pf = [safe_profit_factor(m["profit_factor"]) for m in quarterly]
    q_return = [m["total_return"] for m in quarterly]
    q_dd = [m["max_drawdown"] for m in quarterly]

    full_summary = {
        "full_total_trades": int(trades.summary.get("total_trades", 0) if hasattr(trades, "summary") else len(trades)),
        "min_trades_per_period": min(q_trades),
        "positive_ratio": sum(1 for r in q_return if r > 0) / len(q_return),
        "avg_profit_factor": float(np.mean(q_pf)),
        "min_profit_factor": min(q_pf),
        "avg_quarter_return": float(np.mean(q_return)),     # already in %
        "max_quarter_drawdown_worst": min(q_dd),            # most negative
    }

    score = compute_score_r2(full_summary, quarterly)

    # ---- User attrs -------------------------------------------------
    trial.set_user_attr("full_total_trades", full_summary["full_total_trades"])
    trial.set_user_attr("min_trades_per_period", full_summary["min_trades_per_period"])
    trial.set_user_attr("positive_ratio", full_summary["positive_ratio"])
    trial.set_user_attr("avg_profit_factor", full_summary["avg_profit_factor"])
    trial.set_user_attr("min_profit_factor", full_summary["min_profit_factor"])
    trial.set_user_attr("avg_quarter_return", full_summary["avg_quarter_return"])
    trial.set_user_attr("max_quarter_drawdown_worst", full_summary["max_quarter_drawdown_worst"])

    for m in quarterly:
        qn = m["name"]
        trial.set_user_attr(f"q_{qn}_trades", int(m["total_trades"]))
        trial.set_user_attr(f"q_{qn}_pf", float(safe_profit_factor(m["profit_factor"])))
        trial.set_user_attr(f"q_{qn}_return", float(m["total_return"]))
        trial.set_user_attr(f"q_{qn}_drawdown", float(m["max_drawdown"]))

    # Full summary stats
    train_summary = result.summary
    trial.set_user_attr("total_trades", int(train_summary.get("total_trades", 0)))
    trial.set_user_attr("profit_factor", float(safe_profit_factor(train_summary.get("profit_factor"))))
    trial.set_user_attr("win_rate", float(safe_win_rate(train_summary.get("win_rate"))))
    trial.set_user_attr("max_drawdown_pct", float(train_summary.get("max_drawdown", 0.0)))
    trial.set_user_attr("total_return_pct", float(train_summary.get("total_return", 0.0)))

    # --- Intermediate reporting for MedianPruner ---------------------
    # Report at each quarter boundary so the pruner can assess early progress.
    cumulative_trades = 0
    for i, m in enumerate(quarterly):
        cumulative_trades += m["total_trades"]
        partial_q = quarterly[: i + 1]
        partial_metrics = {
            "full_total_trades": cumulative_trades,
            "min_trades_per_period": min(q_trades[: i + 1]),
            "positive_ratio": sum(1 for r in q_return[: i + 1] if r > 0) / (i + 1),
            "avg_profit_factor": float(np.mean(q_pf[: i + 1])),
            "min_profit_factor": min(q_pf[: i + 1]),
            "avg_quarter_return": float(np.mean(q_return[: i + 1])),
            "max_quarter_drawdown_worst": min(q_dd[: i + 1]),
        }
        partial_score = compute_score_r2(partial_metrics, partial_q)
        trial.report(partial_score, step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return score


# ------------------------------------------------------------------
# Per-seed report
# ------------------------------------------------------------------

def generate_seed_report(study: optuna.Study, seed: int):
    """Generate per-seed CSV, summary, and best-params files."""
    output_dir = REPORT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_df = study.trials_dataframe()
    if not trials_df.empty:
        csv_path = output_dir / f"tpe_round2_trials_seed{seed}.csv"
        trials_df.to_csv(csv_path, index=False)
        print(f"  Trials CSV: {csv_path}")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value if t.value is not None else -float("inf"), reverse=True)

    # ---- Best params JSON -------------------------------------------
    if study.best_trial is not None:
        best = {
            "params": study.best_trial.params,
            "score": study.best_trial.value,
            "user_attrs": study.best_trial.user_attrs,
        }
        best_path = output_dir / f"tpe_round2_best_params_seed{seed}.json"
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Best params: {best_path}")

    # ---- Summary Markdown -------------------------------------------
    lines = [f"# TPE Round 2 — Local Fine-Tuning (Seed {seed})\n"]
    lines.append(f"- **Study**: `tpe_round2_seed{seed}`\n")
    lines.append(f"- **Total trials**: {len(study.trials)}\n")
    lines.append(f"- **Completed**: {len(completed)}\n")
    lines.append(f"- **Pruned**: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")
    lines.append(f"- **Failed**: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}\n")

    if study.best_trial is not None:
        bt = study.best_trial
        lines.append(f"\n## Best Trial (ID {bt.number})\n")
        lines.append(f"- **Score**: {bt.value:.6f}\n")
        lines.append("- **Parameters**:\n")
        for k, v in bt.params.items():
            lines.append(f"  - `{k}`: {v}\n")
        lines.append("- **Key Metrics**:\n")
        attrs = bt.user_attrs
        lines.append(f"  - Full total trades: {attrs.get('full_total_trades', 'N/A')}\n")
        lines.append(f"  - Min trades/period: {attrs.get('min_trades_per_period', 'N/A')}\n")
        lines.append(f"  - Positive ratio: {attrs.get('positive_ratio', 'N/A'):.3f}\n")
        lines.append(f"  - Avg profit factor: {attrs.get('avg_profit_factor', 'N/A'):.3f}\n")
        lines.append(f"  - Min profit factor: {attrs.get('min_profit_factor', 'N/A'):.3f}\n")
        lines.append(f"  - Avg quarterly return %: {attrs.get('avg_quarter_return', 'N/A'):.2f}\n")
        lines.append(f"  - Worst quarterly DD %: {attrs.get('max_quarter_drawdown_worst', 'N/A'):.2f}\n")

    # Top-10 table
    lines.append("\n## Top-10 Results\n")
    if completed:
        header = "| Rank | Trial | Score | AvgPF | MinPF | PosRat | AvgRet% | MinTr | WorstDD% |\n"
        sep = "|------|-------|-------|--------|-------|--------|---------|-------|----------|\n"
        lines.append(header)
        lines.append(sep)
        for rank, t in enumerate(completed[:10], 1):
            attrs = t.user_attrs
            lines.append(
                f"| {rank} | {t.number} | {t.value:.4f} | "
                f"{attrs.get('avg_profit_factor', 0):.2f} | {attrs.get('min_profit_factor', 0):.2f} | "
                f"{attrs.get('positive_ratio', 0):.2f} | {attrs.get('avg_quarter_return', 0):.2f} | "
                f"{attrs.get('min_trades_per_period', 0)} | "
                f"{attrs.get('max_quarter_drawdown_worst', 0):.2f} |\n"
            )

    summary_path = output_dir / f"tpe_round2_summary_seed{seed}.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  Summary: {summary_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TPE Round 2 — Local fine-tuning search")
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data" / "raw" / "BTCUSDT_5m_2024_2025.csv"),
        help="Path to OHLCV CSV",
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=42, help="TPESampler seed")
    args = parser.parse_args()

    # Ensure report directory exists
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    db_path = REPORT_DIR / f"tpe_round2_seed{args.seed}.db"
    db_url = f"sqlite:///{db_path}"

    print("Loading data...")
    df_full = load_ohlcv_csv(args.data)
    print(f"Loaded {len(df_full)} bars, {df_full.index[0]} to {df_full.index[-1]}")

    print(f"Storage: {db_url}")
    print(f"Trials: {args.n_trials}, Seed: {args.seed}")

    study = optuna.create_study(
        study_name=f"tpe_round2_seed{args.seed}",
        direction="maximize",
        sampler=TPESampler(multivariate=True, seed=args.seed, n_startup_trials=30),
        pruner=MedianPruner(),
        storage=db_url,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, df_full),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"  Score = {study.best_trial.value:.6f}")
    print(f"  Params = {study.best_trial.params}")

    generate_seed_report(study, args.seed)


if __name__ == "__main__":
    main()
