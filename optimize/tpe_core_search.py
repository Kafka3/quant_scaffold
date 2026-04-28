#!/usr/bin/env python
"""TPE Phase 1 — 12-parameter core search with quarterly pruning and composite scoring.

Searches the most influential 12 strategy parameters while keeping 7 parameters
fixed to their baseline values.  Evaluates on 6 training quarters (2024-Q1 to
2025-Q2), reports intermediate results for MedianPruner, and records validation
metrics on 2025-Q3.

Outputs
-------
- reports/tpe_core/tpe_core_study.db      (Optuna SQLite storage)
- reports/tpe_core_summary.md             (search summary + top-10)
- reports/tpe_top50.csv                   (top-50 parameters + quarterly detail)
- reports/tpe_best_params.json            (best params, loadable)
"""
import argparse
import copy
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

VAL_PERIOD = ("2025-Q3", "2025-07-01", "2025-10-01")
TEST_PERIOD = ("2025-Q4", "2025-10-01", "2026-01-01")

ALL_TRAIN_VAL_PERIODS = TRAIN_PERIODS + [VAL_PERIOD]

BT_CFG_COST = {
    "initial_cash": 100000.0,
    "fee_per_trade": 5.0,
    "slippage": 5.0,
    "allow_short": True,
}

BT_CFG_NOCOST = {
    "initial_cash": 100000.0,
    "fee_per_trade": 0.0,
    "slippage": 0.0,
    "allow_short": True,
}

STUDY_NAME = "tpe_core_12p"
DEFAULT_N_TRIALS = 500

# ------------------------------------------------------------------
# Config builder
# ------------------------------------------------------------------

def sample_config(trial: optuna.Trial) -> dict:
    """Sample the 12 searched parameters and assemble strategy config."""
    k_period = trial.suggest_int("k_period", 8, 21)
    d_period = trial.suggest_int("d_period", 2, 5)
    smooth = trial.suggest_int("smooth", 1, 3)
    oversold = trial.suggest_int("oversold", 10, 25)
    overbought = trial.suggest_int("overbought", oversold + 50, 90)
    left_bars = trial.suggest_int("left_bars", 2, 6)
    right_bars = trial.suggest_int("right_bars", 2, 5)
    min_separation = trial.suggest_int("min_separation", 3, 15)
    max_separation = trial.suggest_int("max_separation", min_separation + 10, 55)
    ema_period = trial.suggest_int("ema_period", 21, 89)
    min_close_ratio = trial.suggest_float("min_close_ratio", 0.50, 0.95)
    rr_target = trial.suggest_float("rr_target", 1.5, 3.5)

    return {
        "stochastic": {
            "k_period": k_period,
            "d_period": d_period,
            "smooth": smooth,
            "oversold": oversold,
            "overbought": overbought,
        },
        "pivots": {
            "left_bars": left_bars,
            "right_bars": right_bars,
            "min_separation": min_separation,
            "max_separation": max_separation,
            "strict": True,
        },
        "trend": {
            "ema_period": ema_period,
            "lookback_bars": 24,
            "min_close_ratio": min_close_ratio,
        },
        "risk": {
            "atr_period": 14,
            "stop_buffer": 0.0,
            "rr_target": rr_target,
        },
        "setup": {
            "setup_max_bars": 12,
            "replace_same_side_setup": True,
            "invalidate_on_stop_anchor_break": True,
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


def compute_expectancy_r(trades: pd.DataFrame) -> float:
    """Compute average R-multiple from trades."""
    if trades.empty:
        return 0.0
    r_list = []
    for _, row in trades.iterrows():
        if row["side"] == "long":
            risk = row["entry_price"] - row["stop_price"]
        else:
            risk = row["stop_price"] - row["entry_price"]
        if risk <= 0 or pd.isna(risk):
            continue
        r_list.append(row["pnl"] / risk)
    if not r_list:
        return 0.0
    return float(np.mean(r_list))


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
            metrics.append(
                {
                    "name": name,
                    "total_trades": 0,
                    "profit_factor": 0.0,
                    "win_rate": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "expectancy": 0.0,
                }
            )
            continue

        # Trades with entry_time inside the period
        mask = (trades["entry_time"] >= start_ts) & (trades["entry_time"] < end_ts)
        period_trades = trades.loc[mask]

        total_trades = len(period_trades)
        pf = _compute_pf(period_trades)
        wr = _compute_wr(period_trades)
        total_return = _compute_return(eq_slice)
        max_dd = _compute_max_dd(eq_slice)
        expectancy = float(period_trades["pnl"].mean()) if total_trades > 0 else 0.0

        metrics.append(
            {
                "name": name,
                "total_trades": total_trades,
                "profit_factor": pf,
                "win_rate": wr,
                "total_return": total_return,
                "max_drawdown": max_dd,
                "expectancy": expectancy,
            }
        )
    return metrics


def compute_score(summary: dict, quarterly_metrics: list) -> float:
    """Composite score with hard filters and penalties."""
    total_trades = summary.get("total_trades", 0)
    pf = safe_profit_factor(summary.get("profit_factor"))
    max_dd = summary.get("max_drawdown", 0.0)

    # Hard filters
    if total_trades < 10:
        return -9999.0
    if summary.get("total_return", 0.0) < 0:
        return -9999.0

    # Expectancy_R — summary dict may contain it pre-computed
    expectancy_r = summary.get("expectancy_r", 0.0)

    # Quarterly stability
    n_q = len(quarterly_metrics)
    positive_qs = sum(1 for m in quarterly_metrics if m["total_return"] > 0)
    quarterly_positive_ratio = positive_qs / n_q if n_q > 0 else 0.0

    # Base score
    score = (
        expectancy_r * 0.30
        + (min(pf, 5.0) / 5.0) * 0.25
        + quarterly_positive_ratio * 0.20
        + min(total_trades / 50.0, 1.0) * 0.10
        - (abs(max_dd) / 25.0) * 0.15
    )

    # Penalties
    for m in quarterly_metrics:
        if m["total_trades"] < 2:
            score -= 0.05
            break  # deduct once per trial, regardless of how many quarters fail

    if pf > 5.0 and total_trades < 30:
        score -= 0.10

    return score


def _summary_from_slice(trades: pd.DataFrame, equity: pd.Series, end_ts) -> dict:
    """Build a summary dict from trades and equity up to (but not including) end_ts."""
    t = trades[trades["entry_time"] < end_ts]
    eq = equity.loc[:end_ts]
    return {
        "total_trades": len(t),
        "profit_factor": _compute_pf(t),
        "win_rate": _compute_wr(t),
        "max_drawdown": _compute_max_dd(eq),
        "total_return": _compute_return(eq),
    }


# ------------------------------------------------------------------
# Per-trial evaluation
# ------------------------------------------------------------------

def evaluate_trial(cfg: dict, df_full: pd.DataFrame):
    """Run cost/nocost backtests on train and validation periods.

    Returns
    -------
    dict with keys: cost_train, nocost_train, cost_val, quarterly_train
    """
    train_start = TRAIN_PERIODS[0][1]
    train_end = TRAIN_PERIODS[-1][2]
    val_start, val_end = VAL_PERIOD[1], VAL_PERIOD[2]

    df_train = slice_dataframe(df_full, train_start, train_end)
    df_val = slice_dataframe(df_full, val_start, val_end)

    # Train — build once, run cost + nocost
    if len(df_train) < 200:
        return None

    bundle_train = build_signals(df_train, cfg)
    cost_train = run_backtest(df_train, bundle_train, BT_CFG_COST)
    nocost_train = run_backtest(df_train, bundle_train, BT_CFG_NOCOST)

    # Validation
    if len(df_val) < 50:
        cost_val = None
    else:
        bundle_val = build_signals(df_val, cfg)
        cost_val = run_backtest(df_val, bundle_val, BT_CFG_COST)

    quarterly_train = derive_quarterly_metrics(cost_train, TRAIN_PERIODS)

    return {
        "cost_train": cost_train,
        "nocost_train": nocost_train,
        "cost_val": cost_val,
        "quarterly_train": quarterly_train,
    }


# ------------------------------------------------------------------
# Optuna objective
# ------------------------------------------------------------------

def objective(trial: optuna.Trial, df_full: pd.DataFrame):
    cfg = sample_config(trial)

    try:
        res = evaluate_trial(cfg, df_full)
    except Exception as exc:
        # Gracefully handle any strategy/backtest crash
        trial.set_user_attr("error", str(exc))
        return -9999.0

    if res is None:
        return -9999.0

    cost_train = res["cost_train"]
    nocost_train = res["nocost_train"]
    cost_val = res["cost_val"]
    quarterly_train = res["quarterly_train"]

    # --- Intermediate reporting for MedianPruner ------------------
    for i, (q_name, q_start, q_end) in enumerate(TRAIN_PERIODS):
        end_ts = pd.Timestamp(q_end, tz="UTC")
        partial_summary = _summary_from_slice(cost_train.trades, cost_train.equity, end_ts)
        partial_summary["expectancy_r"] = compute_expectancy_r(
            cost_train.trades[cost_train.trades["entry_time"] < end_ts]
        )
        partial_metrics = quarterly_train[: i + 1]
        partial_score = compute_score(partial_summary, partial_metrics)
        trial.report(partial_score, step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # --- Final training metrics -----------------------------------
    train_summary = cost_train.summary
    train_trades = cost_train.trades
    expectancy_r = compute_expectancy_r(train_trades)

    train_summary_enhanced = {
        **train_summary,
        "expectancy_r": expectancy_r,
    }
    score = compute_score(train_summary_enhanced, quarterly_train)

    # --- User attrs -----------------------------------------------
    trial.set_user_attr("total_trades", int(train_summary.get("total_trades", 0)))
    trial.set_user_attr("profit_factor", float(safe_profit_factor(train_summary.get("profit_factor"))))
    trial.set_user_attr("win_rate", float(safe_win_rate(train_summary.get("win_rate"))))
    trial.set_user_attr("expectancy_R", float(expectancy_r))
    trial.set_user_attr("max_drawdown_pct", float(train_summary.get("max_drawdown", 0.0)))
    trial.set_user_attr("total_return", float(train_summary.get("total_return", 0.0)))
    trial.set_user_attr(
        "quarterly_positive_ratio",
        sum(1 for m in quarterly_train if m["total_return"] > 0) / len(TRAIN_PERIODS),
    )

    for m in quarterly_train:
        qn = m["name"]
        trial.set_user_attr(f"q_{qn}_pf", float(safe_profit_factor(m["profit_factor"])))
        trial.set_user_attr(f"q_{qn}_trades", int(m["total_trades"]))
        trial.set_user_attr(f"q_{qn}_return", float(m["total_return"]))
        trial.set_user_attr(f"q_{qn}_drawdown", float(m["max_drawdown"]))

    # No-cost reference
    nocost_summary = nocost_train.summary
    trial.set_user_attr("nocost_total_return", float(nocost_summary.get("total_return", 0.0)))
    trial.set_user_attr("nocost_profit_factor", float(safe_profit_factor(nocost_summary.get("profit_factor"))))
    trial.set_user_attr("nocost_total_trades", int(nocost_summary.get("total_trades", 0)))

    # Validation
    if cost_val is not None:
        val_summary = cost_val.summary
        trial.set_user_attr("val_total_trades", int(val_summary.get("total_trades", 0)))
        trial.set_user_attr("val_total_return", float(val_summary.get("total_return", 0.0)))
        trial.set_user_attr("val_profit_factor", float(safe_profit_factor(val_summary.get("profit_factor"))))
        trial.set_user_attr("val_max_drawdown", float(val_summary.get("max_drawdown", 0.0)))
    else:
        trial.set_user_attr("val_total_trades", 0)
        trial.set_user_attr("val_total_return", 0.0)
        trial.set_user_attr("val_profit_factor", 0.0)
        trial.set_user_attr("val_max_drawdown", 0.0)

    return score


# ------------------------------------------------------------------
# Report generation
# ------------------------------------------------------------------

def generate_reports(study: optuna.Study, output_dir: Path):
    """Generate summary markdown, top-50 CSV, and best-params JSON."""
    trials_df = study.trials_dataframe()
    if trials_df.empty:
        print("Warning: no trials completed, skipping report generation.")
        return

    # ---- Top-50 CSV ------------------------------------------------
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value if t.value is not None else -float("inf"), reverse=True)

    rows = []
    for t in completed[:50]:
        row = {
            "rank": len(rows) + 1,
            "trial_id": t.number,
            "score": t.value,
        }
        row.update(t.params)
        # Attach key user attrs
        attrs = t.user_attrs
        row["total_trades"] = attrs.get("total_trades", 0)
        row["profit_factor"] = attrs.get("profit_factor", 0.0)
        row["win_rate"] = attrs.get("win_rate", 0.0)
        row["expectancy_R"] = attrs.get("expectancy_R", 0.0)
        row["max_drawdown_pct"] = attrs.get("max_drawdown_pct", 0.0)
        row["total_return"] = attrs.get("total_return", 0.0)
        row["quarterly_positive_ratio"] = attrs.get("quarterly_positive_ratio", 0.0)
        row["val_total_return"] = attrs.get("val_total_return", 0.0)
        row["val_profit_factor"] = attrs.get("val_profit_factor", 0.0)
        row["nocost_total_return"] = attrs.get("nocost_total_return", 0.0)
        for q_name, _, _ in TRAIN_PERIODS:
            row[f"q_{q_name}_return"] = attrs.get(f"q_{q_name}_return", 0.0)
            row[f"q_{q_name}_trades"] = attrs.get(f"q_{q_name}_trades", 0)
            row[f"q_{q_name}_pf"] = attrs.get(f"q_{q_name}_pf", 0.0)
        rows.append(row)

    top50_df = pd.DataFrame(rows)
    top50_path = output_dir / "tpe_top50.csv"
    top50_df.to_csv(top50_path, index=False)

    # ---- Best params JSON ------------------------------------------
    if study.best_trial is not None:
        best = {
            "params": study.best_trial.params,
            "score": study.best_trial.value,
            "user_attrs": study.best_trial.user_attrs,
        }
        best_path = output_dir / "tpe_best_params.json"
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False, default=str)

    # ---- Summary Markdown ------------------------------------------
    lines = []
    lines.append("# TPE Phase 1 — Core 12-Parameter Search Summary\n")
    lines.append(f"- **Study name**: `{STUDY_NAME}`\n")
    lines.append(f"- **Total trials**: {len(study.trials)}\n")
    lines.append(f"- **Completed trials**: {len(completed)}\n")
    lines.append(f"- **Pruned trials**: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")
    lines.append(f"- **Failed trials**: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}\n")

    if study.best_trial is not None:
        bt = study.best_trial
        lines.append(f"\n## Best Trial (ID {bt.number})\n")
        lines.append(f"- **Score**: {bt.value:.6f}\n")
        lines.append("- **Parameters**:\n")
        for k, v in bt.params.items():
            lines.append(f"  - `{k}`: {v}\n")
        lines.append("- **Key Metrics**:\n")
        attrs = bt.user_attrs
        lines.append(f"  - Total trades (train): {attrs.get('total_trades', 'N/A')}\n")
        lines.append(f"  - Profit factor (train): {attrs.get('profit_factor', 'N/A'):.3f}\n")
        lines.append(f"  - Win rate (train): {attrs.get('win_rate', 'N/A'):.3f}\n")
        lines.append(f"  - Expectancy R (train): {attrs.get('expectancy_R', 'N/A'):.4f}\n")
        lines.append(f"  - Max drawdown % (train): {attrs.get('max_drawdown_pct', 'N/A'):.2f}\n")
        lines.append(f"  - Total return % (train): {attrs.get('total_return', 'N/A'):.2f}\n")
        lines.append(f"  - Quarterly positive ratio: {attrs.get('quarterly_positive_ratio', 'N/A'):.2f}\n")
        lines.append(f"  - Val return %: {attrs.get('val_total_return', 'N/A'):.2f}\n")
        lines.append(f"  - No-cost return %: {attrs.get('nocost_total_return', 'N/A'):.2f}\n")

    # Top-10 table
    lines.append("\n## Top-10 Results\n")
    if rows:
        top10 = rows[:10]
        header = "| Rank | Trial | Score | PF | WinRate | ExpectR | Trades | Q+Ratio | ValRet | NoCostRet |\n"
        sep = "|------|-------|-------|-----|---------|---------|--------|---------|--------|-----------|\n"
        lines.append(header)
        lines.append(sep)
        for r in top10:
            lines.append(
                f"| {r['rank']} | {r['trial_id']} | {r['score']:.4f} | "
                f"{r['profit_factor']:.2f} | {r['win_rate']:.2f} | {r['expectancy_R']:.3f} | "
                f"{r['total_trades']} | {r['quarterly_positive_ratio']:.2f} | "
                f"{r['val_total_return']:.2f} | {r['nocost_total_return']:.2f} |\n"
            )

    # Parameter importance
    lines.append("\n## Parameter Importance\n")
    try:
        importances = optuna.importance.get_param_importances(study)
        lines.append("| Parameter | Importance |\n|-----------|------------|\n")
        for param, imp in importances.items():
            lines.append(f"| {param} | {imp:.4f} |\n")
    except Exception as exc:
        lines.append(f"Could not compute param importances: {exc}\n")

    # Convergence description
    lines.append("\n## Convergence Curve\n")
    if not trials_df.empty and "value" in trials_df.columns:
        values = trials_df["value"].dropna()
        if len(values) > 0:
            best_so_far = values.cummax()
            lines.append(f"- Initial best (trial 0): {best_so_far.iloc[0]:.4f}\n")
            lines.append(f"- Final best (trial {len(values)-1}): {best_so_far.iloc[-1]:.4f}\n")
            # Find when 90% of improvement happened
            improvement = best_so_far.iloc[-1] - best_so_far.iloc[0]
            if improvement > 0:
                threshold = best_so_far.iloc[0] + 0.9 * improvement
                hit_idx = best_so_far[best_so_far >= threshold].index[0]
                lines.append(f"- 90% of improvement reached by trial ~{hit_idx}\n")
            else:
                lines.append("- No significant improvement observed across trials.\n")

    summary_path = output_dir / "tpe_core_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\nReports saved:")
    print(f"  {top50_path}")
    if study.best_trial is not None:
        print(f"  {output_dir / 'tpe_best_params.json'}")
    print(f"  {summary_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TPE Phase 1 — 12-parameter core search")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "raw" / "BTCUSDT_5m_2024_2025.csv"),
                        help="Path to OHLCV CSV")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=42, help="TPESampler seed")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for Optuna")
    parser.add_argument("--storage", default=str(PROJECT_ROOT / "reports" / "tpe_core" / "tpe_core_study.db"),
                        help="SQLite storage URL or path")
    args = parser.parse_args()

    # Ensure storage directory exists
    storage_path = Path(args.storage)
    if not storage_path.parent.exists():
        storage_path.parent.mkdir(parents=True, exist_ok=True)

    db_url = f"sqlite:///{storage_path}"

    print("Loading data...")
    df_full = load_ohlcv_csv(args.data)
    print(f"Loaded {len(df_full)} bars, {df_full.index[0]} to {df_full.index[-1]}")

    print(f"Storage: {db_url}")
    print(f"Trials: {args.n_trials}, Seed: {args.seed}, Jobs: {args.n_jobs}")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=TPESampler(multivariate=True, seed=args.seed, n_startup_trials=100),
        pruner=MedianPruner(),
        storage=db_url,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, df_full),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"  Score = {study.best_trial.value:.6f}")
    print(f"  Params = {study.best_trial.params}")

    generate_reports(study, PROJECT_ROOT / "reports")


if __name__ == "__main__":
    main()
