"""Phase 6A — Action space and parameter overrides.

Defines 6 actions (skip + 5 trade variants) and utilities to merge action
params with the base tpe_trial_490 config for each action.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ActionDef:
    """Definition of one action in the Top-4 dynamic controller.

    Attributes:
        index:  0-based action index.
        name:   Short unique name (e.g. 'skip', 'base_tpe490').
        label:  Human-readable label.
        params: Dict with keys in {right_bars, rr_target, min_separation,
                min_close_ratio}.  Empty for skip.
    """
    index: int
    name: str
    label: str
    params: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fixed action definitions — matches phase6a_top4_dynamic.yaml
# ---------------------------------------------------------------------------
ACTIONS: List[ActionDef] = [
    ActionDef(index=0, name="skip",                     label="Skip Trade",            params={}),
    ActionDef(index=1, name="base_tpe490",              label="Base tpe_trial_490",    params={"right_bars": 5, "rr_target": 3.14, "min_separation": 15, "min_close_ratio": 0.5124}),
    ActionDef(index=2, name="faster_confirm",           label="Faster Confirmation",   params={"right_bars": 3, "rr_target": 2.4,  "min_separation": 8,  "min_close_ratio": 0.85}),
    ActionDef(index=3, name="balanced",                 label="Balanced",              params={"right_bars": 4, "rr_target": 2.6,  "min_separation": 10, "min_close_ratio": 0.85}),
    ActionDef(index=4, name="quality",                  label="Quality",               params={"right_bars": 4, "rr_target": 2.8,  "min_separation": 12, "min_close_ratio": 0.90}),
    ActionDef(index=5, name="high_quality_low_freq",    label="High Quality Low Freq", params={"right_bars": 5, "rr_target": 3.0,  "min_separation": 14, "min_close_ratio": 0.90}),
]

ACTION_NAMES: List[str] = [a.name for a in ACTIONS]
ACTION_INDEX_MAP: Dict[str, int] = {a.name: a.index for a in ACTIONS}

# The 4 parameters that actions can override
TOP4_PARAMS = ["right_bars", "rr_target", "min_separation", "min_close_ratio"]

# Non-actions (executing actions)
NON_SKIP_ACTIONS: List[ActionDef] = [a for a in ACTIONS if a.name != "skip"]
NON_SKIP_NAMES: List[str] = [a.name for a in NON_SKIP_ACTIONS]


def get_action(name: str) -> ActionDef:
    """Look up action by name. Raises KeyError if not found."""
    for a in ACTIONS:
        if a.name == name:
            return a
    raise KeyError(f"Unknown action: {name}")


def get_action_by_index(idx: int) -> ActionDef:
    """Look up action by index."""
    return ACTIONS[idx]


def build_strategy_config(
    base_config: dict,
    action: ActionDef,
) -> dict:
    """Build a strategy config dict for a given action.

    1. Deep-copy the base tpe_trial_490.yaml config.
    2. Override the top-4 params from the action definition.
    3. Leave all other params untouched.
    4. Return the new config (base config is NOT modified in-place).

    Args:
        base_config: Loaded tpe_trial_490.yaml config dict.
        action:      The action to apply.

    Returns:
        A new strategy config dict with action's params merged in.
    """
    cfg = copy.deepcopy(base_config)

    if action.name == "skip":
        return cfg  # skip action — config won't be used for trading

    strategy = cfg.setdefault("strategy", {})

    # Override pivots
    pivots = strategy.setdefault("pivots", {})
    if "right_bars" in action.params:
        pivots["right_bars"] = action.params["right_bars"]
    if "min_separation" in action.params:
        pivots["min_separation"] = action.params["min_separation"]

    # Override risk
    risk = strategy.setdefault("risk", {})
    if "rr_target" in action.params:
        risk["rr_target"] = action.params["rr_target"]

    # Override trend
    trend = strategy.setdefault("trend", {})
    if "min_close_ratio" in action.params:
        trend["min_close_ratio"] = action.params["min_close_ratio"]

    return cfg


def build_action_configs(base_config: dict) -> Dict[str, dict]:
    """Pre-build all action configs from a base tpe_trial_490 config.

    Returns:
        Dict mapping action_name -> strategy config dict.
        Skip is included but its config is the base config (unused).
    """
    return {
        a.name: build_strategy_config(base_config, a)
        for a in ACTIONS
    }
