"""Phase 6A — Policy implementations.

Three policy types:
  1. always_execute:      Execute every signal with base_tpe490 params.
  2. simple_rule_filter:   Skip if recent_5_trade_avg_r < 0; else base_tpe490.
  3. learned_policy:       sklearn RandomForest predicts best action per signal.

Each policy acts on a DataFrame of signal-level observations (pre-built dataset)
and returns actions + execution results.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from rl.phase6a_actions import (
    ACTION_NAMES,
    NON_SKIP_NAMES,
    NON_SKIP_ACTIONS,
    ActionDef,
    get_action_by_index,
)


# =========================================================================
# Policy Interface
# =========================================================================


class BasePolicy:
    """Base class for Phase 6A policies."""

    name: str = "base"

    def predict_actions(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Given a signal-level dataset, return a copy with 'chosen_action' column.

        Returns:
            DataFrame with original dataset columns + 'chosen_action' (str name).
        """
        raise NotImplementedError

    def fit(self, train_dataset: pd.DataFrame, val_dataset: pd.DataFrame) -> None:
        """Optional training step (used by learned_policy)."""
        pass


# =========================================================================
# Policy 1: Always Execute
# =========================================================================


class AlwaysExecutePolicy(BasePolicy):
    """Execute every signal using base_tpe490 action."""

    name = "always_execute"

    def __init__(self, default_action: str = "base_tpe490"):
        self.default_action = default_action

    def predict_actions(self, dataset: pd.DataFrame) -> pd.DataFrame:
        result = dataset.copy()
        result["chosen_action"] = self.default_action
        return result


# =========================================================================
# Policy 2: Simple Rule Filter
# =========================================================================


class SimpleRuleFilterPolicy(BasePolicy):
    """Skip trade if recent_5_trade_avg_r < 0.

    Otherwise execute with base_tpe490.
    """

    name = "simple_rule_filter"

    def __init__(self, recent_n: int = 5, min_avg_r: float = 0.0, default_action: str = "base_tpe490"):
        self.recent_n = recent_n
        self.min_avg_r = min_avg_r
        self.default_action = default_action

    def predict_actions(self, dataset: pd.DataFrame) -> pd.DataFrame:
        result = dataset.copy()

        # Apply rule: skip if recent avg R < threshold
        col = f"recent_{self.recent_n}_trade_avg_r"
        if col in result.columns:
            cond = result[col].fillna(0.0) < self.min_avg_r
            result["chosen_action"] = np.where(cond, "skip", self.default_action)
        else:
            result["chosen_action"] = self.default_action

        return result


# =========================================================================
# Policy 3: Learned Policy (RandomForest)
# =========================================================================


class LearnedPolicy(BasePolicy):
    """Use sklearn RandomForest to predict the best action at each signal.

    Training:
      - For each signal in train set, we have data on what each action's outcome was.
      - We train a classifier to predict which action gives the highest expected_r.
      - Alternatively, train a regressor per action to predict expected_r, then
        pick argmax.

    Phase 6A v1: Regressor-per-action approach.
      - Train one RandomForestRegressor per action (except skip).
      - At inference: predict expected_r for each non-skip action, pick max.
      - If all predicted expected_r < 0, choose skip.
    """

    name = "learned_policy"

    def __init__(
        self,
        feature_columns: List[str],
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        random_state: int = 42,
        min_predicted_r: float = 0.0,
    ):
        self.feature_columns = feature_columns
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.min_predicted_r = min_predicted_r

        # One regressor per non-skip action
        self.models: Dict[str, RandomForestClassifier] = {}
        self.action_names = NON_SKIP_NAMES
        self.is_fitted = False

    def fit(self, train_dataset: pd.DataFrame, val_dataset: Optional[pd.DataFrame] = None) -> None:
        """Train one RandomForest per action to predict net_r_multiple.

        train_dataset must contain rows with:
          - self.feature_columns (features)
          - 'action_name' (which action this row represents)
          - 'net_r_multiple' (label)

        Each action gets its own model trained on its own subset of rows.
        """
        # Validate columns
        missing = [c for c in self.feature_columns if c not in train_dataset.columns]
        if missing:
            raise ValueError(f"Missing feature columns in train dataset: {missing}")

        for action_name in self.action_names:
            subset = train_dataset[train_dataset["action_name"] == action_name].copy()
            subset = subset.dropna(subset=self.feature_columns + ["net_r_multiple"])

            if len(subset) < 10:
                print(f"  ⚠️  Action '{action_name}': only {len(subset)} training samples, using fallback")
                # Use a dummy model that always predicts the mean
                mean_r = subset["net_r_multiple"].mean() if len(subset) > 0 else 0.0
                # We'll handle this in predict
                self.models[action_name] = None
                self._fallback_mean_r = {**getattr(self, '_fallback_mean_r', {}), action_name: mean_r}
                continue

            X = subset[self.feature_columns].values
            y = subset["net_r_multiple"].values

            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            model.fit(X, y)
            self.models[action_name] = model

        self.is_fitted = True
        if not hasattr(self, '_fallback_mean_r'):
            self._fallback_mean_r = {}

        # Log training summary
        for action_name in self.action_names:
            n = len(train_dataset[train_dataset["action_name"] == action_name])
            print(f"  Model '{action_name}': {n} training rows")

    def predict_actions(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """For each signal, predict expected_r for all non-skip actions and pick best."""
        if not self.is_fitted:
            raise RuntimeError("LearnedPolicy must be fitted before prediction")

        result = dataset.copy()
        result["chosen_action"] = "skip"

        # For each action, predict expected_r
        predictions = {}
        for action_name in self.action_names:
            model = self.models.get(action_name)
            X = dataset[self.feature_columns].fillna(0.0).values

            if model is not None:
                try:
                    pred = model.predict(X)
                    predictions[action_name] = pred
                except Exception:
                    predictions[action_name] = np.zeros(len(dataset))
            else:
                # Fallback: use mean R from training
                mean_r = self._fallback_mean_r.get(action_name, 0.0)
                # For classifier, we convert to 0/1 based on sign
                predictions[action_name] = np.full(len(dataset), 1.0 if mean_r > 0 else 0.0)

        # Choose action with highest predicted R
        best_actions = []
        for i in range(len(dataset)):
            best_action = "skip"
            best_pred_r = self.min_predicted_r - 0.001  # below threshold

            for action_name in self.action_names:
                pred_r = predictions[action_name][i]
                if pred_r > best_pred_r:
                    best_pred_r = pred_r
                    best_action = action_name

            if best_pred_r < self.min_predicted_r:
                best_action = "skip"

            best_actions.append(best_action)

        result["chosen_action"] = best_actions
        return result


# =========================================================================
# Factory
# =========================================================================

POLICY_REGISTRY: Dict[str, type] = {
    "always_execute": AlwaysExecutePolicy,
    "simple_rule_filter": SimpleRuleFilterPolicy,
    "learned_policy": LearnedPolicy,
}


def create_policy(name: str, **kwargs) -> BasePolicy:
    """Create a policy by name with given kwargs."""
    cls = POLICY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown policy: {name}. Available: {list(POLICY_REGISTRY.keys())}")
    return cls(**kwargs)
