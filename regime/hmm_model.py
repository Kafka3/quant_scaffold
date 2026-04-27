import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM


FEATURE_COLUMNS = ["ret", "vol", "range"]


def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ret"] = df["Close"].pct_change().fillna(0.0)
    out["vol"] = out["ret"].rolling(20).std().fillna(0.0)
    out["range"] = ((df["High"] - df["Low"]) / df["Close"]).fillna(0.0)
    return out


def fit_predict_hmm_expanding(features: pd.DataFrame, n_states: int = 3, min_train_size: int = 100) -> pd.Series:
    """
    使用 expanding window 训练 HMM 并预测状态，避免未来函数。

    对每个时间点 t >= min_train_size：
      - 用 features[:t] 训练 HMM
      - 预测 features[t] 的状态
    前 min_train_size 个点返回 NaN（无法训练）。
    """
    n = len(features)
    states = pd.Series(index=features.index, dtype=float, name="regime")
    states.iloc[:min_train_size] = np.nan

    for t in range(min_train_size, n):
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=42)
        train_data = features[FEATURE_COLUMNS].iloc[:t].values
        model.fit(train_data)
        pred = model.predict(features[FEATURE_COLUMNS].iloc[t : t + 1].values)
        states.iloc[t] = pred[0]

    return states
