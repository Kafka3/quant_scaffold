import pandas as pd
from hmmlearn.hmm import GaussianHMM


FEATURE_COLUMNS = ["ret", "vol", "range"]


def build_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ret"] = df["Close"].pct_change().fillna(0.0)
    out["vol"] = out["ret"].rolling(20).std().fillna(0.0)
    out["range"] = ((df["High"] - df["Low"]) / df["Close"]).fillna(0.0)
    return out


def fit_hmm(features: pd.DataFrame, n_states: int = 3) -> GaussianHMM:
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=42)
    model.fit(features[FEATURE_COLUMNS].values)
    return model


def predict_states(model: GaussianHMM, features: pd.DataFrame) -> pd.Series:
    states = model.predict(features[FEATURE_COLUMNS].values)
    return pd.Series(states, index=features.index, name="regime")
