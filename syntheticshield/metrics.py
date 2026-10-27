from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def synthetic_utility(
    synthetic: pd.DataFrame,
    real_test: pd.DataFrame,
    feature_columns: list[str],
    label_column: str = "label",
) -> dict:
    """Train on synthetic, test on real held-out data."""
    x_syn = synthetic[feature_columns].to_numpy(dtype=float)
    y_syn = synthetic[label_column].to_numpy(dtype=int)
    x_test = real_test[feature_columns].to_numpy(dtype=float)
    y_test = real_test[label_column].to_numpy(dtype=int)

    # Guard against degenerate synthetic labels.
    if len(np.unique(y_syn)) < 2:
        majority = int(pd.Series(y_syn).mode().iloc[0]) if len(y_syn) else 0
        pred = np.full_like(y_test, majority)
        return {"utility_auc": 0.5, "utility_accuracy": float(accuracy_score(y_test, pred))}

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs"),
    )
    clf.fit(x_syn, y_syn)
    proba = clf.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "utility_auc": float(roc_auc_score(y_test, proba)),
        "utility_accuracy": float(accuracy_score(y_test, pred)),
    }
