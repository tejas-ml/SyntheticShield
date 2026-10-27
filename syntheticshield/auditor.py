from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def nearest_neighbor_distances(reference: pd.DataFrame, query: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(reference[feature_columns].to_numpy(dtype=float))
    distances, _ = nn.kneighbors(query[feature_columns].to_numpy(dtype=float), return_distance=True)
    return distances[:, 0]


def membership_inference_audit(
    train_members: pd.DataFrame,
    test_nonmembers: pd.DataFrame,
    synthetic: pd.DataFrame,
    feature_columns: list[str],
    fpr_target: float = 0.05,
) -> dict:
    """Nearest-neighbor membership inference audit.

    A sample is predicted as a member if it is unusually close to synthetic data.
    """
    member_d = nearest_neighbor_distances(synthetic, train_members, feature_columns)
    nonmember_d = nearest_neighbor_distances(synthetic, test_nonmembers, feature_columns)

    y_true = np.concatenate([np.ones_like(member_d), np.zeros_like(nonmember_d)])
    scores = -np.concatenate([member_d, nonmember_d])
    auc = roc_auc_score(y_true, scores)

    threshold = np.quantile(nonmember_d, fpr_target)
    tpr = float((member_d <= threshold).mean())
    fpr = float((nonmember_d <= threshold).mean())

    return {
        "membership_auc": float(auc),
        "leakage_rate_at_5pct_fpr": tpr,
        "observed_fpr": fpr,
        "attack_distance_threshold": float(threshold),
        "member_nn_distance_mean": float(member_d.mean()),
        "nonmember_nn_distance_mean": float(nonmember_d.mean()),
    }


def threshold_sweep(
    train_members: pd.DataFrame,
    test_nonmembers: pd.DataFrame,
    synthetic: pd.DataFrame,
    feature_columns: list[str],
    n_thresholds: int = 80,
) -> pd.DataFrame:
    member_d = nearest_neighbor_distances(synthetic, train_members, feature_columns)
    nonmember_d = nearest_neighbor_distances(synthetic, test_nonmembers, feature_columns)
    all_d = np.concatenate([member_d, nonmember_d])
    thresholds = np.quantile(all_d, np.linspace(0.01, 0.99, n_thresholds))
    rows = []
    for thr in thresholds:
        rows.append({
            "distance_threshold": float(thr),
            "tpr": float((member_d <= thr).mean()),
            "fpr": float((nonmember_d <= thr).mean()),
        })
    return pd.DataFrame(rows)


def canary_exposure_audit(
    canaries: pd.DataFrame,
    synthetic: pd.DataFrame,
    feature_columns: list[str],
    epsilon: float = 0.35,
) -> pd.DataFrame:
    d = nearest_neighbor_distances(synthetic, canaries, feature_columns)
    out = canaries[["canary_id", "rarity", "label"]].copy()
    out["nn_distance"] = d
    out["exposed"] = (d <= epsilon).astype(int)
    out["exposure_score"] = -np.log10(np.maximum(d, 1e-8))
    return out
