from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PrivateDataset:
    train: pd.DataFrame
    test: pd.DataFrame
    canaries: pd.DataFrame
    feature_columns: list[str]
    label_column: str = "label"


def _make_base_table(seed: int, n_samples: int, n_features: int) -> pd.DataFrame:
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=max(0, n_features // 4),
        n_repeated=0,
        n_classes=2,
        class_sep=1.0,
        flip_y=0.03,
        random_state=seed,
    )
    cols = [f"x{i}" for i in range(n_features)]
    df = pd.DataFrame(x, columns=cols)
    df["label"] = y.astype(int)
    return df


def _make_canaries(
    seed: int,
    n_features: int,
    n_canaries_per_rarity: int,
    rarity_levels: Iterable[float],
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    cols = [f"x{i}" for i in range(n_features)]
    for rarity in rarity_levels:
        for idx in range(n_canaries_per_rarity):
            # A directional rare pattern: a deterministic offset plus tiny jitter.
            direction = np.zeros(n_features)
            direction[idx % n_features] = 1.0
            direction[(idx + 3) % n_features] = -0.8
            jitter = rng.normal(0.0, 0.04, size=n_features)
            values = rarity * 2.0 * direction + jitter
            row = {col: values[j] for j, col in enumerate(cols)}
            row["label"] = int(idx % 2 == 0)
            row["is_canary"] = 1
            row["canary_id"] = f"r{rarity:.1f}_{idx:03d}"
            row["rarity"] = float(rarity)
            rows.append(row)
    return pd.DataFrame(rows)


def make_private_dataset(
    seed: int = 7,
    n_train: int = 650,
    n_test: int = 650,
    n_features: int = 8,
    n_canaries_per_rarity: int = 12,
    rarity_levels: Iterable[float] = (1.5, 2.5, 3.5, 4.5),
) -> PrivateDataset:
    """Create a private tabular dataset with inserted rare canary records.

    The returned training table contains normal records plus canaries. The held-out
    test table is never given to the generators and acts as the non-member pool.
    """
    base = _make_base_table(seed, n_train + n_test, n_features)
    feature_columns = [f"x{i}" for i in range(n_features)]

    train_base, test = train_test_split(
        base,
        train_size=n_train,
        test_size=n_test,
        random_state=seed,
        stratify=base["label"],
    )
    train_base = train_base.reset_index(drop=True)
    test = test.reset_index(drop=True)

    scaler = StandardScaler()
    train_base[feature_columns] = scaler.fit_transform(train_base[feature_columns])
    test[feature_columns] = scaler.transform(test[feature_columns])

    train_base["is_canary"] = 0
    train_base["canary_id"] = ""
    train_base["rarity"] = 0.0
    test["is_canary"] = 0
    test["canary_id"] = ""
    test["rarity"] = 0.0

    canaries = _make_canaries(seed + 100, n_features, n_canaries_per_rarity, rarity_levels)
    train = pd.concat([train_base, canaries], ignore_index=True)

    return PrivateDataset(
        train=train,
        test=test,
        canaries=canaries.reset_index(drop=True),
        feature_columns=feature_columns,
        label_column="label",
    )
