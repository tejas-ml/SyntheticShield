from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    name: str
    seed: int
    synthetic_size: int


class BootstrapSynthesizer:
    """A deliberately memorization-prone bootstrap synthesizer.

    This baseline mimics a weak synthetic data generator that resamples private
    rows and adds small noise. It is useful for demonstrating membership leakage.
    """

    def __init__(self, noise_scale: float = 0.08, seed: int = 0):
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
        self.train_: pd.DataFrame | None = None
        self.feature_columns_: list[str] | None = None
        self.label_column_: str | None = None

    def fit(self, train: pd.DataFrame, feature_columns: list[str], label_column: str = "label"):
        self.train_ = train.reset_index(drop=True).copy()
        self.feature_columns_ = list(feature_columns)
        self.label_column_ = label_column
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self.train_ is None or self.feature_columns_ is None or self.label_column_ is None:
            raise RuntimeError("fit must be called before sample")
        idx = self.rng.integers(0, len(self.train_), size=n)
        synth = self.train_.iloc[idx].reset_index(drop=True).copy()
        noise = self.rng.normal(0.0, self.noise_scale, size=(n, len(self.feature_columns_)))
        synth[self.feature_columns_] = synth[self.feature_columns_].to_numpy() + noise
        synth["source_row"] = idx
        return synth[self.feature_columns_ + [self.label_column_, "source_row"]]


class DPGaussianSynthesizer:
    """Clipped Gaussian synthesizer with DP-inspired parameter noise.

    This class is intentionally a simple experimental defense. It does not claim
    a formal differential privacy guarantee because no privacy accountant is used.
    """

    def __init__(self, clip: float = 3.0, mean_noise: float = 0.18, cov_noise: float = 0.10, seed: int = 0):
        self.clip = clip
        self.mean_noise = mean_noise
        self.cov_noise = cov_noise
        self.rng = np.random.default_rng(seed)

    def fit(self, train: pd.DataFrame, feature_columns: list[str], label_column: str = "label"):
        self.feature_columns_ = list(feature_columns)
        self.label_column_ = label_column
        x = np.clip(train[self.feature_columns_].to_numpy(dtype=float), -self.clip, self.clip)
        y = train[label_column].to_numpy(dtype=int)
        self.label_rate_ = float(np.clip(y.mean() + self.rng.normal(0, 0.04), 0.05, 0.95))
        mean = x.mean(axis=0) + self.rng.normal(0.0, self.mean_noise, size=x.shape[1])
        cov = np.cov(x, rowvar=False)
        cov += np.eye(cov.shape[0]) * (self.cov_noise + 1e-3)
        self.mean_ = mean
        self.cov_ = cov
        return self

    def sample(self, n: int) -> pd.DataFrame:
        x = self.rng.multivariate_normal(self.mean_, self.cov_, size=n)
        y = self.rng.binomial(1, self.label_rate_, size=n)
        df = pd.DataFrame(x, columns=self.feature_columns_)
        df[self.label_column_] = y.astype(int)
        df["source_row"] = -1
        return df


class DeduplicatedBootstrapSynthesizer:
    """Bootstrap synthesizer after simple deduplication and outlier filtering."""

    def __init__(self, noise_scale: float = 0.22, z_cutoff: float = 4.0, seed: int = 0):
        self.noise_scale = noise_scale
        self.z_cutoff = z_cutoff
        self.rng = np.random.default_rng(seed)
        self.base = BootstrapSynthesizer(noise_scale=noise_scale, seed=seed)

    def fit(self, train: pd.DataFrame, feature_columns: list[str], label_column: str = "label"):
        x = train[feature_columns].to_numpy(dtype=float)
        # Remove exact duplicate feature rows after rounding, then remove extreme outliers.
        rounded = train.assign(**{c: train[c].round(3) for c in feature_columns})
        rounded = rounded.drop_duplicates(subset=feature_columns + [label_column]).reset_index(drop=True)
        z = np.abs(rounded[feature_columns].to_numpy(dtype=float))
        filtered = rounded[(z < self.z_cutoff).all(axis=1)].copy()
        if len(filtered) < max(10, int(0.2 * len(train))):
            filtered = rounded.copy()
        self.filtered_train_ = filtered.reset_index(drop=True)
        self.base.fit(self.filtered_train_, feature_columns, label_column)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        return self.base.sample(n)


def build_generator(defense: str, seed: int):
    if defense == "no_defense":
        return BootstrapSynthesizer(noise_scale=0.055, seed=seed)
    if defense == "dp_training":
        return DPGaussianSynthesizer(clip=2.6, mean_noise=0.22, cov_noise=0.18, seed=seed)
    if defense == "deduplication":
        return DeduplicatedBootstrapSynthesizer(noise_scale=0.18, z_cutoff=3.6, seed=seed)
    raise ValueError(f"Unknown defense: {defense}")
