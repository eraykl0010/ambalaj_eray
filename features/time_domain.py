"""
Zaman Alanı Özellikleri

Her segment (window_samples, n_channels) için kanal bazlı hesaplama:
    - Hjorth Activity, Mobility, Complexity
    - Mean, Skewness, RMS, MAD, Kurtosis, Energy

Tüm fonksiyonlar tek kanal (1D array) üzerinde çalışır.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis


# ── Hjorth Parametreleri ────────────────────────────────────────────

def hjorth_activity(x: np.ndarray) -> float:
    """Hjorth Activity = sinyal varyansı."""
    return float(np.var(x))


def hjorth_mobility(x: np.ndarray) -> float:
    """Hjorth Mobility = 1. türevin std / sinyalin std."""
    dx = np.diff(x)
    std_x = np.std(x)
    if std_x < 1e-15:
        return 0.0
    return float(np.std(dx) / std_x)


def hjorth_complexity(x: np.ndarray) -> float:
    """Hjorth Complexity = mobility(dx) / mobility(x)."""
    dx = np.diff(x)
    mob_x = hjorth_mobility(x)
    if mob_x < 1e-15:
        return 0.0
    mob_dx = hjorth_mobility(dx)
    return float(mob_dx / mob_x)


# ── Temel İstatistikler ────────────────────────────────────────────

def signal_mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def signal_skewness(x: np.ndarray) -> float:
    return float(skew(x, bias=False))


def signal_rms(x: np.ndarray) -> float:
    """Root Mean Square."""
    return float(np.sqrt(np.mean(x ** 2)))


def signal_mad(x: np.ndarray) -> float:
    """Mean Absolute Deviation."""
    return float(np.mean(np.abs(x - np.mean(x))))


def signal_kurtosis(x: np.ndarray) -> float:
    return float(kurtosis(x, fisher=True, bias=False))


def signal_energy(x: np.ndarray) -> float:
    """Sinyal enerjisi = karelerin toplamı."""
    return float(np.sum(x ** 2))


# ── Kanal Bazlı Hesaplama ──────────────────────────────────────────

# Fonksiyon adı → callable eşlemesi
TIME_DOMAIN_FEATURES = {
    "hjorth_activity":   hjorth_activity,
    "hjorth_mobility":   hjorth_mobility,
    "hjorth_complexity": hjorth_complexity,
    "mean":              signal_mean,
    "skewness":          signal_skewness,
    "rms":               signal_rms,
    "mad":               signal_mad,
    "kurtosis":          signal_kurtosis,
    "energy":            signal_energy,
}


def extract_time_features(
    segment: np.ndarray,
    ch_names: list[str],
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Tek segment (window_samples, n_channels) için zaman alanı özelliklerini çıkar.

    Returns:
        dict: "{ch_name}_{feature_name}" → float
    """
    if feature_names is None:
        feature_names = list(TIME_DOMAIN_FEATURES.keys())

    result = {}
    n_channels = segment.shape[1]

    for ch_i in range(n_channels):
        ch = ch_names[ch_i]
        x = segment[:, ch_i]

        for feat_name in feature_names:
            if feat_name not in TIME_DOMAIN_FEATURES:
                continue
            try:
                val = TIME_DOMAIN_FEATURES[feat_name](x)
            except Exception:
                val = np.nan
            result[f"{ch}_{feat_name}"] = val

    return result
