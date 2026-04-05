"""
Entropi Tabanlı Özellikler

Saf NumPy implementasyonları:
    - ApEn    : Approximate Entropy
    - PE      : Permutation Entropy
    - FuzzyEn : Fuzzy Entropy
    - SE      : Sample Entropy
"""

from __future__ import annotations

import numpy as np
from math import factorial


# ── Yardımcı: Gömme Matrisi ────────────────────────────────────────

def _embed(x: np.ndarray, m: int, tau: int = 1) -> np.ndarray:
    """
    Zaman gecikmeli gömme matrisi oluştur.

    Args:
        x:   1D sinyal
        m:   gömme boyutu
        tau: gecikme

    Returns:
        (N - (m-1)*tau, m) matris
    """
    n = len(x)
    N = n - (m - 1) * tau
    if N <= 0:
        return np.array([]).reshape(0, m)

    indices = np.arange(N)[:, None] + np.arange(m)[None, :] * tau
    return x[indices]


# ── Approximate Entropy (ApEn) ──────────────────────────────────────

def approximate_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Approximate Entropy.

    Args:
        x: 1D sinyal
        m: gömme boyutu
        r: tolerans oranı (r * std kullanılır)
    """
    n = len(x)
    if n < m + 2:
        return np.nan

    tolerance = r * np.std(x)
    if tolerance < 1e-15:
        return np.nan

    def _phi(m_val):
        embedded = _embed(x, m_val)
        N = len(embedded)
        if N < 1:
            return np.nan

        counts = np.zeros(N)
        for i in range(N):
            # Chebyshev distance
            dist = np.max(np.abs(embedded - embedded[i]), axis=1)
            counts[i] = np.sum(dist <= tolerance) / N

        # log(0) koruması
        counts = counts[counts > 0]
        if len(counts) == 0:
            return np.nan
        return np.mean(np.log(counts))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    if np.isnan(phi_m) or np.isnan(phi_m1):
        return np.nan

    return float(phi_m - phi_m1)


# ── Sample Entropy (SE) ─────────────────────────────────────────────

def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Sample Entropy.

    ApEn'in bias-düzeltilmiş versiyonu.
    """
    n = len(x)
    if n < m + 2:
        return np.nan

    tolerance = r * np.std(x)
    if tolerance < 1e-15:
        return np.nan

    def _count_matches(m_val):
        embedded = _embed(x, m_val)
        N = len(embedded)
        if N < 2:
            return 0

        count = 0
        for i in range(N - 1):
            # Self-match hariç
            dist = np.max(np.abs(embedded[i + 1:] - embedded[i]), axis=1)
            count += np.sum(dist <= tolerance)
        return count

    B = _count_matches(m)      # m-boyutlu eşleşmeler
    A = _count_matches(m + 1)  # (m+1)-boyutlu eşleşmeler

    if B == 0:
        return np.nan

    return float(-np.log(A / B)) if A > 0 else float(np.nan)


# ── Permutation Entropy (PE) ────────────────────────────────────────

def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1,
                        normalize: bool = True) -> float:
    """
    Permutation Entropy.

    Args:
        x:         1D sinyal
        order:     permütasyon sırası (m)
        delay:     gecikme (tau)
        normalize: True ise [0, 1] arası normalize et
    """
    n = len(x)
    n_permutations = factorial(order)

    if n < order * delay + 1:
        return np.nan

    # Gömme
    embedded = _embed(x, order, delay)
    N = len(embedded)

    if N < 1:
        return np.nan

    # Her satırın sıralama düzenini (ordinal pattern) bul
    patterns = np.argsort(embedded, axis=1)

    # Her pattern'i unique string'e çevir ve say
    pattern_counts = {}
    for i in range(N):
        key = tuple(patterns[i])
        pattern_counts[key] = pattern_counts.get(key, 0) + 1

    # Shannon entropi
    probs = np.array(list(pattern_counts.values()), dtype=float) / N
    pe = -np.sum(probs * np.log2(probs))

    if normalize:
        max_pe = np.log2(n_permutations)
        if max_pe > 0:
            pe /= max_pe

    return float(pe)


# ── Fuzzy Entropy (FuzzyEn) ─────────────────────────────────────────

def fuzzy_entropy(x: np.ndarray, m: int = 2, r: float = 0.2,
                  n_exp: float = 2.0) -> float:
    """
    Fuzzy Entropy.

    Sample Entropy'nin fuzzy membership fonksiyonu ile genelleştirilmesi.
    Heaviside step yerine exp(-d^n / r) kullanır.

    Args:
        x:     1D sinyal
        m:     gömme boyutu
        r:     tolerans oranı (r * std)
        n_exp: fuzzy üstel (varsayılan 2)
    """
    n = len(x)
    if n < m + 2:
        return np.nan

    tolerance = r * np.std(x)
    if tolerance < 1e-15:
        return np.nan

    def _fuzzy_count(m_val):
        embedded = _embed(x, m_val)
        N = len(embedded)
        if N < 2:
            return 0.0

        # Her vektörden kendi ortalamasını çıkar (yerel ortalama çıkarma)
        embedded_centered = embedded - np.mean(embedded, axis=1, keepdims=True)

        total = 0.0
        for i in range(N - 1):
            # Chebyshev distance
            dist = np.max(np.abs(embedded_centered[i + 1:] - embedded_centered[i]),
                          axis=1)
            # Fuzzy membership
            similarity = np.exp(-(dist ** n_exp) / tolerance)
            total += np.sum(similarity)

        # Normalize
        pairs = N * (N - 1) / 2
        return total / pairs if pairs > 0 else 0.0

    phi_m = _fuzzy_count(m)
    phi_m1 = _fuzzy_count(m + 1)

    if phi_m < 1e-15:
        return np.nan

    return float(-np.log(phi_m1 / phi_m)) if phi_m1 > 0 else float(np.nan)


# ── Fonksiyon Eşleme ───────────────────────────────────────────────

ENTROPY_FEATURES = {
    "apen":   approximate_entropy,
    "pe":     permutation_entropy,
    "fuzzen": fuzzy_entropy,
    "se":     sample_entropy,
}


def extract_entropy_features(
    segment: np.ndarray,
    ch_names: list[str],
    feature_names: list[str] | None = None,
    embedding_dim: int = 2,
    tolerance_r: float = 0.2,
    pe_order: int = 3,
    pe_delay: int = 1,
) -> dict[str, float]:
    """
    Tek segment için entropi özelliklerini çıkar.

    Returns:
        dict: "{ch_name}_{feature_name}" → float
    """
    if feature_names is None:
        feature_names = list(ENTROPY_FEATURES.keys())

    result = {}
    n_channels = segment.shape[1]

    for ch_i in range(n_channels):
        ch = ch_names[ch_i]
        x = segment[:, ch_i]

        for feat_name in feature_names:
            try:
                if feat_name == "apen":
                    val = approximate_entropy(x, m=embedding_dim, r=tolerance_r)
                elif feat_name == "se":
                    val = sample_entropy(x, m=embedding_dim, r=tolerance_r)
                elif feat_name == "pe":
                    val = permutation_entropy(x, order=pe_order, delay=pe_delay)
                elif feat_name == "fuzzen":
                    val = fuzzy_entropy(x, m=embedding_dim, r=tolerance_r)
                else:
                    val = np.nan
            except Exception:
                val = np.nan
            result[f"{ch}_{feat_name}"] = val

    return result
