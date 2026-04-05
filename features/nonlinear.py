"""
Doğrusal Olmayan (Nonlinear) Özellikler

Saf NumPy/SciPy implementasyonları (antropy/nolds bağımlılığı yok):
    - LLE       : Largest Lyapunov Exponent (Rosenstein yöntemi)
    - Katz FD   : Katz Fractal Dimension
    - Higuchi FD: Higuchi Fractal Dimension
    - Hurst     : Hurst Exponent (R/S analizi)
    - DFA       : Detrended Fluctuation Analysis
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


# ── Katz Fractal Dimension ──────────────────────────────────────────

def katz_fd(x: np.ndarray) -> float:
    """
    Katz Fractal Dimension.

    FD = log10(L) / log10(d)
    L = toplam yol uzunluğu, d = başlangıçtan en uzak noktaya mesafe
    """
    n = len(x)
    if n < 2:
        return np.nan

    # Ardışık farkların mutlak değerleri toplamı → toplam yol
    dists = np.abs(np.diff(x))
    L = np.sum(dists)

    # Başlangıç noktasından tüm noktalara mesafe → max
    d = np.max(np.abs(x - x[0]))

    if d < 1e-15 or L < 1e-15:
        return np.nan

    a = np.mean(dists)  # ortalama adım
    if a < 1e-15:
        return np.nan

    n_steps = n - 1
    fd = np.log10(n_steps) / (np.log10(n_steps) + np.log10(d / L))

    return float(fd)


# ── Higuchi Fractal Dimension ───────────────────────────────────────

def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    """
    Higuchi Fractal Dimension.

    Args:
        x:    1D sinyal
        kmax: maksimum interval (varsayılan 10)
    """
    n = len(x)
    if n < kmax + 1:
        kmax = n // 2
    if kmax < 2:
        return np.nan

    lk = np.zeros(kmax)

    for k in range(1, kmax + 1):
        lm_k = []
        for m in range(1, k + 1):
            # Alt dizi indeksleri: m, m+k, m+2k, ...
            idx = np.arange(m - 1, n, k)
            if len(idx) < 2:
                continue
            sub = x[idx]
            # Uzunluk hesabı
            length = np.sum(np.abs(np.diff(sub))) * (n - 1) / (k * len(idx) * k)
            # Düzeltme: Normalizasyon
            a = int(np.floor((n - m) / k))
            if a < 1:
                continue
            length_norm = (np.sum(np.abs(np.diff(sub[:a + 1]))) * (n - 1)) / (a * k * k)
            lm_k.append(length_norm)

        if lm_k:
            lk[k - 1] = np.mean(lm_k)

    # log-log regresyon
    valid = lk > 0
    if valid.sum() < 2:
        return np.nan

    k_vals = np.arange(1, kmax + 1)[valid]
    lk_vals = lk[valid]

    log_k = np.log(1.0 / k_vals)
    log_lk = np.log(lk_vals)

    # En küçük kareler regresyon
    slope, _ = np.polyfit(log_k, log_lk, 1)

    return float(slope)


# ── Hurst Exponent (R/S analizi) ────────────────────────────────────

def hurst_exponent(x: np.ndarray) -> float:
    """
    Hurst Exponent — Rescaled Range (R/S) analizi.
    """
    n = len(x)
    if n < 20:
        return np.nan

    # Farklı alt dizi uzunlukları
    max_k = n // 2
    sizes = []
    rs_values = []

    for div in range(2, min(20, max_k)):
        size = n // div
        if size < 10:
            break

        rs_list = []
        for start in range(0, n - size + 1, size):
            sub = x[start:start + size]
            mean_sub = np.mean(sub)
            deviations = np.cumsum(sub - mean_sub)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(sub, ddof=1)
            if S > 1e-15:
                rs_list.append(R / S)

        if rs_list:
            sizes.append(size)
            rs_values.append(np.mean(rs_list))

    if len(sizes) < 3:
        return np.nan

    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)

    slope, _ = np.polyfit(log_sizes, log_rs, 1)

    return float(np.clip(slope, 0.0, 1.0))


# ── DFA (Detrended Fluctuation Analysis) ────────────────────────────

def dfa(x: np.ndarray) -> float:
    """
    Detrended Fluctuation Analysis.

    Returns:
        alpha: DFA skalama üsteli
    """
    n = len(x)
    if n < 16:
        return np.nan

    # Kümülatif toplam (profil)
    y = np.cumsum(x - np.mean(x))

    # Farklı pencere boyutları
    min_box = 4
    max_box = n // 4
    if max_box < min_box + 2:
        return np.nan

    n_boxes = min(15, max_box - min_box)
    box_sizes = np.unique(np.logspace(
        np.log10(min_box), np.log10(max_box), n_boxes
    ).astype(int))

    if len(box_sizes) < 3:
        return np.nan

    fluctuations = []

    for box in box_sizes:
        n_segments = n // box
        if n_segments < 1:
            fluctuations.append(np.nan)
            continue

        rms_values = []
        for seg_i in range(n_segments):
            start = seg_i * box
            end = start + box
            segment = y[start:end]

            # Lineer trend çıkar
            t = np.arange(box)
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            residual = segment - trend

            rms_values.append(np.sqrt(np.mean(residual ** 2)))

        fluctuations.append(np.mean(rms_values))

    fluctuations = np.array(fluctuations)
    valid = ~np.isnan(fluctuations) & (fluctuations > 0)

    if valid.sum() < 3:
        return np.nan

    log_box = np.log(box_sizes[valid])
    log_fluct = np.log(fluctuations[valid])

    alpha, _ = np.polyfit(log_box, log_fluct, 1)

    return float(alpha)


# ── Largest Lyapunov Exponent (Rosenstein) ──────────────────────────

def lle_rosenstein(x: np.ndarray, tau: int = 1, emb_dim: int = 10,
                   min_tsep: int = 50) -> float:
    """
    Largest Lyapunov Exponent — Rosenstein yöntemi (basitleştirilmiş).

    Args:
        x:        1D sinyal
        tau:      gecikme (delay)
        emb_dim:  gömme boyutu
        min_tsep: en yakın komşu aramada minimum zaman ayrımı
    """
    n = len(x)
    N = n - (emb_dim - 1) * tau

    if N < min_tsep + 10:
        return np.nan

    # Faz uzayı yeniden oluşturma
    phase_space = np.zeros((N, emb_dim))
    for i in range(emb_dim):
        phase_space[:, i] = x[i * tau : i * tau + N]

    # Her nokta için en yakın komşuyu bul (min_tsep uzağında)
    max_iter = min(N // 4, 50)
    divergence = np.zeros(max_iter)
    count = np.zeros(max_iter)

    # Hızlı yaklaşım: alt örnekleme
    step = max(1, N // 200)
    sample_indices = np.arange(0, N - max_iter, step)

    for i in sample_indices:
        # i'den en az min_tsep uzaktaki noktalar
        dists = np.sqrt(np.sum((phase_space - phase_space[i]) ** 2, axis=1))
        dists[:max(0, i - min_tsep)] = np.inf  # yakın komşuyu engelle
        dists[i] = np.inf
        dists[min(i + min_tsep, N):] = dists[min(i + min_tsep, N):]  # keep
        # i ± min_tsep arasını engelle
        start_ex = max(0, i - min_tsep)
        end_ex = min(N, i + min_tsep + 1)
        dists[start_ex:end_ex] = np.inf

        j = np.argmin(dists)
        if dists[j] == np.inf:
            continue

        for k in range(max_iter):
            if i + k < N and j + k < N:
                d = np.sqrt(np.sum((phase_space[i + k] - phase_space[j + k]) ** 2))
                if d > 0:
                    divergence[k] += np.log(d)
                    count[k] += 1

    # Ortalama logaritmik diverjans
    valid = count > 0
    if valid.sum() < 5:
        return np.nan

    avg_div = np.zeros_like(divergence)
    avg_div[valid] = divergence[valid] / count[valid]

    # Lineer regresyon (ilk kısım)
    n_fit = min(10, valid.sum())
    t = np.arange(n_fit)
    y = avg_div[:n_fit]

    if np.std(y) < 1e-15:
        return 0.0

    slope, _ = np.polyfit(t, y, 1)

    return float(slope)


# ── Fonksiyon Eşleme ───────────────────────────────────────────────

NONLINEAR_FEATURES = {
    "lle":        lle_rosenstein,
    "katz_fd":    katz_fd,
    "higuchi_fd": higuchi_fd,
    "hurst":      hurst_exponent,
    "dfa":        dfa,
}


def extract_nonlinear_features(
    segment: np.ndarray,
    ch_names: list[str],
    feature_names: list[str] | None = None,
    higuchi_kmax: int = 10,
) -> dict[str, float]:
    """
    Tek segment için nonlineer özellikler.

    Returns:
        dict: "{ch_name}_{feature_name}" → float
    """
    if feature_names is None:
        feature_names = list(NONLINEAR_FEATURES.keys())

    result = {}
    n_channels = segment.shape[1]

    for ch_i in range(n_channels):
        ch = ch_names[ch_i]
        x = segment[:, ch_i]

        for feat_name in feature_names:
            if feat_name not in NONLINEAR_FEATURES:
                continue
            try:
                func = NONLINEAR_FEATURES[feat_name]
                if feat_name == "higuchi_fd":
                    val = func(x, kmax=higuchi_kmax)
                else:
                    val = func(x)
            except Exception:
                val = np.nan
            result[f"{ch}_{feat_name}"] = val

    return result
