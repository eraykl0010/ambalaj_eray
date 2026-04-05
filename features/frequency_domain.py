"""
Frekans Alanı Özellikleri — Sub-band Güçleri

Welch PSD kullanarak her kanal için:
    - Delta  (0.5–4 Hz)
    - Theta  (4–8 Hz)
    - Alpha  (8–13 Hz)
    - Beta   (13–30 Hz)
    - Gamma  (30–45 Hz)

Hem mutlak güç hem de toplam güce oranla (relatif güç) hesaplanır.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid


def compute_bandpowers(
    x: np.ndarray,
    sr: int,
    bands: dict[str, list[float]],
    method: str = "welch",
) -> dict[str, float]:
    """
    Tek kanal (1D) için sub-band güçlerini hesapla.

    Args:
        x:      1D sinyal array
        sr:     örnekleme oranı (Hz)
        bands:  {"delta": [0.5, 4.0], "theta": [4.0, 8.0], ...}
        method: PSD yöntemi ("welch")

    Returns:
        dict: "band_abs" → mutlak güç, "band_rel" → relatif güç
    """
    # PSD hesapla
    nperseg = min(sr * 2, len(x))  # 2 saniyelik pencere veya sinyal uzunluğu
    if nperseg < 4:
        # Çok kısa sinyal — NaN döndür
        result = {}
        for band_name in bands:
            result[f"{band_name}_abs"] = np.nan
            result[f"{band_name}_rel"] = np.nan
        return result

    freqs, psd = welch(x, fs=sr, nperseg=nperseg, noverlap=nperseg // 2)

    # Toplam güç (0.5–45 Hz arası)
    total_mask = (freqs >= 0.5) & (freqs <= 45.0)
    total_power = trapezoid(psd[total_mask], freqs[total_mask])

    if total_power < 1e-20:
        total_power = 1e-20  # sıfıra bölme koruması

    result = {}
    for band_name, (f_low, f_high) in bands.items():
        band_mask = (freqs >= f_low) & (freqs <= f_high)

        if band_mask.sum() < 2:
            result[f"{band_name}_abs"] = np.nan
            result[f"{band_name}_rel"] = np.nan
            continue

        band_power = trapezoid(psd[band_mask], freqs[band_mask])
        result[f"{band_name}_abs"] = float(band_power)
        result[f"{band_name}_rel"] = float(band_power / total_power)

    return result


def extract_frequency_features(
    segment: np.ndarray,
    ch_names: list[str],
    sr: int,
    bands: dict[str, list[float]] | None = None,
) -> dict[str, float]:
    """
    Tek segment (window_samples, n_channels) için frekans özelliklerini çıkar.

    Returns:
        dict: "{ch_name}_{band}_{abs|rel}" → float
    """
    if bands is None:
        bands = {
            "delta": [0.5, 4.0],
            "theta": [4.0, 8.0],
            "alpha": [8.0, 13.0],
            "beta":  [13.0, 30.0],
            "gamma": [30.0, 45.0],
        }

    result = {}
    n_channels = segment.shape[1]

    for ch_i in range(n_channels):
        ch = ch_names[ch_i]
        x = segment[:, ch_i]

        try:
            bp = compute_bandpowers(x, sr, bands)
        except Exception:
            bp = {f"{b}_{t}": np.nan for b in bands for t in ["abs", "rel"]}

        for key, val in bp.items():
            result[f"{ch}_{key}"] = val

    return result
