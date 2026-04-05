"""
Aşama 2 — Ön İşleme: Band-pass & Notch Filtreleme

Görevler:
    1. Band-pass filtre (varsayılan 0.5–45 Hz, Butterworth 3. derece)
    2. Notch filtre (varsayılan 50 Hz, Q=30)
    3. Filtre öncesi/sonrası karşılaştırma grafikleri (lineer eksen)
    4. PSD karşılaştırma grafikleri

Tüm parametreler config.yaml'dan okunur.

Kullanım:
    python -m preprocessing.filters          # standalone test
    from preprocessing.filters import apply_filters
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import load_config, get_active_channels


# ============================================================================
#  1. Filtre Tasarımı
# ============================================================================

def design_bandpass(low_freq: float, high_freq: float, sr: int, order: int = 3):
    """
    Butterworth band-pass filtre katsayılarını döndür.

    Args:
        low_freq:  Alt kesim frekansı (Hz)
        high_freq: Üst kesim frekansı (Hz)
        sr:        Örnekleme oranı (Hz)
        order:     Filtre derecesi (varsayılan 3)

    Returns:
        b, a: filtre katsayıları
    """
    nyquist = sr / 2.0

    # Güvenlik kontrolleri
    if low_freq <= 0:
        raise ValueError(f"low_freq pozitif olmalı, verilen: {low_freq}")
    if high_freq >= nyquist:
        raise ValueError(
            f"high_freq ({high_freq}) Nyquist frekansından ({nyquist}) "
            f"küçük olmalı."
        )
    if low_freq >= high_freq:
        raise ValueError(
            f"low_freq ({low_freq}) < high_freq ({high_freq}) olmalı."
        )

    low = low_freq / nyquist
    high = high_freq / nyquist

    b, a = butter(order, [low, high], btype="band")
    return b, a


def design_notch(freq: float, sr: int, quality_factor: float = 30.0):
    """
    IIR Notch filtre katsayılarını döndür.

    Args:
        freq:           Notch frekansı (Hz)
        sr:             Örnekleme oranı (Hz)
        quality_factor: Q faktörü (yüksek → dar bant)

    Returns:
        b, a: filtre katsayıları
    """
    b, a = iirnotch(freq, quality_factor, sr)
    return b, a


# ============================================================================
#  2. Filtre Uygulama
# ============================================================================

def apply_bandpass(
    data: np.ndarray,
    sr: int,
    low_freq: float = 0.5,
    high_freq: float = 45.0,
    order: int = 3,
) -> np.ndarray:
    """
    Band-pass filtre uygula (zero-phase, filtfilt).

    Args:
        data: (n_samples, n_channels) — ham EEG verisi
        sr:   örnekleme oranı

    Returns:
        filtered: (n_samples, n_channels) — filtrelenmiş veri
    """
    b, a = design_bandpass(low_freq, high_freq, sr, order)

    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        # padlen kontrolü: sinyal uzunluğu yeterli olmalı
        padlen = 3 * max(len(a), len(b))
        if data.shape[0] <= padlen:
            print(
                f"  [UYARI] CH{ch}: sinyal çok kısa ({data.shape[0]} örnek), "
                f"filtre padlen={padlen}. Filtre atlanıyor."
            )
            filtered[:, ch] = data[:, ch]
            continue
        filtered[:, ch] = filtfilt(b, a, data[:, ch])

    return filtered


def apply_notch(
    data: np.ndarray,
    sr: int,
    freq: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """
    Notch filtre uygula (zero-phase, filtfilt).

    Args:
        data: (n_samples, n_channels) — EEG verisi
        sr:   örnekleme oranı

    Returns:
        filtered: (n_samples, n_channels) — notch uygulanmış veri
    """
    b, a = design_notch(freq, sr, quality_factor)

    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        padlen = 3 * max(len(a), len(b))
        if data.shape[0] <= padlen:
            filtered[:, ch] = data[:, ch]
            continue
        filtered[:, ch] = filtfilt(b, a, data[:, ch])

    return filtered


def apply_filters(data: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Config'e göre sırasıyla tüm filtreleri uygula.

    Sıra: Notch → Band-pass
    (Notch önce uygulanır ki 50 Hz şebeke gürültüsü band-pass'ten önce temizlensin.)

    Args:
        data: (n_samples, n_channels) — ham EEG
        cfg:  config sözlüğü

    Returns:
        filtered: (n_samples, n_channels)
    """
    sr = cfg["eeg"]["sampling_rate"]
    filt_cfg = cfg["preprocessing"]["filters"]
    result = data.copy()

    # 1. Notch filtre
    if filt_cfg["notch"]["enabled"]:
        notch_freq = filt_cfg["notch"]["freq"]
        notch_q = filt_cfg["notch"]["quality_factor"]
        print(f"  [FILTRE] Notch: {notch_freq} Hz, Q={notch_q}")
        result = apply_notch(result, sr, notch_freq, notch_q)

    # 2. Band-pass filtre
    if filt_cfg["bandpass"]["enabled"]:
        bp = filt_cfg["bandpass"]
        print(f"  [FILTRE] Band-pass: {bp['low_freq']}–{bp['high_freq']} Hz, "
              f"derece={bp['order']}, tip={bp['filter_type']}")
        result = apply_bandpass(
            result, sr,
            low_freq=bp["low_freq"],
            high_freq=bp["high_freq"],
            order=bp["order"],
        )

    # NaN/inf son kontrol
    nan_count = np.isnan(result).sum()
    inf_count = np.isinf(result).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  [UYARI] Filtre sonrası: {nan_count} NaN, {inf_count} inf → temizleniyor")
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


# ============================================================================
#  3. QC Grafikleri
# ============================================================================

def plot_filter_comparison(
    raw: np.ndarray,
    filtered: np.ndarray,
    ch_names: List[str],
    sr: int,
    title: str = "Filtre Öncesi vs Sonrası",
    save_path: Optional[str | Path] = None,
    channels_to_show: Optional[List[str]] = None,
    duration_sec: float = 5.0,
):
    """
    Filtre öncesi/sonrası overlay grafiği (lineer eksen).

    Args:
        raw:              (n_samples, n_channels) — ham sinyal
        filtered:         (n_samples, n_channels) — filtrelenmiş sinyal
        ch_names:         kanal isimleri
        sr:               örnekleme oranı
        channels_to_show: gösterilecek kanal alt kümesi (None → ilk 6)
        duration_sec:     gösterilecek süre
    """
    if channels_to_show is None:
        # Varsayılan: farklı bölgelerden 6 kanal seç
        default_channels = ["Fp1", "F3", "C3", "P3", "O1", "T7"]
        channels_to_show = [ch for ch in default_channels if ch in ch_names]
        if len(channels_to_show) < 4:
            channels_to_show = ch_names[:6]

    n_show = len(channels_to_show)
    n_samples_show = min(int(duration_sec * sr), raw.shape[0])
    time = np.arange(n_samples_show) / sr

    fig, axes = plt.subplots(n_show, 1, figsize=(16, n_show * 2.2), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ch_name in enumerate(channels_to_show):
        ch_idx = ch_names.index(ch_name)

        raw_seg = raw[:n_samples_show, ch_idx]
        filt_seg = filtered[:n_samples_show, ch_idx]

        # DC offset çıkar (görsel karşılaştırma için)
        raw_centered = raw_seg - np.mean(raw_seg)
        filt_centered = filt_seg - np.mean(filt_seg)

        axes[i].plot(time, raw_centered, color="lightcoral", alpha=0.6,
                     linewidth=0.6, label="Ham")
        axes[i].plot(time, filt_centered, color="steelblue", alpha=0.9,
                     linewidth=0.6, label="Filtrelenmiş")
        axes[i].set_ylabel(ch_name, fontsize=10, fontweight="bold")
        axes[i].grid(alpha=0.2)
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

        if i == 0:
            axes[i].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Zaman (s)", fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def plot_psd_comparison(
    raw: np.ndarray,
    filtered: np.ndarray,
    ch_names: List[str],
    sr: int,
    title: str = "PSD Karşılaştırma",
    save_path: Optional[str | Path] = None,
    channels_to_show: Optional[List[str]] = None,
):
    """
    Filtre öncesi/sonrası PSD (Power Spectral Density) grafiği.
    Y ekseni lineer (µV²/Hz), X ekseni frekans (Hz).

    Args:
        raw:              ham sinyal
        filtered:         filtrelenmiş sinyal
        ch_names:         kanal adları
        sr:               örnekleme oranı
        channels_to_show: gösterilecek kanallar (None → ilk 6)
    """
    if channels_to_show is None:
        default_channels = ["Fp1", "F3", "C3", "P3", "O1", "T7"]
        channels_to_show = [ch for ch in default_channels if ch in ch_names]
        if len(channels_to_show) < 4:
            channels_to_show = ch_names[:6]

    n_show = len(channels_to_show)
    nperseg = min(sr * 2, raw.shape[0])  # 2 saniyelik pencere

    fig, axes = plt.subplots(n_show, 1, figsize=(14, n_show * 2.2), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ch_name in enumerate(channels_to_show):
        ch_idx = ch_names.index(ch_name)

        freqs_r, psd_r = welch(raw[:, ch_idx], fs=sr, nperseg=nperseg)
        freqs_f, psd_f = welch(filtered[:, ch_idx], fs=sr, nperseg=nperseg)

        # Frekans aralığını sınırla (0–60 Hz — notch etkisi görünsün)
        freq_mask_r = freqs_r <= 60
        freq_mask_f = freqs_f <= 60

        axes[i].plot(freqs_r[freq_mask_r], psd_r[freq_mask_r],
                     color="lightcoral", alpha=0.7, linewidth=1.0, label="Ham")
        axes[i].plot(freqs_f[freq_mask_f], psd_f[freq_mask_f],
                     color="steelblue", alpha=0.9, linewidth=1.0, label="Filtrelenmiş")

        axes[i].set_ylabel(f"{ch_name}\n(µV²/Hz)", fontsize=9)
        axes[i].grid(alpha=0.2)
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

        # 50 Hz notch çizgisi
        axes[i].axvline(50, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

        if i == 0:
            axes[i].legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Frekans (Hz)", fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def plot_filtered_all_channels(
    data: np.ndarray,
    ch_names: List[str],
    sr: int,
    title: str = "Filtrelenmiş EEG — Tüm Kanallar",
    save_path: Optional[str | Path] = None,
    duration_sec: float = 10.0,
):
    """
    Filtrelenmiş sinyalin tüm kanallarını tek grafikte göster (lineer eksen).
    """
    n_samples, n_channels = data.shape
    n_show = min(int(duration_sec * sr), n_samples)
    time = np.arange(n_show) / sr

    fig, ax = plt.subplots(figsize=(16, max(8, n_channels * 0.5)))

    # Kanal aralığını hesapla
    centered = []
    for ch_i in range(n_channels):
        sig = data[:n_show, ch_i]
        centered.append(sig - np.mean(sig))

    max_range = max(np.ptp(s) for s in centered) if centered else 1.0
    spacing = max_range * 1.3 if max_range > 0 else 1.0

    yticks = []
    ytick_labels = []

    for ch_i in range(n_channels):
        y_off = -ch_i * spacing
        ax.plot(time, centered[ch_i] + y_off,
                linewidth=0.5, color=plt.cm.tab20(ch_i % 20))
        yticks.append(y_off)
        ytick_labels.append(ch_names[ch_i])

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=7)
    ax.set_xlabel("Zaman (s)", fontsize=11)
    ax.set_ylabel("Kanallar", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(time[0], time[-1])
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def generate_filter_qc(
    raw: np.ndarray,
    filtered: np.ndarray,
    ch_names: List[str],
    sr: int,
    label: str,
    figures_dir: str | Path = "reports/figures/stage2_filtered",
):
    """
    Tek bir kayıt için tüm filtreleme QC grafiklerini üret.
    """
    figures_dir = Path(figures_dir)

    # 1. Filtre öncesi/sonrası overlay
    plot_filter_comparison(
        raw, filtered, ch_names, sr,
        title=f"Filtre Öncesi vs Sonrası — {label}",
        save_path=figures_dir / f"filter_comparison_{label}.png",
    )

    # 2. PSD karşılaştırma
    plot_psd_comparison(
        raw, filtered, ch_names, sr,
        title=f"PSD Karşılaştırma — {label}",
        save_path=figures_dir / f"psd_comparison_{label}.png",
    )

    # 3. Filtrelenmiş tüm kanallar
    plot_filtered_all_channels(
        filtered, ch_names, sr,
        title=f"Filtrelenmiş EEG — {label} (Tüm Kanallar)",
        save_path=figures_dir / f"filtered_all_ch_{label}.png",
    )


# ============================================================================
#  4. Standalone Test
# ============================================================================

if __name__ == "__main__":
    from data.loader import load_eeg

    cfg = load_config()
    sr = cfg["eeg"]["sampling_rate"]

    print("=" * 60)
    print("  AŞAMA 2 — Filtreleme (Band-pass + Notch)")
    print("=" * 60)

    # Örnek dosya
    test_path = Path("test_data/s1_eeg.csv")
    if not test_path.exists():
        # Yüklenen dosyayı dene
        test_path = Path("/mnt/user-data/uploads/s1_eeg.csv")

    if not test_path.exists():
        print("[!] Test EEG dosyası bulunamadı.")
        sys.exit(1)

    # Ham veri yükle
    print(f"\nHam veri yükleniyor: {test_path}")
    raw, ch_names, ch_indices = load_eeg(test_path, cfg, clean=True)
    print(f"  Shape: {raw.shape}, Kanallar: {len(ch_names)}")
    print(f"  Ham değer aralığı: [{raw.min():.2f}, {raw.max():.2f}]")

    # Filtreleme
    print("\nFiltreler uygulanıyor...")
    filtered = apply_filters(raw, cfg)
    print(f"  Filtrelenmiş değer aralığı: [{filtered.min():.4f}, {filtered.max():.4f}]")
    print(f"  Filtrelenmiş std aralığı: [{filtered.std(axis=0).min():.4f}, "
          f"{filtered.std(axis=0).max():.4f}]")

    # QC grafikleri
    print("\nQC grafikleri üretiliyor...")
    generate_filter_qc(
        raw, filtered, ch_names, sr,
        label="202524_test",
        figures_dir="reports/figures/stage2_filtered",
    )

    print("\nAşama 2 testi tamamlandı!")
