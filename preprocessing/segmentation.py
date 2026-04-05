"""
Aşama 4 — Segmentasyon

Görevler:
    1. ICA-temizlenmiş EEG'yi 4s pencereler halinde segmentlere ayır.
    2. %50 overlap (512 örnek kayma) uygula.
    3. Her segment: (1024, n_channels) matris + metadata.
    4. Segment'leri denek/koşul bazlı dosyalara kaydet (.npz).
    5. Segment envanteri (CSV) oluştur.
    6. QC grafikleri üret.

Parametreler config.yaml → segmentation bölümünden okunur.

Kullanım:
    python run.py --stage 4
    python run.py --stage 4 --file "path/to/s1_eeg.csv"
    from preprocessing.segmentation import segment_signal
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import load_config, get_segment_params


# ============================================================================
#  1. Segmentasyon
# ============================================================================

def segment_signal(
    data: np.ndarray,
    cfg: dict,
) -> List[np.ndarray]:
    """
    Sürekli EEG verisini sabit pencerelerle segmentlere ayır.

    Args:
        data:  (n_samples, n_channels) — temizlenmiş EEG
        cfg:   config sözlüğü

    Returns:
        segments: list of np.ndarray, her biri (window_samples, n_channels)
    """
    seg_params = get_segment_params(cfg)
    window_samples = seg_params["window_samples"]   # 1024
    step_samples = seg_params["step_samples"]       # 512
    sr = seg_params["sampling_rate"]                # 256
    min_dur = cfg["segmentation"].get("min_segment_duration", 2.0)
    min_samples = int(min_dur * sr)

    n_samples, n_channels = data.shape

    # Doğrulama
    assert window_samples == int(cfg["segmentation"]["window_seconds"] * sr), \
        f"Window hesaplama hatası: {window_samples} != " \
        f"{cfg['segmentation']['window_seconds']} * {sr}"

    segments = []
    start = 0

    while start + window_samples <= n_samples:
        seg = data[start : start + window_samples, :]  # (1024, n_channels)
        segments.append(seg)
        start += step_samples

    # Kayıt sonunda kalan kısım
    remaining = n_samples - start
    if remaining >= min_samples:
        # Kısa segment: zero-pad ile tam pencereye tamamla
        seg = np.zeros((window_samples, n_channels), dtype=data.dtype)
        seg[:remaining, :] = data[start : start + remaining, :]
        segments.append(seg)
        print(f"  [SEG] Son segment zero-padded: {remaining}/{window_samples} örnek")

    print(f"  [SEG] Toplam segment: {len(segments)} "
          f"(pencere={window_samples}, adım={step_samples}, "
          f"kayıt={n_samples} örnek = {n_samples/sr:.1f}s)")

    return segments


# ============================================================================
#  2. Segment Kaydetme
# ============================================================================

def save_segments(
    segments: List[np.ndarray],
    ch_names: List[str],
    meta: Dict,
    output_dir: str | Path = "outputs/cleaned_segments",
) -> Tuple[Path, pd.DataFrame]:
    """
    Segment'leri .npz dosyasına kaydet + metadata CSV oluştur.

    Dosya adı: {subject_id}_{session}_{condition}.npz
    İçerik:
        - segments: (n_segments, window_samples, n_channels) array
        - ch_names: kanal isimleri
        - subject_id, session, condition

    Returns:
        npz_path:       kaydedilen dosya yolu
        segment_meta:   segment metadata DataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_id = meta.get("subject_id", "unknown")
    session = meta.get("session", "unknown")
    condition = meta.get("condition", "unknown")

    filename = f"{subject_id}_{session}_{condition}"
    npz_path = output_dir / f"{filename}.npz"

    # Segment'leri 3D array'e çevir
    segments_array = np.stack(segments, axis=0)  # (n_seg, 1024, n_ch)

    np.savez_compressed(
        npz_path,
        segments=segments_array,
        ch_names=np.array(ch_names),
        subject_id=subject_id,
        session=session,
        condition=condition,
    )

    print(f"  [KAYIT] {npz_path} → shape={segments_array.shape}")

    # Segment metadata
    rows = []
    for seg_i in range(len(segments)):
        rows.append({
            "subject_id": subject_id,
            "session": session,
            "condition": condition,
            "segment_id": seg_i,
            "segment_file": str(npz_path),
            "n_samples": segments[seg_i].shape[0],
            "n_channels": segments[seg_i].shape[1],
        })

    segment_meta = pd.DataFrame(rows)
    return npz_path, segment_meta


def load_segments(npz_path: str | Path) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Kaydedilmiş segment dosyasını yükle.

    Returns:
        segments:  (n_segments, window_samples, n_channels)
        ch_names:  kanal isimleri
        meta:      dict (subject_id, session, condition)
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    segments = data["segments"]
    ch_names = list(data["ch_names"])
    meta = {
        "subject_id": str(data["subject_id"]),
        "session": str(data["session"]),
        "condition": str(data["condition"]),
    }

    return segments, ch_names, meta


# ============================================================================
#  3. QC Grafikleri
# ============================================================================

def plot_segmentation_overview(
    data: np.ndarray,
    segments: List[np.ndarray],
    ch_names: List[str],
    sr: int,
    cfg: dict,
    title: str = "Segmentasyon Görselleştirme",
    save_path: Optional[str | Path] = None,
    channel_idx: int = 0,
):
    """
    Tek bir kanal üzerinde segment sınırlarını göster.

    Args:
        data:       (n_samples, n_channels) — tam kayıt
        segments:   segment listesi
        ch_names:   kanal isimleri
        sr:         örnekleme oranı
        cfg:        config
        channel_idx: gösterilecek kanal indeksi
    """
    seg_params = get_segment_params(cfg)
    window = seg_params["window_samples"]
    step = seg_params["step_samples"]

    n_samples = data.shape[0]
    time = np.arange(n_samples) / sr
    signal = data[:, channel_idx] - np.mean(data[:, channel_idx])

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.plot(time, signal, color="steelblue", linewidth=0.4, alpha=0.8)

    # Segment sınırlarını çiz
    colors = plt.cm.Set3(np.linspace(0, 1, min(len(segments), 12)))
    for seg_i in range(len(segments)):
        start_sample = seg_i * step
        end_sample = start_sample + window

        if end_sample > n_samples:
            end_sample = n_samples

        color = colors[seg_i % len(colors)]

        # Üstte segment kutusu
        ax.axvspan(
            start_sample / sr, end_sample / sr,
            alpha=0.15, color=color, zorder=0,
        )

        # Segment numarası
        mid_time = (start_sample + end_sample) / 2 / sr
        y_top = ax.get_ylim()[1] * 0.85
        if seg_i % 2 == 0:  # Her 2. segmentte numara (kalabalık olmasın)
            ax.text(mid_time, y_top, f"S{seg_i}", fontsize=6,
                    ha="center", va="top", color="gray", alpha=0.7)

    ax.set_xlabel("Zaman (s)", fontsize=11)
    ax.set_ylabel(f"{ch_names[channel_idx]} (µV)", fontsize=11)
    ax.set_title(
        f"{title}\n"
        f"Pencere={window/sr:.1f}s ({window} örnek), "
        f"Overlap=%{cfg['segmentation']['overlap_ratio']*100:.0f}, "
        f"Toplam={len(segments)} segment",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0, n_samples / sr)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def plot_segment_samples(
    segments: List[np.ndarray],
    ch_names: List[str],
    sr: int,
    title: str = "Örnek Segmentler",
    save_path: Optional[str | Path] = None,
    n_samples_show: int = 4,
    channels_to_show: Optional[List[str]] = None,
):
    """
    Rastgele birkaç segmentin çoklu kanal görünümü.
    """
    if channels_to_show is None:
        default_ch = ["Fp1", "F3", "C3", "P3", "O1", "T7"]
        channels_to_show = [ch for ch in default_ch if ch in ch_names]
        if len(channels_to_show) < 3:
            channels_to_show = ch_names[:4]

    n_seg = min(n_samples_show, len(segments))
    n_ch = len(channels_to_show)

    # Eşit aralıklı segment seç
    indices = np.linspace(0, len(segments) - 1, n_seg, dtype=int)

    fig, axes = plt.subplots(n_ch, n_seg, figsize=(4 * n_seg, n_ch * 1.8),
                             sharex=True, sharey="row")
    if n_ch == 1:
        axes = axes.reshape(1, -1)
    if n_seg == 1:
        axes = axes.reshape(-1, 1)

    window_samples = segments[0].shape[0]
    time = np.arange(window_samples) / sr

    for col, seg_idx in enumerate(indices):
        seg = segments[seg_idx]
        for row, ch_name in enumerate(channels_to_show):
            ch_idx = ch_names.index(ch_name)
            signal = seg[:, ch_idx]

            axes[row, col].plot(time, signal, color="steelblue",
                                linewidth=0.5, alpha=0.9)
            axes[row, col].grid(alpha=0.2)
            axes[row, col].spines["top"].set_visible(False)
            axes[row, col].spines["right"].set_visible(False)

            if col == 0:
                axes[row, col].set_ylabel(ch_name, fontsize=9, fontweight="bold")
            if row == 0:
                axes[row, col].set_title(f"Segment {seg_idx}", fontsize=9)
            if row == n_ch - 1:
                axes[row, col].set_xlabel("Zaman (s)", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def plot_segment_statistics(
    segments: List[np.ndarray],
    ch_names: List[str],
    title: str = "Segment İstatistikleri",
    save_path: Optional[str | Path] = None,
):
    """
    Segment bazlı istatistik dağılımları: ortalama, std, min, max.
    """
    n_seg = len(segments)

    seg_means = np.array([seg.mean() for seg in segments])
    seg_stds = np.array([seg.std() for seg in segments])
    seg_maxabs = np.array([np.abs(seg).max() for seg in segments])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(range(n_seg), seg_means, color="steelblue", alpha=0.7)
    axes[0].set_title("Segment Ortalamaları", fontsize=10)
    axes[0].set_xlabel("Segment")
    axes[0].set_ylabel("Ortalama")

    axes[1].bar(range(n_seg), seg_stds, color="coral", alpha=0.7)
    axes[1].set_title("Segment Std Sapmaları", fontsize=10)
    axes[1].set_xlabel("Segment")
    axes[1].set_ylabel("Std")

    axes[2].bar(range(n_seg), seg_maxabs, color="mediumpurple", alpha=0.7)
    axes[2].set_title("Segment Max |Amplitüd|", fontsize=10)
    axes[2].set_xlabel("Segment")
    axes[2].set_ylabel("|Max|")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def generate_segmentation_qc(
    data: np.ndarray,
    segments: List[np.ndarray],
    ch_names: List[str],
    sr: int,
    cfg: dict,
    label: str,
    figures_dir: str | Path = "reports/figures/stage4_segments",
):
    """Tek kayıt için tüm segmentasyon QC grafiklerini üret."""
    figures_dir = Path(figures_dir)

    # 1. Segment sınırları
    plot_segmentation_overview(
        data, segments, ch_names, sr, cfg,
        title=f"Segmentasyon — {label}",
        save_path=figures_dir / f"seg_overview_{label}.png",
    )

    # 2. Örnek segmentler
    plot_segment_samples(
        segments, ch_names, sr,
        title=f"Örnek Segmentler — {label}",
        save_path=figures_dir / f"seg_samples_{label}.png",
    )

    # 3. Segment istatistikleri
    plot_segment_statistics(
        segments, ch_names,
        title=f"Segment İstatistikleri — {label}",
        save_path=figures_dir / f"seg_stats_{label}.png",
    )


# ============================================================================
#  4. Standalone Test
# ============================================================================

if __name__ == "__main__":
    from data.loader import load_eeg
    from preprocessing.filters import apply_filters
    from preprocessing.ica import apply_ica

    cfg = load_config()
    sr = cfg["eeg"]["sampling_rate"]

    print("=" * 60)
    print("  AŞAMA 4 — Segmentasyon")
    print("=" * 60)

    test_path = Path("/mnt/user-data/uploads/s1_eeg.csv")
    if not test_path.exists():
        print("[!] Test dosyası bulunamadı.")
        sys.exit(1)

    # 1-3. Yükle → Filtrele → ICA
    raw, ch_names, _ = load_eeg(test_path, cfg, clean=True)
    filtered = apply_filters(raw, cfg)
    cleaned, _ = apply_ica(filtered, ch_names, cfg)

    # 4. Segmentasyon
    segments = segment_signal(cleaned, cfg)
    print(f"  Segment shape: ({len(segments)}, {segments[0].shape})")

    # 5. Kaydet
    meta = {"subject_id": "202524", "session": "Session01", "condition": "test"}
    npz_path, seg_meta = save_segments(segments, ch_names, meta)
    print(f"  Segment metadata:\n{seg_meta.head()}")

    # 6. Geri yükle kontrolü
    segs_loaded, ch_loaded, meta_loaded = load_segments(npz_path)
    print(f"\n  Yeniden yükleme: shape={segs_loaded.shape}, "
          f"ch={len(ch_loaded)}, meta={meta_loaded}")

    # 7. QC
    generate_segmentation_qc(
        cleaned, segments, ch_names, sr, cfg,
        label="202524_test",
    )

    print("\nAşama 4 testi tamamlandı!")
