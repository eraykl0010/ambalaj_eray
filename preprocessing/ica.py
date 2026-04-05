"""
Aşama 3 — ICA ile Artifact Azaltma

Akış:
    1. Filtrelenmiş EEG → ICA fit (FastICA)
    2. Otomatik artifact bileşen tespiti:
       a) EOG korelasyonu (Fp1, Fp2, Fpz ile yüksek korelasyon → göz artefaktı)
       b) Kurtosis eşiği (aşırı sivri dağılım → kas/hareket)
       c) Varyans eşiği (aşırı yüksek varyans → gürültü)
    3. Artifact bileşenleri çıkar → temiz sinyal rekonstrüksiyonu
    4. QC grafikleri

Tüm parametreler config.yaml → preprocessing.ica altından okunur.

Kullanım:
    python -m preprocessing.ica                   # standalone test
    from preprocessing.ica import apply_ica
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import pearsonr
from sklearn.decomposition import FastICA

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import load_config, get_active_channels


# ============================================================================
#  10-20 Elektrot Koordinatları (2D topomap için)
# ============================================================================

# Standart 10-20 sistemi — normalize 2D koordinatlar (burun yukarı)
# Kaynak: standart topomap projeksiyonu
ELECTRODE_POS_2D: Dict[str, Tuple[float, float]] = {
    "Fp1":  (-0.139, 0.451),  "Fpz":  (0.000, 0.475),  "Fp2":  (0.139, 0.451),
    "AF3":  (-0.173, 0.380),  "AF4":  (0.173, 0.380),
    "F7":   (-0.383, 0.261),  "F3":   (-0.222, 0.271),  "Fz":   (0.000, 0.285),
    "F4":   (0.222, 0.271),   "F8":   (0.383, 0.261),
    "FC5":  (-0.339, 0.137),  "FC1":  (-0.134, 0.143),
    "FC2":  (0.134, 0.143),   "FC6":  (0.339, 0.137),
    "T7":   (-0.450, 0.000),  "C3":   (-0.248, 0.000),  "Cz":   (0.000, 0.000),
    "C4":   (0.248, 0.000),   "T8":   (0.450, 0.000),
    "CP5":  (-0.339, -0.137), "CP1":  (-0.134, -0.143),
    "CP2":  (0.134, -0.143),  "CP6":  (0.339, -0.137),
    "P7":   (-0.383, -0.261), "P3":   (-0.222, -0.271), "Pz":   (0.000, -0.285),
    "P4":   (0.222, -0.271),  "P8":   (0.383, -0.261),
    "POz":  (0.000, -0.380),
    "O1":   (-0.139, -0.451), "Oz":   (0.000, -0.475),  "O2":   (0.139, -0.451),
}


def get_channel_positions(ch_names: List[str]) -> np.ndarray:
    """
    Kanal isimlerinden 2D koordinat dizisi döndür.

    Returns:
        positions: (n_channels, 2) — [x, y] koordinatları
    """
    positions = []
    for ch in ch_names:
        if ch in ELECTRODE_POS_2D:
            positions.append(ELECTRODE_POS_2D[ch])
        else:
            print(f"  [UYARI] '{ch}' için 2D koordinat bulunamadı, (0,0) atanıyor.")
            positions.append((0.0, 0.0))
    return np.array(positions)


# ============================================================================
#  1. ICA Uygulama
# ============================================================================

def fit_ica(
    data: np.ndarray,
    n_components: Optional[int] = None,
    method: str = "fastica",
    random_state: int = 42,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, FastICA]:
    """
    ICA fit.

    Args:
        data:          (n_samples, n_channels) — filtrelenmiş EEG
        n_components:  bileşen sayısı (None → kanal sayısı)
        method:        ICA yöntemi (şu an "fastica")
        random_state:  seed
        max_iter:      maksimum iterasyon

    Returns:
        sources:       (n_samples, n_components) — ICA kaynak sinyalleri
        mixing_matrix: (n_channels, n_components) — karıştırma matrisi (A)
        ica_model:     fit edilmiş FastICA nesnesi
    """
    n_samples, n_channels = data.shape
    if n_components is None:
        n_components = n_channels

    print(f"  [ICA] Fit başlıyor: {n_components} bileşen, method={method}, "
          f"max_iter={max_iter}")

    import warnings

    ica = FastICA(
        n_components=n_components,
        random_state=random_state,
        max_iter=max_iter,
        whiten="unit-variance",
        tol=1e-3,
    )

    # Fit & transform (convergence uyarısını yakala ama devam et)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sources = ica.fit_transform(data)           # (n_samples, n_components)
        mixing_matrix = ica.mixing_                 # (n_channels, n_components)

        if w and any("did not converge" in str(wi.message) for wi in w):
            print(f"  [ICA] UYARI: FastICA tam yakınsamadı. "
                  f"Kısa kayıtlarda bu normaldir; sonuçlar yine kullanılabilir.")

    print(f"  [ICA] Fit tamamlandı. Sources shape: {sources.shape}")

    return sources, mixing_matrix, ica


# ============================================================================
#  2. Artifact Bileşen Tespiti
# ============================================================================

def detect_eog_components(
    sources: np.ndarray,
    data: np.ndarray,
    ch_names: List[str],
    threshold: float = 0.3,
) -> List[int]:
    """
    EOG artifact bileşenlerini tespit et.

    Strateji: Fp1, Fp2, Fpz kanalları göz hareketlerine en yakın kanallardır.
    Her ICA bileşeninin bu kanallarla mutlak korelasyonunu hesapla.
    Max |r| > threshold olan bileşenler → EOG artifact.

    Returns:
        eog_indices: artifact olan bileşen indeksleri
    """
    # EOG referans kanalları
    eog_channels = ["Fp1", "Fp2", "Fpz"]
    eog_indices = [i for i, ch in enumerate(ch_names) if ch in eog_channels]

    if not eog_indices:
        print("  [EOG] Fp1/Fp2/Fpz kanalları bulunamadı — EOG tespiti atlanıyor.")
        return []

    n_components = sources.shape[1]
    eog_artifacts = []

    for comp_i in range(n_components):
        max_corr = 0.0
        for eog_i in eog_indices:
            r, _ = pearsonr(sources[:, comp_i], data[:, eog_i])
            max_corr = max(max_corr, abs(r))

        if max_corr > threshold:
            eog_artifacts.append(comp_i)

    print(f"  [EOG] {len(eog_artifacts)} bileşen tespit edildi "
          f"(eşik |r|>{threshold}): {eog_artifacts}")

    return eog_artifacts


def detect_kurtosis_components(
    sources: np.ndarray,
    threshold: float = 3.0,
) -> List[int]:
    """
    Yüksek kurtosis'li bileşenleri tespit et.

    Kas artefaktları ve ani spike'lar yüksek kurtosis üretir.
    Z-score > threshold olan bileşenler artifact olarak işaretlenir.
    """
    n_components = sources.shape[1]
    kurt_values = np.array([sp_kurtosis(sources[:, i], fisher=True)
                            for i in range(n_components)])

    # Z-score hesapla
    kurt_mean = np.mean(kurt_values)
    kurt_std = np.std(kurt_values)

    if kurt_std < 1e-10:
        return []

    kurt_z = (kurt_values - kurt_mean) / kurt_std

    artifacts = [i for i in range(n_components) if abs(kurt_z[i]) > threshold]

    print(f"  [KURTOSIS] {len(artifacts)} bileşen tespit edildi "
          f"(z>{threshold}): {artifacts}")

    return artifacts


def detect_variance_components(
    sources: np.ndarray,
    threshold_std: float = 3.0,
) -> List[int]:
    """
    Aşırı yüksek varyanslı bileşenleri tespit et.

    Varyansı ortalamadan threshold_std standart sapma uzakta olan
    bileşenler artifact olarak işaretlenir.
    """
    n_components = sources.shape[1]
    variances = np.var(sources, axis=0)

    var_mean = np.mean(variances)
    var_std = np.std(variances)

    if var_std < 1e-10:
        return []

    var_z = (variances - var_mean) / var_std

    artifacts = [i for i in range(n_components) if var_z[i] > threshold_std]

    print(f"  [VARYANS] {len(artifacts)} bileşen tespit edildi "
          f"(z>{threshold_std}): {artifacts}")

    return artifacts


def detect_artifacts(
    sources: np.ndarray,
    data: np.ndarray,
    ch_names: List[str],
    cfg: dict,
) -> List[int]:
    """
    Tüm artifact tespit yöntemlerini birleştir.

    Returns:
        artifact_indices: çıkarılacak bileşenlerin indeks listesi (sıralı, tekrarsız)
    """
    art_cfg = cfg["preprocessing"]["ica"]["artifact_detection"]

    all_artifacts = set()

    # 1. EOG korelasyonu
    eog = detect_eog_components(
        sources, data, ch_names,
        threshold=art_cfg["eog_threshold"],
    )
    all_artifacts.update(eog)

    # 2. Kurtosis
    kurt = detect_kurtosis_components(
        sources,
        threshold=art_cfg["kurtosis_threshold"],
    )
    all_artifacts.update(kurt)

    # 3. Varyans
    var = detect_variance_components(
        sources,
        threshold_std=art_cfg["variance_threshold_std"],
    )
    all_artifacts.update(var)

    artifact_list = sorted(all_artifacts)

    # Güvenlik: toplam bileşenin yarısından fazlasını çıkarma
    max_remove = sources.shape[1] // 2
    if len(artifact_list) > max_remove:
        print(f"  [UYARI] {len(artifact_list)} artifact tespit edildi ama "
              f"en fazla {max_remove} çıkarılabilir. En belirginler seçiliyor.")
        # Varyans büyüklüğüne göre sırala, en yüksek varyansları çıkar
        variances = np.var(sources, axis=0)
        artifact_list = sorted(artifact_list, key=lambda i: variances[i], reverse=True)
        artifact_list = artifact_list[:max_remove]

    print(f"\n  [ARTIFACT] Toplam çıkarılacak bileşen: {len(artifact_list)} / "
          f"{sources.shape[1]} → {artifact_list}")

    return artifact_list


# ============================================================================
#  3. Rekonstrüksiyon
# ============================================================================

def reconstruct_signal(
    sources: np.ndarray,
    mixing_matrix: np.ndarray,
    ica_model: FastICA,
    artifact_indices: List[int],
    original_mean: np.ndarray,
) -> np.ndarray:
    """
    Artifact bileşenlerini sıfırlayıp sinyali yeniden oluştur.

    Args:
        sources:           (n_samples, n_components)
        mixing_matrix:     (n_channels, n_components)
        ica_model:         fit edilmiş ICA modeli
        artifact_indices:  çıkarılacak bileşen indeksleri
        original_mean:     orijinal verinin kanal bazlı ortalaması

    Returns:
        cleaned: (n_samples, n_channels) — temizlenmiş sinyal
    """
    # Artifact bileşenlerini sıfırla
    sources_clean = sources.copy()
    for idx in artifact_indices:
        sources_clean[:, idx] = 0.0

    # Yeniden oluştur: X_clean = S_clean @ A.T + mean
    cleaned = sources_clean @ mixing_matrix.T + original_mean

    print(f"  [RECON] Sinyal yeniden oluşturuldu. "
          f"Shape: {cleaned.shape}, range=[{cleaned.min():.4f}, {cleaned.max():.4f}]")

    return cleaned


# ============================================================================
#  4. Ana ICA Pipeline
# ============================================================================

def apply_ica(
    data: np.ndarray,
    ch_names: List[str],
    cfg: dict,
) -> Tuple[np.ndarray, Dict]:
    """
    Tam ICA pipeline: fit → tespit → çıkarma → rekonstrüksiyon.

    Args:
        data:      (n_samples, n_channels) — filtrelenmiş EEG
        ch_names:  kanal isimleri
        cfg:       config sözlüğü

    Returns:
        cleaned:   (n_samples, n_channels) — ICA-temizlenmiş sinyal
        ica_info:  dict — ICA detayları (debug/QC için):
            sources, mixing_matrix, artifact_indices, ica_model
    """
    ica_cfg = cfg["preprocessing"]["ica"]

    if not ica_cfg["enabled"]:
        print("  [ICA] Devre dışı (config). Veri değiştirilmeden döndürülüyor.")
        return data, {}

    # Orijinal ortalamaları kaydet (DC bileşen)
    original_mean = np.mean(data, axis=0, keepdims=True)  # (1, n_channels)

    # 1. ICA Fit
    sources, mixing_matrix, ica_model = fit_ica(
        data,
        n_components=ica_cfg.get("n_components"),
        method=ica_cfg.get("method", "fastica"),
        random_state=ica_cfg.get("random_state", 42),
        max_iter=ica_cfg.get("max_iter", 1000),
    )

    # 2. Artifact Tespiti
    artifact_indices = detect_artifacts(sources, data, ch_names, cfg)

    # 3. Rekonstrüksiyon
    if len(artifact_indices) > 0:
        cleaned = reconstruct_signal(
            sources, mixing_matrix, ica_model,
            artifact_indices, original_mean,
        )
    else:
        print("  [ICA] Artifact bileşeni tespit edilmedi — veri değiştirilmedi.")
        cleaned = data.copy()

    # NaN/inf kontrolü
    nan_count = np.isnan(cleaned).sum()
    inf_count = np.isinf(cleaned).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  [UYARI] ICA sonrası: {nan_count} NaN, {inf_count} inf → temizleniyor")
        cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)

    ica_info = {
        "sources": sources,
        "mixing_matrix": mixing_matrix,
        "ica_model": ica_model,
        "artifact_indices": artifact_indices,
        "n_components": sources.shape[1],
    }

    return cleaned, ica_info


# ============================================================================
#  5. QC Grafikleri
# ============================================================================

def plot_ica_components_timeseries(
    sources: np.ndarray,
    sr: int,
    artifact_indices: List[int],
    title: str = "ICA Bileşenleri",
    save_path: Optional[str | Path] = None,
    max_components: int = 20,
    duration_sec: float = 10.0,
):
    """ICA bileşenlerinin zaman serileri. Artifact olanlar kırmızı."""
    n_samples, n_comp = sources.shape
    n_show_comp = min(n_comp, max_components)
    n_show_samples = min(int(duration_sec * sr), n_samples)
    time = np.arange(n_show_samples) / sr

    fig, axes = plt.subplots(n_show_comp, 1,
                             figsize=(16, n_show_comp * 0.8), sharex=True)
    if n_show_comp == 1:
        axes = [axes]

    for i in range(n_show_comp):
        color = "indianred" if i in artifact_indices else "steelblue"
        alpha = 1.0 if i in artifact_indices else 0.7
        label = f"IC{i}" + (" ★ ARTIFACT" if i in artifact_indices else "")

        axes[i].plot(time, sources[:n_show_samples, i],
                     color=color, linewidth=0.4, alpha=alpha)
        axes[i].set_ylabel(f"IC{i}", fontsize=7, fontweight="bold",
                           color="red" if i in artifact_indices else "black")
        axes[i].tick_params(axis="y", labelsize=6)
        axes[i].set_xlim(time[0], time[-1])
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)

        if i in artifact_indices:
            axes[i].set_facecolor("#fff0f0")

    axes[-1].set_xlabel("Zaman (s)", fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def plot_ica_topomaps(
    mixing_matrix: np.ndarray,
    ch_names: List[str],
    artifact_indices: List[int],
    title: str = "ICA Bileşen Topomap'leri",
    save_path: Optional[str | Path] = None,
    max_components: int = 20,
):
    """
    Her ICA bileşeninin topomap'ini çiz.

    Mixing matrix'in her sütunu bir bileşenin kanal ağırlıklarını verir.
    Bu ağırlıklar 2D elektrot konumlarına göre interpolasyonla gösterilir.
    """
    n_channels, n_comp = mixing_matrix.shape
    n_show = min(n_comp, max_components)

    positions = get_channel_positions(ch_names)

    # Grid düzeni
    n_cols = 6
    n_rows = int(np.ceil(n_show / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_rows * n_cols):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        if i >= n_show:
            ax.axis("off")
            continue

        weights = mixing_matrix[:, i]
        is_artifact = i in artifact_indices

        # Basit topomap: scatter + renk kodlama
        _draw_simple_topomap(ax, positions, weights, ch_names)

        label = f"IC{i}"
        if is_artifact:
            label += " ★"
            ax.set_title(label, fontsize=9, fontweight="bold", color="red")
            # Kırmızı çerçeve
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)
        else:
            ax.set_title(label, fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def _draw_simple_topomap(ax, positions, weights, ch_names):
    """
    Basit topomap çizimi: elektrot konumlarına göre ağırlık haritası.
    Interpolasyon ile grid oluşturup imshow ile gösterir.
    """
    from scipy.interpolate import griddata

    x = positions[:, 0]
    y = positions[:, 1]

    # Grid oluştur
    margin = 0.05
    xi = np.linspace(x.min() - margin, x.max() + margin, 100)
    yi = np.linspace(y.min() - margin, y.max() + margin, 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolasyon
    Zi = griddata((x, y), weights, (Xi, Yi), method="cubic")

    # Dairesel maske (kafa şekli)
    center_x, center_y = np.mean(x), np.mean(y)
    radius = max(np.max(np.abs(x - center_x)), np.max(np.abs(y - center_y))) * 1.15
    mask = (Xi - center_x)**2 + (Yi - center_y)**2 > radius**2
    if Zi is not None:
        Zi[mask] = np.nan

    # Simetrik renk skalası
    vmax = np.nanmax(np.abs(weights)) if len(weights) > 0 else 1.0
    vmin = -vmax

    ax.imshow(Zi, extent=[xi[0], xi[-1], yi[0], yi[-1]],
              origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax,
              aspect="equal", interpolation="bilinear")

    # Elektrot noktaları
    ax.scatter(x, y, c="black", s=8, zorder=5)

    # Burun işareti
    nose_x = [center_x - 0.03, center_x, center_x + 0.03]
    nose_y_val = y.max() + margin + 0.02
    nose_y = [y.max() + margin * 0.3, nose_y_val, y.max() + margin * 0.3]
    ax.plot(nose_x, nose_y, "k-", linewidth=1)

    ax.set_xlim(xi[0] - 0.02, xi[-1] + 0.02)
    ax.set_ylim(yi[0] - 0.02, nose_y_val + 0.02)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_ica_before_after(
    before: np.ndarray,
    after: np.ndarray,
    ch_names: List[str],
    sr: int,
    title: str = "ICA Öncesi vs Sonrası",
    save_path: Optional[str | Path] = None,
    channels_to_show: Optional[List[str]] = None,
    duration_sec: float = 5.0,
):
    """ICA öncesi/sonrası sinyal karşılaştırma (lineer eksen)."""
    if channels_to_show is None:
        default_ch = ["Fp1", "Fp2", "F3", "C3", "P3", "O1"]
        channels_to_show = [ch for ch in default_ch if ch in ch_names]
        if len(channels_to_show) < 4:
            channels_to_show = ch_names[:6]

    n_show = len(channels_to_show)
    n_samples = min(int(duration_sec * sr), before.shape[0])
    time = np.arange(n_samples) / sr

    fig, axes = plt.subplots(n_show, 1, figsize=(16, n_show * 2.2), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ch_name in enumerate(channels_to_show):
        ch_idx = ch_names.index(ch_name)
        bef = before[:n_samples, ch_idx]
        aft = after[:n_samples, ch_idx]

        axes[i].plot(time, bef, color="lightcoral", alpha=0.6,
                     linewidth=0.6, label="Filtre sonrası")
        axes[i].plot(time, aft, color="steelblue", alpha=0.9,
                     linewidth=0.6, label="ICA sonrası")

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
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def plot_cleaned_all_channels(
    data: np.ndarray,
    ch_names: List[str],
    sr: int,
    title: str = "ICA-Temizlenmiş EEG — Tüm Kanallar",
    save_path: Optional[str | Path] = None,
    duration_sec: float = 10.0,
):
    """Temizlenmiş sinyalin tüm kanalları (lineer eksen)."""
    n_samples, n_channels = data.shape
    n_show = min(int(duration_sec * sr), n_samples)
    time = np.arange(n_show) / sr

    fig, ax = plt.subplots(figsize=(16, max(8, n_channels * 0.5)))

    centered = []
    for ch_i in range(n_channels):
        sig = data[:n_show, ch_i]
        centered.append(sig - np.mean(sig))

    max_range = max(np.ptp(s) for s in centered) if centered else 1.0
    spacing = max_range * 1.3 if max_range > 0 else 1.0

    yticks, ytick_labels = [], []
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
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def generate_ica_qc(
    filtered_data: np.ndarray,
    cleaned_data: np.ndarray,
    ch_names: List[str],
    sr: int,
    ica_info: Dict,
    label: str,
    figures_dir: str | Path = "reports/figures/stage3_ica",
):
    """Tek kayıt için tüm ICA QC grafiklerini üret."""
    figures_dir = Path(figures_dir)

    if not ica_info:
        print("  [QC] ICA info boş — grafikler atlanıyor.")
        return

    sources = ica_info["sources"]
    mixing_matrix = ica_info["mixing_matrix"]
    artifact_indices = ica_info["artifact_indices"]

    # 1. Bileşen zaman serileri
    plot_ica_components_timeseries(
        sources, sr, artifact_indices,
        title=f"ICA Bileşenleri — {label}",
        save_path=figures_dir / f"ica_components_{label}.png",
        max_components=min(20, sources.shape[1]),
    )

    # 2. Bileşen topomap'leri
    plot_ica_topomaps(
        mixing_matrix, ch_names, artifact_indices,
        title=f"ICA Topomap'ler — {label}",
        save_path=figures_dir / f"ica_topomaps_{label}.png",
        max_components=min(20, mixing_matrix.shape[1]),
    )

    # 3. Öncesi/sonrası karşılaştırma
    plot_ica_before_after(
        filtered_data, cleaned_data, ch_names, sr,
        title=f"ICA Öncesi vs Sonrası — {label}",
        save_path=figures_dir / f"ica_before_after_{label}.png",
    )

    # 4. Temizlenmiş tüm kanallar
    plot_cleaned_all_channels(
        cleaned_data, ch_names, sr,
        title=f"ICA-Temizlenmiş EEG — {label} (Tüm Kanallar)",
        save_path=figures_dir / f"ica_cleaned_all_ch_{label}.png",
    )


# ============================================================================
#  6. Standalone Test
# ============================================================================

if __name__ == "__main__":
    from data.loader import load_eeg
    from preprocessing.filters import apply_filters

    cfg = load_config()
    sr = cfg["eeg"]["sampling_rate"]

    print("=" * 60)
    print("  AŞAMA 3 — ICA (Artifact Azaltma)")
    print("=" * 60)

    test_path = Path("/mnt/user-data/uploads/s1_eeg.csv")
    if not test_path.exists():
        print("[!] Test dosyası bulunamadı.")
        sys.exit(1)

    # 1. Yükle
    raw, ch_names, _ = load_eeg(test_path, cfg, clean=True)
    print(f"  Ham: {raw.shape}")

    # 2. Filtrele
    filtered = apply_filters(raw, cfg)
    print(f"  Filtrelenmiş: {filtered.shape}")

    # 3. ICA
    cleaned, ica_info = apply_ica(filtered, ch_names, cfg)
    print(f"  ICA-temiz: {cleaned.shape}")

    # 4. QC
    generate_ica_qc(
        filtered, cleaned, ch_names, sr, ica_info,
        label="202524_test",
    )

    print("\nAşama 3 testi tamamlandı!")
