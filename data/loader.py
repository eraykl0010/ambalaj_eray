"""
Aşama 1 — Veri Yükleme & Keşif

Görevler:
    1. raw_data_root altındaki dizin yapısını otomatik tarar.
    2. Test_karton* ve Test_camsise* klasörlerini bulur.
    3. s1_eeg.csv veya eeg_eeg.csv dosyasını okur (header yok).
    4. Aktif kanalları seçer, NaN / inf temizler.
    5. Kayıt envanteri (DataFrame) üretir.
    6. QC grafikleri çizer.

Kullanım:
    python -m data.loader                       # standalone çalıştırma
    from data.loader import discover_recordings, load_eeg
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # GUI olmadan çalışabilsin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Proje kök dizinini path'e ekle
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import load_config, get_active_channels


# ============================================================================
#  1. Dizin Keşfi
# ============================================================================

def discover_recordings(cfg: dict) -> pd.DataFrame:
    """
    raw_data_root altındaki tüm uygun EEG kayıtlarını bul.

    Beklenen yapı:
        raw_data_root / 2025XX / SessionXX / Test_karton* veya Test_camsise*
            içinde s1_eeg.csv veya eeg_eeg.csv

    Returns:
        DataFrame — her satır bir kayıt:
            subject_id, session, condition, folder_name, eeg_path,
            info_json_path (varsa), n_samples, duration_sec, n_channels_csv
    """
    root = Path(cfg["paths"]["raw_data_root"])
    if not root.exists():
        raise FileNotFoundError(f"raw_data_root bulunamadı: {root}")

    eeg_file_names = cfg["experiment"]["eeg_file_names"]
    folder_prefixes = cfg["experiment"]["folder_prefixes"]  # dict: karton→prefix, camsise→prefix

    records: List[Dict] = []

    # 2025XX dizinlerini bul (regex: 2025 ile başlayan)
    subject_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and re.match(r"^2025\d+$", d.name)]
    )

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name  # ör. "202524"

        # SessionXX dizinlerini bul
        session_dirs = sorted(
            [d for d in subj_dir.iterdir() if d.is_dir() and re.match(r"^Session\d+$", d.name, re.IGNORECASE)]
        )

        for sess_dir in session_dirs:
            session = sess_dir.name  # ör. "Session01"

            # Test_karton* ve Test_camsise* klasörlerini bul
            for condition, prefix in folder_prefixes.items():
                test_dirs = sorted(
                    [d for d in sess_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)]
                )

                for test_dir in test_dirs:
                    # EEG dosyasını bul (öncelik sırasıyla)
                    eeg_path = None
                    for fname in eeg_file_names:
                        candidate = test_dir / fname
                        if candidate.exists():
                            eeg_path = candidate
                            break

                    if eeg_path is None:
                        # EEG dosyası yoksa bu kaydı atla
                        print(f"  [ATLA] EEG dosyası bulunamadı: {test_dir}")
                        continue

                    # info.json var mı?
                    info_path = test_dir / "info.json"
                    info_json_path = str(info_path) if info_path.exists() else None

                    # Hızlı CSV meta bilgisi (sadece satır ve sütun sayısı)
                    try:
                        # Sadece ilk satırı oku → sütun sayısı
                        first_line = eeg_path.open("r").readline().strip()
                        n_cols = len(first_line.split(","))

                        # Satır sayısı (hızlı yöntem)
                        n_samples = sum(1 for _ in eeg_path.open("r"))
                    except Exception as e:
                        print(f"  [HATA] CSV okunamadı: {eeg_path} → {e}")
                        continue

                    sr = cfg["eeg"]["sampling_rate"]
                    duration_sec = n_samples / sr

                    records.append({
                        "subject_id": subject_id,
                        "session": session,
                        "condition": condition,
                        "folder_name": test_dir.name,
                        "eeg_path": str(eeg_path),
                        "info_json_path": info_json_path,
                        "n_samples": n_samples,
                        "duration_sec": round(duration_sec, 2),
                        "n_channels_csv": n_cols,
                    })

    inventory = pd.DataFrame(records)
    print(f"\n{'='*60}")
    print(f"Toplam kayıt bulunan: {len(inventory)}")
    if len(inventory) > 0:
        print(f"  Denekler : {inventory['subject_id'].nunique()}")
        print(f"  Karton   : {(inventory['condition'] == 'karton').sum()}")
        print(f"  Cam      : {(inventory['condition'] == 'camsise').sum()}")
        print(f"  Süre aralığı: {inventory['duration_sec'].min():.1f}s – {inventory['duration_sec'].max():.1f}s")
    print(f"{'='*60}\n")

    return inventory


# ============================================================================
#  2. EEG Yükleme (Tek Kayıt)
# ============================================================================

def load_eeg(
    eeg_path: str | Path,
    cfg: dict,
    clean: bool = True,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Tek bir EEG CSV dosyasını oku.

    Args:
        eeg_path:  CSV dosya yolu
        cfg:       config sözlüğü
        clean:     True ise NaN/inf → interpolasyon ile temizle

    Returns:
        data:        np.ndarray, shape (n_samples, n_active_channels)
        ch_names:    List[str]   — aktif kanal isimleri
        ch_indices:  List[int]   — orijinal CSV sütun indeksleri
    """
    eeg_path = Path(eeg_path)
    ch_names, ch_indices = get_active_channels(cfg)

    # CSV'yi oku (header yok)
    raw = pd.read_csv(
        eeg_path,
        header=None,
        dtype=np.float64,
        na_values=["", " ", "nan", "NaN", "inf", "-inf"],
    )

    # Toplam sütun kontrolü
    total_cols = raw.shape[1]
    expected_cols = cfg["eeg"]["num_channels_total"]
    if total_cols < expected_cols:
        print(
            f"  [UYARI] CSV'de {total_cols} sütun var, "
            f"beklenen {expected_cols}. Mevcut sütunlar kullanılacak."
        )

    # Aktif kanalları seç (sınır kontrolü ile)
    valid_indices = [i for i in ch_indices if i < total_cols]
    valid_names = [ch_names[j] for j, i in enumerate(ch_indices) if i < total_cols]

    data = raw.iloc[:, valid_indices].values.copy()  # (n_samples, n_channels)

    # --- NaN / inf Temizleme ---
    if clean:
        data = _clean_signal(data, valid_names)

    return data, valid_names, valid_indices


def _clean_signal(data: np.ndarray, ch_names: List[str]) -> np.ndarray:
    """
    NaN ve inf değerleri temizle.

    Strateji:
        1. inf → NaN'a çevir.
        2. Kanal bazlı lineer interpolasyon.
        3. Hâlâ NaN kalan uçlar → ileri/geri doldurma.
        4. Tüm satırı NaN olan kanallar → 0 ile doldur + uyarı.
    """
    # inf → NaN
    inf_count = np.isinf(data).sum()
    if inf_count > 0:
        print(f"  [TEMİZLİK] {inf_count} inf değer NaN'a çevrildi.")
        data[np.isinf(data)] = np.nan

    nan_count = np.isnan(data).sum()
    if nan_count == 0:
        return data

    print(f"  [TEMİZLİK] {nan_count} NaN değer tespit edildi, interpolasyon uygulanıyor...")

    n_samples, n_channels = data.shape
    x = np.arange(n_samples)

    for ch_idx in range(n_channels):
        col = data[:, ch_idx]
        nan_mask = np.isnan(col)

        if nan_mask.all():
            # Tüm kanal NaN → 0 ile doldur
            print(f"    [UYARI] {ch_names[ch_idx]} kanalının tamamı NaN! Sıfırla dolduruldu.")
            data[:, ch_idx] = 0.0
            continue

        if nan_mask.any():
            # Lineer interpolasyon
            valid = ~nan_mask
            data[nan_mask, ch_idx] = np.interp(
                x[nan_mask], x[valid], col[valid]
            )

    # Son kontrol
    remaining = np.isnan(data).sum()
    if remaining > 0:
        print(f"    [UYARI] Interpolasyon sonrası {remaining} NaN kaldı → 0 ile dolduruldu.")
        data[np.where(np.isnan(data))] = 0.0

    return data


# ============================================================================
#  3. Toplu Yükleme (Tüm Envanter)
# ============================================================================

def load_all_recordings(
    inventory: pd.DataFrame,
    cfg: dict,
) -> Dict[str, Dict]:
    """
    Envanterdeki tüm kayıtları yükle.

    Returns:
        dict → key: "{subject_id}_{session}_{condition}_{folder_name}"
               value: {
                   "data": np.ndarray (n_samples, n_channels),
                   "ch_names": List[str],
                   "ch_indices": List[int],
                   "meta": dict (subject_id, session, condition, ...),
               }
    """
    all_data = {}

    for idx, row in inventory.iterrows():
        key = f"{row['subject_id']}_{row['session']}_{row['condition']}_{row['folder_name']}"
        print(f"[{idx+1}/{len(inventory)}] Yükleniyor: {key}")

        try:
            data, ch_names, ch_indices = load_eeg(row["eeg_path"], cfg, clean=True)
            all_data[key] = {
                "data": data,
                "ch_names": ch_names,
                "ch_indices": ch_indices,
                "meta": row.to_dict(),
            }
        except Exception as e:
            print(f"  [HATA] Yükleme başarısız: {e}")

    print(f"\nBaşarıyla yüklenen: {len(all_data)} / {len(inventory)} kayıt")
    return all_data


# ============================================================================
#  4. QC Grafikleri
# ============================================================================

def plot_raw_eeg_sample(
    data: np.ndarray,
    ch_names: List[str],
    sr: int,
    title: str = "Ham EEG Sinyali",
    save_path: Optional[str | Path] = None,
    max_channels: int = 16,
    duration_sec: Optional[float] = 10.0,
):
    """
    Ham EEG sinyalini lineer düzlemde çiz.

    Args:
        data:          (n_samples, n_channels)
        ch_names:      kanal isimleri
        sr:            örnekleme oranı
        title:         grafik başlığı
        save_path:     kaydedilecek dosya yolu (None → sadece göster)
        max_channels:  gösterilecek maksimum kanal sayısı
        duration_sec:  gösterilecek süre (None → tümü)
    """
    n_samples, n_channels = data.shape
    n_show = min(n_channels, max_channels)

    if duration_sec is not None:
        n_show_samples = min(int(duration_sec * sr), n_samples)
    else:
        n_show_samples = n_samples

    time_axis = np.arange(n_show_samples) / sr  # saniye

    # Kanallar arası offset hesapla (okumayı kolaylaştırmak için)
    fig, ax = plt.subplots(figsize=(16, max(8, n_show * 0.5)))

    offsets = []
    for ch_i in range(n_show):
        signal = data[:n_show_samples, ch_i]
        # DC offset'i kaldır (görsel amaçlı)
        signal_centered = signal - np.mean(signal)
        offsets.append(signal_centered)

    # Kanal aralığını belirle
    if len(offsets) > 0:
        max_range = max(np.ptp(s) for s in offsets)
        spacing = max_range * 1.3 if max_range > 0 else 1.0
    else:
        spacing = 1.0

    yticks = []
    ytick_labels = []

    for ch_i in range(n_show):
        y_offset = -ch_i * spacing
        ax.plot(
            time_axis,
            offsets[ch_i] + y_offset,
            linewidth=0.5,
            color=plt.cm.tab20(ch_i % 20),
        )
        yticks.append(y_offset)
        ytick_labels.append(ch_names[ch_i])

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.set_xlabel("Zaman (s)", fontsize=11)
    ax.set_ylabel("Kanallar", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(time_axis[0], time_axis[-1])
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


def plot_channel_stats(
    data: np.ndarray,
    ch_names: List[str],
    title: str = "Kanal İstatistikleri",
    save_path: Optional[str | Path] = None,
):
    """
    Kanal bazlı ortalama ve standart sapma bar grafikleri.
    """
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    n_ch = len(ch_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_ch * 0.25)))

    # Ortalama
    axes[0].barh(range(n_ch), means, color="steelblue", alpha=0.8)
    axes[0].set_yticks(range(n_ch))
    axes[0].set_yticklabels(ch_names, fontsize=7)
    axes[0].set_xlabel("Ortalama (µV)")
    axes[0].set_title("Kanal Ortalamaları")
    axes[0].invert_yaxis()

    # Standart sapma
    axes[1].barh(range(n_ch), stds, color="coral", alpha=0.8)
    axes[1].set_yticks(range(n_ch))
    axes[1].set_yticklabels(ch_names, fontsize=7)
    axes[1].set_xlabel("Std Sapma (µV)")
    axes[1].set_title("Kanal Standart Sapmaları")
    axes[1].invert_yaxis()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def generate_qc_report(
    inventory: pd.DataFrame,
    cfg: dict,
    figures_dir: str | Path = "reports/figures/stage1_raw",
    max_subjects_to_plot: int = 5,
):
    """
    Keşif aşaması QC raporu: her koşuldan birkaç örnek denek grafiği.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sr = cfg["eeg"]["sampling_rate"]
    plotted = 0

    for _, row in inventory.iterrows():
        if plotted >= max_subjects_to_plot:
            break

        try:
            data, ch_names, _ = load_eeg(row["eeg_path"], cfg, clean=True)
        except Exception as e:
            print(f"  [HATA] QC grafik oluşturulamadı: {row['eeg_path']} → {e}")
            continue

        label = f"{row['subject_id']}_{row['session']}_{row['condition']}"

        # Ham sinyal grafiği (ilk 10 sn)
        plot_raw_eeg_sample(
            data, ch_names, sr,
            title=f"Ham EEG — {label}",
            save_path=figures_dir / f"raw_eeg_{label}.png",
            max_channels=32,
            duration_sec=10.0,
        )

        # Kanal istatistikleri
        plot_channel_stats(
            data, ch_names,
            title=f"Kanal İstatistikleri — {label}",
            save_path=figures_dir / f"channel_stats_{label}.png",
        )

        plotted += 1

    print(f"\nQC raporu tamamlandı — {plotted} kayıt için grafikler: {figures_dir}")


# ============================================================================
#  5. Standalone Çalıştırma
# ============================================================================

if __name__ == "__main__":
    cfg = load_config()

    print("=" * 60)
    print("  AŞAMA 1 — Veri Yükleme & Keşif")
    print("=" * 60)

    # 1. Envanter keşfi
    inventory = discover_recordings(cfg)

    if len(inventory) == 0:
        print("[!] Hiç kayıt bulunamadı. raw_data_root yolunu kontrol edin.")
        sys.exit(1)

    # 2. Envanteri kaydet
    inv_path = Path(cfg["paths"]["output_dir"]) / "recording_inventory.csv"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(inv_path, index=False)
    print(f"Envanter kaydedildi: {inv_path}")

    # 3. QC grafikleri
    generate_qc_report(
        inventory, cfg,
        figures_dir=Path(cfg["paths"]["reports_dir"]) / "figures" / "stage1_raw",
        max_subjects_to_plot=5,
    )

    print("\nAşama 1 tamamlandı!")
