"""
Aşama 5 — Feature Çıkarımı Orchestrator

Tüm feature modüllerini birleştirerek her segment için
tek bir feature vektörü oluşturur.

Kullanım:
    python run.py --stage 5
    from features.extract_all import extract_all_features
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config_loader import load_config

from features.time_domain import extract_time_features
from features.frequency_domain import extract_frequency_features
from features.nonlinear import extract_nonlinear_features
from features.entropy import extract_entropy_features


# ============================================================================
#  1. Tek Segment Feature Çıkarımı
# ============================================================================

def extract_segment_features(
    segment: np.ndarray,
    ch_names: List[str],
    sr: int,
    cfg: dict,
) -> Dict[str, float]:
    """
    Tek segment (window_samples, n_channels) için tüm feature'ları çıkar.

    Returns:
        dict: tüm feature'lar birleşik sözlük
    """
    feat_cfg = cfg["features"]
    features = {}

    # 1. Zaman alanı
    if feat_cfg["time_domain"]["enabled"]:
        td = extract_time_features(
            segment, ch_names,
            feature_names=feat_cfg["time_domain"]["features"],
        )
        features.update(td)

    # 2. Frekans alanı
    if feat_cfg["frequency_domain"]["enabled"]:
        bands = {k: v for k, v in feat_cfg["frequency_domain"]["bands"].items()}
        fd = extract_frequency_features(segment, ch_names, sr, bands)
        features.update(fd)

    # 3. Nonlineer
    if feat_cfg["nonlinear"]["enabled"]:
        nl = extract_nonlinear_features(
            segment, ch_names,
            feature_names=feat_cfg["nonlinear"]["features"],
            higuchi_kmax=feat_cfg["nonlinear"].get("higuchi_kmax", 10),
        )
        features.update(nl)

    # 4. Entropi
    if feat_cfg["entropy"]["enabled"]:
        ent = extract_entropy_features(
            segment, ch_names,
            feature_names=feat_cfg["entropy"]["features"],
            embedding_dim=feat_cfg["entropy"]["embedding_dim"],
            tolerance_r=feat_cfg["entropy"]["tolerance_r"],
            pe_order=feat_cfg["entropy"]["pe_order"],
            pe_delay=feat_cfg["entropy"]["pe_delay"],
        )
        features.update(ent)

    return features


# ============================================================================
#  2. Paralel Yardımcı (top-level fonksiyon — pickle uyumlu)
# ============================================================================

def _extract_single_segment(args):
    """
    Tek segment için feature çıkarımı (ProcessPoolExecutor uyumlu).
    Top-level fonksiyon olmalı (pickle serialization).
    """
    seg_data, ch_names, sr, feat_cfg, seg_idx = args

    from features.time_domain import extract_time_features
    from features.frequency_domain import extract_frequency_features
    from features.nonlinear import extract_nonlinear_features
    from features.entropy import extract_entropy_features

    features = {}

    # 1. Zaman alanı
    if feat_cfg["time_domain"]["enabled"]:
        td = extract_time_features(
            seg_data, ch_names,
            feature_names=feat_cfg["time_domain"]["features"],
        )
        features.update(td)

    # 2. Frekans alanı
    if feat_cfg["frequency_domain"]["enabled"]:
        bands = {k: v for k, v in feat_cfg["frequency_domain"]["bands"].items()}
        fd = extract_frequency_features(seg_data, ch_names, sr, bands)
        features.update(fd)

    # 3. Nonlineer
    if feat_cfg["nonlinear"]["enabled"]:
        nl = extract_nonlinear_features(
            seg_data, ch_names,
            feature_names=feat_cfg["nonlinear"]["features"],
            higuchi_kmax=feat_cfg["nonlinear"].get("higuchi_kmax", 10),
        )
        features.update(nl)

    # 4. Entropi
    if feat_cfg["entropy"]["enabled"]:
        ent = extract_entropy_features(
            seg_data, ch_names,
            feature_names=feat_cfg["entropy"]["features"],
            embedding_dim=feat_cfg["entropy"]["embedding_dim"],
            tolerance_r=feat_cfg["entropy"]["tolerance_r"],
            pe_order=feat_cfg["entropy"]["pe_order"],
            pe_delay=feat_cfg["entropy"]["pe_delay"],
        )
        features.update(ent)

    features["_seg_idx"] = seg_idx
    return features


# ============================================================================
#  3. Toplu Feature Çıkarımı (Paralel + Seri)
# ============================================================================

def extract_all_features(
    segments: np.ndarray | List[np.ndarray],
    ch_names: List[str],
    sr: int,
    cfg: dict,
    meta: Optional[Dict] = None,
    verbose: bool = True,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """
    Tüm segmentler için feature çıkarımı.

    Args:
        segments:  (n_segments, window_samples, n_channels) veya list
        ch_names:  kanal isimleri
        sr:        örnekleme oranı
        cfg:       config
        meta:      ek metadata sütunları (subject_id, session, condition)
        verbose:   ilerleme bilgisi göster
        n_jobs:    paralel iş sayısı
                   None → config'den oku (general.n_jobs)
                   1    → seri (paralel yok)
                   -1   → tüm çekirdekler
                   N    → N çekirdek

    Returns:
        DataFrame: satır = segment, sütun = feature'lar + metadata
    """
    if isinstance(segments, np.ndarray) and segments.ndim == 3:
        seg_list = [segments[i] for i in range(segments.shape[0])]
    else:
        seg_list = list(segments)

    n_segments = len(seg_list)
    feat_cfg = cfg["features"]

    # n_jobs belirle
    if n_jobs is None:
        n_jobs = cfg.get("general", {}).get("n_jobs", -1)

    max_cores = multiprocessing.cpu_count()
    if n_jobs == -1:
        n_workers = max_cores
    elif n_jobs <= 0:
        n_workers = max(1, max_cores + n_jobs)
    else:
        n_workers = min(n_jobs, max_cores)

    # Çok az segment varsa paralel overhead'i gereksiz
    if n_segments <= 2:
        n_workers = 1

    t0 = time.time()

    if n_workers > 1:
        # ── PARALEL MOD ──
        if verbose:
            print(f"  [FEATURE] Paralel mod: {n_workers} çekirdek, "
                  f"{n_segments} segment")

        # İş argümanlarını hazırla
        work_args = [
            (seg_list[i], ch_names, sr, feat_cfg, i)
            for i in range(n_segments)
        ]

        results = [None] * n_segments
        completed = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(_extract_single_segment, args): args[4]
                for args in work_args
            }

            for future in as_completed(future_to_idx):
                seg_idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[result["_seg_idx"]] = result
                except Exception as e:
                    print(f"  [HATA] Segment {seg_idx}: {e}")
                    results[seg_idx] = {"_seg_idx": seg_idx}

                completed += 1
                if verbose and (completed % max(1, n_segments // 10) == 0
                                or completed == n_segments):
                    elapsed = time.time() - t0
                    eta = (elapsed / completed) * (n_segments - completed)
                    print(f"  [FEATURE] {completed}/{n_segments} tamamlandı "
                          f"({elapsed:.1f}s geçti, ~{eta:.0f}s kaldı)")

    else:
        # ── SERİ MOD ──
        if verbose:
            print(f"  [FEATURE] Seri mod, {n_segments} segment")

        results = []
        for i in range(n_segments):
            if verbose and (i % 5 == 0 or i == n_segments - 1):
                elapsed = time.time() - t0
                eta = (elapsed / (i + 1)) * (n_segments - i - 1) if i > 0 else 0
                print(f"  [FEATURE] Segment {i+1}/{n_segments} "
                      f"({elapsed:.1f}s geçti, ~{eta:.0f}s kaldı)")

            features = extract_segment_features(seg_list[i], ch_names, sr, cfg)
            features["_seg_idx"] = i
            results.append(features)

    # DataFrame oluştur
    all_rows = []
    for r in results:
        if r is None:
            continue
        row = {}
        if meta:
            row["subject_id"] = meta.get("subject_id", "")
            row["session"] = meta.get("session", "")
            row["condition"] = meta.get("condition", "")
        row["segment_id"] = r.pop("_seg_idx", 0)
        row.update(r)
        all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # Segment sırasını koru
    df = df.sort_values("segment_id").reset_index(drop=True)

    # NaN istatistikleri
    total_vals = df.select_dtypes(include=[np.number]).size
    nan_count = df.select_dtypes(include=[np.number]).isna().sum().sum()
    nan_pct = (nan_count / total_vals * 100) if total_vals > 0 else 0

    elapsed_total = time.time() - t0
    if verbose:
        print(f"\n  [FEATURE] Tamamlandı: {n_segments} segment × "
              f"{len(df.columns)} sütun, {elapsed_total:.1f}s "
              f"({elapsed_total/max(1,n_segments):.1f}s/seg)")
        if n_workers > 1:
            speedup = (elapsed_total * n_workers) / max(elapsed_total, 0.1)
            print(f"  [FEATURE] Çekirdek: {n_workers}, "
                  f"seri tahmini: ~{elapsed_total/max(1,n_segments)*n_segments*n_workers/n_workers:.0f}s")
        print(f"  [FEATURE] NaN: {nan_count}/{total_vals} ({nan_pct:.2f}%)")

    return df


# ============================================================================
#  3. Feature Kaydetme / Yükleme
# ============================================================================

def save_features(
    df: pd.DataFrame,
    output_path: str | Path,
    format: str = "parquet",
):
    """Feature DataFrame'i kaydet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        try:
            df.to_parquet(output_path, index=False)
        except Exception:
            # pyarrow yoksa CSV'ye geri dön
            csv_path = output_path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            print(f"  [UYARI] Parquet yazılamadı, CSV olarak kaydedildi: {csv_path}")
            return csv_path
    elif format == "csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Desteklenmeyen format: {format}")

    print(f"  [KAYIT] Feature dosyası: {output_path} ({len(df)} satır)")
    return output_path


# ============================================================================
#  4. QC Grafikleri
# ============================================================================

def plot_feature_distributions(
    df: pd.DataFrame,
    title: str = "Feature Dağılımları (Örnek)",
    save_path: Optional[str | Path] = None,
    max_features: int = 12,
):
    """Rastgele seçilmiş feature'ların histogram dağılımları."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Metadata sütunlarını çıkar
    feat_cols = [c for c in numeric_cols if c != "segment_id"]

    if len(feat_cols) == 0:
        print("  [QC] Sayısal feature yok — grafik atlanıyor.")
        return

    n_show = min(max_features, len(feat_cols))
    # Farklı kategorilerden örnekle
    sample_cols = []
    categories = ["hjorth", "mean", "rms", "alpha", "beta", "delta",
                   "katz", "hurst", "dfa", "apen", "pe", "se"]
    for cat in categories:
        matches = [c for c in feat_cols if cat in c.lower() and c not in sample_cols]
        if matches:
            sample_cols.append(matches[0])
        if len(sample_cols) >= n_show:
            break
    # Eksik kalırsa rastgele doldur
    remaining = [c for c in feat_cols if c not in sample_cols]
    while len(sample_cols) < n_show and remaining:
        sample_cols.append(remaining.pop(0))

    n_cols = 4
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(sample_cols):
        vals = df[col].dropna()
        if len(vals) > 0:
            axes[i].hist(vals, bins=min(20, len(vals)), color="steelblue",
                         alpha=0.7, edgecolor="white")
        axes[i].set_title(col, fontsize=7, fontweight="bold")
        axes[i].tick_params(labelsize=6)

    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def plot_nan_report(
    df: pd.DataFrame,
    title: str = "NaN Raporu",
    save_path: Optional[str | Path] = None,
):
    """Kanal/feature bazlı NaN oranlarını göster."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feat_cols = [c for c in numeric_cols if c != "segment_id"]

    nan_pcts = df[feat_cols].isna().mean() * 100
    nonzero = nan_pcts[nan_pcts > 0].sort_values(ascending=False)

    if len(nonzero) == 0:
        print("  [QC] Hiç NaN yok — rapor atlanıyor.")
        return

    n_show = min(30, len(nonzero))
    show = nonzero.head(n_show)

    fig, ax = plt.subplots(figsize=(10, max(4, n_show * 0.3)))
    ax.barh(range(len(show)), show.values, color="indianred", alpha=0.7)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show.index, fontsize=6)
    ax.set_xlabel("NaN Oranı (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def generate_feature_qc(
    df: pd.DataFrame,
    label: str,
    figures_dir: str | Path = "reports/figures/stage5_features",
):
    """Feature QC grafiklerini üret."""
    figures_dir = Path(figures_dir)

    plot_feature_distributions(
        df,
        title=f"Feature Dağılımları — {label}",
        save_path=figures_dir / f"feat_dist_{label}.png",
    )

    plot_nan_report(
        df,
        title=f"NaN Raporu — {label}",
        save_path=figures_dir / f"feat_nan_{label}.png",
    )


# ============================================================================
#  5. Standalone Test
# ============================================================================

if __name__ == "__main__":
    from data.loader import load_eeg
    from preprocessing.filters import apply_filters
    from preprocessing.ica import apply_ica
    from preprocessing.segmentation import segment_signal

    cfg = load_config()
    sr = cfg["eeg"]["sampling_rate"]

    print("=" * 60)
    print("  AŞAMA 5 — Feature Çıkarımı")
    print("=" * 60)

    test_path = Path("/mnt/user-data/uploads/s1_eeg.csv")
    if not test_path.exists():
        print("[!] Test dosyası bulunamadı.")
        sys.exit(1)

    # Pipeline: Yükle → Filtrele → ICA → Segment
    raw, ch_names, _ = load_eeg(test_path, cfg, clean=True)
    filtered = apply_filters(raw, cfg)
    cleaned, _ = apply_ica(filtered, ch_names, cfg)
    segments = segment_signal(cleaned, cfg)

    # Feature çıkarımı (ilk 3 segment ile hızlı test)
    print(f"\nHızlı test: ilk 3 segment...")
    meta = {"subject_id": "202524", "session": "Session01", "condition": "test"}
    df = extract_all_features(segments[:3], ch_names, sr, cfg, meta)
    print(f"\nSonuç: {df.shape}")
    print(f"Sütunlar (ilk 20): {list(df.columns[:20])}")

    # Kaydet
    save_features(df, "outputs/feature_datasets/features_test.csv", format="csv")

    # QC
    generate_feature_qc(df, label="202524_test")

    print("\nAşama 5 testi tamamlandı!")
