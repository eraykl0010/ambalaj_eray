"""
Aşama 6 — Etiketleme (Anket Birleştirme)

Görevler:
    1. Anket CSV okuma (subject_id, condition, valence_raw, arousal_raw)
    2. Koşul bazlı median split → Valence High/Low, Arousal High/Low
    3. Feature DataFrame'lerine etiket birleştirme
    4. Etiketli veri setlerini kaydetme (cam ayrı, karton ayrı)
    5. QC: VA scatter / quadrant görseli

Kullanım:
    python run.py --stage 6
    from labels.survey import load_survey, apply_median_split, merge_labels
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

from configs.config_loader import load_config


# ============================================================================
#  1. Anket Okuma
# ============================================================================

def load_survey(survey_path: str | Path) -> pd.DataFrame:
    """
    Anket dosyasını oku (xlsx veya csv).

    Beklenen sütunlar:
        subject_id, condition, valence_raw, arousal_raw
    """
    survey_path = Path(survey_path)
    if not survey_path.exists():
        raise FileNotFoundError(f"Anket dosyası bulunamadı: {survey_path}")

    if survey_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(survey_path)
    else:
        df = pd.read_csv(survey_path)

    # Sütun kontrolü
    required = ["subject_id", "condition", "valence_raw", "arousal_raw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik sütunlar: {missing}. Mevcut: {df.columns.tolist()}")

    # Tip tutarlılığı
    df["subject_id"] = df["subject_id"].astype(str)
    df["condition"] = df["condition"].str.strip().str.lower()

    # NaN temizleme
    nan_count = df[required].isna().sum().sum()
    if nan_count > 0:
        print(f"  [UYARI] {nan_count} NaN değer var — bu satırlar atlanıyor.")
        df = df.dropna(subset=required)

    print(f"  [ANKET] Yüklendi: {len(df)} satır, "
          f"{df['subject_id'].nunique()} denek, "
          f"koşullar: {df['condition'].unique().tolist()}")

    return df


# ============================================================================
#  2. Median Split → Binary Etiketler
# ============================================================================

def apply_median_split(
    survey: pd.DataFrame,
    method: str = "median",
) -> pd.DataFrame:
    """
    Koşul bazlı median split ile binary etiketler oluştur.

    Her koşulun (karton, camsise) kendi median'ı ayrı hesaplanır.

    Args:
        survey:  anket DataFrame
        method:  "median" | "zero"

    Returns:
        survey + valence_label, arousal_label, va_quadrant sütunları
    """
    df = survey.copy()
    df["valence_label"] = ""
    df["arousal_label"] = ""

    thresholds = {}

    print("\n  === Median Split ===")
    for condition in sorted(df["condition"].unique()):
        mask = df["condition"] == condition
        sub = df.loc[mask]

        if method == "median":
            v_thr = sub["valence_raw"].median()
            a_thr = sub["arousal_raw"].median()
        elif method == "zero":
            v_thr = 0.0
            a_thr = 0.0
        else:
            raise ValueError(f"Desteklenmeyen binarizasyon: {method}")

        thresholds[condition] = {"valence": v_thr, "arousal": a_thr}

        df.loc[mask, "valence_label"] = np.where(
            sub["valence_raw"].values >= v_thr, "High", "Low"
        )
        df.loc[mask, "arousal_label"] = np.where(
            sub["arousal_raw"].values >= a_thr, "High", "Low"
        )

        v_h = (df.loc[mask, "valence_label"] == "High").sum()
        v_l = (df.loc[mask, "valence_label"] == "Low").sum()
        a_h = (df.loc[mask, "arousal_label"] == "High").sum()
        a_l = (df.loc[mask, "arousal_label"] == "Low").sum()

        print(f"  {condition.upper():8s}: "
              f"V_thr={v_thr:+.3f} → H:{v_h}/L:{v_l}  |  "
              f"A_thr={a_thr:+.3f} → H:{a_h}/L:{a_l}")

    # VA Quadrant
    def _quad(row):
        v, a = row["valence_label"], row["arousal_label"]
        if v == "High" and a == "High": return "HVHA"
        if v == "High" and a == "Low":  return "HVLA"
        if v == "Low"  and a == "High": return "LVHA"
        return "LVLA"

    df["va_quadrant"] = df.apply(_quad, axis=1)

    print(f"\n  VA Quadrant dağılımı (tüm koşullar birlikte):")
    for q in ["HVHA", "HVLA", "LVHA", "LVLA"]:
        cnt = (df["va_quadrant"] == q).sum()
        print(f"    {q}: {cnt}")
    return df, thresholds


# ============================================================================
#  3. Feature + Etiket Birleştirme
# ============================================================================

def merge_labels(
    features_df: pd.DataFrame,
    labeled_survey: pd.DataFrame,
) -> pd.DataFrame:
    """
    Feature tablosuna anket etiketlerini birleştir.

    Birleştirme: subject_id + condition
    """
    features_df = features_df.copy()
    features_df["subject_id"] = features_df["subject_id"].astype(str)
    labeled_survey = labeled_survey.copy()
    labeled_survey["subject_id"] = labeled_survey["subject_id"].astype(str)

    label_cols = ["subject_id", "condition",
                  "valence_raw", "arousal_raw",
                  "valence_label", "arousal_label", "va_quadrant"]
    survey_sub = labeled_survey[label_cols].drop_duplicates()

    merged = features_df.merge(survey_sub, on=["subject_id", "condition"], how="left")

    n_total = len(merged)
    n_labeled = merged["valence_label"].notna().sum()
    n_unlabeled = n_total - n_labeled

    print(f"\n  [MERGE] Toplam segment: {n_total}")
    print(f"  [MERGE] Etiketli: {n_labeled} ({n_labeled/n_total*100:.1f}%)")

    if n_unlabeled > 0:
        unlabeled_ids = merged.loc[merged["valence_label"].isna(), "subject_id"].unique()
        print(f"  [MERGE] Etiketsiz: {n_unlabeled} segment "
              f"({len(unlabeled_ids)} denek: {unlabeled_ids[:10]})")

    return merged


# ============================================================================
#  4. Toplu Etiketleme Pipeline
# ============================================================================

def label_all_features(
    cfg: dict,
    survey_path: str | Path,
    feat_dir: str | Path | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Tüm koşullar için feature dosyalarını oku, etiketle, kaydet.

    Returns:
        dict: {"karton": DataFrame, "camsise": DataFrame, "all": DataFrame}
    """
    if feat_dir is None:
        feat_dir = Path(cfg["paths"]["output_dir"]) / "feature_datasets"
    feat_dir = Path(feat_dir)

    binarization = cfg["labels"]["binarization"]

    # 1. Anket yükle
    print("1. Anket yükleniyor...")
    survey = load_survey(survey_path)

    # 2. Median split
    print("\n2. Median split uygulanıyor...")
    labeled_survey, thresholds = apply_median_split(survey, method=binarization)

    # 3. Feature dosyalarını oku ve etiketle
    print("\n3. Feature dosyalarına etiket ekleniyor...")
    results = {}

    for condition in ["karton", "camsise"]:
        feat_file = feat_dir / f"features_{condition}.csv"
        if not feat_file.exists():
            print(f"  [UYARI] {feat_file} bulunamadı — atlanıyor.")
            continue

        feat_df = pd.read_csv(feat_file)
        print(f"\n  {condition.upper()}: {len(feat_df)} segment yüklendi")

        merged = merge_labels(feat_df, labeled_survey)

        # Etiketsiz satırları çıkar
        before = len(merged)
        merged = merged.dropna(subset=["valence_label", "arousal_label"])
        after = len(merged)
        if before > after:
            print(f"  [TEMİZ] {before - after} etiketsiz segment çıkarıldı")

        results[condition] = merged

        # Kaydet
        out_path = feat_dir / f"features_{condition}_labeled.csv"
        merged.to_csv(out_path, index=False)
        print(f"  [KAYIT] {out_path} → {len(merged)} satır")

    # 4. Tüm koşullar birleşik
    if results:
        all_df = pd.concat(results.values(), ignore_index=True)
        all_path = feat_dir / "features_all_labeled.csv"
        all_df.to_csv(all_path, index=False)
        print(f"\n  [KAYIT] {all_path} → {len(all_df)} satır")
        results["all"] = all_df

    return results, labeled_survey, thresholds


# ============================================================================
#  5. QC Grafikleri
# ============================================================================

def plot_va_scatter(
    survey: pd.DataFrame,
    title: str = "Valence–Arousal Dağılımı",
    save_path: Optional[str | Path] = None,
):
    """Anket puanlarının VA uzayındaki dağılımı (koşul bazlı)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    conditions = sorted(survey["condition"].unique())
    colors = {"camsise": "#2196F3", "karton": "#4CAF50"}
    markers = {"camsise": "o", "karton": "s"}

    # 1. Koşul bazlı scatter
    ax = axes[0]
    for cond in conditions:
        sub = survey[survey["condition"] == cond]
        ax.scatter(
            sub["valence_raw"], sub["arousal_raw"],
            c=colors.get(cond, "gray"), marker=markers.get(cond, "o"),
            alpha=0.7, s=60, edgecolors="white", linewidth=0.5,
            label=f"{cond.capitalize()} (n={len(sub)})"
        )
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Valence", fontsize=11)
    ax.set_ylabel("Arousal", fontsize=11)
    ax.set_title("Koşul Bazlı", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.grid(alpha=0.15)

    # 2-3. Koşul bazlı quadrant
    for i, cond in enumerate(conditions):
        ax = axes[i + 1]
        sub = survey[survey["condition"] == cond]

        quad_colors = {
            "HVHA": "#F44336",  # kırmızı (excited)
            "HVLA": "#2196F3",  # mavi (calm)
            "LVHA": "#FF9800",  # turuncu (angry)
            "LVLA": "#9E9E9E",  # gri (sad)
        }

        for q, color in quad_colors.items():
            q_sub = sub[sub["va_quadrant"] == q]
            if len(q_sub) > 0:
                ax.scatter(
                    q_sub["valence_raw"], q_sub["arousal_raw"],
                    c=color, s=60, alpha=0.8, edgecolors="white",
                    linewidth=0.5, label=f"{q} (n={len(q_sub)})"
                )

        # Median çizgileri
        v_med = sub["valence_raw"].median()
        a_med = sub["arousal_raw"].median()
        ax.axhline(a_med, color="black", linestyle="-", alpha=0.4, linewidth=1)
        ax.axvline(v_med, color="black", linestyle="-", alpha=0.4, linewidth=1)

        ax.set_xlabel("Valence", fontsize=11)
        ax.set_ylabel("Arousal", fontsize=11)
        ax.set_title(f"{cond.capitalize()} — Quadrant", fontsize=11)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(-2.3, 2.3)
        ax.grid(alpha=0.15)

        # Quadrant isimleri
        ax.text(1.8, 1.8, "HVHA", fontsize=8, ha="center", alpha=0.5, color="#F44336")
        ax.text(-1.8, 1.8, "LVHA", fontsize=8, ha="center", alpha=0.5, color="#FF9800")
        ax.text(1.8, -1.8, "HVLA", fontsize=8, ha="center", alpha=0.5, color="#2196F3")
        ax.text(-1.8, -1.8, "LVLA", fontsize=8, ha="center", alpha=0.5, color="#9E9E9E")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def plot_label_distribution(
    survey: pd.DataFrame,
    title: str = "Etiket Dağılımı",
    save_path: Optional[str | Path] = None,
):
    """Koşul × Valence/Arousal etiket dağılım bar grafikleri."""
    conditions = sorted(survey["condition"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, target in enumerate(["valence_label", "arousal_label"]):
        ax = axes[i]
        data_plot = []
        labels_plot = []
        colors_plot = []

        color_map = {"High": "#4CAF50", "Low": "#F44336"}

        for cond in conditions:
            sub = survey[survey["condition"] == cond]
            for label in ["High", "Low"]:
                cnt = (sub[target] == label).sum()
                data_plot.append(cnt)
                labels_plot.append(f"{cond.capitalize()}\n{label}")
                colors_plot.append(color_map[label])

        bars = ax.bar(range(len(data_plot)), data_plot, color=colors_plot, alpha=0.8,
                       edgecolor="white", linewidth=1)
        ax.set_xticks(range(len(data_plot)))
        ax.set_xticklabels(labels_plot, fontsize=9)
        ax.set_ylabel("Denek Sayısı", fontsize=10)
        ax.set_title(target.replace("_label", "").capitalize(), fontsize=12)

        # Sayıları bar üstüne yaz
        for bar, val in zip(bars, data_plot):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", fontsize=10, fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def plot_segment_label_distribution(
    results: Dict[str, pd.DataFrame],
    title: str = "Segment Bazlı Etiket Dağılımı",
    save_path: Optional[str | Path] = None,
):
    """Segment sayısı bazlı etiket dağılımı (ML dengesizliğini görmek için)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, target in enumerate(["valence_label", "arousal_label"]):
        ax = axes[i]
        data_plot = []
        labels_plot = []
        colors_plot = []

        color_map = {"High": "#4CAF50", "Low": "#F44336"}

        for cond in ["camsise", "karton"]:
            if cond not in results:
                continue
            df = results[cond]
            for label in ["High", "Low"]:
                cnt = (df[target] == label).sum()
                data_plot.append(cnt)
                labels_plot.append(f"{cond.capitalize()}\n{label}")
                colors_plot.append(color_map[label])

        bars = ax.bar(range(len(data_plot)), data_plot, color=colors_plot, alpha=0.8,
                       edgecolor="white", linewidth=1)
        ax.set_xticks(range(len(data_plot)))
        ax.set_xticklabels(labels_plot, fontsize=9)
        ax.set_ylabel("Segment Sayısı", fontsize=10)
        ax.set_title(target.replace("_label", "").capitalize(), fontsize=12)

        for bar, val in zip(bars, data_plot):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(val), ha="center", fontsize=10, fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Grafik kaydedildi: {save_path}")

    plt.close(fig)
    return fig


def generate_label_qc(
    labeled_survey: pd.DataFrame,
    results: Dict[str, pd.DataFrame],
    figures_dir: str | Path = "reports/figures/stage6_labels",
):
    """Tüm etiketleme QC grafiklerini üret."""
    figures_dir = Path(figures_dir)

    plot_va_scatter(
        labeled_survey,
        title="Valence–Arousal Dağılımı (Anket)",
        save_path=figures_dir / "va_scatter.png",
    )

    plot_label_distribution(
        labeled_survey,
        title="Denek Bazlı Etiket Dağılımı",
        save_path=figures_dir / "label_dist_subjects.png",
    )

    plot_segment_label_distribution(
        results,
        title="Segment Bazlı Etiket Dağılımı",
        save_path=figures_dir / "label_dist_segments.png",
    )


# ============================================================================
#  6. Standalone Test
# ============================================================================

if __name__ == "__main__":
    cfg = load_config()

    print("=" * 60)
    print("  AŞAMA 6 — Etiketleme")
    print("=" * 60)

    survey_path = Path("/mnt/user-data/uploads/survey_ratings.csv")
    if not survey_path.exists():
        print("[!] Anket dosyası bulunamadı.")
        sys.exit(1)

    # Pipeline
    results, labeled_survey, thresholds = label_all_features(cfg, survey_path)

    # QC
    generate_label_qc(labeled_survey, results)

    print("\nAşama 6 testi tamamlandı!")
