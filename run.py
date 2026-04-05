"""
Proje Ana Çalıştırıcı (Runner)

Her aşamayı tek komutla çalıştırmanızı sağlar.

Kullanım (PyCharm terminali veya CMD):
    cd C:\\Users\\Eray\\Desktop\\sustainable_packaging_eeg

    python run.py --stage 1        # Aşama 1: Veri Yükleme & Keşif
    python run.py --stage 2        # Aşama 2: Filtreleme
    python run.py --stage 3        # Aşama 3: ICA (henüz eklenmedi)
    ...

    python run.py --stage 2 --subject 202524   # Tek denek test
"""

import argparse
import sys
from pathlib import Path

# Proje kökünü Python path'ine ekle
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_loader import load_config, get_active_channels
from data.loader import (
    discover_recordings,
    load_eeg,
    plot_raw_eeg_sample,
    generate_qc_report,
)
from preprocessing.filters import apply_filters, generate_filter_qc
from preprocessing.ica import apply_ica, generate_ica_qc
from preprocessing.segmentation import (
    segment_signal, save_segments, load_segments, generate_segmentation_qc
)
from features.extract_all import (
    extract_all_features, save_features, generate_feature_qc
)
from labels.survey import (
    load_survey, apply_median_split, label_all_features,
    merge_labels, generate_label_qc,
)
from models.unsupervised import run_unsupervised_analysis
from models.supervised import run_supervised_pipeline

try:
    from models.deep_learning import run_dl_pipeline
    HAS_DL = True
except ImportError:
    HAS_DL = False
    def run_dl_pipeline(*args, **kwargs):
        print("[!] PyTorch bulunamadı — Aşama 9 atlanıyor.")
        print("    pip install torch torchvision ile yükleyin.")
        return {}

from topomaps.generator import (
    generate_topomaps_for_condition, plot_topomap_examples, plot_topomap_stats,
)


def run_stage1(cfg: dict, max_plot: int = 5):
    """Aşama 1 — Veri Yükleme & Keşif"""
    print("=" * 60)
    print("  AŞAMA 1 — Veri Yükleme & Keşif")
    print("=" * 60)

    # 1. Envanter keşfi
    inventory = discover_recordings(cfg)

    if len(inventory) == 0:
        print("[!] Hiç kayıt bulunamadı. config.yaml'daki raw_data_root yolunu kontrol edin:")
        print(f"    Şu anki değer: {cfg['paths']['raw_data_root']}")
        return None

    # 2. Envanteri kaydet
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    inv_path = out_dir / "recording_inventory.csv"
    inventory.to_csv(inv_path, index=False)
    print(f"Envanter kaydedildi: {inv_path}")

    # 3. Envanter özetini göster
    print("\n--- Envanter Özeti ---")
    print(inventory[["subject_id", "session", "condition", "duration_sec", "n_channels_csv"]].to_string())

    # 4. QC grafikleri
    generate_qc_report(
        inventory, cfg,
        figures_dir=Path(cfg["paths"]["reports_dir"]) / "figures" / "stage1_raw",
        max_subjects_to_plot=max_plot,
    )

    return inventory


def run_stage2(cfg: dict, inventory=None, max_plot: int = 5):
    """Aşama 2 — Filtreleme (Band-pass + Notch)"""
    print("=" * 60)
    print("  AŞAMA 2 — Filtreleme (Band-pass + Notch)")
    print("=" * 60)

    sr = cfg["eeg"]["sampling_rate"]
    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "stage2_filtered"

    # Envanter yoksa keşfet
    if inventory is None:
        inv_path = Path(cfg["paths"]["output_dir"]) / "recording_inventory.csv"
        if inv_path.exists():
            import pandas as pd
            inventory = pd.read_csv(inv_path)
            print(f"Envanter yüklendi: {inv_path} ({len(inventory)} kayıt)")
        else:
            print("Envanter bulunamadı — önce Aşama 1'i çalıştırın.")
            print("  python run.py --stage 1")
            return

    plotted = 0
    for idx, row in inventory.iterrows():
        label = f"{row['subject_id']}_{row['session']}_{row['condition']}"
        print(f"\n[{idx+1}/{len(inventory)}] İşleniyor: {label}")

        try:
            # 1. Ham veri yükle
            raw, ch_names, _ = load_eeg(row["eeg_path"], cfg, clean=True)
            print(f"  Ham: shape={raw.shape}, "
                  f"range=[{raw.min():.2f}, {raw.max():.2f}]")

            # 2. Filtreleme
            filtered = apply_filters(raw, cfg)
            print(f"  Filtrelenmiş: range=[{filtered.min():.4f}, {filtered.max():.4f}], "
                  f"std=[{filtered.std(axis=0).min():.4f}, {filtered.std(axis=0).max():.4f}]")

            # 3. QC grafikleri (ilk N denek)
            if plotted < max_plot:
                generate_filter_qc(raw, filtered, ch_names, sr, label, fig_dir)
                plotted += 1

        except Exception as e:
            print(f"  [HATA] {e}")

    print(f"\nAşama 2 tamamlandı! QC grafikleri: {fig_dir}")


def run_stage2_single(cfg: dict, eeg_path: str):
    """Aşama 2 — Tek dosya ile test"""
    print("=" * 60)
    print(f"  AŞAMA 2 — Tek Dosya Test: {eeg_path}")
    print("=" * 60)

    sr = cfg["eeg"]["sampling_rate"]
    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "stage2_filtered"

    eeg_path = Path(eeg_path)
    if not eeg_path.exists():
        print(f"[!] Dosya bulunamadı: {eeg_path}")
        return

    # 1. Yükle
    raw, ch_names, _ = load_eeg(eeg_path, cfg, clean=True)
    print(f"  Ham: shape={raw.shape}, range=[{raw.min():.2f}, {raw.max():.2f}]")

    # 2. Filtrele
    filtered = apply_filters(raw, cfg)
    print(f"  Filtrelenmiş: range=[{filtered.min():.4f}, {filtered.max():.4f}]")

    # 3. QC
    label = eeg_path.parent.name  # klasör adını etiket olarak kullan
    generate_filter_qc(raw, filtered, ch_names, sr, label, fig_dir)

    print(f"\nGrafikler: {fig_dir}")


def run_stage3(cfg: dict, inventory=None, max_plot: int = 5):
    """Aşama 3 — ICA (Artifact Azaltma)"""
    print("=" * 60)
    print("  AŞAMA 3 — ICA (Artifact Azaltma)")
    print("=" * 60)

    sr = cfg["eeg"]["sampling_rate"]
    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "stage3_ica"

    # Envanter yoksa yükle
    if inventory is None:
        inv_path = Path(cfg["paths"]["output_dir"]) / "recording_inventory.csv"
        if inv_path.exists():
            import pandas as pd
            inventory = pd.read_csv(inv_path)
            print(f"Envanter yüklendi: {inv_path} ({len(inventory)} kayıt)")
        else:
            print("Envanter bulunamadı — önce Aşama 1'i çalıştırın.")
            return

    plotted = 0
    for idx, row in inventory.iterrows():
        label = f"{row['subject_id']}_{row['session']}_{row['condition']}"
        print(f"\n[{idx+1}/{len(inventory)}] İşleniyor: {label}")

        try:
            # 1. Yükle
            raw, ch_names, _ = load_eeg(row["eeg_path"], cfg, clean=True)

            # 2. Filtrele
            filtered = apply_filters(raw, cfg)

            # 3. ICA
            cleaned, ica_info = apply_ica(filtered, ch_names, cfg)
            n_art = len(ica_info.get("artifact_indices", []))
            print(f"  Sonuç: {n_art} artifact bileşen çıkarıldı, "
                  f"range=[{cleaned.min():.4f}, {cleaned.max():.4f}]")

            # 4. QC grafikleri (ilk N denek)
            if plotted < max_plot:
                generate_ica_qc(
                    filtered, cleaned, ch_names, sr, ica_info, label, fig_dir
                )
                plotted += 1

        except Exception as e:
            print(f"  [HATA] {e}")

    print(f"\nAşama 3 tamamlandı! QC grafikleri: {fig_dir}")


def run_stage3_single(cfg: dict, eeg_path: str):
    """Aşama 3 — Tek dosya ile test"""
    print("=" * 60)
    print(f"  AŞAMA 3 — Tek Dosya Test: {eeg_path}")
    print("=" * 60)

    sr = cfg["eeg"]["sampling_rate"]
    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "stage3_ica"

    eeg_path = Path(eeg_path)
    if not eeg_path.exists():
        print(f"[!] Dosya bulunamadı: {eeg_path}")
        return

    # 1. Yükle → Filtrele → ICA
    raw, ch_names, _ = load_eeg(eeg_path, cfg, clean=True)
    filtered = apply_filters(raw, cfg)
    cleaned, ica_info = apply_ica(filtered, ch_names, cfg)

    n_art = len(ica_info.get("artifact_indices", []))
    print(f"\n  Sonuç: {n_art} artifact bileşen çıkarıldı")
    print(f"  Artifact bileşenler: {ica_info.get('artifact_indices', [])}")

    # QC
    label = eeg_path.parent.name
    generate_ica_qc(filtered, cleaned, ch_names, sr, ica_info, label, fig_dir)
    print(f"\nGrafikler: {fig_dir}")


def run_stage4(cfg: dict, inventory=None, max_plot: int = 5):
    """Aşama 4 — Segmentasyon"""
    print("=" * 60)
    print("  AŞAMA 4 — Segmentasyon")
    print("=" * 60)

    sr = cfg["eeg"]["sampling_rate"]
    seg_dir = Path(cfg["paths"]["output_dir"]) / "cleaned_segments"
    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "stage4_segments"

    # Envanter
    if inventory is None:
        inv_path = Path(cfg["paths"]["output_dir"]) / "recording_inventory.csv"
        if inv_path.exists():
            import pandas as pd
            inventory = pd.read_csv(inv_path)
            print(f"Envanter yüklendi: {inv_path} ({len(inventory)} kayıt)")
        else:
            print("Envanter bulunamadı — önce Aşama 1'i çalıştırın.")
            return

    all_seg_meta = []
    plotted = 0

    for idx, row in inventory.iterrows():
        label = f"{row['subject_id']}_{row['session']}_{row['condition']}"
        print(f"\n[{idx+1}/{len(inventory)}] İşleniyor: {label}")

        try:
            # 1-3. Pipeline
            raw, ch_names, _ = load_eeg(row["eeg_path"], cfg, clean=True)
            filtered = apply_filters(raw, cfg)
            cleaned, _ = apply_ica(filtered, ch_names, cfg)

            # 4. Segmentasyon
            segments = segment_signal(cleaned, cfg)

            # 5. Kaydet
            meta = {
                "subject_id": row["subject_id"],
                "session": row["session"],
                "condition": row["condition"],
            }
            npz_path, seg_meta = save_segments(segments, ch_names, meta, seg_dir)
            all_seg_meta.append(seg_meta)

            # 6. QC (ilk N denek)
            if plotted < max_plot:
                generate_segmentation_qc(
                    cleaned, segments, ch_names, sr, cfg, label, fig_dir
                )
                plotted += 1

        except Exception as e:
            print(f"  [HATA] {e}")

    # Toplam segment envanteri
    if all_seg_meta:
        import pandas as pd
        full_meta = pd.concat(all_seg_meta, ignore_index=True)
        meta_path = Path(cfg["paths"]["output_dir"]) / "segment_inventory.csv"
        full_meta.to_csv(meta_path, index=False)
        print(f"\nSegment envanteri kaydedildi: {meta_path}")
        print(f"Toplam segment: {len(full_meta)}")

    print(f"\nAşama 4 tamamlandı! Segmentler: {seg_dir}, QC: {fig_dir}")


def run_stage4_single(cfg: dict, eeg_path: str):
    """Aşama 4 — Tek dosya ile test"""
    print("=" * 60)
    print(f"  AŞAMA 4 — Tek Dosya Test: {eeg_path}")
    print("=" * 60)

    sr = cfg["eeg"]["sampling_rate"]
    seg_dir = Path(cfg["paths"]["output_dir"]) / "cleaned_segments"
    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "stage4_segments"

    eeg_path = Path(eeg_path)
    if not eeg_path.exists():
        print(f"[!] Dosya bulunamadı: {eeg_path}")
        return

    # 1-3. Pipeline
    raw, ch_names, _ = load_eeg(eeg_path, cfg, clean=True)
    filtered = apply_filters(raw, cfg)
    cleaned, _ = apply_ica(filtered, ch_names, cfg)

    # 4. Segmentasyon
    segments = segment_signal(cleaned, cfg)

    # 5. Kaydet
    meta = {"subject_id": "test", "session": "test", "condition": "test"}
    npz_path, seg_meta = save_segments(segments, ch_names, meta, seg_dir)

    print(f"\n  Toplam segment: {len(segments)}, her biri: {segments[0].shape}")
    print(f"  Dosya: {npz_path}")

    # 6. QC
    label = eeg_path.parent.name
    generate_segmentation_qc(cleaned, segments, ch_names, sr, cfg, label, fig_dir)
    print(f"\nGrafikler: {fig_dir}")


def run_stage5(cfg: dict, inventory=None, max_plot: int = 5):
    """Aşama 5 — Feature Çıkarımı"""
    print("=" * 60)
    print("  AŞAMA 5 — Feature Çıkarımı")
    print("=" * 60)

    sr = cfg["eeg"]["sampling_rate"]
    seg_dir = Path(cfg["paths"]["output_dir"]) / "cleaned_segments"
    feat_dir = Path(cfg["paths"]["output_dir"]) / "feature_datasets"
    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "stage5_features"

    # Envanter
    if inventory is None:
        inv_path = Path(cfg["paths"]["output_dir"]) / "recording_inventory.csv"
        if inv_path.exists():
            import pandas as pd
            inventory = pd.read_csv(inv_path)
            print(f"Envanter yüklendi: {inv_path} ({len(inventory)} kayıt)")
        else:
            print("Envanter bulunamadı — önce Aşama 1'i çalıştırın.")
            return

    # Koşul bazlı feature toplama
    condition_dfs = {"karton": [], "camsise": []}
    plotted = 0

    for idx, row in inventory.iterrows():
        label = f"{row['subject_id']}_{row['session']}_{row['condition']}"
        print(f"\n[{idx+1}/{len(inventory)}] İşleniyor: {label}")

        # Segment dosyası var mı? (stage 4 çıktısı)
        npz_file = seg_dir / f"{row['subject_id']}_{row['session']}_{row['condition']}.npz"

        try:
            if npz_file.exists():
                # Önceden segmentlenmiş veriyi yükle
                segments, ch_names, meta = load_segments(npz_file)
                meta_dict = meta
                print(f"  Segment yüklendi: {segments.shape}")
            else:
                # Tam pipeline çalıştır
                raw, ch_names, _ = load_eeg(row["eeg_path"], cfg, clean=True)
                filtered = apply_filters(raw, cfg)
                cleaned, _ = apply_ica(filtered, ch_names, cfg)
                seg_list = segment_signal(cleaned, cfg)
                segments = seg_list
                meta_dict = {
                    "subject_id": str(row["subject_id"]),
                    "session": str(row["session"]),
                    "condition": str(row["condition"]),
                }

            # Feature çıkarımı
            df = extract_all_features(
                segments, ch_names, sr, cfg, meta=meta_dict, verbose=True
            )

            condition = row["condition"]
            if condition in condition_dfs:
                condition_dfs[condition].append(df)

            # QC (ilk N)
            if plotted < max_plot:
                generate_feature_qc(df, label, fig_dir)
                plotted += 1

        except Exception as e:
            print(f"  [HATA] {e}")

    # Koşul bazlı birleştir ve kaydet
    import pandas as pd
    for condition, dfs in condition_dfs.items():
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            out_path = feat_dir / f"features_{condition}.csv"
            save_features(merged, out_path, format="csv")
            print(f"\n  {condition}: {len(merged)} satır → {out_path}")

    # Tüm koşullar birleşik
    all_dfs = [df for dfs in condition_dfs.values() for df in dfs]
    if all_dfs:
        full = pd.concat(all_dfs, ignore_index=True)
        save_features(full, feat_dir / "features_all.csv", format="csv")

    print(f"\nAşama 5 tamamlandı! Feature'lar: {feat_dir}")


def run_stage5_single(cfg: dict, eeg_path: str):
    """Aşama 5 — Tek dosya ile test"""
    print("=" * 60)
    print(f"  AŞAMA 5 — Tek Dosya Test: {eeg_path}")
    print("=" * 60)

    sr = cfg["eeg"]["sampling_rate"]
    feat_dir = Path(cfg["paths"]["output_dir"]) / "feature_datasets"
    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "stage5_features"

    eeg_path = Path(eeg_path)
    if not eeg_path.exists():
        print(f"[!] Dosya bulunamadı: {eeg_path}")
        return

    # Pipeline
    raw, ch_names, _ = load_eeg(eeg_path, cfg, clean=True)
    filtered = apply_filters(raw, cfg)
    cleaned, _ = apply_ica(filtered, ch_names, cfg)
    segments = segment_signal(cleaned, cfg)

    # Feature çıkarımı
    meta = {"subject_id": "test", "session": "test", "condition": "test"}
    df = extract_all_features(segments, ch_names, sr, cfg, meta, verbose=True)

    # Kaydet
    save_features(df, feat_dir / "features_single_test.csv", format="csv")

    # QC
    label = eeg_path.parent.name
    generate_feature_qc(df, label, fig_dir)

    print(f"\nDataFrame: {df.shape}")
    print(f"Feature'lar: {feat_dir}, Grafikler: {fig_dir}")


def run_stage6(cfg: dict):
    """Aşama 6 — Etiketleme (Anket Birleştirme)"""
    print("=" * 60)
    print("  AŞAMA 6 — Etiketleme (Anket Birleştirme)")
    print("=" * 60)

    survey_path = cfg["paths"].get("survey_csv")
    if not survey_path:
        print("[!] config.yaml'da survey_csv yolu tanımlı değil.")
        print("    Lütfen config.yaml → paths → survey_csv'ye dosya yolunu ekleyin.")
        print('    Örnek: survey_csv: "C:\\Users\\Eray\\...\\survey_ratings.csv"')
        return

    survey_path = Path(survey_path)
    if not survey_path.exists():
        print(f"[!] Anket dosyası bulunamadı: {survey_path}")
        return

    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "stage6_labels"

    # Tam pipeline
    results, labeled_survey, thresholds = label_all_features(cfg, survey_path)

    # QC grafikleri
    generate_label_qc(labeled_survey, results, fig_dir)

    # Özet
    print("\n=== ÖZET ===")
    for cond in ["camsise", "karton"]:
        if cond in results:
            df = results[cond]
            v_h = (df["valence_label"] == "High").sum()
            v_l = (df["valence_label"] == "Low").sum()
            a_h = (df["arousal_label"] == "High").sum()
            a_l = (df["arousal_label"] == "Low").sum()
            print(f"  {cond.upper()}: {len(df)} segment — "
                  f"V(H:{v_h}/L:{v_l}), A(H:{a_h}/L:{a_l})")

    print(f"\nAşama 6 tamamlandı! Grafikler: {fig_dir}")


def run_stage7(cfg: dict):
    """Aşama 7 — Unsupervised Keşif (PCA, t-SNE, UMAP, K-Means, GMM)"""
    print("=" * 60)
    print("  AŞAMA 7 — Unsupervised Keşif")
    print("=" * 60)

    feat_dir = Path(cfg["paths"]["output_dir"]) / "feature_datasets"
    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "stage7_unsupervised"

    # Etiketli veri yükle
    all_path = feat_dir / "features_all_labeled.csv"
    if not all_path.exists():
        print("[!] features_all_labeled.csv bulunamadı — önce Aşama 6'yı çalıştırın.")
        return

    import pandas as pd
    df_all = pd.read_csv(all_path)
    print(f"Yüklendi: {all_path} → {df_all.shape}")

    # 1. Tüm veri
    run_unsupervised_analysis(df_all, cfg, label="all", figures_dir=fig_dir)

    # 2. Koşul bazlı
    for cond in ["karton", "camsise"]:
        cond_path = feat_dir / f"features_{cond}_labeled.csv"
        if cond_path.exists():
            df_cond = pd.read_csv(cond_path)
            run_unsupervised_analysis(df_cond, cfg, label=cond, figures_dir=fig_dir)
        else:
            sub = df_all[df_all["condition"] == cond]
            if len(sub) > 20:
                run_unsupervised_analysis(sub, cfg, label=cond, figures_dir=fig_dir)

    print(f"\nAşama 7 tamamlandı! Grafikler: {fig_dir}")


def run_stage8(cfg: dict):
    """Aşama 8 — Supervised ML"""
    print("=" * 60)
    print("  AŞAMA 8 — Supervised ML")
    print("=" * 60)

    feat_dir = Path(cfg["paths"]["output_dir"]) / "feature_datasets"
    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "stage8_supervised"
    met_dir = Path(cfg["paths"]["reports_dir"]) / "metrics"

    import pandas as pd

    # Koşul bazlı
    for condition in ["karton", "camsise"]:
        path = feat_dir / f"features_{condition}_labeled.csv"
        if path.exists():
            df = pd.read_csv(path)
            print(f"\n{condition.upper()}: {df.shape}")
            run_supervised_pipeline(df, cfg, condition=condition,
                                    figures_dir=fig_dir, metrics_dir=met_dir)
        else:
            print(f"  [UYARI] {path} bulunamadı — atlanıyor.")

    # Tüm veri
    all_path = feat_dir / "features_all_labeled.csv"
    if all_path.exists():
        df_all = pd.read_csv(all_path)
        print(f"\nTÜM VERİ: {df_all.shape}")
        run_supervised_pipeline(df_all, cfg, condition="all",
                                figures_dir=fig_dir, metrics_dir=met_dir)

    print(f"\nAşama 8 tamamlandı! Grafikler: {fig_dir}, Metrikler: {met_dir}")


def run_stage9(cfg: dict):
    """Aşama 9 — Deep Learning (1D CNN, CNN-LSTM, EEGNet)"""
    print("=" * 60)
    print("  AŞAMA 9 — Deep Learning")
    print("=" * 60)

    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "stage9_deeplearning"
    met_dir = Path(cfg["paths"]["reports_dir"]) / "metrics"

    for condition in ["karton", "camsise"]:
        print(f"\n{'#'*50}")
        print(f"  Koşul: {condition.upper()}")
        print(f"{'#'*50}")
        run_dl_pipeline(cfg, condition=condition,
                        figures_dir=fig_dir, metrics_dir=met_dir)

    print(f"\nAşama 9 tamamlandı! Grafikler: {fig_dir}, Metrikler: {met_dir}")


def run_stage10(cfg: dict):
    """Aşama 10 — Topomap Üretimi"""
    print("=" * 60)
    print("  AŞAMA 10 — Topomap Üretimi")
    print("=" * 60)

    fig_dir = Path(cfg["paths"]["reports_dir"]) / "figures" / "stage10_topomaps"
    topo_cfg = cfg.get("topomaps", {})
    resolution = topo_cfg.get("resolution", 64)
    save_format = topo_cfg.get("save_format", "both")

    all_stats = {}
    for condition in ["karton", "camsise"]:
        print(f"\n{'='*40}")
        print(f"  Koşul: {condition.upper()}")
        print(f"{'='*40}")

        stats = generate_topomaps_for_condition(
            cfg, condition, resolution=resolution, save_format=save_format,
        )
        all_stats[condition] = stats

        plot_topomap_examples(cfg, condition,
                              save_path=fig_dir / f"topomap_examples_{condition}.png")

    plot_topomap_stats(all_stats, save_path=fig_dir / "topomap_stats.png")

    print(f"\nAşama 10 tamamlandı! Grafikler: {fig_dir}")


# ====================================================================
#  CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sürdürülebilir Ambalaj EEG — Aşama Çalıştırıcı"
    )
    parser.add_argument(
        "--stage", type=int, required=True,
        help="Çalıştırılacak aşama numarası (1, 2, 3, ...)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config dosyası yolu (varsayılan: configs/config.yaml)"
    )
    parser.add_argument(
        "--max-plot", type=int, default=5,
        help="QC grafiği üretilecek maksimum denek sayısı (varsayılan: 5)"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Tek dosya ile test (ör. --file path/to/s1_eeg.csv)"
    )
    args = parser.parse_args()

    # Config yükle
    cfg = load_config(args.config)
    print(f"Config yüklendi. Aktif kanal sayısı: {len(get_active_channels(cfg)[0])}")

    # Aşamayı çalıştır
    if args.stage == 1:
        run_stage1(cfg, max_plot=args.max_plot)

    elif args.stage == 2:
        if args.file:
            run_stage2_single(cfg, args.file)
        else:
            run_stage2(cfg, max_plot=args.max_plot)

    elif args.stage == 3:
        if args.file:
            run_stage3_single(cfg, args.file)
        else:
            run_stage3(cfg, max_plot=args.max_plot)

    elif args.stage == 4:
        if args.file:
            run_stage4_single(cfg, args.file)
        else:
            run_stage4(cfg, max_plot=args.max_plot)

    elif args.stage == 5:
        if args.file:
            run_stage5_single(cfg, args.file)
        else:
            run_stage5(cfg, max_plot=args.max_plot)

    elif args.stage == 6:
        run_stage6(cfg)

    elif args.stage == 7:
        run_stage7(cfg)

    elif args.stage == 8:
        run_stage8(cfg)

    elif args.stage == 9:
        run_stage9(cfg)

    elif args.stage == 10:
        run_stage10(cfg)

    # elif args.stage == 11:
    #     run_stage11(cfg, ...)

    else:
        print(f"[!] Aşama {args.stage} henüz tanımlı değil.")
        print("Mevcut aşamalar: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")


if __name__ == "__main__":
    main()
