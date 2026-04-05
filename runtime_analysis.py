r"""
Empirical Runtime Analysis — EEG Emotion Recognition Pipeline
==============================================================
Uses the actual project data from:
  C:\Users\Eray\Desktop\Surdurulebilir Ambalaj Deney Verileri\EEG Signals

Directory structure:
  EEG Signals / 2025XX / SessionXX / Test_karton* or Test_camsise* / s1_eeg.csv

Usage:
    python runtime_analysis.py
    python runtime_analysis.py --n_subjects 10

Output:
    - runtime_results.csv
    - Console summary table
"""

import time
import numpy as np
import os
import sys
import csv
import re
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import skew, kurtosis as kurt_func
from sklearn.decomposition import FastICA

# ============================================================
# CONFIGURATION
# ============================================================
RAW_DATA_ROOT = r"C:\Users\Eray\Desktop\Surdurulebilir Ambalaj Deney Verileri\EEG Signals"
FS = 256
N_CHANNELS = 32
WINDOW_SEC = 4
OVERLAP = 0.5
N_FEATURES_PER_CH = 28
N_SELECTED = 30

CHANNEL_NAMES = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F3", "Fz",
    "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3",
    "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7",
    "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2"
]

EEG_FILE_NAMES = ["s1_eeg.csv", "eeg_eeg.csv"]
FOLDER_PREFIXES = {"karton": "Test_karton", "camsise": "Test_camsise"}


# ============================================================
# DATA DISCOVERY & LOADING
# ============================================================
def discover_recordings(root_path, max_per_condition=None):
    """
    Scan directory tree and find all EEG recordings.
    Returns list of dicts: {subject_id, condition, eeg_path}
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root}")

    records = []
    subject_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and re.match(r"^2025\d+$", d.name)]
    )

    for subj_dir in subject_dirs:
        session_dirs = sorted(
            [d for d in subj_dir.iterdir()
             if d.is_dir() and re.match(r"^Session\d+$", d.name, re.IGNORECASE)]
        )

        for sess_dir in session_dirs:
            for condition, prefix in FOLDER_PREFIXES.items():
                test_dirs = sorted(
                    [d for d in sess_dir.iterdir()
                     if d.is_dir() and d.name.startswith(prefix)]
                )

                for test_dir in test_dirs:
                    eeg_path = None
                    for fname in EEG_FILE_NAMES:
                        candidate = test_dir / fname
                        if candidate.exists():
                            eeg_path = candidate
                            break

                    if eeg_path is None:
                        continue

                    records.append({
                        "subject_id": subj_dir.name,
                        "condition": condition,
                        "eeg_path": str(eeg_path),
                    })

    if max_per_condition is not None:
        limited = []
        count = {"karton": 0, "camsise": 0}
        for r in records:
            if count[r["condition"]] < max_per_condition:
                limited.append(r)
                count[r["condition"]] += 1
        records = limited

    print(f"Discovered {len(records)} recordings "
          f"({sum(1 for r in records if r['condition']=='karton')} karton, "
          f"{sum(1 for r in records if r['condition']=='camsise')} camsise)")
    return records


def load_eeg_csv(filepath):
    """
    Load a single EEG CSV file.
    No header, 32 columns, comma-separated, float64.
    """
    import pandas as pd
    df = pd.read_csv(filepath, header=None, dtype=np.float64)
    data = df.values

    if np.any(np.isinf(data)):
        data[np.isinf(data)] = np.nan
    if np.any(np.isnan(data)):
        for ch in range(data.shape[1]):
            col = data[:, ch]
            mask = np.isnan(col)
            if mask.any() and not mask.all():
                x = np.arange(len(col))
                data[mask, ch] = np.interp(x[mask], x[~mask], col[~mask])
            elif mask.all():
                data[:, ch] = 0.0

    return data


# ============================================================
# PIPELINE STAGES
# ============================================================
def stage_filtering(signal):
    """Bandpass 0.5-45 Hz + 50 Hz Notch"""
    b_bp, a_bp = butter(4, [0.5, 45], btype='band', fs=FS)
    filtered = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        filtered[:, ch] = filtfilt(b_bp, a_bp, signal[:, ch])

    b_n, a_n = iirnotch(50, 30, FS)
    for ch in range(signal.shape[1]):
        filtered[:, ch] = filtfilt(b_n, a_n, filtered[:, ch])

    return filtered


def stage_ica(signal, n_components=25):
    """FastICA artifact removal"""
    ica = FastICA(n_components=n_components, max_iter=500, random_state=42)
    sources = ica.fit_transform(signal)

    n_remove = 7
    kurt_vals = kurt_func(sources, axis=0)
    artifact_idx = np.argsort(np.abs(kurt_vals))[-n_remove:]
    sources[:, artifact_idx] = 0

    reconstructed = ica.inverse_transform(sources)
    return reconstructed


def stage_segmentation(signal):
    """4s windows, 50% overlap"""
    window_samples = WINDOW_SEC * FS
    step = int(window_samples * (1 - OVERLAP))

    segments = []
    start = 0
    while start + window_samples <= signal.shape[0]:
        segments.append(signal[start:start + window_samples, :])
        start += step

    return segments


def _hjorth(x):
    dx = np.diff(x)
    ddx = np.diff(dx)
    activity = np.var(x)
    mobility = np.sqrt(np.var(dx) / (activity + 1e-10))
    complexity = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-10)) / (mobility + 1e-10)
    return activity, mobility, complexity


def _hurst(x, max_lag=20):
    n = len(x)
    lags = range(2, min(max_lag, n // 2))
    tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
    if len(tau) < 2:
        return 0.5
    poly = np.polyfit(np.log(list(lags)), np.log(np.array(tau) + 1e-10), 1)
    return poly[0]


def _dfa(x, min_box=4, max_box=None):
    n = len(x)
    if max_box is None:
        max_box = n // 4
    y = np.cumsum(x - np.mean(x))
    box_sizes = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 10).astype(int))
    fluct = []
    for bs in box_sizes:
        n_boxes = n // bs
        if n_boxes < 1:
            continue
        rms_list = []
        for i in range(n_boxes):
            segment = y[i * bs:(i + 1) * bs]
            coef = np.polyfit(np.arange(bs), segment, 1)
            trend = np.polyval(coef, np.arange(bs))
            rms_list.append(np.sqrt(np.mean((segment - trend) ** 2)))
        fluct.append(np.mean(rms_list))
    if len(fluct) < 2:
        return 0.5
    poly = np.polyfit(np.log(box_sizes[:len(fluct)]), np.log(np.array(fluct) + 1e-10), 1)
    return poly[0]


def _approx_entropy(x, m=2, r_frac=0.2):
    r = r_frac * np.std(x)
    n = len(x)
    def count_matches(tl):
        templates = np.array([x[i:i + tl] for i in range(n - tl)])
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dist <= r) - 1
        return count / (len(templates) * (len(templates) - 1) + 1e-10)
    phi_m = np.log(count_matches(m) + 1e-10)
    phi_m1 = np.log(count_matches(m + 1) + 1e-10)
    return phi_m - phi_m1


def _perm_entropy(x, order=3, delay=1):
    from itertools import permutations
    n = len(x)
    perms = list(permutations(range(order)))
    counts = {p: 0 for p in perms}
    for i in range(n - (order - 1) * delay):
        idx = tuple(np.argsort(x[i:i + order * delay:delay]))
        if idx in counts:
            counts[idx] += 1
    total = sum(counts.values())
    if total == 0:
        return 0
    probs = np.array([c / total for c in counts.values() if c > 0])
    return -np.sum(probs * np.log2(probs)) / np.log2(len(perms))


def _fuzzy_entropy(x, m=2, r_frac=0.2, n_exp=2):
    r = r_frac * np.std(x)
    n = len(x)
    def phi(tl):
        templates = np.array([x[i:i + tl] for i in range(n - tl)])
        templates = templates - templates.mean(axis=1, keepdims=True)
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates - templates[i]), axis=1)
            sim = np.exp(-((dist / (r + 1e-10)) ** n_exp))
            count += (np.sum(sim) - 1)
        return count / (len(templates) * (len(templates) - 1) + 1e-10)
    return -np.log(phi(m + 1) / (phi(m) + 1e-10) + 1e-10)


def stage_feature_extraction(segments):
    """Extract 28 features per channel per segment"""
    all_features = []

    for seg in segments:
        seg_features = []
        for ch in range(seg.shape[1]):
            x = seg[:, ch]
            feats = []

            # Time domain (9)
            act, mob, comp = _hjorth(x)
            feats.extend([act, mob, comp])
            feats.append(np.mean(x))
            feats.append(skew(x))
            feats.append(np.sqrt(np.mean(x ** 2)))
            feats.append(np.mean(np.abs(x - np.mean(x))))
            feats.append(kurt_func(x))
            feats.append(np.sum(x ** 2))

            # Frequency domain (10)
            freqs, psd = welch(x, fs=FS, nperseg=min(256, len(x)))
            bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
            total_power = np.trapezoid(psd, freqs) + 1e-10
            for low, high in bands:
                idx = (freqs >= low) & (freqs <= high)
                bp = np.trapezoid(psd[idx], freqs[idx])
                feats.append(bp)
                feats.append(bp / total_power)

            # Nonlinear (5)
            feats.append(0.0)  # LLE placeholder
            katz_fd = np.log10(len(x)) / (np.log10(len(x)) + np.log10(
                np.max(np.abs(np.diff(x))) / (np.sum(np.abs(np.diff(x))) + 1e-10) + 1e-10))
            feats.append(katz_fd)
            feats.append(katz_fd)  # Higuchi simplified
            feats.append(_hurst(x))
            feats.append(_dfa(x))

            # Entropy (4)
            feats.append(_approx_entropy(x[:200], m=2, r_frac=0.2))
            feats.append(_perm_entropy(x, order=3))
            feats.append(_fuzzy_entropy(x[:200], m=2, r_frac=0.2))
            se = psd / (total_power + 1e-10)
            se = se[se > 0]
            feats.append(-np.sum(se * np.log2(se + 1e-10)))

            seg_features.extend(feats)

        all_features.append(seg_features)

    return np.array(all_features)


# ============================================================
# MAIN
# ============================================================
def run_timing_analysis(n_subjects=5):
    print("=" * 70)
    print("EMPIRICAL RUNTIME ANALYSIS — EEG Emotion Recognition Pipeline")
    print("=" * 70)
    print(f"Data root: {RAW_DATA_ROOT}")
    print(f"Config: {FS} Hz, {N_CHANNELS} ch, {WINDOW_SEC}s window, {OVERLAP*100:.0f}% overlap")
    print(f"Features: {N_FEATURES_PER_CH}/ch x {N_CHANNELS} ch = {N_FEATURES_PER_CH * N_CHANNELS}")
    print("-" * 70)

    recordings = discover_recordings(RAW_DATA_ROOT, max_per_condition=n_subjects)

    if len(recordings) == 0:
        print("[ERROR] No recordings found. Check RAW_DATA_ROOT path.")
        sys.exit(1)

    timings = {
        'Data Loading': [],
        'Filtering': [],
        'ICA': [],
        'Segmentation': [],
        'Feature Extraction': [],
    }
    segment_counts = []

    for i, rec in enumerate(recordings):
        subj = rec["subject_id"]
        cond = rec["condition"]
        path = rec["eeg_path"]
        print(f"\n  [{i+1}/{len(recordings)}] {subj} - {cond}")

        t0 = time.perf_counter()
        signal = load_eeg_csv(path)
        timings['Data Loading'].append(time.perf_counter() - t0)
        print(f"    Loaded: {signal.shape} ({signal.shape[0]/FS:.1f}s)")

        t0 = time.perf_counter()
        filtered = stage_filtering(signal)
        timings['Filtering'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        cleaned = stage_ica(filtered)
        timings['ICA'].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        segments = stage_segmentation(cleaned)
        timings['Segmentation'].append(time.perf_counter() - t0)
        segment_counts.append(len(segments))

        t0 = time.perf_counter()
        features = stage_feature_extraction(segments)
        timings['Feature Extraction'].append(time.perf_counter() - t0)

        print(f"    Segments: {len(segments)}, Features: {features.shape}")

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS: Per-Recording Average Runtime")
    print("=" * 70)

    header = f"{'Stage':<25} {'Mean (s)':>10} {'Std (s)':>10} {'Per-Seg (ms)':>14} {'% Total':>10}"
    print(header)
    print("-" * 70)

    total_mean = sum(np.mean(v) for v in timings.values())
    avg_segs = np.mean(segment_counts)

    results = []
    for stage, times in timings.items():
        mean_t = np.mean(times)
        std_t = np.std(times)
        per_seg = (mean_t / avg_segs) * 1000 if avg_segs > 0 else 0
        pct = (mean_t / total_mean) * 100 if total_mean > 0 else 0

        print(f"{stage:<25} {mean_t:>10.4f} {std_t:>10.4f} {per_seg:>14.2f} {pct:>9.1f}%")
        results.append({
            'Stage': stage,
            'Mean_sec': round(mean_t, 4),
            'Std_sec': round(std_t, 4),
            'Per_segment_ms': round(per_seg, 2),
            'Percent_total': round(pct, 1)
        })

    print("-" * 70)
    print(f"{'TOTAL':<25} {total_mean:>10.4f} {'':>10} {'':>14} {'100.0%':>10}")
    print(f"{'Avg segments/recording':<25} {avg_segs:>10.1f}")
    print(f"{'Recordings tested':<25} {len(recordings):>10}")

    n_total = 112
    print(f"\nFull dataset estimate ({n_total} recordings):")
    print(f"  Total: {total_mean * n_total:.1f}s ({total_mean * n_total / 60:.1f} min)")

    print(f"\nMemory estimates:")
    print(f"  Single recording (raw):  {45 * FS * N_CHANNELS * 8 / 1e6:.2f} MB")
    print(f"  Feature matrix (full):   {2464 * N_FEATURES_PER_CH * N_CHANNELS * 8 / 1e6:.2f} MB")
    print(f"  After ReliefF ({N_SELECTED}):    {2464 * N_SELECTED * 8 / 1e6:.2f} MB")
    print(f"  Reduction ratio:         {N_FEATURES_PER_CH * N_CHANNELS / N_SELECTED:.1f}x")

    with open('runtime_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: runtime_results.csv")


if __name__ == "__main__":
    n_subj = 5
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == '--n_subjects' and i + 1 < len(sys.argv):
                n_subj = int(sys.argv[i + 1])

    run_timing_analysis(n_subjects=n_subj)