"""
Konfigürasyon yükleyici.

Kullanım:
    from configs.config_loader import load_config, get_active_channels
    cfg = load_config()                  # config.yaml'ı sözlük olarak döner
    ch_names, ch_indices = get_active_channels(cfg)  # excluded kanallar çıkarılmış
"""

import os
from pathlib import Path
from typing import Tuple, List

import yaml


_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """YAML konfigürasyon dosyasını oku ve sözlük olarak döndür."""
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_active_channels(cfg: dict) -> Tuple[List[str], List[int]]:
    """
    Aktif kanal isimlerini ve indekslerini döndür.

    excluded_channels listesindeki kanallar çıkarılır.

    Returns:
        channel_names: List[str]  — aktif kanal isimleri
        channel_indices: List[int] — CSV sütun indeksleri (0-based)
    """
    all_channels = cfg["eeg"]["channel_names"]
    excluded = set(cfg["eeg"].get("excluded_channels", []))

    channel_names = []
    channel_indices = []
    for idx, ch in enumerate(all_channels):
        if ch not in excluded:
            channel_names.append(ch)
            channel_indices.append(idx)

    if not channel_names:
        raise ValueError("Tüm kanallar excluded_channels'a eklenmiş — aktif kanal yok!")

    return channel_names, channel_indices


def get_segment_params(cfg: dict) -> dict:
    """Segmentasyon parametrelerini hesapla ve doğrula."""
    sr = cfg["eeg"]["sampling_rate"]
    win_sec = cfg["segmentation"]["window_seconds"]
    overlap = cfg["segmentation"]["overlap_ratio"]

    window_samples = int(win_sec * sr)
    step_samples = int(window_samples * (1 - overlap))

    # Temel doğrulamalar
    assert window_samples > 0, f"window_samples sıfır olamaz (sr={sr}, win={win_sec}s)"
    assert step_samples > 0, f"step_samples sıfır olamaz (overlap={overlap})"
    assert 0.0 <= overlap < 1.0, f"overlap_ratio [0, 1) aralığında olmalı, değer: {overlap}"

    return {
        "window_samples": window_samples,
        "step_samples": step_samples,
        "window_seconds": win_sec,
        "overlap_ratio": overlap,
        "sampling_rate": sr,
    }


if __name__ == "__main__":
    # Hızlı test
    cfg = load_config()
    ch_names, ch_idx = get_active_channels(cfg)
    seg = get_segment_params(cfg)

    print(f"Aktif kanallar ({len(ch_names)}): {ch_names}")
    print(f"Kanal indeksleri: {ch_idx}")
    print(f"Segmentasyon: {seg}")
    print(f"Excluded: {cfg['eeg']['excluded_channels']}")
