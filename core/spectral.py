# core/spectral.py
import numpy as np
import librosa
import librosa.feature as librosa_features
from typing import Dict, Any, List, Tuple

def compute_stft(y: np.ndarray, n_fft: int, hop_length: int, window: str = "hann") -> np.ndarray:
    """Zwraca kompleksowy STFT (f × t)."""
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window, center=True)

def compute_spectral_features(y: np.ndarray,
                              sr: int,
                              n_fft: int,
                              hop_length: int,
                              n_mfcc: int = 13,
                              n_mels: int = 128,
                              roll_percent: float = 0.85) -> List[Dict[str, Any]]:
    """
    Oblicza cechy spektralne i MFCC dla całego sygnału.
    Zwraca listę słowników (po jednej pozycji na ramkę) z polami:
      time, spectral_centroid, spectral_bandwidth, rolloff, flatness,
      spectral_contrast (lista), chroma (lista), mfcc (lista), mfcc_delta (lista)
    """
    # 1) STFT (absolutna amplituda)
    S_complex = compute_stft(y, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(S_complex)  # magnituda (f × t)

    # 2) podstawowe cechy (zwracane jako (1 × t) arrays)
    centroid = librosa_features.spectral.spectral_centroid(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length).squeeze()
    bandwidth = librosa_features.spectral.spectral_bandwidth(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length).squeeze()
    rolloff = librosa_features.spectral.spectral_rolloff(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=roll_percent).squeeze()
    flatness = librosa_features.spectral.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length).squeeze()
    contrast = librosa_features.spectral.spectral_contrast(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)  # shape (n_bands+1, t)
    chroma = librosa_features.spectral.chroma_stft(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length)  # (12, t)

    # 3) MFCC
    mfcc = librosa_features.spectral.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)

    # 4) times
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

    # 5) złożenie wyników per-frame
    features_per_frame: List[Dict[str, Any]] = []
    for i, t in enumerate(times):
        frame_dict: Dict[str, Any] = {
            "time": float(t),
            # podstawowe cechy (liczby)
            "spectral_centroid": float(centroid[i]) if i < centroid.shape[0] else None,
            "spectral_bandwidth": float(bandwidth[i]) if i < bandwidth.shape[0] else None,
            "spectral_rolloff": float(rolloff[i]) if i < rolloff.shape[0] else None,
            "spectral_flatness": float(flatness[i]) if i < flatness.shape[0] else None,
            # spectral_contrast jako lista (n_bands+1)
            "spectral_contrast": [float(x) for x in contrast[:, i].tolist()] if i < contrast.shape[1] else [],
            # chroma jako lista 12 elementów
            "chroma": [float(x) for x in chroma[:, i].tolist()] if i < chroma.shape[1] else [],
            # MFCC oraz delta (listy)
            "mfcc": [float(x) for x in mfcc[:, i].tolist()] if i < mfcc.shape[1] else [],
            "mfcc_delta": [float(x) for x in mfcc_delta[:, i].tolist()] if i < mfcc_delta.shape[1] else []
        }
        features_per_frame.append(frame_dict)

    return features_per_frame


# --- opcjonalne funkcje pomocnicze do normalizacji/podsumowań ---

def compute_feature_track_stats(feature_list: List[Dict[str, Any]], keys: List[str]):
    """
    Oblicza mean/std per-track dla wybranych kluczy liczbowych (np. spectral_centroid).
    Zwraca dict {key: {"mean":..., "std":...}}
    """
    stats = {}
    for k in keys:
        vals = np.array([f[k] for f in feature_list if f.get(k) is not None], dtype=float)
        if vals.size == 0:
            stats[k] = {"mean": None, "std": None}
        else:
            stats[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return stats

def zscore_normalize(feature_list: List[Dict[str, Any]], key: str):
    """
    Zwraca nową listę z dodanym polem f'{key}_z' będącym z-score.
    """
    vals = np.array([f[key] for f in feature_list], dtype=float)
    mu, sigma = np.mean(vals), np.std(vals)
    for f in feature_list:
        f[f"{key}_z"] = None if sigma == 0 else float((f[key] - mu) / sigma)
    return feature_list
