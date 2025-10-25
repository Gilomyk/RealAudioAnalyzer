import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import librosa

SR = None

def load_audio(path: str, sr=SR):
    """
    Wczytuje plik audio i zwraca (y, sr).
    y: 1D numpy float32 (mono) - macierz zwracająca wartości sygnału w każdej próbce
    sr: Częstotliwosc próbkowania (sampling rate) - default 22050 Hz
    """

    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

def merge_frame_data(rms_frames, fft_frames, time_tolerance=1e-3):
    merged = []
    j = 0  # indeks dla FFT
    for rms_frame in rms_frames:
        t_rms = rms_frame["time"]
        # szukamy najbliższego fft_frame o zbliżonym czasie
        while j + 1 < len(fft_frames) and abs(fft_frames[j + 1]["time"] - t_rms) < abs(fft_frames[j]["time"] - t_rms):
            j += 1
        if abs(fft_frames[j]["time"] - t_rms) < time_tolerance:
            merged.append({
                "time": t_rms,
                "rms": rms_frame["rms"],
                "bands": fft_frames[j]["bands"],
                "fft_peaks": fft_frames[j]["fft_peaks"]
            })
        else:
            # jeśli nie znaleziono dopasowania – tylko RMS
            merged.append({
                "time": t_rms,
                "rms": rms_frame["rms"],
                "bands": None,
                "fft_peaks": []
            })
    return merged


def save_json(obj: Dict[str, Any], out_path: str, indent: int = 2) -> None:
    """
    Zapisuje strukturę Python->JSON. Tworzy katalogi po drodze.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # JSON nie obsługuje numpy types bezpośrednio — konwertujemy float32/np.int64 -> native
    def _convert(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return o

    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_convert)