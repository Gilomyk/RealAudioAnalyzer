import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
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


def merge_frame_data_with_rhythm(frames: List[Dict[str, Any]],
                                 onset_times: List[float],
                                 beat_times: List[float],
                                 plp_times: List[float],
                                 plp_values: List[float],
                                 time_tolerance: float = 1e-3) -> List[Dict[str, Any]]:
    """
    Dla każdej ramki dodaje flagi is_onset, is_beat i interpoluje lokalne tempo (PLP)
    """
    merged = []
    plp_values = np.array(plp_values)
    plp_times = np.array(plp_times)

    for frame in frames:
        t = frame["time"]
        # is_onset
        is_onset = any(abs(t - ot) <= time_tolerance for ot in onset_times)
        # is_beat
        is_beat = any(abs(t - bt) <= time_tolerance for bt in beat_times)
        # local_tempo: interpolacja PLP
        if t <= plp_times[0]:
            local_tempo = float(plp_values[0])
        elif t >= plp_times[-1]:
            local_tempo = float(plp_values[-1])
        else:
            local_tempo = float(np.interp(t, plp_times, plp_values))

        frame_copy = frame.copy()
        frame_copy.update({
            "is_onset": is_onset,
            "is_beat": is_beat,
            "local_tempo": local_tempo
        })
        merged.append(frame_copy)
    return merged


def merge_spectral_features(frames: List[Dict[str, Any]],
                            spectral_frames: List[Dict[str, Any]],
                            time_tolerance: float = 1e-3) -> List[Dict[str, Any]]:
    """
    Scala cechy spektralne z ramkami RMS/FFT w oparciu o dopasowanie czasowe.
    Zakłada, że spectral_frames to lista słowników z kluczem 'time'.

    Args:
        frames: lista ramek (z RMS/FFT)
        spectral_frames: lista ramek zwrócona z compute_spectral_features()
        time_tolerance: maksymalna różnica czasów (sekundy), aby uznać ramki za odpowiadające sobie

    Returns:
        Lista ramek z dodanymi cechami spektralnymi.
    """
    merged = []

    # Wydobywamy czasy ramek spektralnych
    spectral_times = np.array([sf["time"] for sf in spectral_frames])

    for frame in frames:
        t = frame["time"]
        # Znajdź najbliższą ramkę spektralną względem czasu
        idx = np.argmin(np.abs(spectral_times - t))
        if abs(spectral_times[idx] - t) > time_tolerance:
            # jeśli zbyt daleko — nie łączymy
            merged.append(frame)
            continue

        frame_copy = frame.copy()

        # Pobierz dane spektralne z dopasowanej ramki
        sf = spectral_frames[idx]

        # Dodaj poszczególne cechy — dynamicznie (żeby nie zakładać z góry kluczy)
        for key, val in sf.items():
            if key == "time":
                continue
            frame_copy[key] = val

        merged.append(frame_copy)

    return merged




def save_json(obj: Dict[str, Any],
              input_path: str,
              fft_mode: str = "raw",
              out_dir: str = "output",
              suffix: str = "",
              indent: int = 2) -> Path:
    """
    Zapisuje strukturę Python->JSON. Tworzy katalogi po drodze.
    """
    input_name = Path(input_path).stem  # np. song.wav -> "song"
    file_name = f"analysis_{input_name}_{fft_mode}{suffix}.json"
    out_path = Path(out_dir) / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON nie obsługuje numpy types bezpośrednio — konwertujemy float32/np.int64 -> native
    def _convert(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return o

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_convert)

    return out_path
