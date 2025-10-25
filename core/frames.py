import json
from pathlib import Path
from typing import List, Dict

import numpy as np

from core.io_utils import load_audio

FRAME_DURATION = 0.05   # sekundy (50 ms)
HOP_DURATION = 0.025    # sekundy (hop = 25 ms => 50% overlap)

def frames_from_signal(y: np.ndarray, sr: int, frame_duration=FRAME_DURATION, hop_duration=HOP_DURATION):
    """
    Dzieli sygnał na ramki (bez okna — prosty slicing).
    Zwraca listę ramek oraz czas początku każdej ramki (w sekundach).
    """

    frame_len = int(round(frame_duration * sr))
    hop_len = int(round(hop_duration * sr))
    if frame_len <= 0 or hop_len <= 0:
        raise ValueError("frame_duration i hop_duration muszą być > 0 i odpowiednie do sr")

    frames = []
    times = []

    for start in range(0, max(1, len(y) - frame_len + 1), hop_len):
        frame = y[start:start + frame_len]
        frames.append(frame)
        times.append(start / sr)

    # Obsługa ostatniej ramki (jeśli krótsza niż frame_len, opcjonalnie dopadamy zero-paddingiem)
    if len(y) % hop_len != 0 and (len(y) - frame_len) < 0:
        # krótszy plik niż frame_len -> jedna ramka
        frames = [y]
        times = [0.0]

    elif (len(y) - frame_len) % hop_len != 0:
        # dodajemy ewentualnie ostatnią ramkę z dopelnieniem zerami
        last_start = ((len(y) - frame_len) // hop_len + 1) * hop_len
        if last_start < len(y):
            last_frame = y[last_start:last_start + frame_len]
            if len(last_frame) < frame_len:
                last_frame = np.pad(last_frame, (0, frame_len - len(last_frame)))
            frames.append(last_frame)
            times.append(last_start / sr)
    return frames, times

def rms_of_frame(frame: np.ndarray) -> float:
    """
    RMS = sqrt(mean(x^2))
    """
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))

def analyze_rms_from_signal(y: np.ndarray, sr: int, frame_duration: float = FRAME_DURATION, hop_duration: float = HOP_DURATION):
    """
    Analiza RMS bez operacji I/O.
    Zwraca dict z metadanymi i listą ramek: {"sr":..., "frame_duration":..., "hop_duration":..., "frames":[{"time":..., "rms":...}, ...]}
    """
    frames, times = frames_from_signal(y, sr, frame_duration, hop_duration)
    results: List[Dict] = []
    for frame, t in zip(frames, times):
        rms = rms_of_frame(frame)
        results.append({
            "time": float(t),
            "rms": float(rms)
        })
    return {"sr": sr, "frame_duration": frame_duration, "hop_duration": hop_duration, "frames": results}

def analyze_rms(path: str, out_json: str = None, frame_duration: float = FRAME_DURATION, hop_duration: float = HOP_DURATION):
    """
    Wrapper: wczytuje plik, deleguje do analyze_rms_from_signal i zwraca wyniki.
    Nie zapisuje pliku (zapisamy to centralnie w analyze.py / io_utils.save_json).
    """
    y, sr = load_audio(path)   # zakładam, że load_audio jest w core.io_utils i importowalne
    return analyze_rms_from_signal(y, sr, frame_duration, hop_duration)
