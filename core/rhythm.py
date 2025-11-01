# core/rhythm.py
import numpy as np
import librosa
import librosa.feature.rhythm as rhythm
from typing import List, Dict, Any

def compute_onset_envelope(y, sr, hop_length):
    """Zwraca onset envelope (spectral flux)."""
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=hop_length)
    return oenv, times

def detect_onsets(y, sr, hop_length, backtrack=False):
    """
    Wykrywa onsety (w jednostce czasu [s] jeśli units='time' lub frames).
    Jeżeli backtrack=True, dopasowuje onsets do najbliższego lokalnego minimum energii.
    """
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr,
                                              hop_length=hop_length, units='frames', backtrack=False)
    if backtrack:
        onset_frames = librosa.onset.onset_backtrack(onset_frames, oenv)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return onset_times.tolist(), onset_frames.tolist(), oenv

def estimate_tempo_global(y=None, sr=22050, onset_envelope=None, hop_length=512):
    """
    Zwraca estymowane tempo globalne (BPM) przy użyciu librosa.beat.tempo
    """
    tempo = rhythm.tempo(y=y, sr=sr, onset_envelope=onset_envelope, hop_length=hop_length)
    # tempo może zwrócić ndarray — bierzemy pierwszą wartość (mono)
    return float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)

def compute_plp(y, sr, hop_length):
    """
    Predominant Local Pulse (PLP) — krzywa lokalnego tempa / pulsu.
    Zwraca plp_curve (w jednostkach frames) i odpowiadające czasy.
    """
    # librosa.beat.plp expects onset_envelope or y; we pass y and hop_length
    plp_curve = librosa.beat.plp(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(plp_curve)), sr=sr, hop_length=hop_length)
    return plp_curve.tolist(), times.tolist()

def beat_track(y, sr, hop_length, start_bpm=120.0, tightness=100):
    """
    Beat tracking returns (tempo, beat_times, beat_frames)
    tempo: estimated global bpm
    """
    print("Beat tracking...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=None,
                                                 hop_length=hop_length, start_bpm=start_bpm, tightness=tightness)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    return float(tempo), beat_times.tolist(), beat_frames.tolist()
