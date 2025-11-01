import numpy as np
import librosa
import librosa.feature.spectral as spectral

import config.config as config

VALID_FFT_MODES = {
    "raw": "surowe amplitudy FFT",
    "rms_normalized": "normalizacja względem RMS",
    "power_normalized": "normalizacja mocy pasm",
    "log_db": "skala logarytmiczna (dB)",
    "rms_weighted": "amplitudy ważone RMS-em"
}

def compute_fft_for_frames(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 1024):
    """
    Oblicza krótkoczasową transformatę Fouriera (STFT) dla sygnału.
    Zwraca:
        - amplitudy (macierz: częstotliwość × czas)
        - wektor częstotliwości
        - wektor czasów
    """
    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length, window='hann', center=True)
    magnitude = np.abs(stft)  # moduł zespolonych wartości -> amplitudy
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=hop_length)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    return magnitude, freqs, times, rms

def apply_fft_mode(magnitude: np.ndarray, rms_per_frame: np.ndarray, fft_mode: str):
    """
    Modyfikuje macierz amplitud STFT w zależności od trybu FFT.
    """
    eps = 1e-10

    if fft_mode not in VALID_FFT_MODES:
        raise ValueError(f"Nieznany tryb FFT: {fft_mode}. Dozwolone: {list(VALID_FFT_MODES.keys())}")

    if fft_mode == "raw":
        return magnitude

    elif fft_mode == "rms_normalized":
        # dzielimy każdą kolumnę przez RMS danej ramki
        return magnitude / (rms_per_frame[np.newaxis, :] + eps)

    elif fft_mode == "power_normalized":
        power = magnitude ** 2
        return power / (np.sum(power, axis=0, keepdims=True) + eps)

    elif fft_mode == "log_db":
        return 20 * np.log10(magnitude + eps)

    elif fft_mode == "rms_weighted":
        return magnitude * rms_per_frame[np.newaxis, :]

    return magnitude



def compute_band_energies(stft_matrix: np.ndarray, sr: int, bands: dict, n_fft: int = 2048):
    """
    Sumuje energie w określonych pasmach częstotliwości.
    Zwraca słownik {band_name: [lista energii w czasie]}.
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power_spectrum = np.square(stft_matrix)

    band_energies = {}
    for band_name, (low, high) in bands.items():
        idx = np.where((freqs >= low) & (freqs < high))[0]
        band_energy = np.sum(power_spectrum[idx, :], axis=0)
        band_energies[band_name] = band_energy.tolist()

    return band_energies


def compute_fft_peaks(stft_matrix: np.ndarray, freqs: np.ndarray, top_n: int = 5):
    """
    Dla każdej ramki wybiera N najsilniejszych częstotliwości (peaków).
    Zwraca listę list: [ [ (freq, amplitude), ... ], ... ]
    """
    peaks_per_frame = []
    magnitude_T = stft_matrix.T  # iterujemy po ramach (czas w wierszach)

    for frame in magnitude_T:
        top_indices = np.argsort(frame)[-top_n:][::-1]
        peaks = [(float(freqs[i]), float(frame[i])) for i in top_indices]
        peaks_per_frame.append(peaks)

    return peaks_per_frame


def analyze_fft(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 1024, fft_mode: str = config.FFT_MODE):
    """
    Analizuje sygnał audio w dziedzinie częstotliwości.
    Zwraca listę słowników z polami:
        time, bands, fft_peaks
    """
    bands = config.BANDS
    print(f"Analyzing FFT for type: {fft_mode}")
    magnitude, freqs, times, rms_per_frame = compute_fft_for_frames(y, sr, frame_length, hop_length)

    magnitude = apply_fft_mode(magnitude, rms_per_frame, fft_mode)

    #TODO: ability to use BANDS or BANDS_SIMPLE argument in a proper way

    band_energies = compute_band_energies(magnitude, sr, bands, n_fft=frame_length)
    fft_peaks = compute_fft_peaks(magnitude, freqs, top_n=5)

    results = []
    for i, t in enumerate(times):
        frame_data = {
            "time": float(t),
            "bands": {band: float(band_energies[band][i]) for band in bands},
            "fft_peaks": fft_peaks[i]
        }
        results.append(frame_data)

    return results
