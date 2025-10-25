import numpy as np
import librosa

from config.config import BANDS, BANDS_SIMPLE, PEAK_COUNT

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
    return magnitude, freqs, times


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


def analyze_fft(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 1024):
    """
    Analizuje sygnał audio w dziedzinie częstotliwości.
    Zwraca listę słowników z polami:
        time, bands, fft_peaks
    """
    magnitude, freqs, times = compute_fft_for_frames(y, sr, frame_length, hop_length)

    #TODO: ability to use BANDS or BANDS_SIMPLE argument in a proper way
    bands = BANDS

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
