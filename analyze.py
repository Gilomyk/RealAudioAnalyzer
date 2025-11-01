import argparse
from pathlib import Path

from core.io_utils import load_audio, save_json, merge_frame_data, merge_spectral_features, merge_frame_data_with_rhythm
from core.frames import analyze_rms_from_signal  # funkcja zwracająca dict
from core.compute_fft import analyze_fft
from core.rhythm import detect_onsets, estimate_tempo_global, compute_plp, beat_track
from core.spectral import compute_spectral_features
from analyze_stats import append_stats_to_json

import config.config as cfg

# --- Funkcje pomocnicze ---
def validate_fft_mode(value: str):
    """Sprawdza, czy podany tryb FFT jest poprawny."""
    if value not in cfg.VALID_FFT_MODES.keys():
        valid = ", ".join([str(k) for k in cfg.VALID_FFT_MODES.keys() if k is not None])
        raise argparse.ArgumentTypeError(f"Niepoprawny tryb FFT: '{value}'. Dozwolone: {valid}")
    return value

# --- Jeśli uruchamiasz skrypt bezpośrednio ---
def main():
    parser = argparse.ArgumentParser(description="Audio analysis pipeline (RMS -> FFT -> tempo/beat/onset -> spectral -> save JSON -> optional: apply stats)")
    parser.add_argument("--input", required=True, help="Ścieżka do pliku audio (np. .wav)")
    parser.add_argument("--fft_mode", type=validate_fft_mode, help="Tryb analizy FFT. Dostępne: " + ", ".join([str(k) for k in cfg.VALID_FFT_MODES.keys() if k is not None]))
    parser.add_argument("--frame", type=float, help="Długość ramki (s) dla RMS i jako bazowe n_fft dla FFT", default=cfg.FRAME_DURATION)
    parser.add_argument("--hop", type=float, help="Hop (s) dla RMS i FFT", default=cfg.HOP_DURATION)
    parser.add_argument("--n_peaks", type=int, help="Ile pików FFT zwrócić (per frame)", default=cfg.PEAK_COUNT)
    parser.add_argument("--stats", action="store_true", help="Czy liczymy statystyki dla pliku audio")
    args = parser.parse_args()

    # Wczytywanie audio
    print(f"Loading audio: {args.input}")
    y, sr = load_audio(args.input, sr=None)

    #  Obliczanie RMS (z ramkami w sekundach)
    rms_result = analyze_rms_from_signal(y, sr, frame_duration=args.frame, hop_duration=args.hop)

    #  Przygotowanie parametrów FFT (przeliczamy frame/hop w s -> próbki)
    frame_len_samples = int(round(args.frame * sr))
    hop_len_samples = int(round(args.hop * sr))
    n_fft = frame_len_samples if cfg.FFT_N_FFT is None else cfg.FFT_N_FFT

    if n_fft < 16:
        n_fft = 2048

    # Analiza FFT
    fft_frames = analyze_fft(y, sr, frame_length=n_fft, hop_length=hop_len_samples, fft_mode=args.fft_mode)

    # Połączenie RMS i FFT
    merged_frames = merge_frame_data(rms_result["frames"], fft_frames)


    # Obliczanie tempa, detekcja onset i uderzeń
    onset_times, onset_frames, oenv = detect_onsets(y, sr, hop_len_samples, backtrack=True)
    global_tempo = estimate_tempo_global(y=y, sr=sr, onset_envelope=oenv, hop_length=hop_len_samples)
    plp_curve, plp_times = compute_plp(y, sr, hop_len_samples)
    tempo_est, beat_times, beat_frames = beat_track(y, sr, hop_len_samples)

    merged_frames = merge_frame_data_with_rhythm(merged_frames, onset_times, beat_times, plp_times, plp_curve)

    spectral_data = compute_spectral_features(
        y, sr,
        n_fft=n_fft,
        hop_length=hop_len_samples,
        n_mfcc=13,
        n_mels=128,
        roll_percent=0.85
    )

    merged_frames = merge_spectral_features(merged_frames, spectral_data)

    output = {
        "source": str(Path(args.input).resolve()),
        "sr": sr,
        "global_tempo_bpm": global_tempo,
        "frames": merged_frames,
        "onset_frames": onset_frames,
        "beat_frames": beat_frames,
    }

    # 6) Zapis do pliku
    output_path = save_json(output, args.input, fft_mode=args.fft_mode)
    print(f"Saved results to {output_path}")
    print(f"RMS frames: {len(rms_result['frames'])}, FFT frames: {len(fft_frames)}")

    if args.stats:
        append_stats_to_json(output_path)


if __name__ == "__main__":
    main()