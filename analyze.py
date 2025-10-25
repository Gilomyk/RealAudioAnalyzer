import argparse
from pathlib import Path

from core.io_utils import load_audio, save_json, merge_frame_data
from core.frames import analyze_rms_from_signal, analyze_rms  # funkcja zwracająca dict
from core.compute_fft import analyze_fft

import config.config as cfg

# --- Funkcje pomocnicze ---



# --- Jeśli uruchamiasz skrypt bezpośrednio ---
def main():
    parser = argparse.ArgumentParser(description="Audio analysis pipeline (RMS -> FFT -> save JSON)")
    parser.add_argument("input", help="Ścieżka do pliku audio (np. .wav)", default=cfg.DEFAULT_INPUT_EXAMPLE)
    parser.add_argument("--out", help="Ścieżka do pliku JSON z wynikami", default=cfg.DEFAULT_OUTPUT_PATH)
    parser.add_argument("--frame", type=float, help="Długość ramki (s) dla RMS i jako bazowe n_fft dla FFT", default=cfg.FRAME_DURATION)
    parser.add_argument("--hop", type=float, help="Hop (s) dla RMS i FFT", default=cfg.HOP_DURATION)
    parser.add_argument("--n_peaks", type=int, help="Ile pików FFT zwrócić (per frame)", default=cfg.PEAK_COUNT)
    args = parser.parse_args()

    # Wczytywanie audio
    y, sr = load_audio(args.input, sr=None)

    #  Obliczanie RMS (z ramkami w sekundach)
    rms_result = analyze_rms_from_signal(y, sr, frame_duration=args.frame, hop_duration=args.hop)
    # rms_result: {"sr": sr, "frame_duration":..., "hop_duration":..., "frames":[{"time":..., "rms":...}, ...]}

    #  Przygotowanie parametrów FFT (przeliczamy frame/hop w s -> próbki)
    frame_len_samples = int(round(args.frame * sr))
    hop_len_samples = int(round(args.hop * sr))
    n_fft = frame_len_samples if cfg.FFT_N_FFT is None else cfg.FFT_N_FFT

    if n_fft < 16:
        n_fft = 2048

    # Analiza FFT
    fft_frames = analyze_fft(y, sr, frame_length=n_fft, hop_length=hop_len_samples)

    merged_frames = merge_frame_data(rms_result["frames"], fft_frames)

    output = {
        "source": str(Path(args.input).resolve()),
        "sr": sr,
        "frames": merged_frames
    }

    # Tworzenie struktury JSON
    # output = {
    #     "source": str(Path(args.input).resolve()),
    #     "sr": sr,
    #     "rms_analysis": rms_result,          # cała struktura z ramkami RMS
    #     "fft_analysis": {
    #         "n_fft": n_fft,
    #         "hop_length": hop_len_samples,
    #         "frame_samples": frame_len_samples,
    #         "frames": fft_frames              # lista słowników z compute_fft.analyze_fft
    #     },
    #     # na przyszłość: tu można dopisać: "onsets":..., "beats":..., "mfcc":...
    # }

    # 6) Zapis do pliku
    save_json(output, args.out)
    print(f"Zapisano wyniki do {args.out}")
    print(f"RMS frames: {len(rms_result['frames'])}, FFT frames: {len(fft_frames)}")


if __name__ == "__main__":
    main()