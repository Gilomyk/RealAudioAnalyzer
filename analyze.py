"""
analyze.py
Podstawowy etap: dzielenie na ramki i obliczanie RMS.
Uruchamianie:
    python analyze.py path/to/file.wav --frame 0.05 --sr 44100 --out out.json
"""
from core.frames import analyze_rms

# --- Parametry domyślne (możesz je zmienić) ---
FRAME_DURATION = 0.05   # sekundy (50 ms)
HOP_DURATION = 0.025    # sekundy (hop = 25 ms => 50% overlap)
SR = None               # jeśli None => użyj oryginalnego sr z pliku

# --- Funkcje pomocnicze ---



# --- Jeśli uruchamiasz skrypt bezpośrednio ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analiza audio — etap 1 (RMS per frame)")
    parser.add_argument("input", help="Ścieżka do pliku .wav (lub inny format obsługiwany przez librosa)")
    parser.add_argument("--out", help="Ścieżka do pliku JSON z wynikami", default="outputs/analysis_rms.json")
    parser.add_argument("--frame", type=float, help="Długość ramki w sekundach", default=FRAME_DURATION)
    parser.add_argument("--hop", type=float, help="Hop (przesunięcie) w sekundach", default=HOP_DURATION)
    args = parser.parse_args()
    res = analyze_rms(args.input, args.out, frame_duration=args.frame, hop_duration=args.hop)
    print(f"Zapisano wyniki do {args.out} — liczba ramek: {len(res['frames'])}")