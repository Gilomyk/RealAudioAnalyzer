# config/config.py

# --- Ogólne parametry czasu ---
FRAME_DURATION = 0.05   # s (50 ms)
HOP_DURATION = 0.025     # s (25 ms = 50% overlap)

# --- FFT ---
FFT_N_FFT = None  # None -> automatycznie = frame_len
PEAK_COUNT = 5           # ile pików zachować z FFT
WINDOW_TYPE = "hann"     # okno FFT (np. "hann", "hamming")

# --- Podział na pasma (Hz) ---
BANDS = {
    "sub_bass": (20, 60),
    "bass": (60, 250),
    "low_mid": (250, 500),
    "mid": (500, 2000),
    "high_mid": (2000, 4000),
    "presence": (4000, 6000),
    "brilliance": (6000, 12000),
    "air": (12000, 20000)
}

BANDS_SIMPLE = {
    "bass": (20, 250),
    "mid": (250, 4000),
    "treble": (4000, 12000)
}


# --- Ogólne ścieżki domyślne ---
DEFAULT_OUTPUT_PATH = "outputs/analysis_full.json"
DEFAULT_INPUT_EXAMPLE = "examples/slowa.wav"

# --- Debug / logowanie ---
VERBOSE = True