import librosa

SR = None

def load_audio(path: str, sr=SR):
    """
    Wczytuje plik audio i zwraca (y, sr).
    y: 1D numpy float32 (mono) - jeżeli plik stereo, zostanie zsumowany/zmieszany na mono
    """

    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr