"""M1 - Audio ingestion, validation, and normalization."""
import os
import numpy as np
import soundfile as sf
import librosa


TARGET_SR = 22050
TARGET_DURATION = 30.0
TARGET_LUFS = -14.0


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio file to mono numpy array at TARGET_SR."""
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True, duration=TARGET_DURATION)
    return y, sr


def normalize_lufs(y: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """Approximate loudness normalization (RMS-based, simple proxy for LUFS)."""
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-9:
        return y
    # Convert target LUFS to approximate RMS target
    target_rms = 10 ** (target_lufs / 20.0)
    gain = target_rms / rms
    return np.clip(y * gain, -1.0, 1.0)


def pad_or_trim(y: np.ndarray, sr: int, duration: float = TARGET_DURATION) -> np.ndarray:
    """Ensure audio is exactly `duration` seconds."""
    target_len = int(duration * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def ingest(path: str) -> tuple[np.ndarray, int]:
    """Full ingestion pipeline: load → trim → normalize."""
    y, sr = load_audio(path)
    y = pad_or_trim(y, sr)
    y = normalize_lufs(y, sr)
    return y, sr


def save_audio(y: np.ndarray, sr: int, path: str):
    """Save numpy array to WAV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr)
