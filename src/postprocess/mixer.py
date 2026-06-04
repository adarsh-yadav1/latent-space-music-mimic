"""M8 - Post-processing: mix stems, normalize, light reverb."""
import numpy as np
from scipy.signal import fftconvolve


SR = 22050

# Stem mix levels (relative amplitudes)
STEM_LEVELS = {
    "drums":  0.7,
    "bass":   0.8,
    "melody": 1.0,
    "other":  0.5,
}


def mix_stems(stems: dict[str, np.ndarray]) -> np.ndarray:
    """Mix all stem arrays into a single mono signal."""
    max_len = max(len(v) for v in stems.values())
    mix = np.zeros(max_len, dtype=np.float32)
    for name, audio in stems.items():
        gain = STEM_LEVELS.get(name, 0.7)
        padded = np.pad(audio, (0, max_len - len(audio)))
        mix += padded * gain
    return mix


def apply_reverb(y: np.ndarray, sr: int = SR, room_scale: float = 0.3) -> np.ndarray:
    """Synthetic reverb via a simple exponential impulse response."""
    decay_time = 0.5 + room_scale * 1.5   # 0.5 to 2 seconds
    ir_len = int(decay_time * sr)
    t = np.arange(ir_len) / sr
    ir = np.random.randn(ir_len) * np.exp(-t * (6.9 / decay_time))
    ir[0] = 1.0   # Direct sound
    ir /= np.sum(np.abs(ir)) + 1e-9

    wet = fftconvolve(y, ir)[:len(y)]
    return (y * 0.7 + wet * 0.3).astype(np.float32)


def normalize_loudness(y: np.ndarray, target_lufs: float = -14.0) -> np.ndarray:
    """RMS-based loudness normalization."""
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-9:
        return y
    target_rms = 10 ** (target_lufs / 20.0)
    gain = target_rms / rms
    return np.clip(y * gain, -1.0, 1.0).astype(np.float32)


def postprocess(stems: dict[str, np.ndarray], sr: int = SR,
                room_scale: float = 0.3) -> np.ndarray:
    """Full post-processing pipeline."""
    mix = mix_stems(stems)
    mix = apply_reverb(mix, sr, room_scale)
    mix = normalize_loudness(mix)
    return mix
