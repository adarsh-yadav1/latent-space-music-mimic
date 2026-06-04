"""M2 - Source separation using librosa HPSS + frequency-band splitting.

No Demucs (too heavy to install in 24h). Uses:
  - HPSS: separates harmonic (melody/harmony) vs percussive (drums)
  - Frequency band splits: bass (<250Hz), mid (250-4kHz), high (>4kHz)

Produces 4 pseudo-stems: drums, bass, melody, other
"""
import numpy as np
import librosa


def separate_stems(y: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """
    Separate input mono audio into 4 stems.
    Returns dict: {drums, bass, melody, other} each as np.ndarray at same sr.
    """
    # HPSS: harmonic/percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=3.0)

    # STFT for frequency-band splitting of harmonic content
    D = librosa.stft(y_harmonic)
    freqs = librosa.fft_frequencies(sr=sr)

    # Frequency masks
    bass_mask = freqs < 250
    mid_mask = (freqs >= 250) & (freqs < 4000)
    high_mask = freqs >= 4000

    def apply_mask(D, mask):
        D_masked = np.zeros_like(D)
        D_masked[mask, :] = D[mask, :]
        return librosa.istft(D_masked, length=len(y))

    bass = apply_mask(D, bass_mask)
    melody = apply_mask(D, mid_mask)
    other = apply_mask(D, high_mask)

    # Normalize each stem
    def norm(x):
        peak = np.max(np.abs(x))
        return x / peak if peak > 1e-6 else x

    return {
        "drums": norm(y_percussive),
        "bass": norm(bass),
        "melody": norm(melody),
        "other": norm(other),
    }
