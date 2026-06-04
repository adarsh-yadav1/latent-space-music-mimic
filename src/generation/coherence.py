"""M6 - Coherence & Structure Manager.

Imposes macro-structure on 5-min output:
  intro (0-30s) → development (30s-3:30) → climax (3:30-4:30) → outro (4:30-5:00)
"""
import numpy as np


SR = 22050


def _energy_envelope(n_samples: int, section: str) -> np.ndarray:
    """Return a gain envelope for a given section."""
    t = np.linspace(0, 1, n_samples)
    if section == "intro":
        return np.linspace(0.2, 0.7, n_samples)
    elif section == "development":
        return 0.6 + 0.2 * np.sin(t * np.pi * 3)
    elif section == "climax":
        return np.linspace(0.8, 1.0, n_samples)
    elif section == "outro":
        return np.linspace(0.9, 0.1, n_samples)
    return np.ones(n_samples)


def get_section(t_sec: float, total_dur: float = 300.0) -> str:
    ratio = t_sec / total_dur
    if ratio < 0.1:
        return "intro"
    elif ratio < 0.7:
        return "development"
    elif ratio < 0.9:
        return "climax"
    else:
        return "outro"


def apply_structure(audio: np.ndarray, sr: int = SR, total_dur: float = 300.0) -> np.ndarray:
    """Apply section-based energy envelopes to full-length audio."""
    n = len(audio)
    envelope = np.ones(n, dtype=np.float32)

    sections = [
        ("intro",       0.0,  0.1),
        ("development", 0.1,  0.7),
        ("climax",      0.7,  0.9),
        ("outro",       0.9,  1.0),
    ]

    for name, start_ratio, end_ratio in sections:
        s = int(start_ratio * n)
        e = int(end_ratio * n)
        if e > s:
            envelope[s:e] = _energy_envelope(e - s, name)

    return (audio * envelope).astype(np.float32)


def crossfade_chunks(chunks: list[np.ndarray], overlap_samples: int) -> np.ndarray:
    """Overlap-add list of audio chunks with cosine crossfade."""
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    # Calculate total output length
    total = len(chunks[0])
    for c in chunks[1:]:
        total += len(c) - overlap_samples
    output = np.zeros(total, dtype=np.float32)

    # Fade windows
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap_samples)) ** 2
    fade_in  = np.cos(np.linspace(np.pi / 2, 0, overlap_samples)) ** 2

    pos = 0
    for i, chunk in enumerate(chunks):
        end = pos + len(chunk)
        if end > total:
            chunk = chunk[:total - pos]
            end = total

        if i == 0:
            output[pos:end] += chunk
        else:
            # Fade in the new chunk
            fade_len = min(overlap_samples, len(chunk))
            chunk_copy = chunk.copy()
            chunk_copy[:fade_len] *= fade_in[:fade_len]
            # Fade out the tail of what's already in the buffer
            o_start = pos
            o_end = min(o_start + fade_len, total)
            output[o_start:o_end] *= fade_out[:o_end - o_start]
            output[pos:end] += chunk_copy[:end - pos]

        pos += len(chunk) - overlap_samples

    return output[:total]
