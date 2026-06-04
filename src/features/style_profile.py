"""StyleProfile — typed container for all extracted style features."""
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np
import json


@dataclass
class StyleProfile:
    stem: str                          # "drums" | "bass" | "melody" | "other"
    bpm: float = 120.0
    beat_strength: float = 0.5        # avg onset strength
    groove_feel: str = "straight"     # "straight" | "swing" | "shuffle"

    # Timbre
    mfcc_mean: list = field(default_factory=list)   # (13,)
    mfcc_std: list = field(default_factory=list)    # (13,)
    spectral_centroid_mean: float = 2000.0
    spectral_bandwidth_mean: float = 1500.0
    spectral_rolloff_mean: float = 3500.0
    zero_crossing_rate: float = 0.05

    # Dynamics
    rms_mean: float = 0.05
    rms_std: float = 0.02
    dynamic_range_db: float = 20.0

    # Melody (only for melody stem)
    f0_mean: Optional[float] = None
    f0_std: Optional[float] = None
    pitch_range_semitones: Optional[float] = None
    melodic_density: float = 0.5      # notes per beat (estimated)

    # Phrasing
    avg_phrase_length_s: float = 4.0
    phrase_count: int = 4

    # Key / mode
    key: str = "C"
    mode: str = "major"              # "major" | "minor"
    chroma_vector: list = field(default_factory=list)   # (12,)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_text_prompt(self) -> str:
        """Convert style profile to a text description for MusicGen conditioning."""
        genre_hint = self._infer_genre_hint()
        mode_str = self.mode
        energy = "energetic" if self.rms_mean > 0.07 else "calm" if self.rms_mean < 0.03 else "moderate"
        density = "sparse" if self.melodic_density < 0.5 else "dense" if self.melodic_density > 2.0 else "flowing"
        brightness = "bright" if self.spectral_centroid_mean > 3000 else "warm" if self.spectral_centroid_mean < 1500 else "balanced"
        tempo_feel = "slow" if self.bpm < 80 else "fast" if self.bpm > 140 else "mid-tempo"
        groove = f"{self.groove_feel} groove" if self.stem == "drums" else ""

        parts = [
            f"{genre_hint}",
            f"{tempo_feel} at {self.bpm:.0f} BPM",
            f"{energy} {brightness} {self.stem}",
            f"{density} phrasing",
            f"{mode_str} tonality in {self.key}",
        ]
        if groove:
            parts.append(groove)
        return ", ".join(p for p in parts if p)

    def _infer_genre_hint(self) -> str:
        """Heuristic genre inference from features."""
        # Swing feel → jazz
        if self.groove_feel == "swing":
            return "jazz"
        # High BPM + high centroid → electronic
        if self.bpm > 128 and self.spectral_centroid_mean > 3500:
            return "electronic"
        # Low centroid, warm → classical or ambient
        if self.spectral_centroid_mean < 1500:
            return "ambient classical"
        # Default
        return "instrumental"
