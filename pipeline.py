"""pipeline.py — Orchestrates all modules end-to-end.

Usage:
    python pipeline.py --input <audio_file> --output <output_wav> [--duration 300]
"""
import os
import sys
import time
import argparse
import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.ingest.audio_input import ingest, save_audio
from src.separation.separator import separate_stems
from src.features.extractor import extract
from src.generation.composer import generate_stem
from src.generation.coherence import apply_structure, crossfade_chunks
from src.postprocess.mixer import postprocess

SR = 22050
CHUNK_DURATION = 30.0   # seconds per chunk
OVERLAP = 3.0           # seconds of crossfade overlap


def run(input_path: str, output_path: str, target_duration: float = 300.0,
        seed: int = 42, verbose: bool = True) -> str:
    """
    Full pipeline:
      1. Ingest 30s clip
      2. Separate into stems
      3. Extract style features per stem
      4. Generate new audio per stem for target_duration
      5. Mix, apply structure, post-process
      6. Save output

    Returns path to output WAV.
    """
    def log(msg):
        if verbose:
            print(f"[pipeline] {msg}")

    t0 = time.time()

    # ── 1. Ingest ──────────────────────────────────────────────────────────
    log(f"Loading {input_path} ...")
    y, sr = ingest(input_path)
    log(f"  Loaded {len(y)/sr:.1f}s @ {sr}Hz")

    # ── 2. Separate ────────────────────────────────────────────────────────
    log("Separating stems (HPSS + frequency bands)...")
    stems = separate_stems(y, sr)
    log(f"  Stems: {list(stems.keys())}")

    # ── 3. Extract features ────────────────────────────────────────────────
    log("Extracting style features...")
    profiles = {}
    for name, stem_audio in stems.items():
        profiles[name] = extract(name, stem_audio, sr)
        p = profiles[name]
        log(f"  {name}: BPM={p.bpm:.1f}, Key={p.key}{p.mode[0].upper()}, "
            f"centroid={p.spectral_centroid_mean:.0f}Hz, groove={p.groove_feel}")
        log(f"    → prompt: \"{p.to_text_prompt()}\"")

    # ── 4. Generate ────────────────────────────────────────────────────────
    log(f"Generating {target_duration:.0f}s of new audio per stem...")
    generated_stems = {}

    for name, profile in profiles.items():
        log(f"  Generating {name}...")
        stem_audio = generate_stem(profile, target_duration, sr=SR, seed=seed + hash(name) % 1000)
        generated_stems[name] = stem_audio
        log(f"    Done: {len(stem_audio)/SR:.1f}s")

    # ── 5. Apply structure + post-process ─────────────────────────────────
    log("Applying structure and mixing...")
    for name in generated_stems:
        generated_stems[name] = apply_structure(generated_stems[name], SR, target_duration)

    # Estimate room scale from source dynamics
    source_dr = profiles.get("melody", profiles.get("other", list(profiles.values())[0]))
    room_scale = np.clip(source_dr.dynamic_range_db / 40.0, 0.1, 0.9)

    final_audio = postprocess(generated_stems, SR, room_scale=room_scale)
    log(f"  Mixed: {len(final_audio)/SR:.1f}s, peak={np.max(np.abs(final_audio)):.3f}")

    # ── 6. Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    sf.write(output_path, final_audio, SR)
    elapsed = time.time() - t0
    log(f"Saved to {output_path} ({elapsed:.1f}s elapsed)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Latent Space Music Mimic")
    parser.add_argument("--input",    required=True,  help="Input audio file (WAV/MP3, ~30s)")
    parser.add_argument("--output",   default="outputs/output.wav", help="Output WAV path")
    parser.add_argument("--duration", type=float, default=300.0, help="Target duration in seconds (default 300)")
    parser.add_argument("--seed",     type=int,   default=42,    help="Random seed")
    args = parser.parse_args()

    out = run(args.input, args.output, args.duration, args.seed, verbose=True)
    print(f"\nDone! Output: {out}")


if __name__ == "__main__":
    main()
