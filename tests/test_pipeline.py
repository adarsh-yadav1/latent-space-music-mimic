"""tests/ — Smoke tests for all pipeline modules."""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SR = 22050
DUR = 5.0  # short duration for fast tests

@pytest.fixture
def synthetic_audio():
    """30-second synthetic audio clip."""
    n = int(30.0 * SR)
    t = np.linspace(0, 30.0, n)
    y = 0.5 * np.sin(2 * np.pi * 220 * t)
    y += 0.3 * np.sin(2 * np.pi * 440 * t)
    # Drum bursts
    for beat in np.arange(0, 30, 0.5):
        idx = int(beat * SR)
        if idx + int(0.05*SR) < n:
            y[idx:idx+int(0.05*SR)] += 0.4 * np.random.randn(int(0.05*SR))
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.8
    return y.astype(np.float32), SR


# ── Ingestion ──────────────────────────────────────────────────────────────

def test_ingest_normalize(tmp_path, synthetic_audio):
    import soundfile as sf
    from src.ingest.audio_input import ingest
    y, sr = synthetic_audio
    p = tmp_path / "test.wav"
    sf.write(str(p), y, sr)
    y2, sr2 = ingest(str(p))
    assert sr2 == SR
    assert len(y2) == int(30.0 * SR)
    assert np.max(np.abs(y2)) <= 1.0


# ── Separation ─────────────────────────────────────────────────────────────

def test_separation_returns_four_stems(synthetic_audio):
    from src.separation.separator import separate_stems
    y, sr = synthetic_audio
    stems = separate_stems(y, sr)
    assert set(stems.keys()) == {"drums", "bass", "melody", "other"}
    for name, s in stems.items():
        assert len(s) == len(y), f"{name} length mismatch"
        assert np.isfinite(s).all(), f"{name} has non-finite values"


# ── Feature Extraction ─────────────────────────────────────────────────────

def test_feature_extraction(synthetic_audio):
    from src.separation.separator import separate_stems
    from src.features.extractor import extract
    y, sr = synthetic_audio
    stems = separate_stems(y, sr)
    for name, s in stems.items():
        p = extract(name, s, sr)
        assert p.stem == name
        assert 20 <= p.bpm <= 300, f"{name} BPM out of range: {p.bpm}"
        assert p.key in ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        assert p.mode in ("major", "minor")
        assert len(p.mfcc_mean) == 13
        assert len(p.chroma_vector) == 12
        assert isinstance(p.to_text_prompt(), str)
        assert len(p.to_text_prompt()) > 10


# ── StyleProfile text prompt ───────────────────────────────────────────────

def test_style_profile_prompt():
    from src.features.style_profile import StyleProfile
    p = StyleProfile(stem="melody", bpm=132.0, key="A", mode="minor",
                     groove_feel="swing", spectral_centroid_mean=3200.0,
                     rms_mean=0.08, melodic_density=1.5)
    prompt = p.to_text_prompt()
    assert "132" in prompt
    assert "minor" in prompt
    assert "A" in prompt


# ── Generation ─────────────────────────────────────────────────────────────

def test_generate_melody_stem(synthetic_audio):
    from src.separation.separator import separate_stems
    from src.features.extractor import extract
    from src.generation.composer import generate_stem
    y, sr = synthetic_audio
    stems = separate_stems(y, sr)
    profile = extract("melody", stems["melody"], sr)
    audio = generate_stem(profile, target_duration=DUR, sr=SR, seed=0)
    assert len(audio) == int(DUR * SR)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_generate_drums_stem(synthetic_audio):
    from src.separation.separator import separate_stems
    from src.features.extractor import extract
    from src.generation.composer import generate_stem
    y, sr = synthetic_audio
    stems = separate_stems(y, sr)
    profile = extract("drums", stems["drums"], sr)
    audio = generate_stem(profile, target_duration=DUR, sr=SR, seed=1)
    assert len(audio) == int(DUR * SR)
    assert np.isfinite(audio).all()


def test_generate_bass_stem(synthetic_audio):
    from src.separation.separator import separate_stems
    from src.features.extractor import extract
    from src.generation.composer import generate_stem
    y, sr = synthetic_audio
    stems = separate_stems(y, sr)
    profile = extract("bass", stems["bass"], sr)
    audio = generate_stem(profile, target_duration=DUR, sr=SR, seed=2)
    assert len(audio) == int(DUR * SR)
    assert np.isfinite(audio).all()


# ── Coherence ──────────────────────────────────────────────────────────────

def test_apply_structure():
    from src.generation.coherence import apply_structure
    audio = np.ones(int(300 * SR), dtype=np.float32)
    out = apply_structure(audio, SR, 300.0)
    assert len(out) == len(audio)
    assert np.isfinite(out).all()
    # Intro should be quieter than climax
    intro_rms = np.sqrt(np.mean(out[:int(15*SR)]**2))
    climax_rms = np.sqrt(np.mean(out[int(220*SR):int(270*SR)]**2))
    assert climax_rms > intro_rms


def test_crossfade_chunks():
    from src.generation.coherence import crossfade_chunks
    chunks = [np.ones(int(5*SR), dtype=np.float32) for _ in range(3)]
    overlap = int(0.5*SR)
    out = crossfade_chunks(chunks, overlap)
    expected_len = 3*int(5*SR) - 2*overlap
    assert abs(len(out) - expected_len) <= 2


# ── Post-processing ────────────────────────────────────────────────────────

def test_mix_and_postprocess(synthetic_audio):
    from src.separation.separator import separate_stems
    from src.features.extractor import extract
    from src.generation.composer import generate_stem
    from src.postprocess.mixer import postprocess
    y, sr = synthetic_audio
    stems_raw = separate_stems(y, sr)
    profiles = {n: extract(n, s, sr) for n, s in stems_raw.items()}
    generated = {n: generate_stem(p, DUR, SR, seed=99) for n, p in profiles.items()}
    out = postprocess(generated, SR, room_scale=0.3)
    assert len(out) == int(DUR * SR)
    assert np.max(np.abs(out)) <= 1.0
    assert np.isfinite(out).all()


# ── Full pipeline ──────────────────────────────────────────────────────────

def test_full_pipeline_e2e(tmp_path, synthetic_audio):
    import soundfile as sf
    from pipeline import run
    y, sr = synthetic_audio
    input_path = str(tmp_path / "input.wav")
    output_path = str(tmp_path / "output.wav")
    sf.write(input_path, y, sr)

    result = run(input_path, output_path, target_duration=10.0, seed=7, verbose=False)
    assert os.path.exists(result)
    y_out, sr_out = sf.read(result)
    assert sr_out == SR
    duration = len(y_out) / sr_out
    assert abs(duration - 10.0) < 0.5
    assert np.max(np.abs(y_out)) <= 1.0
