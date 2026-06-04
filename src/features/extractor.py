"""M3 - Feature extraction per stem → StyleProfile."""
import numpy as np
import librosa
from src.features.style_profile import StyleProfile


def extract(stem_name: str, y: np.ndarray, sr: int) -> StyleProfile:
    """Extract all style features from a stem."""
    profile = StyleProfile(stem=stem_name)

    # ── Rhythm ─────────────────────────────────────────────────────────────
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    raw_bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    # Fall back to onset-based BPM if beat tracker fails
    if raw_bpm < 20:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo2, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        raw_bpm2 = float(tempo2) if np.isscalar(tempo2) else float(tempo2[0])
        raw_bpm = raw_bpm2 if raw_bpm2 >= 20 else 120.0
    profile.bpm = raw_bpm

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    profile.beat_strength = float(np.mean(onset_env))

    # Swing detection: compare even vs odd inter-beat intervals
    if len(beats) > 4:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        ibi = np.diff(beat_times)
        even_ibi = ibi[0::2]
        odd_ibi = ibi[1::2]
        min_len = min(len(even_ibi), len(odd_ibi))
        if min_len > 0:
            ratio = np.mean(even_ibi[:min_len]) / (np.mean(odd_ibi[:min_len]) + 1e-9)
            if ratio > 1.25:
                profile.groove_feel = "swing"
            elif ratio > 1.1:
                profile.groove_feel = "shuffle"

    # ── Timbre ─────────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    profile.mfcc_mean = mfcc.mean(axis=1).tolist()
    profile.mfcc_std = mfcc.std(axis=1).tolist()

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    profile.spectral_centroid_mean = float(np.mean(centroid))

    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    profile.spectral_bandwidth_mean = float(np.mean(bw))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    profile.spectral_rolloff_mean = float(np.mean(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y)
    profile.zero_crossing_rate = float(np.mean(zcr))

    # ── Dynamics ───────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]
    profile.rms_mean = float(np.mean(rms))
    profile.rms_std = float(np.std(rms))
    rms_db = librosa.amplitude_to_db(rms + 1e-9)
    profile.dynamic_range_db = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))

    # ── Pitch / Melody ─────────────────────────────────────────────────────
    if stem_name in ("melody", "other"):
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr, fill_na=None
        )
        if f0 is not None:
            voiced = f0[voiced_flag == 1] if voiced_flag is not None else f0[~np.isnan(f0)]
            if len(voiced) > 10:
                profile.f0_mean = float(np.nanmean(voiced))
                profile.f0_std = float(np.nanstd(voiced))
                semitones = librosa.hz_to_midi(voiced)
                profile.pitch_range_semitones = float(np.ptp(semitones))
                profile.melodic_density = float(np.sum(voiced_flag == 1)) / max(len(beats), 1)

    # ── Phrasing ───────────────────────────────────────────────────────────
    # Simple phrase segmentation: find silences / low-energy regions
    frame_len = int(sr * 0.1)
    hop = int(sr * 0.05)
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    threshold = np.percentile(rms_frames, 20)
    silence = rms_frames < threshold
    # Count transitions from non-silence to silence as phrase ends
    transitions = np.diff(silence.astype(int))
    phrase_ends = np.where(transitions == 1)[0]
    if len(phrase_ends) > 1:
        phrase_lengths = np.diff(phrase_ends) * (hop / sr)
        profile.avg_phrase_length_s = float(np.mean(phrase_lengths))
        profile.phrase_count = len(phrase_ends)
    else:
        profile.avg_phrase_length_s = 4.0
        profile.phrase_count = int(30.0 / 4.0)

    # ── Key / Chroma ───────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    profile.chroma_vector = chroma_mean.tolist()

    # Key detection via template matching
    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    best_score = -np.inf
    best_key = "C"
    best_mode = "major"
    for i in range(12):
        maj_score = np.dot(np.roll(major_template, i), chroma_mean)
        min_score = np.dot(np.roll(minor_template, i), chroma_mean)
        if maj_score > best_score:
            best_score = maj_score
            best_key = notes[i]
            best_mode = "major"
        if min_score > best_score:
            best_score = min_score
            best_key = notes[i]
            best_mode = "minor"

    profile.key = best_key
    profile.mode = best_mode

    return profile
