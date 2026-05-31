#!/usr/bin/env python3
"""
video_label_assist.py — PadelLabs ML Toolchain
================================================
Detects stroke hit timestamps in a padel video using audio transient detection,
synchronizes them with IMU spike timestamps from the Watch session, and outputs
a per-spike review CSV so you can label strokes from 2-second clips instead of
scrubbing a 2-hour video manually.

How it works:
  1. ffmpeg extracts the audio track from the MOV → temp WAV
  2. scipy detects percussive ball-hit transients in the audio
  3. IMU spike timestamps are loaded from the session JSON
  4. Cross-correlation of the two timestamp sequences finds the video↔IMU offset
  5. Each IMU spike is matched to the nearest audio hit within a tolerance window
  6. Output: review CSV + (optionally) 2-second video clips for each spike

Usage:
    # Basic: auto-detect sync offset
    python3 video_label_assist.py \\
        --video "19 April.MOV" \\
        --session-dir "/path/to/DataCollection/19 April" \\
        --session-id "4AF757C9-5E73-4E54-A9EF-F2ACCD12DC88" \\
        --output-dir "review/19-april"

    # If auto-sync fails, provide the offset manually (seconds):
    #   video_time = imu_time + offset
    python3 video_label_assist.py ... --sync-offset 12.5

    # Also extract 2-second clips for every spike (slow — ~1 hour per session):
    python3 video_label_assist.py ... --extract-clips

    # Skip audio detection, only extract clips for spikes already in a review CSV:
    python3 video_label_assist.py --extract-clips-from review/19-april/review.csv \\
        --video "19 April.MOV"

Requirements (all in padel-ml venv):
    scipy, numpy, pandas
    ffmpeg (brew install ffmpeg)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal


# ── Audio transient detection ─────────────────────────────────────────────────

def extract_audio(video_path: Path, out_wav: Path, sample_rate: int = 22050) -> None:
    """Use ffmpeg to extract mono audio from video at given sample rate."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                        # no video
        "-ac", "1",                   # mono
        "-ar", str(sample_rate),      # sample rate
        "-acodec", "pcm_s16le",       # 16-bit PCM
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode()}")


def detect_hit_transients(
    wav_path: Path,
    min_gap_s: float = 0.20,       # ignore hits closer than this
    highpass_hz: int = 800,        # ball hit click is high-frequency
    energy_window_ms: int = 10,    # short-time energy window
    threshold_factor: float = 4.0, # peak must be N× median energy
) -> np.ndarray:
    """
    Returns sorted array of hit timestamps (seconds from video start).
    Works by:
      1. High-pass filtering to isolate the percussive click
      2. Computing short-time energy envelope
      3. Peak-picking with minimum gap constraint
    """
    sample_rate, audio = wavfile.read(str(wav_path))

    # Normalize to float [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0

    # High-pass filter: isolate ball hit click
    sos = signal.butter(4, highpass_hz, btype="high", fs=sample_rate, output="sos")
    filtered = signal.sosfilt(sos, audio)

    # Short-time energy envelope
    win_samples = max(1, int(energy_window_ms * sample_rate / 1000))
    energy = np.convolve(filtered ** 2, np.ones(win_samples) / win_samples, mode="same")
    energy = np.sqrt(energy)  # RMS-like

    # Peak detection
    median_e = np.median(energy)
    min_height = threshold_factor * median_e
    min_samples = int(min_gap_s * sample_rate)

    peaks, _ = signal.find_peaks(energy, height=min_height, distance=min_samples)
    timestamps = peaks / sample_rate

    print(f"  Audio: detected {len(timestamps)} hit transients over "
          f"{len(audio)/sample_rate:.1f}s of video")
    return timestamps


# ── IMU spike loading ─────────────────────────────────────────────────────────

def load_imu_spikes(session_dir: Path, session_id: str) -> pd.DataFrame:
    """
    Load spike markers from session JSON.
    Returns DataFrame with columns: timestamp_s, strokeType
    where timestamp_s is seconds from session start (Watch time).
    """
    json_path = session_dir / f"{session_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Session JSON not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    markers = data.get("markers", [])
    if not markers:
        raise ValueError("No markers found in session JSON")

    df = pd.DataFrame([
        {"timestamp_s": m["timestamp"], "strokeType": m.get("strokeType", "unknown")}
        for m in markers
    ])
    df = df.sort_values("timestamp_s").reset_index(drop=True)

    # Burst-group markers within 1.2s (same physical stroke)
    groups, group_id = [], 0
    last_ts = -np.inf
    for ts in df["timestamp_s"]:
        if ts - last_ts > 1.2:
            group_id += 1
        groups.append(group_id)
        last_ts = ts
    df["burst_id"] = groups

    # One representative per burst (use max-accel sample — take the first for now)
    df = df.groupby("burst_id").first().reset_index(drop=True)

    print(f"  IMU: {len(df)} stroke events (after burst-grouping {len(markers)} raw markers)")
    return df


# ── Video ↔ IMU timestamp synchronization ────────────────────────────────────

def estimate_sync_offset(
    audio_hits: np.ndarray,
    imu_spikes: np.ndarray,
    search_range_s: float = 120.0,  # search ±2 minutes
    grid_ms: float = 100.0,         # 100ms grid resolution
) -> float:
    """
    Find the offset O such that  video_time = imu_time + O
    by maximizing cross-correlation of the two sparse event sequences.

    Returns offset in seconds.
    """
    # Build binary vectors on a common grid
    grid_step = grid_ms / 1000.0
    max_time = max(audio_hits.max(), imu_spikes.max() + search_range_s)
    n_bins = int(max_time / grid_step) + 1

    audio_vec = np.zeros(n_bins)
    for t in audio_hits:
        idx = int(t / grid_step)
        if 0 <= idx < n_bins:
            audio_vec[idx] = 1.0

    imu_vec = np.zeros(n_bins)
    for t in imu_spikes:
        idx = int(t / grid_step)
        if 0 <= idx < n_bins:
            imu_vec[idx] = 1.0

    # Cross-correlate: imu_vec shifted by lag vs audio_vec
    xcorr = signal.correlate(audio_vec, imu_vec, mode="full")
    lags = signal.correlation_lags(len(audio_vec), len(imu_vec), mode="full") * grid_step

    # Restrict search to ±search_range_s
    mask = np.abs(lags) <= search_range_s
    best_lag_idx = np.argmax(xcorr * mask)
    best_offset = lags[best_lag_idx]
    best_score = xcorr[best_lag_idx]

    print(f"  Sync: best offset = {best_offset:+.2f}s  "
          f"(score {best_score:.0f} events matched on {grid_ms}ms grid)")
    return float(best_offset)


def match_spikes_to_hits(
    imu_df: pd.DataFrame,
    audio_hits: np.ndarray,
    offset_s: float,
    tolerance_s: float = 0.40,
) -> pd.DataFrame:
    """
    For each IMU spike, find the nearest audio hit after applying the sync offset.
    video_time = imu_time + offset_s
    """
    video_times = imu_df["timestamp_s"].values + offset_s

    matched_video_ts = []
    matched_audio_idx = []
    match_delta = []

    for vt in video_times:
        if len(audio_hits) == 0:
            matched_video_ts.append(vt)
            matched_audio_idx.append(-1)
            match_delta.append(np.nan)
            continue
        diffs = np.abs(audio_hits - vt)
        best = np.argmin(diffs)
        delta = audio_hits[best] - vt
        if abs(delta) <= tolerance_s:
            matched_video_ts.append(audio_hits[best])
            matched_audio_idx.append(best)
            match_delta.append(delta)
        else:
            matched_video_ts.append(vt)   # use predicted position
            matched_audio_idx.append(-1)
            match_delta.append(np.nan)

    result = imu_df.copy()
    result["video_time_s"] = matched_video_ts
    result["audio_match_delta_s"] = match_delta
    result["audio_matched"] = [i >= 0 for i in matched_audio_idx]
    result["video_frame"] = (np.array(matched_video_ts) * 30).astype(int)  # assume 30fps

    matched = result["audio_matched"].sum()
    print(f"  Match: {matched}/{len(result)} IMU spikes matched to audio hit "
          f"(within ±{tolerance_s*1000:.0f}ms)")
    return result


# ── Clip extraction ───────────────────────────────────────────────────────────

def extract_clips(
    video_path: Path,
    review_df: pd.DataFrame,
    output_dir: Path,
    clip_duration_s: float = 2.0,
    pre_s: float = 0.5,           # seconds before hit
) -> None:
    """Extract a short video clip around each stroke event."""
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting {len(review_df)} clips to {clips_dir} ...")
    for i, row in review_df.iterrows():
        start = max(0, row["video_time_s"] - pre_s)
        label = row["strokeType"].replace(" ", "_")
        imu_ts = f"{row['timestamp_s']:.2f}".replace(".", "_")
        fname = clips_dir / f"spike_{i:04d}_{imu_ts}s_{label}.mp4"

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video_path),
            "-t", f"{clip_duration_s:.1f}",
            "-c:v", "libx264", "-crf", "23",
            "-c:a", "aac", "-b:a", "64k",
            "-loglevel", "error",
            str(fname),
        ]
        subprocess.run(cmd, capture_output=True)
        if i % 20 == 0:
            print(f"  {i+1}/{len(review_df)}", end="\r", flush=True)

    print(f"\n  Done. Clips in: {clips_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Video ↔ IMU sync + stroke label assist")
    parser.add_argument("--video", required=True, help="Path to session .MOV")
    parser.add_argument("--session-dir", required=True, help="Path to folder containing session CSV+JSON")
    parser.add_argument("--session-id", required=True,
                        help="Session UUID (without extension), e.g. 4AF757C9-...")
    parser.add_argument("--output-dir", default="review_output",
                        help="Where to write review CSV + clips")
    parser.add_argument("--sync-offset", type=float, default=None,
                        help="Manual sync offset in seconds: video_time = imu_time + offset. "
                             "If omitted, auto-detected via cross-correlation.")
    parser.add_argument("--tolerance-ms", type=float, default=400,
                        help="Max allowed audio↔IMU delta to count as a match (default 400ms)")
    parser.add_argument("--extract-clips", action="store_true",
                        help="Extract 2-second clips for every spike (slow)")
    parser.add_argument("--extract-clips-from",
                        help="Skip detection; just extract clips for spikes in this review CSV")
    parser.add_argument("--highpass-hz", type=int, default=800,
                        help="High-pass filter cutoff for audio (default 800 Hz)")
    parser.add_argument("--threshold", type=float, default=4.0,
                        help="Energy peak threshold factor above median (default 4.0)")
    parser.add_argument("--min-gap-ms", type=float, default=200,
                        help="Minimum gap between detected hits in ms (default 200)")
    args = parser.parse_args()

    video_path = Path(args.video)
    session_dir = Path(args.session_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Clip-only mode ────────────────────────────────────────────────────────
    if args.extract_clips_from:
        review_df = pd.read_csv(args.extract_clips_from)
        extract_clips(video_path, review_df, output_dir)
        return

    print(f"\n{'='*60}")
    print(f"  PadelLabs — Video Label Assist")
    print(f"  Video:   {video_path.name}")
    print(f"  Session: {args.session_id[:8]}...")
    print(f"{'='*60}\n")

    # ── Step 1: Extract audio + detect hits ───────────────────────────────────
    print("Step 1: Extracting audio from video...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = Path(f.name)

    try:
        extract_audio(video_path, wav_path)
        audio_hits = detect_hit_transients(
            wav_path,
            min_gap_s=args.min_gap_ms / 1000.0,
            highpass_hz=args.highpass_hz,
            threshold_factor=args.threshold,
        )
    finally:
        wav_path.unlink(missing_ok=True)

    # ── Step 2: Load IMU spikes ────────────────────────────────────────────────
    print("\nStep 2: Loading IMU spike markers...")
    imu_df = load_imu_spikes(session_dir, args.session_id)

    # ── Step 3: Sync ──────────────────────────────────────────────────────────
    print("\nStep 3: Synchronizing video ↔ IMU timestamps...")
    if args.sync_offset is not None:
        offset = args.sync_offset
        print(f"  Using manual offset: {offset:+.2f}s")
    else:
        offset = estimate_sync_offset(audio_hits, imu_df["timestamp_s"].values)

    # ── Step 4: Match spikes ──────────────────────────────────────────────────
    print("\nStep 4: Matching IMU spikes to audio hits...")
    review_df = match_spikes_to_hits(
        imu_df, audio_hits, offset, tolerance_s=args.tolerance_ms / 1000.0
    )

    # ── Step 5: Write review CSV ──────────────────────────────────────────────
    out_csv = output_dir / "review.csv"
    review_df.to_csv(out_csv, index=False)
    print(f"\nReview CSV: {out_csv}")

    # ── Step 5b: Write videoSyncOffset back into session JSON ─────────────────
    # Labeling tool convention: video_time = marker.timestamp - syncOffset
    # Our convention:           video_time = marker.timestamp + offset
    # Therefore:                syncOffset = -offset
    labeling_tool_sync = -offset
    json_path = session_dir / f"{args.session_id}.json"
    if json_path.exists():
        with open(json_path) as f:
            session_json = json.load(f)
        session_json["videoSyncOffset"] = round(labeling_tool_sync, 3)
        with open(json_path, "w") as f:
            json.dump(session_json, f, indent=2)
        print(f"\nWrote videoSyncOffset={labeling_tool_sync:+.2f}s into {json_path.name}")
        print(f"  → Open the labeling tool and load this session — sync slider auto-sets to {labeling_tool_sync:+.1f}s")
    else:
        print(f"\n⚠ Could not find {json_path} to write videoSyncOffset")

    # Summary
    matched_pct = review_df["audio_matched"].mean() * 100
    print(f"\n  Match rate: {matched_pct:.0f}% of IMU spikes have audio confirmation")
    if matched_pct < 50:
        print(f"  ⚠ Low match rate — try adjusting --sync-offset or --threshold")
        print(f"  Hint: run with --sync-offset to manually specify the alignment")
        # Print first few audio hits and first few IMU+offset for manual inspection
        print(f"\n  First 10 audio hits (video seconds):")
        print("   ", np.round(audio_hits[:10], 2))
        print(f"\n  First 10 IMU spikes + estimated offset (video seconds):")
        print("   ", np.round(imu_df["timestamp_s"].values[:10] + offset, 2))

    # ── Step 6: Extract clips (optional) ──────────────────────────────────────
    if args.extract_clips:
        extract_clips(video_path, review_df, output_dir)
    else:
        print(f"\n  Tip: add --extract-clips to extract 2s video clips for every spike.")
        print(f"  Or extract for a subset: --extract-clips-from {out_csv}")


if __name__ == "__main__":
    main()
