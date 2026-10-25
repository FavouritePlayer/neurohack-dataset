"""NeuroCompass: eight-channel EEG reference maps + live/replay comparison.

Designed for the standard g.tec Unicorn Hybrid Black montage:
    Fz C3 Cz C4 Pz PO7 Oz PO8 (250 Hz)

Requires: pip install mne numpy scipy matplotlib
For optional LSL acquisition: pip install pylsl

Run in this order:
    python3 neurocompass_eeg_reference.py build /Users/rafa/Desktop/hackathon/raw_pilot_
    python3 neurocompass_eeg_reference.py replay /Users/rafa/Desktop/hackathon/raw_pilot_/CON_01.vhdr
    python3 neurocompass_eeg_reference.py live --lsl-name UniCornDeviceDataStream

The program creates:
    neurocompass_figures/eeg_reference_templates.npz
    neurocompass_figures/eeg_reference_subjects.csv
    neurocompass_figures/eeg_reference_topomaps.png
    neurocompass_figures/eeg_reference_vs_live.png
    neurocompass_figures/eeg_live_state.json

These are DESCRIPTIVE, NON-DIAGNOSTIC spatial PSD maps, not a validated visual
neglect classifier. In particular: old recordings and a live Unicorn differ in
age, amplifier, impedance, artifacts, and behavioral task. EEG resemblance
must NOT set treatment or screen settings automatically. Adapt only based on
measured within-person performance under randomized visual-display settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
BANDS = {"Theta": (4.0, 8.0), "Alpha": (8.0, 13.0), "Beta": (13.0, 30.0)}
GROUPS = ["CON", "N-", "N+"]
GROUP_TITLES = ["Healthy", "Stroke, no neglect", "Stroke + neglect"]
WINDOW_SECONDS = 4.0
STEP_SECONDS = 2.0
MAX_PTP_UV = 250.0


def montage_info() -> mne.Info:
    info = mne.create_info(CHANNELS, sfreq=250.0, ch_types="eeg")
    # New MNE calls this colin27_1020; older installations used standard_1020.
    try:
        montage = mne.channels.make_standard_montage("colin27_1020")
    except ValueError:
        montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)
    return info


def window_features(data_v: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """(3 bands, 8 electrodes): integrated power µV² and relative spatial dB.

    Input is continuous, approximately 4-second data, volts, (8, samples).
    The average reference removes the common reference signal, and the 1-35 Hz
    filter/PSD pipeline is identical for archived and live data.
    """
    if data_v.shape[0] != len(CHANNELS):
        raise ValueError(f"Expected {len(CHANNELS)} channels, got {data_v.shape}")
    centered = data_v - data_v.mean(axis=0, keepdims=True)
    sos = butter(4, [1.0, 35.0], btype="bandpass", fs=sfreq, output="sos")
    cleaned = sosfiltfilt(sos, centered, axis=-1)
    frequencies, densities = welch(
        cleaned, fs=sfreq, nperseg=min(round(sfreq * 2), cleaned.shape[-1]),
        noverlap=min(round(sfreq), max(0, cleaned.shape[-1] // 2 - 1)), axis=-1,
    )
    df = float(frequencies[1] - frequencies[0])
    power = []
    for low, high in BANDS.values():
        indices = (frequencies >= low) & (frequencies < high)
        power.append(densities[:, indices].sum(axis=-1) * df * 1e12)
    power = np.asarray(power)
    # Center the 8 spatial log-powers *within each frequency band*.
    db = 10.0 * np.log10(np.maximum(power, 1e-12))
    relative_db = db - db.mean(axis=1, keepdims=True)
    return power, relative_db


def quality_ok(data_v: np.ndarray) -> bool:
    if not np.all(np.isfinite(data_v)):
        return False
    return bool(np.max(np.ptp(data_v, axis=1)) * 1e6 < MAX_PTP_UV)


def load_eeg(vhdr: Path) -> tuple[np.ndarray, float]:
    """Read only the eight comparable channels, without modifying source files.

    Some source .vhdr files incorrectly name their .eeg companion; when the
    same-stem .eeg exists, build a temporary corrected header and symlinks.
    """
    header = vhdr.read_text(encoding="latin-1")
    match = re.search(r"(?m)^DataFile=([^\r\n]+)", header)
    if match is None:
        raise ValueError("Missing DataFile header")
    data_name = match.group(1).strip()

    if (vhdr.parent / data_name).is_file():
        raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="ERROR")
        if not all(ch in raw.ch_names for ch in CHANNELS):
            raise ValueError("Missing required Unicorn-aligned EEG channel(s)")
        raw.pick(CHANNELS).reorder_channels(CHANNELS)
        data = raw.get_data()
        sfreq = float(raw.info["sfreq"])
        raw.close()
        return data, sfreq

    true_eeg = vhdr.with_suffix(".eeg")
    true_markers = vhdr.with_suffix(".vmrk")
    if not true_eeg.is_file() or not true_markers.is_file():
        raise FileNotFoundError(f"Companion EEG or marker file missing for {vhdr.name}")

    print(f"  Repairing misleading DataFile reference in {vhdr.name} (temporary only)")
    with tempfile.TemporaryDirectory(prefix="nc_brainvision_") as tmp:
        base = Path(tmp)
        (base / true_eeg.name).symlink_to(true_eeg.resolve())
        (base / true_markers.name).symlink_to(true_markers.resolve())
        corrected = re.sub(r"(?m)^DataFile=[^\r\n]+", f"DataFile={true_eeg.name}", header)
        corrected = re.sub(r"(?m)^MarkerFile=[^\r\n]+", f"MarkerFile={true_markers.name}", corrected)
        patched_header = base / vhdr.name
        patched_header.write_text(corrected, encoding="latin-1")
        raw = mne.io.read_raw_brainvision(patched_header, preload=False, verbose="ERROR")
        if not all(ch in raw.ch_names for ch in CHANNELS):
            raise ValueError("Missing required Unicorn-aligned EEG channel(s)")
        raw.pick(CHANNELS).reorder_channels(CHANNELS)
        data = raw.get_data()
        sfreq = float(raw.info["sfreq"])
        raw.close()
    return data, sfreq


def recording_features(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray, int, int]:
    n_window = round(WINDOW_SECONDS * sfreq)
    n_step = round(STEP_SECONDS * sfreq)
    powers, maps = [], []
    total = 0
    for start in range(0, data.shape[-1] - n_window + 1, n_step):
        chunk = data[:, start:start+n_window]
        total += 1
        if not quality_ok(chunk):
            continue
        power, relative_map = window_features(chunk, sfreq)
        powers.append(power)
        maps.append(relative_map)
    if len(maps) < 3:
        raise ValueError(f"Only {len(maps)} usable 4-second windows from {total}")
    return np.median(powers, axis=0), np.median(maps, axis=0), len(maps), total


def extract_subject(file_stem: str) -> tuple[str, str] | None:
    match = re.match(r"^(CON|N\+|N-)_(\d+)", file_stem)
    if not match:
        return None
    return match.group(1), "_".join(match.groups())


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_reference(group_maps: np.ndarray, target: Path,
                     fourth_map: np.ndarray | None = None,
                     fourth_title: str = "Live / replay") -> None:
    """Scalp maps share one color scale within each row (same eight electrodes)."""
    cols = 3 if fourth_map is None else 4
    fig, axes = plt.subplots(3, cols, figsize=(4.2 * cols, 11), layout="constrained")
    info = montage_info()

    for band_i, band in enumerate(BANDS):
        data_to_plot = [group_maps[i, band_i] for i in range(3)]
        if fourth_map is not None:
            data_to_plot.append(fourth_map[band_i])
        finite = np.concatenate([x[np.isfinite(x)] for x in data_to_plot])
        lim = np.percentile(np.abs(finite), 98) if len(finite) else 2.0
        lim = max(1.0, float(lim))

        for i, data in enumerate(data_to_plot):
            ax = axes[band_i, i]
            if np.all(np.isfinite(data)):
                image, _ = mne.viz.plot_topomap(
                    data, info, axes=ax, show=False, cmap="RdBu_r", vlim=(-lim, lim),
                    extrapolate="head", contours=3, sensors=True,
                )
            else:
                ax.text(.5, .5, "Not available", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
            ax.set_title(f"{band} — {GROUP_TITLES[i] if i < 3 else fourth_title}", fontsize=11)
        if len(finite):
            bar = fig.colorbar(image, ax=list(axes[band_i]), shrink=.52, pad=.015)
            bar.set_label("Relative scalp power (dB)")

    fig.suptitle("Eight matched EEG sites — spatial PSD shapes (exploratory)", fontsize=15)
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)


def output_root(data_dir: Path) -> Path:
    root = data_dir.resolve().parent / "neurocompass_figures"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build(data_folder: Path) -> None:
    output = output_root(data_folder)
    files = sorted(data_folder.rglob("*.vhdr"))
    if not files:
        raise FileNotFoundError(f"No BrainVision .vhdr files found under {data_folder}")
    subject_runs = {}
    used, skipped = 0, 0

    for i, path in enumerate(files, start=1):
        identity = extract_subject(path.stem)
        if identity is None:
            continue
        group, subject = identity
        print(f"[{i}/{len(files)}] {path.name}", flush=True)
        try:
            volts, fs = load_eeg(path)
            if abs(fs - 250.0) > .1:
                raise ValueError(f"Unexpected sample rate: {fs} Hz")
            power, rel, accepted, total = recording_features(volts, fs)
        except (OSError, ValueError, RuntimeError) as exc:
            skipped += 1
            print(f"  SKIPPED: {exc}", flush=True)
            continue
        used += 1
        subject_runs.setdefault(subject, []).append((power, rel))
        print(f"  accepted {accepted}/{total} four-second windows", flush=True)

    group_maps = np.full((3, 3, 8), np.nan)
    subject_rows = []
    maps_by_group = {group: [] for group in GROUPS}

    for subject, runs in sorted(subject_runs.items()):
        group = extract_subject(subject)[0]
        power = np.median(np.asarray([item[0] for item in runs]), axis=0)
        relative = np.median(np.asarray([item[1] for item in runs]), axis=0)
        maps_by_group[group].append(relative)
        row = {"participant": subject, "group": group, "sessions": len(runs)}
        for b, band in enumerate(BANDS):
            for j, ch in enumerate(CHANNELS):
                row[f"{band}_{ch}_uV2"] = round(float(power[b, j]), 5)
                row[f"{band}_{ch}_relative_dB"] = round(float(relative[b, j]), 5)
        subject_rows.append(row)

    for i, group in enumerate(GROUPS):
        if maps_by_group[group]:
            group_maps[i] = np.median(np.asarray(maps_by_group[group]), axis=0)
        print(f"{group}: {len(maps_by_group[group])} distinct participants", flush=True)

    if used == 0:
        raise RuntimeError("No usable recordings. Check file companions and EEG units.")
    np.savez_compressed(output / "eeg_reference_templates.npz",
                        group_maps=group_maps,
                        channels=np.array(CHANNELS),
                        bands=np.array(list(BANDS)),
                        groups=np.array(GROUPS))
    save_csv(output / "eeg_reference_subjects.csv", subject_rows)
    render_reference(group_maps, output / "eeg_reference_topomaps.png")
    print(f"\nBuilt references from {used} recordings; skipped {skipped}.")
    print(f"Outputs saved to: {output}")
    print("NOTE: Relative maps are not diagnostic biomarkers of neglect.")


def get_templates(output: Path) -> np.ndarray:
    filename = output / "eeg_reference_templates.npz"
    if not filename.is_file():
        raise FileNotFoundError(f"Run 'build' first: {filename}")
    with np.load(filename) as saved:
        channels = list(saved["channels"])
        if channels != CHANNELS:
            raise ValueError("Template montage is not the expected Unicorn eight-channel montage")
        return saved["group_maps"].copy()


def show_comparison(output: Path, live_map: np.ndarray,
                    live_powers: np.ndarray, title: str, status: str) -> None:
    references = get_templates(output)
    render_reference(references, output / "eeg_reference_vs_live.png",
                     fourth_map=live_map, fourth_title=title)

    distances = {}
    for i, group in enumerate(GROUPS):
        if np.all(np.isfinite(references[i])):
            distances[group] = round(float(np.linalg.norm(live_map-references[i])), 4)
        else:
            distances[group] = None
    state = {
        "status": status,
        "timestamp_unix": time.time(),
        "channels": CHANNELS,
        "units_absolute": "uV^2",
        "units_spatial": "dB relative to mean of eight channels",
        "bandpowers_uV2": {band: dict(zip(CHANNELS, map(float, live_powers[i])))
                            for i, band in enumerate(BANDS)},
        "relative_maps_db": {band: dict(zip(CHANNELS, map(float, live_map[i])))
                             for i, band in enumerate(BANDS)},
        "distances_descriptive_not_diagnostic": distances,
        "shader_command": None,
        "note": "No automated treatment decision: compare participant behavioral responses under settings.",
    }
    temp_path = output / "eeg_live_state.tmp.json"
    temp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp_path.replace(output / "eeg_live_state.json")
    print(f"Updated {output / 'eeg_reference_vs_live.png'}", flush=True)


def replay(path: Path, start_time: float) -> None:
    data, fs = load_eeg(path)
    first = int(start_time * fs)
    last = first + int(WINDOW_SECONDS * fs)
    if first < 0 or last > data.shape[1]:
        raise ValueError("Replay start time is outside the recorded signal")
    segment = data[:, first:last]
    if not quality_ok(segment):
        print("WARNING: replay segment exceeds EEG peak-to-peak artifact threshold")
    powers, maps = window_features(segment, fs)
    output = output_root(path.parent)
    show_comparison(output, maps, powers, title="EEG replay (not live)", status="replay")


def live(data_dir: Path, lsl_name: str | None, units: str, duration: float) -> None:
    try:
        from pylsl import StreamInlet, resolve_byprop
    except ImportError as exc:
        raise RuntimeError("Install pylsl first: python3 -m pip install pylsl") from exc

    if lsl_name:
        available = resolve_byprop("name", lsl_name, timeout=6)
    else:
        available = resolve_byprop("type", "EEG", timeout=6)
    if not available:
        raise RuntimeError("No matching LSL stream. Start Unicorn LSL Interface first.")
    meta = available[0]
    print(f"Connected LSL: {meta.name()} ({meta.channel_count()} channels, "
          f"{meta.nominal_srate()} Hz). Expect first 8 channels: {CHANNELS}")
    if meta.channel_count() < 8:
        raise ValueError("LSL stream contains fewer than 8 channels")
    if abs(meta.nominal_srate()-250) > 1:
        raise ValueError("This demo requires the Unicorn 250 Hz stream")
    print(f"Assumed incoming EEG units: {units}; VERIFY using the device stream docs.")
    inlet = StreamInlet(meta)
    buffer = deque(maxlen=round(WINDOW_SECONDS * 250))
    started = time.monotonic()
    last_update = started
    output = output_root(data_dir)

    while duration <= 0 or time.monotonic()-started < duration:
        values, _timestamps = inlet.pull_chunk(timeout=.3, max_samples=128)
        if values:
            array = np.asarray(values, dtype=float)
            for sample in array[:, :8]:
                buffer.append(sample)
        if len(buffer) < round(WINDOW_SECONDS*250):
            continue
        if time.monotonic()-last_update < 2.0:
            continue
        last_update = time.monotonic()
        live_data = np.asarray(buffer).T
        if units == "uv":
            live_data *= 1e-6
        if not quality_ok(live_data):
            print("Artifact / large movement: holding prior map (no state update).")
            continue
        powers, maps = window_features(live_data, 250)
        show_comparison(output, maps, powers,
                        title="Live Unicorn (4 s window)", status="live")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Create 3 group EEG reference maps")
    build_parser.add_argument("folder", type=Path)
    replay_parser = commands.add_parser("replay", help="Compare a recorded 4-s window with references")
    replay_parser.add_argument("vhdr", type=Path)
    replay_parser.add_argument("--start", type=float, default=100.0)
    live_parser = commands.add_parser("live", help="Compare Unicorn LSL streaming EEG with references")
    live_parser.add_argument("--folder", type=Path, default=Path.cwd() / "raw_pilot_")
    live_parser.add_argument("--lsl-name", type=str, default=None)
    live_parser.add_argument("--units", choices=["uv", "v"], default="uv")
    live_parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()
    if args.command == "build":
        build(args.folder)
    elif args.command == "replay":
        replay(args.vhdr, args.start)
    elif args.command == "live":
        live(args.folder, args.lsl_name, args.units, args.duration)


if __name__ == "__main__":
    main()
