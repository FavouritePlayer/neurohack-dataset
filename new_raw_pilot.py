"""NeuroCompass: inspect, clean, and compare one BrainVision EEG recording.

Run from the hackathon virtual environment:
    python3 raw_pilot_fixed.py

The BrainVision .vhdr, .eeg and .vmrk files must remain together.
Default input: ~/Desktop/hackathon/raw_pilot_/CON_01.vhdr
Default output: ~/Desktop/hackathon/neurocompass_figures/

For testing on a different computer, set EEG_DATA_DIR and/or EEG_OUTPUT_DIR.

IMPORTANT SCIENCE:
  * CON_01 is a healthy control, not a participant with neglect.
  * Event codes 11 and 12 are called "Cue 11" and "Cue 12" until their
    experimental left/right meanings are verified against the paper/codebook.
  * This script demonstrates EEG preprocessing and exploratory comparisons;
    it cannot demonstrate that a screen shader treats spatial neglect.
"""

from collections import Counter
from pathlib import Path
import csv
import os

# Avoid the occasional macOS interactive Matplotlib backend error, previously
# reported as "SystemError: NULL object passed to Py_BuildValue". That traceback
# alone does not establish the exact cause. Saving PNG files avoids needing
# interactive plot windows; open the saved images in Finder after the script.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np


# ============================================================================
# 0. SETTINGS
# ============================================================================

DATA_DIR = Path(
    os.environ.get(
        "EEG_DATA_DIR", str(Path.home() / "Desktop/hackathon/raw_pilot_")
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "EEG_OUTPUT_DIR", str(Path.home() / "Desktop/hackathon/neurocompass_figures")
    )
)
RECORDING = "CON_01"
HEADER = DATA_DIR / f"{RECORDING}.vhdr"

CUE_IDS = {"Cue 11": 11, "Cue 12": 12}
POSTERIOR = ["P5", "P6", "PO7", "PO8", "PO1", "PO2", "O1", "O2"]
INSPECT_CHANNELS = ["Fp1", "Fp2", "AF4", "PO7", "PO8", "LVEOG", "RHEOG"]
BANDS = {"Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}
EOG_CHANNELS = ["LVEOG", "RHEOG"]

# The visual review for THIS PARTICULAR CON_01 ICA fit identified ICA002 as
# an ocular component. Never carry component index 2 over to another subject:
# ICA ordering changes with data, channel selection, and fitting choices.
EYE_COMPONENT = 2

# 250 samples = 1-second Welch FFT at 250 Hz; frequency resolution = 1 Hz.
# Each cue epoch is 0.2..1.6 s long (351 samples, including endpoints).
WELCH_N_FFT = 250


def save_figure(fig, name):
    """Save plots reproducibly even if graphical windows do not work."""
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def make_epochs(inst, events):
    """All selected cue epochs. No automatic artifact rejection is claimed."""
    epochs = mne.Epochs(
        inst,
        events,
        event_id=CUE_IDS,
        tmin=0.2,
        tmax=1.6,
        baseline=None,
        preload=True,
        verbose=False,
    )
    return epochs


def compute_bands(epochs):
    """Return band powers in µV², with shape (trials, EEG channels)."""
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=4,
        fmax=30,
        n_fft=WELCH_N_FFT,
        verbose=False,
    )
    frequencies = spectrum.freqs
    # MNE outputs V²/Hz. Multiply by (10^6 µV/V)^2 = 10^12.
    psd_uv2_per_hz = spectrum.get_data() * 1e12
    print("PSD shape (trials, electrodes, frequencies):", psd_uv2_per_hz.shape)

    result = {}
    for band, (low, high) in BANDS.items():
        inside = (frequencies >= low) & (frequencies <= high)
        result[band] = np.trapezoid(
            psd_uv2_per_hz[:, :, inside],
            frequencies[inside],
            axis=-1,
        )
    return result


def plot_recording(raw):
    """Show why the original recordings needed ocular-artifact inspection."""
    start_sec, stop_sec = 13.0, 23.0
    sfreq = raw.info["sfreq"]
    values = raw.get_data(
        picks=INSPECT_CHANNELS,
        start=int(start_sec * sfreq),
        stop=int(stop_sec * sfreq),
    ) * 1e6
    times = np.arange(values.shape[1]) / sfreq + start_sec

    fig, axes = plt.subplots(len(INSPECT_CHANNELS), 1, figsize=(13, 11), sharex=True)
    for axis, channel, signal in zip(axes, INSPECT_CHANNELS, values):
        axis.plot(times, signal, linewidth=0.7)
        axis.set_ylabel(f"{channel}\nµV", rotation=0, labelpad=33)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{RECORDING}: original EEG and EOG, 13–23 s")
    fig.tight_layout()
    save_figure(fig, "01_original_EEG_EOG_13_23s.png")


def plot_alpha_boxplots(epochs, alpha_power, filename):
    """Two posterior electrodes × two cue codes (NOT left/right labels)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, channel in zip(axes, ["PO7", "PO8"]):
        index = epochs.ch_names.index(channel)
        values11 = alpha_power[epochs.events[:, 2] == 11, index]
        values12 = alpha_power[epochs.events[:, 2] == 12, index]
        axis.boxplot([values11, values12])
        axis.set_xticks([1, 2], ["Cue 11", "Cue 12"])
        axis.set(title=f"{channel} — Alpha Power", ylabel="Alpha power (µV²)")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        print(
            f"{channel} alpha medians (cue 11 / cue 12): "
            f"{np.median(values11):.3f} / {np.median(values12):.3f} µV²"
        )
    fig.tight_layout()
    save_figure(fig, filename)


def plot_topomaps(epochs, band_power, label, scales):
    """Three frequency bands × two cues, with fixed before/after scales."""
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), layout="constrained")

    for row, band in enumerate(BANDS):
        for col, code in enumerate([11, 12]):
            mean_map = band_power[band][epochs.events[:, 2] == code].mean(axis=0)
            image, _ = mne.viz.plot_topomap(
                mean_map,
                epochs.info,
                axes=axes[row, col],
                vlim=(0, scales[band]),
                cmap="viridis",
                contours=0,
                extrapolate="head",
                outlines="head",
                show=False,
            )
            axes[row, col].set_title(f"{band} — Cue {code}")
        fig.colorbar(
            image,
            ax=list(axes[row, :]),
            shrink=0.7,
            label="Band power (µV²)",
        )

    fig.suptitle(f"{RECORDING}: {label} — same scale before/after ICA")
    save_figure(fig, f"{label.lower()}_theta_alpha_beta_topomaps.png")


def mean_band_power(epochs, power, code, channel):
    channel_idx = epochs.ch_names.index(channel)
    return float(power[epochs.events[:, 2] == code, channel_idx].mean())


def alpha_laterality(left, right):
    """Simple relative electrode asymmetry, NOT a validated neglect score."""
    denominator = left + right
    return (right - left) / denominator if denominator > 0 else float("nan")


def print_comparison(epochs_raw, bands_raw, epochs_clean, bands_clean):
    print("\n" + "=" * 65)
    print("BAND-POWER COMPARISON — ORIGINAL VS ICA-CLEANED")
    print("Units are µV². Results depend on the selected recording/ICA fit.")
    print("=" * 65)

    with (OUTPUT_DIR / "band_power_all_channels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["stage", "band", "cue_code", "electrode", "mean_power_uv2"])
        for stage, epochs, powers in [
            ("original", epochs_raw, bands_raw),
            ("ICA002_removed", epochs_clean, bands_clean),
        ]:
            for band in BANDS:
                for code in [11, 12]:
                    for channel in epochs.ch_names:
                        value = mean_band_power(epochs, powers[band], code, channel)
                        writer.writerow([stage, band, code, channel, f"{value:.8f}"])

    for code in [11, 12]:
        print(f"\nCue {code} (128 expected in CON_01)")
        for band, channel in [
            ("Theta", "Fp1"),
            ("Alpha", "Fp1"),
            ("Beta", "AF4"),
            ("Alpha", "PO7"),
            ("Alpha", "PO8"),
        ]:
            before = mean_band_power(epochs_raw, bands_raw[band], code, channel)
            after = mean_band_power(epochs_clean, bands_clean[band], code, channel)
            change = 100 * (after / before - 1) if before else float("nan")
            print(
                f"  {channel:>3s} {band:<5s}: "
                f"{before:8.2f} -> {after:8.2f} µV² ({change:+6.1f}%)"
            )

        po7 = mean_band_power(epochs_clean, bands_clean["Alpha"], code, "PO7")
        po8 = mean_band_power(epochs_clean, bands_clean["Alpha"], code, "PO8")
        print(f"  (PO8-PO7)/(PO8+PO7) = {alpha_laterality(po7, po8):+.4f}")

    print("\nSaved per-channel results:", OUTPUT_DIR / "band_power_all_channels.csv")
    print(
        "CAUTION: a persistent PO8/PO7 difference is not proof of neglect; "
        "the cue-dependent *difference* is the relevant first comparison."
    )


def main():
    # ========================================================================
    # 1. OPEN THE ENTIRE UNFILTERED RECORDING
    # ========================================================================
    if not HEADER.is_file():
        raise FileNotFoundError(
            f"Cannot find {HEADER}. Keep .vhdr, .eeg and .vmrk together."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Data folder:", DATA_DIR)
    print("EEG files:", len(list(DATA_DIR.glob("*.eeg"))))
    raw = mne.io.read_raw_brainvision(
        HEADER,
        preload=False,
        eog=["LVEOG", "RHEOG", "VEOG", "HEOG"],
        verbose=False,
    )
    print(raw)
    print("Sampling rate:", raw.info["sfreq"])
    print("Channels:", len(raw.ch_names))
    print("Duration (minutes):", round(raw.times[-1] / 60, 3))
    print("\nALL CHANNELS\n", raw.ch_names)
    print("\nPOSTERIOR ELECTRODES")
    for channel in POSTERIOR:
        print(f"  {channel}: {channel in raw.ch_names}")

    if any(name not in raw.ch_names for name in INSPECT_CHANNELS):
        missing = set(INSPECT_CHANNELS) - set(raw.ch_names)
        raise ValueError(f"Missing channels needed for CON_01 diagnostic: {missing}")

    # ========================================================================
    # 2. EXAMINE EVENT MARKERS; DO NOT ASSUME THEIR DIRECTION
    # ========================================================================
    print("\nEVENT COUNTS")
    for description, count in sorted(Counter(raw.annotations.description).items()):
        print(f"  {description}: {count}")
    events, event_ids = mne.events_from_annotations(raw, verbose=False)
    print("\nEVENT MAPPING:", event_ids)
    print("EVENT ARRAY SHAPE:", events.shape)
    print("\nFIRST 30 EVENTS")
    for onset, description in zip(
        raw.annotations.onset[:30], raw.annotations.description[:30]
    ):
        print(f"  {onset:8.2f} s | {description}")

    # ========================================================================
    # 3. DEFINE THE SAME 53 MAPPED EEG SENSORS USED BEFORE
    # ========================================================================
    montage = mne.channels.make_standard_montage("colin27_1005")
    types = dict(zip(raw.ch_names, raw.get_channel_types()))
    excluded_from_map = {"M2", "I5", "I3", "Iz", "I4", "I6", "SI3", "SIz", "SI4"}
    good_channels = [
        name
        for name in raw.ch_names
        if types[name] == "eeg"
        and name in montage.ch_names
        and name not in excluded_from_map
    ]
    print("\nMapped EEG channels:", len(good_channels))
    print("Excluded from 53-channel map:", sorted(excluded_from_map))
    if len(good_channels) != 53:
        raise ValueError(
            f"Expected 53 CON_01 mapped EEG channels, got {len(good_channels)}. "
            "Check recording/channel montage before using ICA002."
        )

    # One original, *unfiltered* copy is shared by all comparisons.
    raw_compare = raw.copy().pick(good_channels + EOG_CHANNELS).load_data(verbose=False)
    raw_compare.set_montage(montage, on_missing="ignore")
    plot_recording(raw)

    # ========================================================================
    # 4. FIT ICA ON A SEPARATELY FILTERED COPY
    # ========================================================================
    raw_ica = raw_compare.copy()
    raw_ica.filter(l_freq=1.0, h_freq=None, picks="eeg", verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=20,
        method="fastica",
        random_state=97,
        max_iter="auto",
    )
    ica.fit(raw_ica, picks="eeg", decim=3, verbose=False)

    vertical_candidates, _ = ica.find_bads_eog(
        raw_ica, ch_name="LVEOG", verbose=False
    )
    horizontal_candidates, _ = ica.find_bads_eog(
        raw_ica, ch_name="RHEOG", verbose=False
    )
    candidates = sorted(set(vertical_candidates + horizontal_candidates))
    print("\nVertical EOG candidates:", vertical_candidates)
    print("Horizontal EOG candidates:", horizontal_candidates)
    print("Combined candidates:", candidates)

    # This gate prevents accidentally removing an unrelated ICA002 if the ICA
    # results change because of a code/data/library change.
    if EYE_COMPONENT not in candidates:
        raise RuntimeError(
            "ICA002 was not flagged as ocular in this fit. "
            "Inspect components again; do not automatically delete index 2."
        )

    # ========================================================================
    # 5. REVIEW ICA002 BEFORE CHANGING THE RECORDING
    # ========================================================================
    comp_fig = ica.plot_components(picks=[EYE_COMPONENT], show=False)
    save_figure(comp_fig, "02_ICA002_topography.png")

    property_figs = ica.plot_properties(
        raw_ica, picks=[EYE_COMPONENT], show=False, verbose=False
    )
    for index, fig in enumerate(property_figs, start=1):
        save_figure(fig, f"03_ICA002_properties_{index}.png")

    # IMPORTANT: 13.0/23.0 are floats = seconds. In MNE plot_overlay,
    # start=13 and stop=23 as integers can be interpreted as sample positions:
    # 13/250=.052 s and 23/250=.092 s; this is the OLD failed attempt.
    over = ica.plot_overlay(
        raw_compare,
        exclude=[EYE_COMPONENT],
        picks=["Fp1", "Fp2", "AF4", "PO7", "PO8"],
        start=13.0,
        stop=23.0,
        show=False,
    )
    save_figure(over, "04_before_after_ICA002_13_23s.png")

    # ========================================================================
    # 6. APPLY ICA **ONCE** TO THE UNFILTERED COPY
    # ========================================================================
    raw_clean = raw_compare.copy()
    ica.apply(raw_clean, exclude=[EYE_COMPONENT], verbose=False)
    print("\nApplied ICA002 exclusion ONCE to a copy of the unfiltered recording.")

    eeg_before = raw_compare.copy().pick(good_channels)
    eeg_clean = raw_clean.copy().pick(good_channels)

    # ========================================================================
    # 7. RECOMPUTE EPOCHS AND PSD: ORIGINAL VS CLEANED
    # ========================================================================
    epochs_before = make_epochs(eeg_before, events)
    epochs_clean = make_epochs(eeg_clean, events)
    print("\nORIGINAL EPOCHS:", epochs_before)
    print("CLEANED EPOCHS:", epochs_clean)
    for name, epochs in [("Original", epochs_before), ("Cleaned", epochs_clean)]:
        print(
            f"  {name}: Cue 11 = {len(epochs['Cue 11'])} epochs; "
            f"Cue 12 = {len(epochs['Cue 12'])} epochs"
        )
    print(
        "NOTE: '0 bad epochs dropped' only means no trials were removed. "
        "We supplied no amplitude-rejection threshold; it does NOT prove "
        "that all 256 epochs are clean."
    )

    before_bands = compute_bands(epochs_before)
    clean_bands = compute_bands(epochs_clean)

    print("\nORIGINAL alpha (unfiltered EEG, before ICA)")
    plot_alpha_boxplots(
        epochs_before, before_bands["Alpha"], "05_original_alpha_boxplots.png"
    )
    print("\nCLEANED alpha (ICA002 excluded)")
    plot_alpha_boxplots(
        epochs_clean, clean_bands["Alpha"], "06_cleaned_alpha_boxplots.png"
    )

    print_comparison(epochs_before, before_bands, epochs_clean, clean_bands)

    # A fixed vlim across ORIGINAL and CLEANED ensures honest comparison:
    # a dark cleaned map really indicates lower power on the same color scale.
    scales = {}
    for band in BANDS:
        condition_means = [
            matrix[epochs.events[:, 2] == code].mean(axis=0).max()
            for epochs, matrix in [
                (epochs_before, before_bands[band]),
                (epochs_clean, clean_bands[band]),
            ]
            for code in [11, 12]
        ]
        scales[band] = float(max(condition_means))

    plot_topomaps(epochs_before, before_bands, "ORIGINAL", scales)
    plot_topomaps(epochs_clean, clean_bands, "CLEANED", scales)
    print("\nDONE. Open this folder in Finder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()


# ============================================================================
# HISTORICAL ATTEMPTS (PRESERVED AS REQUESTED, BUT NEVER EXECUTED)
# All sections below are inactive triple-quoted strings. The numeric examples
# were independently reproduced for CON_01 using 53 EEG channels, 250 Hz,
# 20 ICA components, and ICA002 excluded. Numbers can vary across runs/versions.
# ============================================================================

"""
FAILED ATTEMPT A — ALPHA PSD CALCULATED BEFORE CLEANING:

epochs = mne.Epochs(
    raw, events, event_id={"Cue_11": 11, "Cue_12": 12},
    tmin=0.2, tmax=1.6, baseline=None,
    picks=["PO7", "PO8"], preload=True,
)
spectrum = epochs.compute_psd(method="welch", fmin=8, fmax=13, n_fft=250)

WHY IT DID NOT ANSWER THE CLEANED-EEG QUESTION:
- This code ran BEFORE the ICA fit; the name "epochs" referenced original EEG.
- 128 Cue 11 + 128 Cue 12 = 256 selected epochs (1.4 seconds each).
- "0 bad epochs dropped" did NOT mean the 256 trials passed a quality test:
  no reject threshold was specified, so there was no amplitude screening.
- Alpha power was correctly computed mathematically, but it was PRE-cleaning.
- Original CON_01 averages: PO7 4.106 vs 4.035 µV² for cues 11/12;
  PO8 17.662 vs 17.552 µV². Similarity does not identify cue directions.

FIX:
- We preserve the original for comparison AND recreate epochs AFTER ICA002
  exclusion using eeg_clean. Every PSD, boxplot, and topomap gets the correct
  input recording explicitly (original or cleaned).
"""

"""
FAILED ATTEMPT B — ORIGINAL TOPOGRAPHIC MAPS TREATED AS NEURAL PATTERNS:

eeg = raw.copy().pick(good_channels)
epochs_all = mne.Epochs(
    eeg, events, event_id={"Cue 11": 11, "Cue 12": 12},
    tmin=0.2, tmax=1.6, baseline=None, preload=True,
)
# Then compute PSD and plot theta/alpha/beta topomaps.

WHY WE COULD NOT INTERPRET THE BRIGHT FRONTAL POWER AS NEURAL ATTENTION:
- Original Fp1 theta was 204.76 µV² for Cue 11 and 130.01 for Cue 12.
- After ocular-component exclusion, it became ~24.06 and ~21.38 µV²:
  about 88% and 84% lower. The original frontal theta map was largely
  contaminated by an ocular-associated source.
- Fp1 alpha went from 47.62 to 22.05 µV² for Cue 11 (~54% lower).
- However, AF4 beta stayed ~290.10 -> 283.47 µV² for Cue 11 (~2% lower),
  so removing ONE eye-related ICA component is not total artifact removal.
- Posterior Cue 11 alpha changed little: PO7 4.11 -> 4.07 µV² and
  PO8 17.66 -> 17.42 µV². That is useful but not proof of attention decoding.

FIX:
- Compute maps once on original EEG and once on ICA-cleaned EEG.
- Keep the same color scale for ORIGINAL and CLEANED within each frequency band;
  otherwise a "bright" pixel on two differently scaled maps is misleading.
- Preserve frontal electrodes: we removed a fitted source, not whole channels.
"""

"""
FAILED ATTEMPT C — PREVIEWING THE WRONG TIME INTERVAL:

ica.plot_overlay(raw_ica, exclude=[2], start=13, stop=23)

WHY:
- For the MNE version tested, integer start/stop were interpreted as
  sample indices rather than seconds. With sampling rate = 250 samples/s,
  sample 13 = 0.052 s and sample 23 = 0.092 s: only 0.040 seconds.
- That was not the intended ten-second window (13 to 23 s) containing the
  large ocular deflections.

FIX:
- Use start=13.0, stop=23.0 (floating-point seconds) and check the x-axis.
- Preview removal on the original UNFILTERED data, as well as fitting ICA
  on a separate 1-Hz high-pass-filtered copy.
"""

"""
FAILED ATTEMPT D — APPLYING ICA THREE SEPARATE TIMES AND OVERWRITING NAMES:

raw_clean = raw.copy().load_data()
raw_clean.pick(good_channels + ["LVEOG", "RHEOG"])
ica.apply(raw_clean, exclude=[2])
eeg_clean = raw_clean.copy().pick(good_channels)

raw_clean = raw_compare.copy()
ica.apply(raw_clean, exclude=[2])
eeg_clean = raw_clean.copy().pick(good_channels)

raw_clean = raw.copy().load_data()
raw_clean.pick(good_channels + ["LVEOG", "RHEOG"])
ica.apply(raw_clean, exclude=[2])
eeg_clean = raw_clean.copy().pick(good_channels)

WHY:
- The original file applied the SAME component three times to three newly
  created copies, overwriting raw_clean and eeg_clean on each pass. That is
  redundant (not three successive passes on one signal), wastes RAM/CPU,
  obscures the output used for analysis, and does not update the earlier maps.
- The code also excluded ICA002 before printing the candidate indices, so
  a change in ICA fitting could accidentally remove a nonocular component.

FIX:
- Identify and inspect candidates first; require index 2 to be flagged.
- Keep raw_compare unchanged and do just ONE:
      raw_clean = raw_compare.copy()
      ica.apply(raw_clean, exclude=[2])
- Recompute epochs and maps from eeg_clean AFTER that line.
"""

"""
LIMITATION — THIS IS NOT YET A NEUROCOMPASS VALIDATION:

- Cue 11 and Cue 12 are event codes, not yet verified as left/right cues.
- PO8 alpha is about four times PO7 alpha in BOTH conditions. Raw cue-11
  alpha lateralization (PO8-PO7)/(PO8+PO7) is ~0.623. Cue-12 is also ~0.626.
  Stable electrode asymmetry alone cannot tell us whether attention shifted.
- CON_01 is one healthy person, not one of the stroke-neglect recordings.
- We have not yet tested personalized shaders or target-detection accuracy.
- Our next scientifically meaningful analysis is within-person, condition-
  dependent alpha modulation and/or cue-locked ERP with behavioral results,
  once the publication's marker codebook is established.
"""
