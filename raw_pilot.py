from pathlib import Path
import mne
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

folder = (
    Path.home() / "Desktop/hackathon/raw_pilot_"
)
print(Path.home())

print("Folder exists:", folder.exists())
print("EEG files:", len(list(folder.glob("*.eeg"))))

raw = mne.io.read_raw_brainvision(
    folder / "CON_01.vhdr",
    preload=False,
    eog=["LVEOG", "RHEOG", "VEOG", "HEOG"]
)

print(raw)
print("Sampling rate:", raw.info["sfreq"])
print("Channels:", len(raw.ch_names))
print("Duration (minutes):", raw.times[-1] / 60)



# ============================================================
# 1. INSPECT CHANNEL NAMES
# ============================================================

print("\nALL CHANNELS")
print(raw.ch_names)

# ============================================================
# 2. CHECK POSTERIOR ELECTRODES
# ============================================================

posterior = [
    "P5", "P6",
    "PO7", "PO8",
    "PO1", "PO2",
    "O1", "O2"
]

print("\nPOSTERIOR ELECTRODES")

for ch in posterior:
    print(f"{ch}: {ch in raw.ch_names}")

# ============================================================
# 3. COUNT EVENT TYPES
# ============================================================

print("\nEVENT COUNTS")

counts = Counter(raw.annotations.description)

for event, count in sorted(counts.items()):
    print(f"{event}: {count}")

# ============================================================
# 4. CONVERT ANNOTATIONS INTO MNE EVENTS
# ============================================================

events, event_id = mne.events_from_annotations(raw)

print("\nEVENT ID MAPPING")
print(event_id)

print("\nEVENTS ARRAY SHAPE")
print(events.shape)

# ============================================================
# 5. SHOW FIRST 30 EVENTS WITH TIMESTAMPS
# ============================================================

print("\nFIRST 30 EVENTS")

for onset, description in zip(
    raw.annotations.onset[:30],
    raw.annotations.description[:30]
):
    print(f"{onset:8.2f} s | {description}")

    # ============================================================
# CREATE EEG EPOCHS FOR TWO CUE CONDITIONS
# ============================================================

cue_ids = {
    "Cue_11": 11,
    "Cue_12": 12
}

epochs = mne.Epochs(
    raw,
    events,
    event_id=cue_ids,
    tmin=0.2,
    tmax=1.6,
    baseline=None,
    picks=["PO7", "PO8"],
    preload=True
)

print("\nEPOCHS")
print(epochs)

print("Cue 11 trials:", len(epochs["Cue_11"]))
print("Cue 12 trials:", len(epochs["Cue_12"]))

# ============================================================
# COMPUTE ALPHA PSD
# ============================================================

spectrum = epochs.compute_psd(
    method="welch",
    fmin=8,
    fmax=13,
    n_fft=250
)

freqs = spectrum.freqs

# Shape: trials × electrodes × frequencies
psd = spectrum.get_data()

# Integrate PSD to obtain alpha-band power in µV²
alpha_power = np.trapezoid(
    psd,
    freqs,
    axis=-1
) * 1e12

print("\nALPHA POWER SHAPE")
print(alpha_power.shape)

# ============================================================
# COMPARE THE TWO CONDITIONS
# ============================================================

for label, code in cue_ids.items():

    mask = epochs.events[:, 2] == code

    mean_power = alpha_power[mask].mean(axis=0)

    print(f"\n{label}")
    print(f"PO7: {mean_power[0]:.3f} µV²")
    print(f"PO8: {mean_power[1]:.3f} µV²")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for i, channel in enumerate(["PO7", "PO8"]):

    cue11 = alpha_power[
        epochs.events[:, 2] == 11, i
    ]

    cue12 = alpha_power[
        epochs.events[:, 2] == 12, i
    ]

    axes[i].boxplot([cue11, cue12])

    axes[i].set_xticks(
        [1, 2],
        ["Cue 11", "Cue 12"]
    )

    axes[i].set(
        title=f"{channel} — Alpha Power",
        ylabel="Alpha power (µV²)"
    )

    axes[i].set_yscale("log")
    axes[i].grid(alpha=0.2)

    print(f"\n{channel}")
    print(f"Cue 11 median: {np.median(cue11):.3f} µV²")
    print(f"Cue 12 median: {np.median(cue12):.3f} µV²")

plt.tight_layout()
plt.show()

# ============================================================
# 1. ASSIGN STANDARD ELECTRODE POSITIONS
# ============================================================

montage = mne.channels.make_standard_montage(
    "colin27_1005"
)

# Identify EEG channels
channel_types = dict(
    zip(raw.ch_names, raw.get_channel_types())
)

# Exclude EOG, mastoid and inferior channels
# from this conventional scalp visualization.
excluded = {
    "M2", "I5", "I3", "Iz", "I4", "I6",
    "SI3", "SIz", "SI4"
}

good_channels = [
    ch for ch in raw.ch_names
    if channel_types[ch] == "eeg"
    and ch in montage.ch_names
    and ch not in excluded
]

eeg = raw.copy().pick(good_channels)

eeg.set_montage(montage)

print("Mapped scalp electrodes:", len(good_channels))

# ============================================================
# 2. CREATE EPOCHS USING ALL MAPPED ELECTRODES
# ============================================================

epochs_all = mne.Epochs(
    eeg,
    events,
    event_id={"Cue 11": 11, "Cue 12": 12},
    tmin=0.2,
    tmax=1.6,
    baseline=None,
    preload=True
)

# ============================================================
# 3. CALCULATE PSD ACROSS FREQUENCIES
# ============================================================

spectrum = epochs_all.compute_psd(
    method="welch",
    fmin=4,
    fmax=30,
    n_fft=250
)

freqs = spectrum.freqs

# Convert V²/Hz to µV²/Hz
psd = spectrum.get_data() * 1e12

print("PSD shape:", psd.shape)

# ============================================================
# 4. DEFINE FREQUENCY BANDS
# ============================================================

bands = {
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30)
}

# ============================================================
# 5. CREATE TOPOGRAPHIC MAPS
# ============================================================

fig, axes = plt.subplots(
    len(bands),
    2,
    figsize=(10, 11),
    layout="constrained"
)

for row, (band_name, (fmin, fmax)) in enumerate(
    bands.items()
):

    # Select frequency bins
    mask = (freqs >= fmin) & (freqs <= fmax)

    # Integrate PSD across the frequency band
    # Result: trials × electrodes
    band_power = np.trapezoid(
        psd[:, :, mask],
        freqs[mask],
        axis=-1
    )

    # Average each condition across its trials
    maps = [
        band_power[
            epochs_all.events[:, 2] == code
        ].mean(axis=0)
        for code in [11, 12]
    ]

    # Same color scale for both cue conditions
    vmax = max(power.max() for power in maps)

    for col, (cue, power) in enumerate(
        zip(["Cue 11", "Cue 12"], maps)
    ):

        ax = axes[row, col]

        im, _ = mne.viz.plot_topomap(
            power,
            epochs_all.info,
            axes=ax,
            vlim=(0, vmax),
            cmap="viridis",
            contours=0,
            show=False
        )

        ax.set_title(f"{band_name} — {cue}")

    fig.colorbar(
        im,
        ax=list(axes[row, :]),
        shrink=0.75,
        label="Band power (µV²)"
    )

plt.show()

# ============================================================
# INSPECT RAW EEG AND EOG
# ============================================================

channels = [
    "Fp1",
    "Fp2",
    "AF4",
    "PO7",
    "PO8",
    "LVEOG",
    "RHEOG"
]

sfreq = raw.info["sfreq"]

start = 13
end = 23

X = raw.get_data(
    picks=channels,
    start=int(start * sfreq),
    stop=int(end * sfreq)
) * 1e6

times = np.arange(X.shape[1]) / sfreq + start

fig, axes = plt.subplots(
    len(channels),
    1,
    figsize=(13, 10),
    sharex=True
)

for ax, channel, signal in zip(
    axes, channels, X
):

    ax.plot(times, signal, linewidth=0.8)

    ax.set_ylabel(
        f"{channel}\nµV",
        rotation=0,
        labelpad=35
    )

    ax.grid(alpha=0.2)

axes[-1].set_xlabel("Time (s)")

fig.suptitle(
    "CON_01 — Frontal EEG, Posterior EEG, and EOG"
)

plt.tight_layout()
plt.show()

# ============================================================
# PREPARE A COPY FOR ICA
# ============================================================

raw_ica = raw.copy().load_data()

# Keep mapped EEG channels plus the EOG references
raw_ica.pick(
    good_channels + ["LVEOG", "RHEOG"]
)

raw_ica.set_montage(
    montage,
    on_missing="ignore"
)

# High-pass filter EEG for ICA fitting
raw_ica.filter(
    l_freq=1.0,
    h_freq=None,
    picks="eeg"
)

# ============================================================
# FIT ICA
# ============================================================

ica = mne.preprocessing.ICA(
    n_components=20,
    method="fastica",
    random_state=97,
    max_iter="auto"
)

ica.fit(
    raw_ica,
    picks="eeg",
    decim=3
)
# ==========================================
# APPLY ICA CLEANING
# ==========================================

# Make a copy of the original recording
raw_clean = raw.copy().load_data()

# Use the same EEG channels used to fit ICA
raw_clean.pick(good_channels + ["LVEOG", "RHEOG"])

# Add electrode positions
raw_clean.set_montage(
    montage,
    on_missing="ignore"
)

# Remove the eye-movement component
ica.apply(raw_clean, exclude=[2])

# Keep EEG channels for analysis
eeg_clean = raw_clean.copy().pick(good_channels)

print("ICA cleaning completed!")
print(eeg_clean)

# ============================================================
# IDENTIFY COMPONENTS ASSOCIATED WITH EOG
# ============================================================

vertical_indices, vertical_scores = ica.find_bads_eog(
    raw_ica,
    ch_name="LVEOG"
)

horizontal_indices, horizontal_scores = ica.find_bads_eog(
    raw_ica,
    ch_name="RHEOG"
)

candidates = sorted(
    set(vertical_indices + horizontal_indices)
)

print("Vertical EOG candidates:", vertical_indices)
print("Horizontal EOG candidates:", horizontal_indices)
print("Combined candidates:", candidates)

# Inspect their scalp patterns
if candidates:
    ica.plot_components(picks=candidates)
else:
    print("No components automatically flagged.")

# Inspect ICA002's detailed properties
ica.plot_properties(
    raw_ica,
    picks=[2]
)

# Compare the signal before and after
# excluding ICA002 (preview only)
ica.plot_overlay(
    raw_ica,
    exclude=[2],
    picks=["Fp1", "Fp2", "AF4", "PO7", "PO8"],
    start=13.0,
    stop=23.0
)

# Copy the original, unfiltered recording
raw_compare = raw.copy().load_data()

raw_compare.pick(
    good_channels + ["LVEOG", "RHEOG"]
)

raw_compare.set_montage(
    montage,
    on_missing="ignore"
)

# Preview removal of ICA002 over 13–23 seconds
ica.plot_overlay(
    raw_compare,
    exclude=[2],
    picks=["Fp1", "Fp2", "AF4", "PO7", "PO8"],
    start=13.0,
    stop=23.0
)



## now corrected 
# Preserve the original recording
raw_clean = raw_compare.copy()

# Remove ICA002's contribution
ica.apply(
    raw_clean,
    exclude=[2]
)

# Select EEG electrodes for analysis
eeg_clean = raw_clean.copy().pick(
    good_channels
)

print(eeg_clean)


# Keep the original untouched
raw_clean = raw.copy().load_data()

# Match the channels used to fit ICA
raw_clean.pick(good_channels + ["LVEOG", "RHEOG"])

raw_clean.set_montage(
    montage,
    on_missing="ignore"
)

# Exclude the ocular component
ica.apply(raw_clean, exclude=[2])

# EEG-only copy for spectral analysis
eeg_clean = raw_clean.copy().pick(good_channels)