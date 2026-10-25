import mne
import numpy as np
import matplotlib.pyplot as plt
from eegdash import EEGDashDataset


# ============================================================
# 1. LOAD SCHNEIDER EEG DATASET
# ============================================================

dataset = EEGDashDataset(
    dataset="ds002034",
    cache_dir="./data",
    query={"subject": "01"}
)

print("Number of recordings:", len(dataset.datasets))

raw = dataset.datasets[0].raw.copy()

raw.load_data()

print("\nSampling frequency:", raw.info["sfreq"])
print("Original channels:", len(raw.ch_names))


# ============================================================
# 2. FIX ELECTRODE NAMES
# ============================================================

raw.rename_channels({
    "FP1": "Fp1",
    "FPz": "Fpz",
    "FP2": "Fp2"
})

# Reference electrodes are not scalp recording locations
raw.drop_channels(["REF1", "REF2"])

# Assign electrode coordinates
raw.set_montage(
    "colin27_1020",
    on_missing="warn"
)


# ============================================================
# 3. SELECT GOOD EEG CHANNELS
# ============================================================

bad_channels = raw.info["bads"].copy()

print("\nBad channels:", bad_channels)

# Retain EEG channels only, excluding marked bad channels.
# This also removes EXG, TRIG, and Status from the analysis.
eeg = raw.copy().pick(
    picks="eeg",
    exclude="bads"
)

print("Good EEG channels:", len(eeg.ch_names))


# ============================================================
# 4. CALCULATE ALPHA POWER
# ============================================================

def calculate_alpha_power(recording):

    spectrum = recording.compute_psd(
        method="welch",
        fmin=8,
        fmax=13,
        picks="eeg",
        n_fft=2048
    )

    psd = spectrum.get_data()

    freqs = spectrum.freqs

    # Integrate PSD across 8–13 Hz
    alpha_power = np.trapezoid(
        psd,
        freqs,
        axis=-1
    )

    # Convert V² to µV²
    alpha_power_uv2 = alpha_power * 1e12

    return alpha_power_uv2


# ============================================================
# 5. ALPHA POWER BEFORE AVERAGE REFERENCING
# ============================================================

print("\nBEFORE AVERAGE REFERENCING")

alpha_before = calculate_alpha_power(eeg)

print("Minimum:", np.min(alpha_before))
print("Median:", np.median(alpha_before))
print("Maximum:", np.max(alpha_before))


# ============================================================
# 6. APPLY AVERAGE REFERENCE
# ============================================================

# Create a separate copy so the original remains unchanged
eeg_avg = eeg.copy()

eeg_avg.set_eeg_reference(
    ref_channels="average",
    projection=False
)

print("\nAverage reference applied.")


# ============================================================
# 7. ALPHA POWER AFTER AVERAGE REFERENCING
# ============================================================

print("\nAFTER AVERAGE REFERENCING")

alpha_after = calculate_alpha_power(eeg_avg)

print("Minimum:", np.min(alpha_after))
print("Median:", np.median(alpha_after))
print("Maximum:", np.max(alpha_after))


# ============================================================
# 8. PO7 AND PO8 ALPHA POWER
# ============================================================

print("\nPOSTERIOR ALPHA POWER")

channels = eeg_avg.ch_names

for channel in ["PO7", "PO8"]:

    index = channels.index(channel)

    print(
        f"{channel}: "
        f"{alpha_after[index]:.4f} µV²"
    )


# ============================================================
# 9. POSTERIOR ALPHA LATERALIZATION
# ============================================================

po7 = alpha_after[channels.index("PO7")]
po8 = alpha_after[channels.index("PO8")]

lateralization = (
    (po8 - po7) /
    (po8 + po7)
)

print("\nPosterior Alpha Lateralization:", lateralization)


# ============================================================
# 10. PLOT BEFORE AND AFTER TOPOGRAPHIES
# ============================================================

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

maps = [
    (alpha_before, "Before Average Reference"),
    (alpha_after, "After Average Reference")
]

for ax, (power, title) in zip(axs, maps):

    power_db = 10 * np.log10(
        np.maximum(power, 1e-12)
    )

    im, cn = mne.viz.plot_topomap(
        power_db,
        eeg.info if "Before" in title else eeg_avg.info,
        axes=ax,
        cmap="RdBu_r",
        contours=6,
        sensors=True,
        extrapolate="head",
        show=False
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Alpha Power (dB re 1 µV²)",
        shrink=0.75
    )

    ax.set_title(title)

fig.suptitle(
    "Subject 01 — Alpha Power (8–13 Hz)",
    fontsize=15
)

plt.show()


# ============================================================
# 11. PRINT ALL ELECTRODE POWERS
# ============================================================

print("\nALPHA POWER BY ELECTRODE")

for ch, power in sorted(
    zip(channels, alpha_after),
    key=lambda item: item[1],
    reverse=True
):

    print(f"{ch:5s}: {power:.3f} µV²")

print(raw.annotations)

for description in np.unique(raw.annotations.description):
    print(description)