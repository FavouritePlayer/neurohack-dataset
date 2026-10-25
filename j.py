
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import mne

from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# 1. CONFIGURATION
# ============================================================

FILE = Path(
    "neurocompass_emotiv32_decoder_results/participant_eeg_bandpowers_emotiv32.npz"
)

OUT = Path("neurocompass_32_results")
OUT.mkdir(exist_ok=True)

GROUPS = ["CON", "N-", "N+"]

GROUP_NAMES = {
    "CON": "Healthy",
    "N-": "Stroke, no neglect",
    "N+": "Stroke + neglect"
}

GROUP_COLORS = {
    "CON": "#31936d",
    "N-": "#e8a100",
    "N+": "#db641d"
}

BANDS = ["Theta", "Alpha", "Beta"]

# These correspond to the bands in our previous
# EEG power extraction, not newly computed filters.
BAND_RANGES = {
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30)
}

# Previously examined 8-site reference.
# We will select these sites FROM the same 32-site data.
EIGHT_CHANNELS = [
    "Fz", "C3", "Cz", "C4",
    "Pz", "PO7", "Oz", "PO8"
]

# Electrode pairs selected for spatial attention analysis.
ROIS = {
    "Parietal": {
        "left":  ["P7", "CP5"],
        "right": ["P8", "CP6"]
    },
    "Occipital": {
        "left":  ["PO7", "O1"],
        "right": ["PO8", "O2"]
    },
    "Posterior": {
        "left":  ["P7", "CP5", "PO7", "O1"],
        "right": ["P8", "CP6", "PO8", "O2"]
    },
    "Frontal": {
        "left":  ["F7", "FC5"],
        "right": ["F8", "FC6"]
    }
}

PRIMARY_ROIS = ["Parietal", "Occipital", "Posterior"]

N_BOOT = 3000
RNG = np.random.default_rng(42)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10
})


# ============================================================
# 2. LOAD REAL PARTICIPANT BANDPOWERS
# ============================================================

with np.load(FILE, allow_pickle=False) as data:

    subjects = data["subjects"].astype(str)

    groups = data["groups"].astype(str)

    channels = data["channels"].astype(str).tolist()

    powers = data["powers32"].astype(float)

    counts = data["counts"]


print("Participants:", len(subjects))
print("Channels:", len(channels))
print("Power array:", powers.shape)
print("Cue epochs:", counts.sum())

for group in GROUPS:
    print(group, ":", np.sum(groups == group))


# Expected:
#
# powers:
# (participants, cue conditions, bands, electrodes)
#
# In our archive:
# (14, 6, 3, 32)

assert powers.ndim == 4
assert powers.shape[0] == len(subjects)
assert powers.shape[1] == 6
assert powers.shape[2] == 3
assert powers.shape[3] == len(channels)
assert len(channels) == 32

if not np.all(np.isfinite(powers)):
    raise ValueError("Dataset contains invalid bandpowers.")

if np.any(powers <= 0):
    raise ValueError("Bandpowers must be positive for log conversion.")


# Convert integrated bandpower from µV² to dB.
#
# This is:
# 10 log10(power / 1 µV²)
#
# It is NOT a PSD in dB/Hz.
# The PSD was integrated across each frequency band.

power_db = 10 * np.log10(powers)


# ============================================================
# 3. SPATIAL NORMALIZATION
# ============================================================

# We want two complementary representations:
#
# A) Absolute bandpower:
#    How much power exists at each electrode?
#
# B) Spatial-relative bandpower:
#    Where is power higher or lower relative to the
#    participant's scalp-wide distribution?
#
# Fp1 and Fp2 are excluded from the spatial centering
# because they can be strongly affected by blinks.
#
# IMPORTANT:
# This does not remove blink artifacts.
# It is also NOT an EEG voltage re-reference.

center_channels = [
    ch for ch in channels
    if ch not in ["Fp1", "Fp2"]
]

center_idx = [
    channels.index(ch)
    for ch in center_channels
]

spatial_center = np.median(
    power_db[..., center_idx],
    axis=-1,
    keepdims=True
)

spatial_db = power_db - spatial_center


# ============================================================
# 4. PARTICIPANT-LEVEL AGGREGATION
# ============================================================

# Each participant contributes ONE summary per band/site.
#
# Never treat thousands of epochs as thousands
# of independent people.
#
# Median across the six cue conditions.

person_absolute = np.median(power_db, axis=1)

person_spatial = np.median(spatial_db, axis=1)

# Shapes:
# (participants, 3 bands, 32 electrodes)


def group_medians(person_values):

    return np.stack([
        np.median(
            person_values[groups == group],
            axis=0
        )
        for group in GROUPS
    ])


group_absolute = group_medians(person_absolute)

group_spatial = group_medians(person_spatial)

# Shapes:
# (3 groups, 3 bands, 32 electrodes)


# ============================================================
# 5. CREATE THE REAL 32-ELECTRODE MONTAGE
# ============================================================

montage = mne.channels.make_standard_montage(
    "standard_1020"
)

info32 = mne.create_info(
    ch_names=channels,
    sfreq=256,
    ch_types="eeg"
)

info32.set_montage(
    montage,
    on_missing="raise"
)

idx8 = [
    channels.index(ch)
    for ch in EIGHT_CHANNELS
]

info8 = mne.create_info(
    ch_names=EIGHT_CHANNELS,
    sfreq=256,
    ch_types="eeg"
)

info8.set_montage(
    montage,
    on_missing="raise"
)


def plot_head(values, info, ax, vmin, vmax, cmap="RdBu_r"):

    mne.viz.plot_topomap(
        values,
        info,
        axes=ax,

        cmap=cmap,
        vlim=(vmin, vmax),

        # Circular head visualization
        outlines="head",
        extrapolate="head",

        # Head sphere (meters)
        sphere=(0.0, 0.0, 0.0, 0.105),

        # Smooth interpolation without cubic overshoot
        image_interp="cubic",

        # Avoid artificial edge extremes
        border="mean",

        # Electrode locations
        sensors=True,

        # Contour lines
        contours=0,

        # Image resolution
        res=200,

        show=False
    )

    ax.set_aspect("equal")


def add_colorbar(fig, axes, vmin, vmax, label, cmap="RdBu_r"):

    sm = ScalarMappable(
        norm=Normalize(vmin=vmin, vmax=vmax),
        cmap=cmap
    )

    fig.colorbar(
        sm,
        ax=axes,
        shrink=0.75,
        label=label
    )


def save_figure(fig, filename):

    fig.savefig(
        OUT / filename,
        dpi=180,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


# ============================================================
# FIGURE 1 — ABSOLUTE BANDPOWER
# ============================================================

fig, axs = plt.subplots(
    3, 3,
    figsize=(12, 10),
    layout="constrained"
)

for bi, band in enumerate(BANDS):

    # Shared scale across all 3 groups within a band.
    band_values = group_absolute[:, bi, :]

    vmin = np.percentile(band_values, 2)
    vmax = np.percentile(band_values, 98)

    for gi, group in enumerate(GROUPS):

        ax = axs[gi, bi]

        plot_head(
            group_absolute[gi, bi],
            info32,
            ax,
            vmin,
            vmax,
            cmap="viridis"
        )

        ax.set_title(
            f"{GROUP_NAMES[group]}\n{band}",
            fontsize=11
        )

    add_colorbar(
        fig,
        axs[:, bi],
        vmin,
        vmax,
        f"{band} power (dB re 1 µV²)",
        cmap="viridis"
    )

fig.suptitle(
    "Absolute EEG bandpower — 32 electrodes",
    fontsize=17
)

save_figure(
    fig,
    "01_absolute_bandpower_topomaps.png"
)


# ============================================================
# FIGURE 2 — SPATIAL-RELATIVE TOPOGRAPHIES
# ============================================================

# Important:
# Red means relatively higher power compared
# with that participant's spatial median.
#
# Blue means relatively lower power.
#
# BLUE DOES NOT MEAN INACTIVE BRAIN TISSUE.

lim = max(
    1.0,
    np.percentile(np.abs(group_spatial), 98)
)

fig, axs = plt.subplots(
    3, 3,
    figsize=(12, 11),
    layout="constrained"
)

for gi, group in enumerate(GROUPS):

    for bi, band in enumerate(BANDS):

        ax = axs[gi, bi]

        plot_head(
            group_spatial[gi, bi],
            info32,
            ax,
            -lim,
            lim
        )

        ax.set_title(
            f"{GROUP_NAMES[group]}\n{band}"
        )

add_colorbar(
    fig,
    axs,
    -lim,
    lim,
    "Spatial-relative bandpower (dB)"
)

fig.suptitle(
    "Spatial EEG distributions — 32 electrodes",
    fontsize=17
)

save_figure(
    fig,
    "02_spatial_relative_topomaps.png"
)


# ============================================================
# FIGURE 3 — THREE IMPORTANT GROUP CONTRASTS
# ============================================================

# First:
# Stroke without neglect minus healthy
#
# Second:
# Stroke with neglect minus stroke without neglect
#
# Third:
# Stroke with neglect minus healthy
#
# These are observational cohort contrasts.
# They do not prove that a particular pattern
# is caused by neglect.

stroke_effect = (
    group_spatial[1] - group_spatial[0]
)

neglect_associated = (
    group_spatial[2] - group_spatial[1]
)

total_difference = (
    group_spatial[2] - group_spatial[0]
)

contrasts = np.stack([
    stroke_effect,
    neglect_associated,
    total_difference
])

contrast_names = [
    "Stroke only − Healthy",
    "Stroke + neglect − Stroke only",
    "Stroke + neglect − Healthy"
]

lim = max(
    1.0,
    np.percentile(np.abs(contrasts), 98)
)

fig, axs = plt.subplots(
    3, 3,
    figsize=(12, 11),
    layout="constrained"
)

for ci, contrast_name in enumerate(contrast_names):

    for bi, band in enumerate(BANDS):

        ax = axs[ci, bi]

        plot_head(
            contrasts[ci, bi],
            info32,
            ax,
            -lim,
            lim
        )

        ax.set_title(
            f"{contrast_name}\n{band}",
            fontsize=10
        )

add_colorbar(
    fig,
    axs,
    -lim,
    lim,
    "Between-cohort spatial difference (dB)"
)

fig.suptitle(
    "Separating stroke-cohort and neglect-cohort differences",
    fontsize=16
)

save_figure(
    fig,
    "03_three_group_contrasts.png"
)


# ============================================================
# FIGURE 4 — 8 ELECTRODES VS 32 ELECTRODES
# ============================================================

# We use the SAME dataset and SAME EEG reference.
#
# The only difference in this visualization is
# how many electrode locations are available
# to construct the spatial map.
#
# This is a comparison of spatial sampling,
# NOT proof that a 32-channel system decodes
# neglect more accurately.

difference = neglect_associated

lim = max(
    1.0,
    np.percentile(np.abs(difference), 98)
)

fig, axs = plt.subplots(
    2, 3,
    figsize=(12, 8),
    layout="constrained"
)

for bi, band in enumerate(BANDS):

    # Only 8 electrodes used for interpolation.

    plot_head(
        difference[bi, idx8],
        info8,
        axs[0, bi],
        -lim,
        lim
    )

    axs[0, bi].set_title(
        f"{band}\n8 electrode locations"
    )

    # All 32 electrodes used for interpolation.

    plot_head(
        difference[bi],
        info32,
        axs[1, bi],
        -lim,
        lim
    )

    axs[1, bi].set_title(
        f"{band}\n32 electrode locations"
    )

add_colorbar(
    fig,
    axs,
    -lim,
    lim,
    "N+ − N− spatial contrast (dB)"
)

fig.suptitle(
    "What changes when we visualize the same EEG contrast\n"
    "using 8 versus 32 electrode locations?",
    fontsize=16
)

save_figure(
    fig,
    "04_eight_vs_thirtytwo_electrodes.png"
)


# ============================================================
# 6. COMPUTE RIGHT–LEFT ROI LATERALIZATION
# ============================================================

# Lateralization:
#
# right dB power - left dB power
#
# Positive = greater right power
# Negative = greater left power
#
# Compute it separately for EVERY PARTICIPANT.

roi_values = {}

roi_values_by_cue = {}

for name, sites in ROIS.items():

    left_idx = [
        channels.index(ch)
        for ch in sites["left"]
    ]

    right_idx = [
        channels.index(ch)
        for ch in sites["right"]
    ]

    left_db = np.mean(
        power_db[..., left_idx],
        axis=-1
    )

    right_db = np.mean(
        power_db[..., right_idx],
        axis=-1
    )

    # Shape:
    # participant × cue × band

    asymmetry_by_cue = right_db - left_db

    roi_values_by_cue[name] = asymmetry_by_cue

    # Aggregate inside each person first.
    # Shape: participant × band

    roi_values[name] = np.median(
        asymmetry_by_cue,
        axis=1
    )


# ============================================================
# FIGURE 5 — EVERY PARTICIPANT'S LATERALIZATION
# ============================================================

fig, axs = plt.subplots(
    3, 3,
    figsize=(13, 11),
    layout="constrained"
)

for ri, roi in enumerate(PRIMARY_ROIS):

    values = roi_values[roi]

    for bi, band in enumerate(BANDS):

        ax = axs[ri, bi]

        for gi, group in enumerate(GROUPS):

            mask = groups == group

            person_values = values[mask, bi]

            jitter = RNG.uniform(
                -0.10,
                0.10,
                size=len(person_values)
            )

            ax.scatter(
                gi + jitter,
                person_values,
                s=45,
                color=GROUP_COLORS[group],
                alpha=0.85
            )

            group_median = np.median(person_values)

            ax.plot(
                [gi - 0.24, gi + 0.24],
                [group_median, group_median],
                color="black",
                linewidth=2.4
            )

        ax.axhline(
            0,
            color="gray",
            linewidth=1
        )

        ax.set_xticks(
            range(3),
            ["CON", "N−", "N+"]
        )

        ax.set_title(
            f"{roi} — {band}"
        )

        ax.grid(
            axis="y",
            alpha=0.2
        )

        if bi == 0:
            ax.set_ylabel(
                "Right − left power (dB)"
            )

fig.suptitle(
    "Participant-level spatial EEG asymmetry",
    fontsize=17
)

save_figure(
    fig,
    "05_individual_lateralization.png"
)


# ============================================================
# 7. BOOTSTRAP GROUP DIFFERENCES
# ============================================================

# IMPORTANT:
# Resample PARTICIPANTS, not individual EEG epochs.
#
# Here, we estimate how much the observed cohort
# contrast could change when different participants
# are sampled.
#
# With 4 N+ and 5 N− participants, uncertainty
# will remain substantial.

Nplus = person_spatial[groups == "N+"]

Nminus = person_spatial[groups == "N-"]

plus_idx = RNG.integers(
    0,
    len(Nplus),
    size=(N_BOOT, len(Nplus))
)

minus_idx = RNG.integers(
    0,
    len(Nminus),
    size=(N_BOOT, len(Nminus))
)

boot_plus = np.median(
    Nplus[plus_idx],
    axis=1
)

boot_minus = np.median(
    Nminus[minus_idx],
    axis=1
)

boot_differences = boot_plus - boot_minus

ci_low = np.quantile(
    boot_differences,
    0.025,
    axis=0
)

ci_high = np.quantile(
    boot_differences,
    0.975,
    axis=0
)

# Shape:
# band × electrode


# ============================================================
# FIGURE 6 — ELECTRODE-WISE UNCERTAINTY
# ============================================================

fig, axs = plt.subplots(
    1, 3,
    figsize=(15, 11),
    sharey=True,
    layout="constrained"
)

y = np.arange(len(channels))

for bi, band in enumerate(BANDS):

    ax = axs[bi]

    observed = neglect_associated[bi]

    low = ci_low[bi]
    high = ci_high[bi]

    left_error = np.maximum(
        observed - low,
        0
    )

    right_error = np.maximum(
        high - observed,
        0
    )

    errors = np.vstack([
        left_error,
        right_error
    ])

    ax.errorbar(
        observed,
        y,
        xerr=errors,
        fmt="o",
        markersize=3,
        color="#1b4f72",
        ecolor="#98a7b3",
        elinewidth=1,
        capsize=2
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1
    )

    ax.set_title(band)

    ax.set_xlabel(
        "N+ − N− difference (dB)"
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

axs[0].set_yticks(
    y,
    channels
)

axs[0].invert_yaxis()

fig.suptitle(
    "Neglect-associated electrode differences\n"
    "Participant-bootstrap 95% intervals (exploratory)",
    fontsize=16
)

save_figure(
    fig,
    "06_channelwise_bootstrap_intervals.png"
)


# ============================================================
# FIGURE 7 — CUE-DEPENDENT ALPHA TOPOGRAPHIES
# ============================================================

# We have six recorded cue conditions.
#
# IMPORTANT:
# A/B should NOT be assumed to mean actual
# left/right visual space without verifying
# the dataset's event definitions.
#
# This follows our existing array ordering:
#
# index 0: Valid A
# index 1: Valid B
# index 2: Neutral A
# index 3: Neutral B
# index 4: Invalid B
# index 5: Invalid A
#
# Confirm these mappings against the original
# BrainVision marker/event documentation
# before scientific interpretation.

CUE_PAIRS = {
    "Valid": (0, 1),
    "Neutral": (2, 3),
    "Invalid": (5, 4)
}

ALPHA = BANDS.index("Alpha")

cue_maps = np.zeros((
    len(GROUPS),
    len(CUE_PAIRS),
    len(channels)
))

for gi, group in enumerate(GROUPS):

    mask = groups == group

    for ci, (condition, (a, b)) in enumerate(
        CUE_PAIRS.items()
    ):

        # Per-subject B−A spatial change.

        person_change = (
            spatial_db[mask, b, ALPHA, :]
            -
            spatial_db[mask, a, ALPHA, :]
        )

        cue_maps[gi, ci] = np.median(
            person_change,
            axis=0
        )


lim = max(
    0.5,
    np.percentile(np.abs(cue_maps), 98)
)

fig, axs = plt.subplots(
    3, 3,
    figsize=(12, 11),
    layout="constrained"
)

for gi, group in enumerate(GROUPS):

    for ci, condition in enumerate(CUE_PAIRS):

        ax = axs[gi, ci]

        plot_head(
            cue_maps[gi, ci],
            info32,
            ax,
            -lim,
            lim
        )

        ax.set_title(
            f"{GROUP_NAMES[group]}\n"
            f"{condition}: B−A"
        )

add_colorbar(
    fig,
    axs,
    -lim,
    lim,
    "Within-group alpha power contrast (dB)"
)

fig.suptitle(
    "Cue-dependent alpha spatial changes\n"
    "(A/B physical meaning requires verification)",
    fontsize=16
)

save_figure(
    fig,
    "07_cue_dependent_alpha_topographies.png"
)


# ============================================================
# 8. EXPORT CANDIDATE FEEDBACK METRICS
# ============================================================

rows = []

for roi, values in roi_values.items():

    for bi, band in enumerate(BANDS):

        medians = {
            group: np.median(
                values[groups == group, bi]
            )
            for group in GROUPS
        }

        rows.append({
            "ROI": roi,
            "Band": band,

            "Healthy_RminusL_dB": medians["CON"],

            "Stroke_only_RminusL_dB": medians["N-"],

            "Stroke_neglect_RminusL_dB": medians["N+"],

            "Stroke_minus_healthy_dB":
                medians["N-"] - medians["CON"],

            "Neglect_minus_stroke_dB":
                medians["N+"] - medians["N-"],

            "Neglect_minus_healthy_dB":
                medians["N+"] - medians["CON"]
        })

summary = pd.DataFrame(rows)

summary.to_csv(
    OUT / "candidate_feedback_metrics.csv",
    index=False
)


# Participant-level table for honest inspection.

person_rows = []

for roi, values in roi_values.items():

    for bi, band in enumerate(BANDS):

        for i, subject in enumerate(subjects):

            person_rows.append({
                "Subject": subject,
                "Group": groups[i],
                "ROI": roi,
                "Band": band,
                "RminusL_dB": values[i, bi]
            })

person_table = pd.DataFrame(person_rows)

person_table.to_csv(
    OUT / "participant_lateralization.csv",
    index=False
)


# Save references for the later live demo.
# These are descriptive references, not trained models.

reference_values = np.stack([
    group_medians(
        roi_values[roi]
    )
    for roi in PRIMARY_ROIS
], axis=1)

# Shape:
# group × ROI × band

np.savez_compressed(
    OUT / "neurocompass_32_reference.npz",
    groups=np.array(GROUPS),
    rois=np.array(PRIMARY_ROIS),
    bands=np.array(BANDS),
    channels=np.array(channels),
    reference_lateralization_db=reference_values,
    group_spatial_reference_db=group_spatial
)


print("\nCandidate feedback metrics:")
print(summary.round(2).to_string(index=False))

print("\nSaved figures and reference measurements to:")
print(OUT.resolve())
