
from pathlib import Path
from collections import defaultdict
import os
import re
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mne


# ============================================================
# NEUROCOMPASS — HEALTHY VS STROKE VS VISUAL NEGLECT HEATMAPS
# ============================================================

ROOT = Path(
    os.environ.get(
        "NEUROCOMPASS_DATA_FOLDER",
        "/Users/rafa/Desktop/hackathon/raw_pilot_"
    )
)

OUT = ROOT.parent / "files"
OUT.mkdir(exist_ok=True)

GROUPS = ["CON", "N-", "N+"]

GROUP_NAMES = [
    "Healthy",
    "Stroke, no neglect",
    "Stroke + neglect"
]

# Match the common g.tec Unicorn eight-channel montage.
ELECTRODES = [
    "Fz", "C3", "Cz", "C4",
    "Pz", "PO7", "Oz", "PO8"
]

BANDS = {
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30)
}

CUES = [11, 12, 13, 14, 15, 16]

CUE_LABELS = [
    "Valid / A",
    "Valid / B",
    "Neutral / A*",
    "Neutral / B*",
    "Invalid / B*",
    "Invalid / A*"
]


def identity(name):
    match = re.match(r"^(CON|N\+|N-)_(\d+)", name)

    if match:
        group = match.group(1)
        person = f"{group}_{match.group(2)}"
        return group, person

    return None, None


# ============================================================
# PART 1 — BEHAVIORAL HEATMAP
#
# Reconstructed event-code mapping:
#
# Cue 11 -> Target 21: valid, target side A
# Cue 12 -> Target 22: valid, target side B
#
# Cue 13 -> Target 23: neutral, future target side A
# Cue 14 -> Target 24: neutral, future target side B
#
# Cue 15 -> Target 26: invalid, target side B
# Cue 16 -> Target 25: invalid, target side A
#
# Cue 17/18/19: likely catch/no-target trials
#
# Response 50: approximately target detected
# Response 51: approximately target missed
#
# IMPORTANT:
# This mapping was inferred from marker sequences.
# A/B physical left/right still needs codebook verification.
# ============================================================

trials = defaultdict(
    lambda: defaultdict(lambda: [0, 0])
)

for marker_path in sorted(ROOT.glob("*.vmrk")):

    group, participant = identity(marker_path.stem)

    if group is None:
        continue

    marker_text = marker_path.read_text(encoding="latin-1")

    markers = [
        (int(sample), int(code))
        for code, sample in re.findall(
            r"(?m)^Mk\d+=Stimulus,S\s*(\d+),(\d+)",
            marker_text
        )
    ]

    for index, (_, code) in enumerate(markers):

        if code not in range(11, 17):
            continue

        following = []

        for _, next_code in markers[index + 1:]:

            if 11 <= next_code <= 19:
                break

            following.append(next_code)

        responses = [
            c for c in following
            if c in (50, 51)
        ]

        # Keep trials with exactly one interpretable response.
        if len(responses) != 1:
            continue

        if code in (11, 12):
            condition = "Valid"

        elif code in (13, 14):
            condition = "Neutral"

        else:
            condition = "Invalid"

        # These are TARGET sides, not necessarily cue sides.
        side = "A" if code in (11, 13, 16) else "B"

        detected = int(responses[0] == 50)

        trials[group][(condition, side)][0] += detected
        trials[group][(condition, side)][1] += 1


conditions = ["Valid", "Neutral", "Invalid"]

behavior = np.full((3, 6), np.nan)

for gi, group in enumerate(GROUPS):

    for ci, condition in enumerate(conditions):

        for si, side in enumerate(("A", "B")):

            hits, total = trials[group][(condition, side)]

            if total:
                behavior[gi, 2 * ci + si] = (
                    100 * hits / total
                )


fig, ax = plt.subplots(
    figsize=(13, 5),
    layout="constrained"
)

im = ax.imshow(
    behavior,
    cmap="viridis",
    vmin=35,
    vmax=100,
    aspect="auto"
)

ax.set(
    yticks=np.arange(3),
    yticklabels=GROUP_NAMES,
    xticks=np.arange(6),
    xticklabels=[
        f"{condition}\n{side}"
        for condition in conditions
        for side in ("A", "B")
    ],
    title="Target detection by condition — behavioral results"
)

for gi, group in enumerate(GROUPS):

    for ci, condition in enumerate(conditions):

        for si, side in enumerate(("A", "B")):

            hits, total = trials[group][(condition, side)]
            value = behavior[gi, 2 * ci + si]

            label = (
                f"{value:.1f}%\n{hits}/{total}"
                if total
                else "NA"
            )

            ax.text(
                2 * ci + si,
                gi,
                label,
                ha="center",
                va="center",
                weight="bold",
                color="white" if value < 82 else "black"
            )

fig.colorbar(
    im,
    ax=ax,
    label="Targets detected (%)"
)

fig.savefig(
    OUT / "behavior_all_groups_conditions.png",
    dpi=180
)

plt.close(fig)


print("\nBEHAVIORAL SIGNATURE")
print("Side B minus Side A detection, in percentage points")

for gi, group in enumerate(GROUPS):

    print(f"\n{GROUP_NAMES[gi]}:")

    for ci, condition in enumerate(conditions):

        side_a, side_b = behavior[
            gi,
            2 * ci:2 * ci + 2
        ]

        print(
            f"  {condition:7s}: "
            f"A={side_a:5.1f}%  "
            f"B={side_b:5.1f}%  "
            f"gap={side_b-side_a:+5.1f} points"
        )


# ============================================================
# PART 2 — READ EIGHT IDENTICAL ELECTRODES IN EVERY SUBJECT
#
# An occasional BrainVision header points to the wrong
# .eeg filename. The function below can correct this in a
# TEMPORARY header without changing original recordings.
# ============================================================

def open_brainvision(header_path):

    header = header_path.read_text(encoding="latin-1")

    found = re.search(
        r"(?m)^DataFile=([^\r\n]+)",
        header
    )

    if not found:
        raise ValueError("No DataFile reference")

    data_name = found.group(1).strip()

    if (header_path.parent / data_name).exists():

        raw = mne.io.read_raw_brainvision(
            header_path,
            preload=False,
            verbose="ERROR"
        )

        if not all(
            channel in raw.ch_names
            for channel in ELECTRODES
        ):
            raise ValueError(
                "Missing some of the eight requested electrodes"
            )

        raw.pick(ELECTRODES)
        raw.reorder_channels(ELECTRODES)
        raw.load_data()

        return raw

    # Repair only the temporary header.
    actual_eeg = header_path.with_suffix(".eeg")
    actual_markers = header_path.with_suffix(".vmrk")

    if not (
        actual_eeg.exists()
        and actual_markers.exists()
    ):
        raise FileNotFoundError(
            "Missing matching .eeg or .vmrk"
        )

    with tempfile.TemporaryDirectory() as tmp:

        temporary_folder = Path(tmp)

        (
            temporary_folder / actual_eeg.name
        ).symlink_to(actual_eeg.resolve())

        (
            temporary_folder / actual_markers.name
        ).symlink_to(actual_markers.resolve())

        header = re.sub(
            r"(?m)^DataFile=[^\r\n]+",
            f"DataFile={actual_eeg.name}",
            header
        )

        header = re.sub(
            r"(?m)^MarkerFile=[^\r\n]+",
            f"MarkerFile={actual_markers.name}",
            header
        )

        temporary_header = (
            temporary_folder / header_path.name
        )

        temporary_header.write_text(
            header,
            encoding="latin-1"
        )

        raw = mne.io.read_raw_brainvision(
            temporary_header,
            preload=False,
            verbose="ERROR"
        )

        if not all(
            channel in raw.ch_names
            for channel in ELECTRODES
        ):
            raise ValueError(
                "Missing some of the eight requested electrodes"
            )

        raw.pick(ELECTRODES)
        raw.reorder_channels(ELECTRODES)
        raw.load_data()

        return raw


# ============================================================
# PART 3 — COMPUTE THETA / ALPHA / BETA POWER
#
# Shape:
# Each epoch: 8 channels x frequency bins
#
# For each band:
#   integrate PSD across frequency
#   -> band power in µV² per electrode
#
# Also convert to dB and subtract the eight-site mean.
# This helps compare spatial PATTERNS across people.
#
# Not ICA-cleaned here; residual EOG is possible.
# Epochs exceeding 250 µV peak-to-peak are rejected.
# ============================================================

per_person = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(list)
    )
)

for path in sorted(ROOT.glob("*.vhdr")):

    group, person = identity(path.stem)

    if group is None:
        continue

    try:

        raw = open_brainvision(path)

        # Consistent reference across all recordings.
        raw.set_eeg_reference(
            "average",
            verbose=False
        )

        # Spectral analysis bandpass.
        raw.filter(
            1.,
            35.,
            picks="eeg",
            verbose=False
        )

        events, _ = mne.events_from_annotations(
            raw,
            verbose=False
        )

        events = events[
            np.isin(events[:, 2], CUES)
        ]

        epochs = mne.Epochs(
            raw,
            events,
            event_id={
                str(code): code
                for code in CUES
            },
            tmin=0.2,
            tmax=1.6,
            baseline=None,
            preload=True,
            reject=dict(eeg=250e-6),
            verbose=False
        )

        if len(epochs) == 0:
            raise ValueError(
                "All epochs rejected"
            )

        psd = epochs.compute_psd(
            method="welch",
            fmin=4,
            fmax=30,
            n_fft=250,
            n_per_seg=250,
            verbose=False
        )

        # trials x electrodes x frequencies
        densities = psd.get_data()

        freqs = psd.freqs

        df = freqs[1] - freqs[0]

        for cue in CUES:

            selected_trials = (
                epochs.events[:, 2] == cue
            )

            if selected_trials.sum() < 5:
                continue

            for band, (low, high) in BANDS.items():

                frequency_mask = (
                    (freqs >= low)
                    & (freqs < high)
                )

                # Integrate power spectral density.
                # Conversion: V² -> µV².
                bandpower = (
                    densities[selected_trials][
                        :, :, frequency_mask
                    ].sum(axis=-1)
                    * df
                    * 1e12
                )

                # One power value per electrode.
                power_by_channel = np.median(
                    bandpower,
                    axis=0
                )

                # Convert to decibels.
                logpower = 10 * np.log10(
                    np.maximum(
                        power_by_channel,
                        1e-12
                    )
                )

                # Relative spatial distribution.
                relative_map = (
                    logpower - logpower.mean()
                )

                per_person[
                    (group, person)
                ][(cue, band)]["maps"].append(
                    relative_map
                )

                per_person[
                    (group, person)
                ][(cue, band)]["power"].append(
                    power_by_channel
                )

        print(
            f"OK {path.name}: "
            f"{len(epochs)} usable epochs"
        )

    except (
        OSError,
        ValueError,
        RuntimeError
    ) as error:

        print(
            f"SKIP {path.name}: {error}"
        )


# ============================================================
# PART 4 — GROUP AGGREGATION
#
# First combine a participant's sessions.
# Then calculate the median across participants.
#
# This avoids counting 500 trials from one patient
# as if they were 500 independent patients.
# ============================================================

# group x cue x band x electrode
group_maps = np.full(
    (3, 6, 3, 8),
    np.nan
)

group_powers = np.full(
    (3, 6, 3, 8),
    np.nan
)

for gi, group in enumerate(GROUPS):

    people = sorted(
        person
        for g, person in per_person
        if g == group
    )

    print(
        f"{group}: "
        f"{len(people)} participants "
        f"with usable eight-channel EEG"
    )

    for ci, cue in enumerate(CUES):

        for bi, band in enumerate(BANDS):

            person_maps = []
            person_powers = []

            for person in people:

                entry = per_person[
                    (group, person)
                ][(cue, band)]

                if entry["maps"]:

                    person_maps.append(
                        np.median(
                            entry["maps"],
                            axis=0
                        )
                    )

                    person_powers.append(
                        np.median(
                            entry["power"],
                            axis=0
                        )
                    )

            if person_maps:

                group_maps[
                    gi, ci, bi
                ] = np.median(
                    person_maps,
                    axis=0
                )

                group_powers[
                    gi, ci, bi
                ] = np.median(
                    person_powers,
                    axis=0
                )


# ============================================================
# PART 5 — CREATE EEG TOPOGRAPHIC MAPS
#
# Rows:
#   Healthy
#   Stroke without neglect
#   Stroke with neglect
#
# Columns:
#   Cue 11
#   Cue 12
#   Cue 13
#   Cue 14
#   Cue 15
#   Cue 16
#
# Separate figure for theta, alpha, beta.
# ============================================================

try:

    standard = mne.channels.make_standard_montage(
        "colin27_1020"
    )

except ValueError:

    standard = mne.channels.make_standard_montage(
        "standard_1020"
    )

info = mne.create_info(
    ELECTRODES,
    sfreq=250,
    ch_types="eeg"
)

info.set_montage(standard)


for bi, band in enumerate(BANDS):

    fig, axs = plt.subplots(
        3,
        6,
        figsize=(20, 10),
        layout="constrained"
    )

    finite = group_maps[
        :, :, bi
    ][
        np.isfinite(group_maps[:, :, bi])
    ]

    if finite.size:
        vmax = max(
            1.0,
            np.percentile(
                np.abs(finite),
                95
            )
        )
    else:
        vmax = 1.0

    for gi in range(3):

        for ci in range(6):

            ax = axs[gi, ci]

            values = group_maps[
                gi,
                ci,
                bi
            ]

            if np.isfinite(values).all():

                mne.viz.plot_topomap(
                    values,
                    info,
                    axes=ax,
                    show=False,
                    cmap="RdBu_r",
                    vlim=(-vmax, vmax),
                    sensors=True,
                    extrapolate="head",
                    contours=0
                )

            else:

                ax.text(
                    0.5,
                    0.5,
                    "No usable\nEEG",
                    ha="center",
                    va="center",
                    transform=ax.transAxes
                )

                ax.axis("off")

            ax.set_title(
                f"{GROUP_NAMES[gi]}\n"
                f"{CUE_LABELS[ci]}",
                fontsize=10
            )

    color = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(
            -vmax,
            vmax
        ),
        cmap="RdBu_r"
    )

    fig.colorbar(
        color,
        ax=axs,
        shrink=0.5,
        label=(
            "Band power relative to "
            "eight-electrode mean (dB)"
        )
    )

    fig.suptitle(
        f"{band} spatial maps — "
        "group × cue code",
        fontsize=17
    )

    fig.savefig(
        OUT / (
            f"EEG_{band.lower()}"
            "_group_condition_topomaps.png"
        ),
        dpi=150
    )

    plt.close(fig)


# ============================================================
# PART 6 — DIRECT DIFFERENCE:
# STROKE WITH NEGLECT MINUS HEALTHY
#
# Positive / red:
#   N+ has MORE relative band power at this position.
#
# Negative / blue:
#   N+ has LESS relative band power at this position.
#
# This is descriptive and exploratory, NOT diagnostic.
# ============================================================

fig, axs = plt.subplots(
    3,
    6,
    figsize=(20, 10),
    layout="constrained"
)

# N+ is group index 2.
# Healthy is group index 0.
difference = (
    group_maps[2] - group_maps[0]
)

finite = difference[
    np.isfinite(difference)
]

if finite.size:
    limit = max(
        1.0,
        np.percentile(
            np.abs(finite),
            95
        )
    )
else:
    limit = 1.0


for bi, band in enumerate(BANDS):

    for ci, cue in enumerate(CUES):

        ax = axs[bi, ci]

        values = difference[
            ci,
            bi
        ]

        if np.isfinite(values).all():

            mne.viz.plot_topomap(
                values,
                info,
                axes=ax,
                show=False,
                cmap="RdBu_r",
                vlim=(-limit, limit),
                sensors=True,
                extrapolate="head",
                contours=2
            )

        else:

            ax.text(
                0.5,
                0.5,
                "No usable\ncomparison",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

            ax.axis("off")

        ax.set_title(
            f"{band} — {CUE_LABELS[ci]}",
            fontsize=10
        )

scale = mpl.cm.ScalarMappable(
    norm=mpl.colors.Normalize(
        -limit,
        limit
    ),
    cmap="RdBu_r"
)

fig.colorbar(
    scale,
    ax=axs,
    shrink=0.5,
    label=(
        "N+ minus healthy "
        "spatial band-power difference (dB)"
    )
)

fig.suptitle(
    "Exploratory neglect-vs-healthy EEG differences",
    fontsize=17
)

fig.savefig(
    OUT / "EEG_Nplus_minus_healthy_topomaps.png",
    dpi=150
)

plt.close(fig)


# ============================================================
# PART 7 — SAVE ACTUAL NUMBERS
#
# powers_uV2:
#   absolute theta/alpha/beta power in µV²
#
# spatial_db:
#   relative band-power maps in dB
#
# Each shape:
#   3 groups × 6 cues × 3 bands × 8 electrodes
# ============================================================

np.savez_compressed(
    OUT / "EEG_group_condition_values.npz",
    powers_uV2=group_powers,
    spatial_db=group_maps,
    groups=GROUPS,
    cues=CUES,
    bands=list(BANDS),
    electrodes=ELECTRODES
)


# ============================================================
# PART 8 — EXPLORATORY POSTERIOR ALPHA ASYMMETRY
#
# One candidate EEG feature for NeuroCompass:
#
#   PO8 alpha - PO7 alpha
#
# Compare this between cue 11 and cue 12.
#
# NOTE: Larger resting PO8 power, by itself, does NOT
# establish visual neglect. Cue-dependent modulation
# may be more informative.
# ============================================================

print(
    "\nEXPLORATORY EEG MEASURE: "
    "CUE-DEPENDENT ALPHA LATERALIZATION"
)

for gi, group in enumerate(GROUPS):

    # Index 1 = alpha.
    # Index 5 = PO7.
    # Index 7 = PO8.

    cue11_asymmetry = (
        group_maps[gi, 0, 1, 7]
        - group_maps[gi, 0, 1, 5]
    )

    cue12_asymmetry = (
        group_maps[gi, 1, 1, 7]
        - group_maps[gi, 1, 1, 5]
    )

    cue_modulation = (
        cue11_asymmetry
        - cue12_asymmetry
    )

    print(
        f"{group}: "
        f"Cue 11={cue11_asymmetry:+.2f} dB, "
        f"Cue 12={cue12_asymmetry:+.2f} dB, "
        f"modulation={cue_modulation:+.2f} dB"
    )


print("\nDONE.")

print(
    "All heatmaps and numerical values saved to:",
    OUT
)

print(
    "\nIMPORTANT: Neutral A/B indicates the FUTURE "
    "target side, not a directional neutral cue."
)

print(
    "IMPORTANT: During invalid trials, the cue points "
    "opposite the final target side."
)

print(
    "IMPORTANT: These scalp maps are exploratory "
    "group comparisons, not a clinical classifier."
)

# Target detection asymmetry:
# Side B percentage minus Side A percentage

asymmetry = behavior[:, 1::2] - behavior[:, 0::2]

fig, ax = plt.subplots(figsize=(9, 5))

colors = ["#2E8B57", "#E69F00", "#D55E00"]

for i, group in enumerate(GROUP_NAMES):

    ax.plot(
        conditions,
        asymmetry[i],
        marker="o",
        markersize=9,
        linewidth=2.5,
        color=colors[i],
        label=group
    )

ax.axhline(0, color="black", linewidth=1)

ax.set(
    xlabel="Cue condition",
    ylabel="Side B − Side A (percentage points)",
    title="Spatial detection asymmetry across cue conditions"
)

ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    OUT / "behavior_detection_asymmetry.png",
    dpi=200
)

plt.show()