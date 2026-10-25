"""NeuroCompass: evidence-first 32-electrode feedback feature analysis.

NO GROUP CLASSIFIER, NO TRAIN/TEST SPLIT, NO CLAIM OF DIAGNOSIS.

Input: participant_eeg_bandpowers_emotiv32.npz generated from raw BrainVision
EEG with a 32-channel average reference and 1-35 Hz filter.

Creates participant-level group reference plots, descriptive cohort contrasts,
left-right posterior bandpower metrics, cue-dependent measurements,
subject-level bootstrap intervals, and reusable reference features.

Run:
    python neurocompass_feedback_features.py

Optional for scalp topomaps:
    pip install mne

Limitations:
- Archive contains *band-integrated powers*, not entire frequency spectra or
  raw continuous EEG. Revisit raw .vhdr/.eeg for blink ICA, clean PSD spectra,
  accurate cue/target epoching, and baseline-normalized ERD/ERS.
- Cue A/B is unverified as physical left/right. B-A is an experimental
  contrast, not automatically right-minus-left visual target location.
- Group summaries are descriptive, not calibrated normative target values.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "upload" / "participant_eeg_bandpowers_emotiv32.npz"
OUTPUT = ROOT / "neurocompass_feedback_analysis"

SEED = 42
N_BOOT = 5000
GROUPS = ["CON", "N-", "N+"]
GROUP_LABELS = ["Healthy", "Stroke only", "Stroke + neglect"]
GROUP_COLORS = ["#2c9674", "#e7a400", "#dc5d1b"]
BANDS = ["Theta", "Alpha", "Beta"]
LIMITS = [(4, 8), (8, 13), (13, 30)]
CUES = [11, 12, 13, 14, 15, 16]
PAIRS = {"Valid": (11, 12), "Neutral": (13, 14), "Invalid": (16, 15)}

# Six ROIs from the source stroke/neglect attention-cue paper:
# Lasaponara et al. (2018), J Neurosci. Laterality computed R minus L.
ROIS = {
    "Parietal": (["P7", "CP5"], ["P8", "CP6"]),
    "Occipital": (["PO7", "O1"], ["PO8", "O2"]),
    "Posterior": (["P7", "CP5", "PO7", "O1"], ["P8", "CP6", "PO8", "O2"]),
    # Frontal power is useful as a sensitivity/QC check, but do not use as
    # primary closed-loop feedback without EOG / blink removal validation.
    "Frontal (QC)": (["F7", "FC5"], ["F8", "FC6"]),
}
PRIMARY_ROIS = ["Parietal", "Occipital", "Posterior"]


def median_ci(x, rng, iterations=N_BOOT):
    """Person-level bootstrap interval for median; assumes independent people."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    draws = x[rng.integers(0, len(x), (iterations, len(x)))]
    return np.quantile(np.median(draws, axis=1), [0.025, 0.975])


def contrast_ci(x, y, rng, iterations=N_BOOT):
    """Bootstrap: median group x minus median group y, resampling PEOPLE."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan
    bx = np.median(x[rng.integers(0, len(x), (iterations, len(x)))], axis=1)
    by = np.median(y[rng.integers(0, len(y), (iterations, len(y)))], axis=1)
    differences = bx - by
    lo, hi = np.quantile(differences, [0.025, 0.975])
    observed = np.median(x) - np.median(y)
    sign_stability = np.mean(np.sign(differences) == np.sign(observed))
    return float(lo), float(hi), float(sign_stability)


def load_features():
    with np.load(INPUT, allow_pickle=False) as data:
        groups = data["groups"].astype(str)
        people = data["subjects"].astype(str)
        channels = data["channels"].astype(str).tolist()
        powers = data["powers32"].astype(float)
        counts = data["counts"]
    assert powers.shape == (len(people), 6, 3, 32)
    assert len(channels) == len(set(channels)) == 32
    for left, right in ROIS.values():
        assert set(left + right).issubset(channels)

    logpower = 10 * np.log10(np.maximum(powers, 1e-12))  # dB re 1 µV²
    # This spatial reference removes each person's *across-electrode* dB
    # offset in each cue/band; it does NOT remove volume conduction/artifacts.
    spatial = logpower - np.nanmean(logpower, axis=-1, keepdims=True)

    # Important: pool trials and cues *inside each participant*, then pool
    # participant statistics (never pretend epochs are independent people).
    individual_spatial = np.nanmedian(spatial, axis=1)  # person × band × site
    group_spatial = np.stack([
        np.nanmedian(individual_spatial[groups == g], axis=0)
        for g in GROUPS
    ])

    roi_laterality = {}
    roi_left = {}
    roi_right = {}
    for name, (left, right) in ROIS.items():
        a = np.nanmean(logpower[..., [channels.index(x) for x in left]], axis=-1)
        b = np.nanmean(logpower[..., [channels.index(x) for x in right]], axis=-1)
        roi_left[name], roi_right[name] = a, b
        # right-left logpower is a *dB power ratio*, not a phase measurement.
        roi_laterality[name] = b - a  # person × cue × band

    return groups, people, channels, powers, counts, group_spatial, roi_laterality


def plot_measured_electrode_heatmaps(channels, templates):
    fig, axes = plt.subplots(3, 1, figsize=(17, 8.7), layout="constrained")
    vlim = max(3, float(np.nanpercentile(np.abs(templates), 98)))
    for bi, (ax, band) in enumerate(zip(axes, BANDS)):
        ax.imshow(templates[:, bi], aspect="auto", cmap="RdBu_r",
                  vmin=-vlim, vmax=vlim)
        ax.set_title(f"{band} — electrode values (not scalp interpolation)")
        ax.set_yticks(range(3), GROUP_LABELS)
        ax.set_xticks(range(len(channels)), channels, rotation=75, fontsize=8)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which="minor", axis="y", color="#e2e7e7", linewidth=1.1)
    fig.colorbar(plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=plt.Normalize(-vlim, vlim)),
        ax=axes, shrink=.78, label="Spatial relative bandpower (dB)")
    fig.suptitle("Three cohort reference maps — each cell is an actual electrode")
    fig.savefig(OUTPUT / "01_group_electrode_band_heatmaps.png", dpi=160)
    plt.close(fig)


def plot_residual_electrodes(channels, templates):
    # Between-cohort contrasts are DESCRIPTIVE and may be confounded by
    # participants, lesions, artifacts, etc.; they are not causal isolation.
    comparisons = [templates[1] - templates[0],
                   templates[2] - templates[1]]
    titles = ["Stroke cohort vs healthy: N− minus healthy",
              "Neglect cohort vs stroke-only: N+ minus N−"]
    vlim = max(1.0, float(np.nanpercentile(np.abs(comparisons), 98)))
    fig, axs = plt.subplots(2, 1, figsize=(17, 6), layout="constrained")
    for ax, data, title in zip(axs, comparisons, titles):
        ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_yticks(range(3), BANDS)
        ax.set_xticks(range(len(channels)), channels, rotation=75, fontsize=8)
        ax.set_title(title)
    fig.colorbar(plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=plt.Normalize(-vlim, vlim)),
        ax=axs, shrink=.75, label="Between-group median difference (dB)")
    fig.suptitle("Separate general stroke contrast from a candidate neglect contrast")
    fig.savefig(OUTPUT / "02_stroke_vs_neglect_residual_electrodes.png", dpi=160)
    plt.close(fig)


def scatter_subjects(ax, group, values, rng, title, ylabel=False):
    for gi, gg in enumerate(GROUPS):
        v = values[group == gg]
        jitter = rng.uniform(-.12, .12, size=len(v))
        ax.scatter(np.full(len(v), gi) + jitter, v, s=37,
                   color=GROUP_COLORS[gi], alpha=.85, edgecolor="white", linewidth=.4)
        median = np.nanmedian(v)
        ax.plot([gi-.26, gi+.26], [median]*2, color="#0f2431", lw=2)
    ax.axhline(0, color="#7b8790", lw=1)
    ax.set_xticks(range(3), ["CON", "N−", "N+"], fontsize=9)
    if ylabel:
        ax.set_ylabel("Right − left bandpower (dB)")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=.18)


def plot_roi_laterality(groups, features):
    rng = np.random.default_rng(SEED)
    fig, axs = plt.subplots(3, 3, figsize=(13, 11), layout="constrained")
    for ri, name in enumerate(PRIMARY_ROIS):
        vals = np.nanmedian(features[name], axis=1)  # per person
        for bi, band in enumerate(BANDS):
            scatter_subjects(axs[ri, bi], groups, vals[:, bi], rng,
                             f"{name} · {band}", ylabel=(bi == 0))
    fig.suptitle("Left–right lateralization: every dot is a participant")
    fig.savefig(OUTPUT / "03_subject_level_posterior_lateralization.png", dpi=165)
    plt.close(fig)


def plot_cue_changes(groups, features):
    rng = np.random.default_rng(SEED)
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for bi, band in enumerate(["Theta", "Alpha"]):
        for ci, (condition, (a, b)) in enumerate(PAIRS.items()):
            arr = features["Posterior"]
            vals = arr[:, CUES.index(b), bi] - arr[:, CUES.index(a), bi]
            scatter_subjects(axs[bi, ci], groups, vals, rng,
                             f"{band}: {condition} B−A",
                             ylabel=(ci == 0))
    fig.suptitle("Within-person cue effects — A/B physical side unverified")
    fig.savefig(OUTPUT / "04_posterior_cue_modulation.png", dpi=165)
    plt.close(fig)


def feature_table_and_plot(groups, subjects, features):
    rng = np.random.default_rng(SEED)
    rows = []
    person_rows = []
    for name, arr in features.items():
        person = np.nanmedian(arr, axis=1)
        for bi, band in enumerate(BANDS):
            v = person[:, bi]
            med = [float(np.nanmedian(v[groups == g])) for g in GROUPS]
            lo, hi, stability = contrast_ci(
                v[groups == "N+"], v[groups == "N-"], rng)
            rows.append([name, band, *med,
                         med[1]-med[0], med[2]-med[1],
                         lo, hi, stability])
            for subject, group, value in zip(subjects, groups, v):
                person_rows.append([subject, group, name, band, value])

    hdr = ["ROI", "Band", "CON_median_dB", "Nminus_median_dB",
           "Nplus_median_dB", "Stroke_minus_healthy_dB",
           "Neglect_minus_stroke_dB", "bootstrap_lo_dB",
           "bootstrap_hi_dB", "bootstrap_direction_fraction"]
    with (OUTPUT / "candidate_feedback_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(hdr)
        writer.writerows(rows)
    with (OUTPUT / "participant_roi_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "group", "ROI", "band", "right_minus_left_dB"])
        writer.writerows(person_rows)

    # Honest display of effect size and uncertainty, not a significance claim.
    core = [row for row in rows if row[0] in PRIMARY_ROIS]
    names = [f"{r[0]} {r[1]}" for r in core]
    values = np.array([r[6] for r in core])
    lows = np.array([r[7] for r in core])
    highs = np.array([r[8] for r in core])
    ypos = np.arange(len(core))
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    ax.barh(ypos, values, color=np.where(values >= 0, "#b96551", "#438cba"),
            alpha=.8)
    ax.errorbar(values, ypos, xerr=np.array([values-lows, highs-values]),
                fmt="none", ecolor="#162631", capsize=4, lw=1.5)
    ax.axvline(0, color="#2c3b47")
    ax.set_yticks(ypos, names)
    ax.invert_yaxis()
    ax.set_xlabel("N+ minus N− median laterality (dB); participant bootstrap 95% interval")
    ax.set_title("Candidate EEG feedback metrics — exploratory and uncertain")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(OUTPUT / "05_candidate_features_bootstrap.png", dpi=170)
    plt.close(fig)

    print("\nROI right-minus-left group medians (dB):")
    for row in rows:
        print(f"{row[0]:13s} {row[1]:5s} CON={row[2]:+5.2f} "
              f"N-={row[3]:+5.2f} N+={row[4]:+5.2f} "
              f"Δneglect={row[6]:+5.2f} "
              f"bootstrap95=[{row[7]:+.2f}, {row[8]:+.2f}]")
    return rows


def save_live_references(groups, subjects, channels, features):
    """Reference features for visualization, NOT a clinical classifier.

    Group median and group MAD have shape [group, ROI, band].
    Posterior feedback should first be normalized to the live person's own
    resting and attention-task baseline before making any display decisions.
    """
    roi_names = list(ROIS)
    person_vals = np.stack([
        np.nanmedian(features[name], axis=1) for name in roi_names
    ], axis=1)  # person × ROI × band
    group_median = np.stack([
        np.nanmedian(person_vals[groups == g], axis=0) for g in GROUPS
    ])
    group_mad = np.stack([
        np.nanmedian(np.abs(person_vals[groups == g] -
                            group_median[i]), axis=0)
        for i, g in enumerate(GROUPS)
    ])
    np.savez_compressed(
        OUTPUT / "feedback_roi_references.npz",
        channels=np.array(channels), subjects=np.array(subjects),
        participant_groups=np.array(groups), groups=np.array(GROUPS),
        rois=np.array(roi_names), bands=np.array(BANDS),
        person_right_minus_left_db=person_vals,
        group_median_right_minus_left_db=group_median,
        group_mad_right_minus_left_db=group_mad,
    )


def plot_optional_mne_topomaps(channels, templates):
    """True MNE 10-20 electrode placement when MNE is available."""
    try:
        import mne
    except ImportError:
        print("MNE not installed: skipping optional 32-electrode scalp plots.")
        return
    info = mne.create_info(channels, 256, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    fig, axs = plt.subplots(3, 3, figsize=(11, 10), layout="constrained")
    vmax = max(2.0, np.nanpercentile(np.abs(templates), 98))
    for gi, gg in enumerate(GROUPS):
        for bi, band in enumerate(BANDS):
            mne.viz.plot_topomap(
                templates[gi, bi], info, axes=axs[gi, bi],
                cmap="RdBu_r", vlim=(-vmax, vmax),
                contours=0, sensors=True, show=False, extrapolate="local",
            )
            axs[gi, bi].set_title(f"{GROUP_LABELS[gi]} · {band}")
    fig.colorbar(plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax)),
        ax=axs, shrink=.7, label="Spatial-relative power (dB)")
    fig.savefig(OUTPUT / "06_group_topomaps_32_electrodes.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(11, 7.5), layout="constrained")
    diff = np.stack([templates[1]-templates[0], templates[2]-templates[1]])
    lim = max(1.0, np.nanpercentile(np.abs(diff), 98))
    for i in range(2):
        for bi, band in enumerate(BANDS):
            mne.viz.plot_topomap(
                diff[i, bi], info, axes=axs[i, bi], cmap="RdBu_r",
                vlim=(-lim, lim), contours=0, sensors=True,
                show=False, extrapolate="local",
            )
            axs[i, bi].set_title(
                ("N− − healthy" if i == 0 else "N+ − N−") + f" · {band}")
    fig.colorbar(plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=plt.Normalize(-lim, lim)),
        ax=axs, shrink=.7, label="Between-group median difference (dB)")
    fig.savefig(OUTPUT / "07_stroke_vs_neglect_topomaps.png", dpi=160)
    plt.close(fig)


def live_window_roi_laterality(window_v, sfreq, channels):
    """Compute the SAME type of ROI features for one clean live EEG window.

    window_v: [32 channels, samples] EEG VOLTS, ordered as 'channels'.
    Expects already notch/bandpassed (1–35 Hz), good contact and no gross
    blink/muscle artifact. Does *not* do EOG/ICA or clinical classification.
    The original pilot used Welch 1-second segments in 1.4-second epochs.
    """
    window_v = np.asarray(window_v, float)
    archive_channels = {
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1",
        "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1",
        "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO7", "PO3",
        "PO4", "PO8", "O1", "Oz", "O2",
    }
    if len(channels) != 32 or set(channels) != archive_channels:
        raise ValueError(
            "Live data need the same 32 EEG sites as the archive. "
            "For a 29-site headset, reprocess the archive with the matching "
            "channel set and EEG reference before comparing values."
        )
    if window_v.ndim != 2 or window_v.shape[0] != len(channels):
        raise ValueError("window_v must have shape [32 channels, samples]")
    if not np.all(np.isfinite(window_v)):
        raise ValueError("Bad/NaN samples: reject or clean this window first")
    if window_v.shape[1] < int(sfreq):
        raise ValueError("Need >= 1 second of EEG")
    data = window_v - np.mean(window_v, axis=0, keepdims=True)
    nseg = min(int(round(sfreq)), data.shape[-1])
    freq, psd = welch(data, fs=sfreq, nperseg=nseg,
                      noverlap=nseg//2, axis=-1)
    df = freq[1]-freq[0]
    power = np.stack([
        psd[:, (freq>=lo)&(freq<hi)].sum(axis=1)*df*1e12
        for lo,hi in LIMITS
    ])  # bands × 32 sites in µV²
    db = 10 * np.log10(np.maximum(power, 1e-12))
    laterality = {}
    for name, (left, right) in ROIS.items():
        laterality[name] = np.mean(
            db[:, [channels.index(x) for x in right]], axis=1
        ) - np.mean(
            db[:, [channels.index(x) for x in left]], axis=1
        )
    return laterality


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    groups, people, channels, powers, counts, templates, features = load_features()
    print("Participants:", {g: int(sum(groups == g)) for g in GROUPS})
    print("Channels:", len(channels), "Cue epochs:", int(counts.sum()))
    plot_measured_electrode_heatmaps(channels, templates)
    plot_residual_electrodes(channels, templates)
    plot_roi_laterality(groups, features)
    plot_cue_changes(groups, features)
    feature_table_and_plot(groups, people, features)
    save_live_references(groups, people, channels, features)
    plot_optional_mne_topomaps(channels, templates)
    print("\nNo classifier was trained. Output:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
