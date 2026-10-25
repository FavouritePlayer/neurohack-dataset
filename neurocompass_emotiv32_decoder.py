"""NeuroCompass 32-site EMOTIV Flex pilot EEG analysis and held-out decoder.

Run in the existing hackathon venv:
    pip install mne numpy scipy matplotlib scikit-learn
    python neurocompass_emotiv32_decoder.py

The data location can be changed with NEUROCOMPASS_DATA_FOLDER.
Outputs and a participant cache go in neurocompass_emotiv32_decoder_results.

EMOTIV Flex / Flex 2 uses configurable electrode positions. CHANNELS below is
one balanced 32-site 10-20 layout, NOT a guaranteed factory wire assignment.
Match CHANNELS to the actual 32 cap positions before training on live EEG.

Important: marker A/B physical side semantics remain unverified. This script
classifies research groups; it is NOT a clinical diagnosis or a neglect score.
"""

from collections import defaultdict
from pathlib import Path
import csv
import os
import re
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, balanced_accuracy_score, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get(
    "NEUROCOMPASS_DATA_FOLDER", "/Users/rafa/Desktop/hackathon/raw_pilot_"
))
OUT = ROOT.parent / "neurocompass_emotiv32_decoder_results"
# A symmetric 32-site 10-20 arrangement for a configurable EMOTIV Flex cap.
# It must match the actual wired sensor positions in your Flex session.
CHANNELS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8",
    "CP5", "CP1", "CP2", "CP6",
    "P7", "P3", "Pz", "P4", "P8",
    "PO7", "PO3", "PO4", "PO8",
    "O1", "Oz", "O2",
]
GROUPS = ["CON", "N-", "N+"]
NAMES = ["Healthy", "Stroke, no neglect", "Stroke + neglect"]
BANDS = {"Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}
CUES = [11, 12, 13, 14, 15, 16]
# The ordered pair is A-target cue, B-target cue, not physical L/R yet.
CUE_PAIRS = {"Valid": (11, 12), "Neutral": (13, 14), "Invalid": (16, 15)}
FORCE_REPROCESS = False  # True: recompute BrainVision features rather than use cache.
MAX_BAD_CAP_CHANNELS_TO_INTERPOLATE = 3  # 0 disables interpolation.
SEED = 42


def identity(filename):
    m = re.match(r"^(CON|N\+|N-)_(\d+)", filename)
    if not m:
        return None
    return m.group(1), f"{m.group(1)}_{m.group(2)}"


def read_brainvision(path):
    """Read BrainVision; tolerate header DataFile/MarkerFile name mistakes."""
    import mne

    header = path.read_text(encoding="latin-1")
    df = re.search(r"(?m)^DataFile=([^\r\n]+)", header)
    mf = re.search(r"(?m)^MarkerFile=([^\r\n]+)", header)
    if df is None or mf is None:
        raise ValueError("BrainVision header has no DataFile/MarkerFile")
    if (path.parent / df.group(1).strip()).exists() and (
        path.parent / mf.group(1).strip()
    ).exists():
        return mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")

    eeg_path, vmrk_path = path.with_suffix(".eeg"), path.with_suffix(".vmrk")
    if not eeg_path.exists() or not vmrk_path.exists():
        raise FileNotFoundError(f"Missing matching .eeg or .vmrk for {path.name}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / eeg_path.name).symlink_to(eeg_path.resolve())
        (tmp_path / vmrk_path.name).symlink_to(vmrk_path.resolve())
        header = re.sub(r"(?m)^DataFile=[^\r\n]+", f"DataFile={eeg_path.name}", header)
        header = re.sub(r"(?m)^MarkerFile=[^\r\n]+", f"MarkerFile={vmrk_path.name}", header)
        fixed = tmp_path / path.name
        fixed.write_text(header, encoding="latin-1")
        return mne.io.read_raw_brainvision(fixed, preload=True, verbose="ERROR")


def epoch_bandpowers(data, sfreq):
    """Data [epochs, channels, time] in V -> [epochs, 3, channels] µV²."""
    nperseg = min(int(round(sfreq)), data.shape[-1])
    frequencies, psd = welch(
        data, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1
    )
    df = frequencies[1] - frequencies[0]
    values = []
    for lo, hi in BANDS.values():
        mask = (frequencies >= lo) & (frequencies < hi)
        values.append(psd[..., mask].sum(axis=-1) * df * 1e12)
    return np.stack(values, axis=1)


def make_participant_cache():
    """Return [subject, cue, band, 32 electrodes] for both references."""
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    scalp_positions = {n.casefold() for n in montage.ch_names}
    sessions = defaultdict(lambda: defaultdict(lambda: {"cap": [], "full": [], "count": []}))

    files = sorted(ROOT.glob("*.vhdr"))
    if not files:
        raise FileNotFoundError(f"No .vhdr files found in {ROOT}")

    for i, path in enumerate(files, start=1):
        ident = identity(path.stem)
        if ident is None:
            continue
        try:
            raw = read_brainvision(path)
            missing = [ch for ch in CHANNELS if ch not in raw.ch_names]
            if missing:
                raise ValueError(f"Missing 32-site cap positions: {missing}")
            # True scalp EEG only. Exclude VEOG, HEOG, M1, M2, I3, SI3, etc.
            selected_bads = [ch for ch in CHANNELS if ch in raw.info["bads"]]
            scalp = [
                ch for ch in raw.ch_names
                if ch.casefold() in scalp_positions
                and ch not in ("M1", "M2")
                and (ch not in raw.info["bads"] or ch in selected_bads)
            ]
            if not all(ch in scalp for ch in CHANNELS):
                raise ValueError("Some 32 cap electrodes are not in the scalp montage")
            if len(selected_bads) > MAX_BAD_CAP_CHANNELS_TO_INTERPOLATE:
                raise ValueError(
                    f"Too many bad 32-site electrodes ({len(selected_bads)}): "
                    f"{selected_bads}"
                )
            raw.pick(scalp)
            if selected_bads:
                # Interpolate only a small number of explicitly marked bad channels.
                # Their values are estimates, not measured EEG; report them.
                raw.set_montage(montage, on_missing="ignore")
                raw.interpolate_bads(reset_bads=True, verbose=False)
                print(f"  WARNING: interpolated {selected_bads} in {path.name}")
            # No ICA; same 1–35 Hz filter applied before BOTH reference choices.
            raw.filter(1.0, 35.0, picks="eeg", verbose=False)
            events, _ = mne.events_from_annotations(raw, verbose=False)
            events = events[np.isin(events[:, 2], CUES)]
            if not len(events):
                raise ValueError("No marker events 11–16")
            present_cues = sorted(set(events[:, 2]))
            epochs = mne.Epochs(
                raw, events, event_id={str(c): int(c) for c in present_cues},
                tmin=0.2, tmax=1.6, baseline=None, preload=True,
                reject=None, verbose=False,
            )
            x = epochs.get_data(copy=True)  # [trials, mapped scalp electrodes, samples]
            idx = [raw.ch_names.index(ch) for ch in CHANNELS]
            x32 = x[:, idx, :]

            # Two references from the SAME raw epochs:
            # 1) average of 32 cap positions; 2) all mapped scalp electrodes.
            cap_ref = x32 - x32.mean(axis=1, keepdims=True)
            full_ref = x32 - x.mean(axis=1, keepdims=True)

            # A transparent amplitude check; no claim that all surviving trials are clean.
            ptp = np.ptp(cap_ref, axis=-1).max(axis=1)
            good = np.isfinite(x).all(axis=(1, 2)) & (ptp < 250e-6)
            if not good.any():
                raise ValueError("No epochs passed the 250 µV peak-to-peak check")
            cue_ids = epochs.events[good, 2]
            p8 = epoch_bandpowers(cap_ref[good], raw.info["sfreq"])
            pf = epoch_bandpowers(full_ref[good], raw.info["sfreq"])

            group, subject = ident
            for cue in CUES:
                selected = cue_ids == cue
                if selected.sum() < 3:
                    continue
                sessions[(group, subject)][cue]["cap"].append(np.median(p8[selected], axis=0))
                sessions[(group, subject)][cue]["full"].append(np.median(pf[selected], axis=0))
                sessions[(group, subject)][cue]["count"].append(int(selected.sum()))
            print(
                f"[{i}/{len(files)}] {path.name}: {int(good.sum())} epochs, "
                f"{len(scalp)} full-reference scalp sites"
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"SKIP {path.name}: {exc}")

    people = sorted(sessions, key=lambda x: (GROUPS.index(x[0]), x[1]))
    powers32 = np.full((len(people), len(CUES), 3, len(CHANNELS)), np.nan)
    powersfull = powers32.copy()
    counts = np.zeros((len(people), len(CUES)), dtype=int)
    for pi, person in enumerate(people):
        for ci, cue in enumerate(CUES):
            entry = sessions[person][cue]
            if entry["cap"]:
                powers32[pi, ci] = np.median(entry["cap"], axis=0)
                powersfull[pi, ci] = np.median(entry["full"], axis=0)
                counts[pi, ci] = sum(entry["count"])

    if not people:
        raise RuntimeError("No usable participants; check raw data and event marker codes")
    return (
        np.array([p[0] for p in people]), np.array([p[1] for p in people]),
        powers32, powersfull, counts,
    )


def spatial_db(powers):
    """Log bandpower relative to the 32-site geometric mean; not absolute."""
    db = 10 * np.log10(np.maximum(powers, 1e-12))
    return db - np.nanmean(db, axis=-1, keepdims=True)


def diagnostics(group, people, p8, pfull, counts):
    import mne

    # Transparency: one row per participant, not one row per correlated EEG trial.
    with (OUT / "participant_features.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant", "group", "usable_cue_epochs", "reference", "band",
                         *CHANNELS])
        for i in range(len(people)):
            for ref_name, cube in (("32-site", p8), ("full-scalp", pfull)):
                for bi, band in enumerate(BANDS):
                    x = np.nanmedian(cube[i, :, bi, :], axis=0)
                    writer.writerow([people[i], group[i], int(counts[i].sum()), ref_name,
                                     band, *np.round(x, 6)])

    # 1: Actual µV² values before spatial log-mean normalization.
    fig, axs = plt.subplots(3, 1, figsize=(19, 12), layout="constrained")
    x = np.arange(len(CHANNELS))
    colors = ["#2f8f69", "#e7a000", "#dc5800"]
    for bi, (ax, band) in enumerate(zip(axs, BANDS)):
        for gi, label in enumerate(GROUPS):
            participant_power = np.nanmedian(p8[group == label, :, bi, :], axis=1)
            grp_power = np.nanmedian(participant_power, axis=0)
            ax.bar(x + (gi - 1) * 0.26, grp_power, width=0.25,
                   color=colors[gi], alpha=.86, label=NAMES[gi])
        ax.set(xticks=x, xticklabels=CHANNELS, ylabel="Band power (µV²)",
               title=f"{band}: per-person medians, then group median (32-site reference)")
        ax.tick_params(axis="x", labelrotation=65, labelsize=8)
        ax.grid(axis="y", alpha=.2)
        if bi == 0:
            ax.legend()
    fig.savefig(OUT / "01_true_bandpower_electrodes.png", dpi=170)
    plt.close(fig)

    # 2: Are alpha maps sensitive to how the EEG was referenced?
    info = mne.create_info(CHANNELS, sfreq=250, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    maps = [spatial_db(np.nanmedian(cube, axis=1)) for cube in (p8, pfull)]
    vmax = max(1., np.nanpercentile(np.abs(np.stack(maps)), 95))
    fig, axs = plt.subplots(3, 2, figsize=(8, 11), layout="constrained")
    for gi, label in enumerate(GROUPS):
        for ri, ref_name in enumerate(("32-site reference", "Full-scalp reference")):
            group_map = np.nanmedian(maps[ri][group == label, 1], axis=0)
            mne.viz.plot_topomap(group_map, info, axes=axs[gi, ri], show=False,
                                 cmap="RdBu_r", vlim=(-vmax, vmax), sensors=True,
                                 extrapolate="head")
            axs[gi, ri].set_title(f"{NAMES[gi]}\n{ref_name}", fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(-vmax, vmax), cmap="RdBu_r"
    ), ax=axs, shrink=.6, label="Alpha, relative spatial dB")
    fig.savefig(OUT / "02_alpha_reference_sensitivity.png", dpi=170)
    plt.close(fig)

    # 3: Contrast target side B against A within participants, for each cue type.
    relative = spatial_db(p8)
    contrasts = {}
    for name, (cue_a, cue_b) in CUE_PAIRS.items():
        contrasts[name] = relative[:, CUES.index(cue_b), 1, :] - relative[
            :, CUES.index(cue_a), 1, :]
    all_diffs = np.stack(list(contrasts.values()))
    vmax = max(.5, np.nanpercentile(np.abs(all_diffs), 95))
    fig, axs = plt.subplots(3, 3, figsize=(11, 10), layout="constrained")
    for gi, label in enumerate(GROUPS):
        for ci, (condition, arr) in enumerate(contrasts.items()):
            z = np.nanmedian(arr[group == label], axis=0)
            mne.viz.plot_topomap(z, info, axes=axs[gi, ci], show=False,
                                 cmap="RdBu_r", vlim=(-vmax, vmax), sensors=True,
                                 extrapolate="head")
            axs[gi, ci].set_title(f"{NAMES[gi]}\n{condition}: B − A", fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(-vmax, vmax), cmap="RdBu_r"
    ), ax=axs, shrink=.6, label="Within-participant relative alpha difference (dB)")
    fig.savefig(OUT / "03_alpha_cue_side_differences.png", dpi=170)
    plt.close(fig)

    # 4: Show individual posterior alpha asymmetry, not just group medians.
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    po8, po7 = CHANNELS.index("PO8"), CHANNELS.index("PO7")
    lateralization = np.nanmedian(
        relative[:, :, 1, po8] - relative[:, :, 1, po7], axis=1
    )
    plot_rng = np.random.default_rng(SEED)
    for gi, label in enumerate(GROUPS):
        v = lateralization[group == label]
        jitter = plot_rng.uniform(-.12, .12, size=len(v))
        ax.scatter(np.full(len(v), gi) + jitter, v, color=colors[gi], s=60, alpha=.8)
        ax.plot([gi-.25, gi+.25], [np.nanmedian(v)]*2, color="black", lw=2.4)
    ax.axhline(0, color="gray", lw=1)
    ax.set(xticks=range(3), xticklabels=NAMES, ylabel="PO8 − PO7 alpha (dB)",
           title="Posterior alpha asymmetry: every dot is a participant")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "04_individual_alpha_lateralization.png", dpi=170)
    plt.close(fig)

    # 5: The neglect-associated contrast the demo is actually interested in:
    #    stroke+neglect MINUS stroke-without-neglect (NOT a pure causal neglect effect).
    by_person = np.nanmedian(relative, axis=1)  # [person, band, 32 sites]
    nplus = np.nanmedian(by_person[group == "N+"], axis=0)
    nminus = np.nanmedian(by_person[group == "N-"], axis=0)
    contrast = nplus - nminus
    contrast_lim = max(0.5, float(np.nanmax(np.abs(contrast))))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for bi, (ax, name) in enumerate(zip(axes, BANDS)):
        mne.viz.plot_topomap(
            contrast[bi], info, axes=ax, show=False, cmap="RdBu_r",
            vlim=(-contrast_lim, contrast_lim), sensors=True,
            extrapolate="head",
        )
        ax.set_title(f"{name}: N+ minus N-", fontsize=11)
    fig.colorbar(plt.cm.ScalarMappable(
        norm=plt.Normalize(-contrast_lim, contrast_lim), cmap="RdBu_r"
    ), ax=axes, shrink=.7, label="Neglect-associated spatial power contrast (dB)")
    fig.suptitle("32-site contrast within stroke cohorts (exploratory)")
    fig.savefig(OUT / "06_Nplus_minus_Nminus_topomaps.png", dpi=170)
    plt.close(fig)
    print("Saved five diagnostic figures and participant_features.csv")


def feature_matrix(power_cube, relative=False):
    # One independent feature vector PER PERSON, median over six cue types.
    pooled = np.nanmedian(power_cube, axis=1)  # [subjects, bands, channels]
    logp = 10 * np.log10(np.maximum(pooled, 1e-12))
    if relative:
        logp = logp - np.nanmean(logp, axis=-1, keepdims=True)
    return logp.reshape(logp.shape[0], -1)  # 3 * 32 = 96 features


def save_group_templates(groups, powers):
    """Save 32-electrode references for later visual comparison with live EEG.

    These are group summaries, not diagnostic norms or calibrated live targets.
    Dimensions: [group, band, electrode] and [group, cue, band, electrode].
    """
    person_power = np.nanmedian(powers, axis=1)
    absolute_db = 10 * np.log10(np.maximum(person_power, 1e-12))
    relative_db = spatial_db(person_power)
    cue_relative_db = spatial_db(powers)
    group_absolute = np.stack([
        np.nanmedian(absolute_db[groups == g], axis=0) for g in GROUPS
    ])
    group_relative = np.stack([
        np.nanmedian(relative_db[groups == g], axis=0) for g in GROUPS
    ])
    group_by_cue = np.stack([
        np.nanmedian(cue_relative_db[groups == g], axis=0) for g in GROUPS
    ])
    np.savez_compressed(
        OUT / "group_reference_emotiv32.npz",
        channels=np.array(CHANNELS), groups=np.array(GROUPS),
        bands=np.array(list(BANDS)), cues=np.array(CUES),
        absolute_db=group_absolute,
        spatial_db=group_relative,
        spatial_db_by_cue=group_by_cue,
        nplus_minus_nminus=group_relative[2] - group_relative[1],
        participant_count=np.array([np.count_nonzero(groups == g) for g in GROUPS]),
    )
    print("Saved group_reference_emotiv32.npz for 32-channel visual comparisons")


def build_model(c):
    if c == "LDA":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        clf = LogisticRegression(C=float(c), class_weight="balanced", max_iter=2500)
    # All learned transformations happen INSIDE each cross-validation fold.
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)


def loocv_predictions(x, y, model_choice):
    preds = np.empty(len(y), dtype=object)
    for train_ix, validation_ix in LeaveOneOut().split(x):
        model = build_model(model_choice)
        model.fit(x[train_ix], y[train_ix])
        preds[validation_ix] = model.predict(x[validation_ix])
    return preds


def decoder(groups, subjects, p8, pfull):
    # IMPORTANT: split SUBJECTS once, before scaling/model fitting/hyperparameter tuning.
    rng = np.random.default_rng(SEED)
    heldout = []
    for label in GROUPS:
        ids = np.flatnonzero(groups == label)
        if len(ids) < 3:
            raise ValueError(
                f"Need at least 3 people in group {label} for an untouched "
                f"participant holdout + leave-one-person-out training CV; got {len(ids)}"
            )
        ids = rng.permutation(ids)
        n_test = min(max(1, round(.20 * len(ids))), len(ids) - 2)
        heldout.extend(ids[:n_test])
    test_ix = np.array(sorted(heldout), dtype=int)
    train_ix = np.setdiff1d(np.arange(len(groups)), test_ix)
    print("\nUNTOUCHED TEST PARTICIPANTS RESERVED:")
    print([(subjects[i], groups[i]) for i in test_ix])
    print(f"Training: {len(train_ix)} people | Test: {len(test_ix)} people")

    # Train with the SAME 32-site reference that the real EMOTIV cap can compute.
    # Full-scalp reference is a research sensitivity check only.
    representations = {
        "32-site absolute dB": feature_matrix(p8),
        "32-site spatial dB": feature_matrix(p8, relative=True),
    }
    choices = [0.03, 0.3, 3.0, "LDA"]
    records = []
    for name, x in representations.items():
        for choice in choices:
            try:
                predictions = loocv_predictions(x[train_ix], groups[train_ix], choice)
                score = balanced_accuracy_score(groups[train_ix], predictions)
                records.append((score, name, choice, predictions))
            except ValueError as exc:
                print(f"Model failed: {name} / {choice}: {exc}")

    if not records:
        raise RuntimeError("No decoder could be fitted")
    # Highest leave-one-person-out balanced accuracy; ties preserve prior order.
    records.sort(key=lambda item: -item[0])
    cv_score, best_name, best_choice, cv_pred = records[0]
    print("\nTRAIN-ONLY LEAVE-ONE-PERSON-OUT MODEL COMPARISON")
    for score, name, choice, _ in records:
        print(f"  {name:30s} model={str(choice):4s}  balanced accuracy={score:.3f}")
    print(f"\nSELECTED: {best_name} / {best_choice}")
    print(f"Train-only LOPO balanced accuracy: {cv_score:.3f}")
    print("LOPO cross-validation predictions (training participants only):")
    print(classification_report(groups[train_ix], cv_pred, labels=GROUPS,
                                target_names=NAMES, zero_division=0))

    # ONE final untouched evaluation. Do not tune features or models after seeing it.
    model = build_model(best_choice)
    x = representations[best_name]
    model.fit(x[train_ix], groups[train_ix])
    test_pred = model.predict(x[test_ix])
    test_score = balanced_accuracy_score(groups[test_ix], test_pred)
    print("\n======== FINAL UNTOUCHED TEST ========")
    print(f"Participant-held-out balanced accuracy: {test_score:.3f}")
    print("Chance benchmark for three equally weighted classes: ~0.333")
    print(classification_report(groups[test_ix], test_pred, labels=GROUPS,
                                target_names=NAMES, zero_division=0))
    probabilities = model.predict_proba(x[test_ix])
    for row, ix in enumerate(test_ix):
        formatted = ", ".join(
            f"{label}={probabilities[row, list(model.classes_).index(label)]:.2f}"
            for label in GROUPS
        )
        print(f"  {subjects[ix]}: actual={groups[ix]} predicted={test_pred[row]} [{formatted}]")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, true_labels, predicted, title in (
        (axes[0], groups[train_ix], cv_pred, "Training LOPO predictions"),
        (axes[1], groups[test_ix], test_pred, "Untouched participants"),
    ):
        cm = confusion_matrix(true_labels, predicted, labels=GROUPS)
        ConfusionMatrixDisplay(cm, display_labels=GROUPS).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d"
        )
        ax.set_title(title)
    fig.suptitle(f"EEG-only group decoding (best: {best_name}, {best_choice})")
    fig.savefig(OUT / "05_subject_heldout_decoder_confusions.png", dpi=170)
    plt.close(fig)

    with (OUT / "decoder_holdout_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant", "actual_group", "predicted_group",
                         *[f"prob_{g}" for g in GROUPS]])
        for row, ix in enumerate(test_ix):
            writer.writerow([subjects[ix], groups[ix], test_pred[row], *[
                probabilities[row, list(model.classes_).index(g)] for g in GROUPS
            ]])
    print("\nIMPORTANT: this classifies historical group labels, NOT neglect severity,")
    print("attention direction, shader effectiveness, or a live user's diagnosis.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "participant_eeg_bandpowers_emotiv32.npz"
    if cache.exists() and not FORCE_REPROCESS:
        print(f"Reading previously extracted EEG data: {cache}")
        with np.load(cache, allow_pickle=False) as z:
            if "channels" not in z or list(z["channels"]) != CHANNELS:
                raise ValueError(
                    "Cached montage differs from CHANNELS. Set "
                    "FORCE_REPROCESS=True to recompute with this 32-site layout."
                )
            groups, subjects = z["groups"], z["subjects"]
            p8, pfull, counts = z["powers32"], z["powersfull"], z["counts"]
    else:
        groups, subjects, p8, pfull, counts = make_participant_cache()
        np.savez_compressed(cache, groups=groups, subjects=subjects,
                            channels=np.array(CHANNELS),
                            powers32=p8, powersfull=pfull, counts=counts)

    print("\nSUBJECT COUNTS:", {g: int(sum(groups == g)) for g in GROUPS})
    usable = np.isfinite(np.nanmedian(p8, axis=1)).all(axis=(1, 2)) & np.isfinite(
        np.nanmedian(pfull, axis=1)
    ).all(axis=(1, 2))
    if (~usable).any():
        print("Dropping participants without enough valid EEG for all 96 pooled features:",
              subjects[~usable].tolist())
    groups, subjects = groups[usable], subjects[usable]
    p8, pfull, counts = p8[usable], pfull[usable], counts[usable]
    diagnostics(groups, subjects, p8, pfull, counts)
    save_group_templates(groups, p8)
    decoder(groups, subjects, p8, pfull)
    print(f"\nDONE. Output folder: {OUT}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
