"""Decode visual-target detections from the NeuroCompass BrainVision .vmrk files.

Usage (from the hackathon folder on Rafa's Mac):
    python3 neurocompass_behavior.py /Users/rafa/Desktop/hackathon/raw_pilot_

Writes:
    neurocompass_figures/behavior_target_detection.png
    neurocompass_figures/behavior_trials.csv
    neurocompass_figures/behavior_subjects.csv

IMPORTANT: The experimental codebook below is reconstructed from marker
sequences and frequency ratios, NOT confirmed against the acquisition script.
Therefore scalp side labels are 'Side A' and 'Side B', not definitive left/right.

Observed in CON_01 and all 29 recordings:
  11 -> 21  valid A       12 -> 22  valid B
  13 -> 23  neutral A     14 -> 24  neutral B
  15 -> 26  invalid B     16 -> 25  invalid A
  17,18,19 -> no target (catch)

  50 -> target-reported/detected, inferred from target trials
  51 -> no-target-reported/not-detected, inferred from catch trials

Counts in CON_01: 128 each valid, 80 each neutral, 32 each invalid,
32 each catch. Among directional target trials, 128:32 = 80%:20% valid:invalid.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


CUE_CODES = set(range(11, 20))
TARGET_CODES = set(range(21, 27))
RESPONSE_CODES = {50, 51}
SIDE_A_TARGETS = {21, 23, 25}
SIDE_B_TARGETS = {22, 24, 26}

GROUPS = ["CON", "N-", "N+"]
GROUP_LABELS = {
    "CON": "Healthy controls",
    "N-": "Stroke without neglect",
    "N+": "Stroke with neglect",
}


def load_markers(path: Path) -> list[tuple[int, int]]:
    """Return (sample_index, stimulus_code) pairs from a BrainVision marker file."""
    markers = []
    for line in path.read_text(encoding="latin-1").splitlines():
        match = re.match(r"Mk\d+=Stimulus,S\s*(\d+),(\d+)", line)
        if match:
            code, sample = map(int, match.groups())
            markers.append((sample, code))
    return sorted(markers)


def decode_recording(path: Path) -> list[dict]:
    """Split marker stream into cue-initiated trials, retaining uncertainty."""
    subject_match = re.match(r"^(CON|N\+|N-)_(\d+)", path.stem)
    if not subject_match:
        return []

    group, participant_number = subject_match.groups()
    participant = f"{group}_{participant_number}"
    events = load_markers(path)
    cue_indices = [i for i, (_, code) in enumerate(events) if code in CUE_CODES]
    rows = []

    for trial_index, current_i in enumerate(cue_indices):
        end_i = cue_indices[trial_index + 1] if trial_index + 1 < len(cue_indices) else len(events)
        cue_sample, cue_code = events[current_i]
        events_in_trial = events[current_i + 1 : end_i]
        targets = [(s, code) for s, code in events_in_trial if code in TARGET_CODES]
        responses = [(s, code) for s, code in events_in_trial if code in RESPONSE_CODES]

        if cue_code in (11, 12):
            condition = "valid"
        elif cue_code in (13, 14):
            condition = "neutral"
        elif cue_code in (15, 16):
            condition = "invalid"
        else:
            condition = "catch"

        target = targets[0][1] if len(targets) == 1 else None
        response = responses[0][1] if len(responses) == 1 else None
        side = "A" if target in SIDE_A_TARGETS else "B" if target in SIDE_B_TARGETS else "none"

        rows.append({
            "recording": path.stem,
            "group": group,
            "participant": participant,
            "trial": trial_index + 1,
            "cue_code": cue_code,
            "cue_sample": cue_sample,
            "condition_inferred": condition,
            "target_code": target if target is not None else "",
            "side_inferred": side,
            "response_code": response if response is not None else "",
            "detected_inferred": int(response == 50) if response in RESPONSE_CODES else "",
            "target_marker_count": len(targets),
            "response_marker_count": len(responses),
        })
    return rows


def detection_rate(rows: list[dict], group: str | None = None,
                   participant: str | None = None, side: str | None = None,
                   condition: str | None = None) -> tuple[int, int, float]:
    chosen = [r for r in rows if r["side_inferred"] in ("A", "B")
              and r["response_code"] in RESPONSE_CODES
              and (group is None or r["group"] == group)
              and (participant is None or r["participant"] == participant)
              and (side is None or r["side_inferred"] == side)
              and (condition is None or r["condition_inferred"] == condition)]
    successes = sum(r["response_code"] == 50 for r in chosen)
    return successes, len(chosen), (successes / len(chosen) if chosen else np.nan)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_folder", type=Path, help="Unarchived folder containing .vmrk files")
    args = parser.parse_args()

    marker_files = sorted(args.data_folder.rglob("*.vmrk"))
    if not marker_files:
        parser.error(f"No .vmrk files found beneath {args.data_folder}")

    rows = []
    for path in marker_files:
        rows.extend(decode_recording(path))
    if not rows:
        parser.error("Markers found, but no recordings matched CON_*, N-_*, or N+_*.")

    output_folder = args.data_folder.resolve().parent / "neurocompass_figures"
    output_folder.mkdir(parents=True, exist_ok=True)
    save_csv(output_folder / "behavior_trials.csv", rows)

    subjects = []
    for group in GROUPS:
        group_rows = [r for r in rows if r["group"] == group]
        participants = sorted(set(r["participant"] for r in group_rows))
        recordings = sorted(set(r["recording"] for r in group_rows))
        print(f"\n{GROUP_LABELS[group]}: {len(participants)} participants, {len(recordings)} recordings")
        for side in ("A", "B"):
            hits, n, rate = detection_rate(rows, group=group, side=side)
            print(f"  Side {side}: {hits}/{n} detections = {100 * rate:.1f}%")
        for condition in ("valid", "neutral", "invalid"):
            a_hits, a_n, a_rate = detection_rate(rows, group=group, side="A", condition=condition)
            b_hits, b_n, b_rate = detection_rate(rows, group=group, side="B", condition=condition)
            print(f"  {condition:7s}: A={a_hits}/{a_n} ({a_rate:.1%})  "
                  f"B={b_hits}/{b_n} ({b_rate:.1%})")
        catches = [r for r in group_rows if r["condition_inferred"] == "catch"
                   and r["response_code"] in RESPONSE_CODES]
        false_alarms = sum(r["response_code"] == 50 for r in catches)
        print(f"  Catch-trial response 50: {false_alarms}/{len(catches)} "+
              "(consistent with false alarms)")

        for participant in participants:
            a_hits, a_n, a = detection_rate(rows, participant=participant, side="A")
            b_hits, b_n, b = detection_rate(rows, participant=participant, side="B")
            subjects.append({
                "group": group,
                "participant": participant,
                "a_hits": a_hits,
                "a_trials": a_n,
                "a_rate_percent": round(100 * a, 3),
                "b_hits": b_hits,
                "b_trials": b_n,
                "b_rate_percent": round(100 * b, 3),
                "b_minus_a_percentage_points": round(100 * (b - a), 3),
            })

    save_csv(output_folder / "behavior_subjects.csv", subjects)

    # Two independent observations: group-level asymmetry and person-to-person variation.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    x = np.arange(len(GROUPS))
    width = 0.35
    for i, (side, offset, color) in enumerate((("A", -width / 2, "#277DA1"),
                                              ("B", +width / 2, "#F8961E"))):
        rates = [100 * detection_rate(rows, group=group, side=side)[2] for group in GROUPS]
        bars = ax1.bar(x + offset, rates, width, label=f"Side {side}", color=color)
        for bar, rate in zip(bars, rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                     f"{rate:.1f}%", ha="center", va="bottom", fontsize=10)
    ax1.set_xticks(x, ["Healthy\ncontrols", "Stroke\nno neglect", "Stroke\nwith neglect"])
    ax1.set_ylabel("Targets detected (%)")
    ax1.set_ylim(0, 108)
    ax1.set_title("Visual target detection by recording group")
    ax1.legend()
    ax1.grid(axis="y", alpha=.2)
    ax1.set_axisbelow(True)

    patient_subjects = [r for r in subjects if r["group"] == "N+"]
    x2 = np.arange(len(patient_subjects))
    ax2.bar(x2 - width / 2, [r["a_rate_percent"] for r in patient_subjects], width,
            color="#277DA1", label="Side A")
    ax2.bar(x2 + width / 2, [r["b_rate_percent"] for r in patient_subjects], width,
            color="#F8961E", label="Side B")
    ax2.set_xticks(x2, [r["participant"] for r in patient_subjects])
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Targets detected (%)")
    ax2.set_title("Different asymmetries in neglect participants")
    ax2.grid(axis="y", alpha=.2)
    ax2.set_axisbelow(True)

    fig.suptitle("NeuroCompass: detection from recorded behavioral markers", fontsize=15)
    fig.text(.5, -.015, "Codebook inferred, not verified. Side A/B are unlabeled physical sides. "
             "Rates are descriptive, not independent trial-level clinical evidence.",
             ha="center", fontsize=9)
    plot_path = output_folder / "behavior_target_detection.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    problems = [r for r in rows if r["target_marker_count"] > 1 or r["response_marker_count"] > 1]
    missing = [r for r in rows if r["response_code"] == ""]
    print(f"\nTotal cue trials: {len(rows)}; ambiguous trials: {len(problems)}; "
          f"without response markers: {len(missing)}")
    print(f"\nSaved: {plot_path}")
    print(f"Saved: {output_folder / 'behavior_trials.csv'}")
    print(f"Saved: {output_folder / 'behavior_subjects.csv'}")
    print("\nNEXT: Check the acquisition protocol to verify LEFT/RIGHT assignment.")
    print("Do not claim these recordings measure the effect of a NeuroCompass shader.")


if __name__ == "__main__":
    main()
