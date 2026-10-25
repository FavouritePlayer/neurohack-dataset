"""NeuroCompass: closed-loop left-contrast adaptation from live/mock EEG swing.

Designed for the standard g.tec Unicorn Hybrid Black montage:
    Fz C3 Cz C4 Pz PO7 Oz PO8 (250 Hz)

Reuses window_features() / quality_ok() from neurocompass_eeg_reference.py so
montage, 1–35 Hz filter, average reference, Welch PSD, and 4 s windows match
the eight-channel reference pipeline EXACTLY.

Posterior alpha index (o.py convention, after avg-ref 8–13 Hz power):
    index = (alpha_PO8 - alpha_PO7) / (alpha_PO8 + alpha_PO7 + eps)
Live cued swing (NOT from archived group templates):
    swing = index(attend_right) - index(attend_left)
Loss toward a healthy target:
    loss = (swing_person - swing_healthy_target)**2

Healthy-target source priority (always printed; no silent magic numbers):
    (a) --healthy-baseline file from a healthy teammate on this rig
    (b) --healthy-target CLI float
    (c) documented fallback: Posterior Alpha Healthy_RminusL_dB from
        neurocompass_32_results/candidate_feedback_metrics.csv

Run:
    python3 neurocompass_adapt.py --mock
    python3 neurocompass_adapt.py --mock --trials-per-condition 6 --held-out-size 4
    python3 neurocompass_adapt.py --lsl-name UniCornDeviceDataStream

Creates (under neurocompass_figures/adapt_<stamp>/ by default):
    eeg_live_state.json          live state (+ shader / swing / loss / grid)
    trials.csv                   per-trial event + alpha log
    raw_eeg_windows.npz          raw anticipatory windows BEFORE filtering
    loop.json copy (optional)    ScreenShade-style left-contrast command

CAVEATS — exploratory engineering demo only:
- Not diagnostic. Not a neglect classifier. Not a treatment decision.
- Cue A/B physical side in archived 32-ch analyses is unverified; this live
  loop uses explicit attend_left / attend_right labels on THIS session only.
- Group CSV fallback is a documented exploratory number (ROI dB laterality),
  not a calibrated normative swing for the Unicorn index.
- Adapt only from measured within-person swing under randomized settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np

from neurocompass_eeg_reference import (
    BANDS,
    CHANNELS,
    WINDOW_SECONDS,
    quality_ok,
    window_features,
)

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "neurocompass_figures"
METRICS_CSV = ROOT / "neurocompass_32_results" / "candidate_feedback_metrics.csv"
DEFAULT_GRID = (0.85, 1.00, 1.15, 1.30)
SFREQ = 250.0
EPS = 1e-12
PO7 = CHANNELS.index("PO7")
PO8 = CHANNELS.index("PO8")
ALPHA_BAND = list(BANDS).index("Alpha")
MOCK_OPTIMUM = 1.15  # documented mock peak (matches NeuroCompass MockEEGSource)
MOCK_GAIN_WIDTH = 0.22
LOSS_NOISE_EPS_DEFAULT = 0.05
TRIAL_CSV_FIELDS = [
    "participant_id",
    "trial_id",
    "block",
    "mode",
    "shader_multiplier",
    "cue_side",
    "target_side",
    "target_present",
    "cue_onset",
    "anticipatory_start",
    "anticipatory_end",
    "target_onset",
    "response_onset",
    "correct",
    "reaction_time_ms",
    "artifact_flag",
    "alpha_po7",
    "alpha_po8",
    "lateralization",
    "eeg_source",
    "quality_ok",
]


# ---------------------------------------------------------------------------
# Alpha index / swing / loss (reuse reference window_features for power)
# ---------------------------------------------------------------------------

def alpha_powers_uv2(data_v: np.ndarray, sfreq: float = SFREQ) -> tuple[float, float]:
    """Return (alpha_PO7, alpha_PO8) in µV² via the shared window_features pipeline."""
    power, _relative = window_features(data_v, sfreq)
    return float(power[ALPHA_BAND, PO7]), float(power[ALPHA_BAND, PO8])


def posterior_alpha_index(alpha_po7: float, alpha_po8: float, eps: float = EPS) -> float:
    """o.py: (PO8 - PO7) / (PO8 + PO7 + eps) on alpha power after avg-ref Welch."""
    return (alpha_po8 - alpha_po7) / (alpha_po8 + alpha_po7 + eps)


def swing_from_indices(index_attend_right: float, index_attend_left: float) -> float:
    """Live cued swing: index(attend_right) - index(attend_left)."""
    return index_attend_right - index_attend_left


def loss_vs_healthy(swing_person: float, swing_healthy_target: float) -> float:
    return (swing_person - swing_healthy_target) ** 2


# ---------------------------------------------------------------------------
# Healthy target resolution (priority a → b → c; always printed)
# ---------------------------------------------------------------------------

@dataclass
class HealthyTarget:
    value: float
    source: str
    detail: str


def load_healthy_baseline(path: Path) -> HealthyTarget:
    """(a) Baseline healthy teammate file measured on this rig.

    Accepts JSON with key swing / swing_healthy / swing_healthy_target,
    or a one-column / labeled CSV with the same field names.
    """
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".json" or text.startswith("{"):
        payload = json.loads(text)
        for key in ("swing_healthy_target", "swing_healthy", "swing"):
            if key in payload:
                return HealthyTarget(
                    float(payload[key]),
                    "baseline_file",
                    f"{path} key={key}",
                )
        raise ValueError(
            f"Baseline JSON {path} must contain swing / swing_healthy / "
            "swing_healthy_target"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        if "," in sample or "swing" in sample.lower():
            reader = csv.DictReader(handle)
            row = next(reader)
            for key in ("swing_healthy_target", "swing_healthy", "swing"):
                if key in row and row[key] not in (None, ""):
                    return HealthyTarget(
                        float(row[key]),
                        "baseline_file",
                        f"{path} CSV key={key}",
                    )
            raise ValueError(f"Baseline CSV {path} missing swing column")
        value = float(text.splitlines()[0].strip())
        return HealthyTarget(value, "baseline_file", f"{path} plain float")


def fallback_healthy_from_metrics(csv_path: Path = METRICS_CSV) -> HealthyTarget:
    """(c) Documented fallback: Posterior Alpha Healthy_RminusL_dB.

    This is ROI right-minus-left dB laterality from the 32-ch archive summary —
    exploratory, not a calibrated Unicorn swing index. Printed as such.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing metrics CSV for fallback healthy target: {csv_path}"
        )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            roi = (row.get("ROI") or "").strip()
            band = (row.get("Band") or "").strip()
            if roi == "Posterior" and band == "Alpha":
                key = "Healthy_RminusL_dB"
                if key not in row:
                    raise KeyError(f"{csv_path} missing column {key}")
                value = float(row[key])
                return HealthyTarget(
                    value,
                    "metrics_csv_fallback",
                    (
                        f"{csv_path.name} Posterior/Alpha {key}={value:.6f} "
                        "(exploratory ROI dB laterality; not a Unicorn swing "
                        "calibration)"
                    ),
                )
    raise ValueError(f"No Posterior/Alpha row in {csv_path}")


def resolve_healthy_target(
    baseline: Path | None,
    cli_value: float | None,
) -> HealthyTarget:
    if baseline is not None:
        target = load_healthy_baseline(baseline)
        print(f"Healthy target source: (a) baseline file → {target.value:.6f}")
        print(f"  detail: {target.detail}")
        return target
    if cli_value is not None:
        target = HealthyTarget(
            float(cli_value),
            "cli",
            f"--healthy-target {cli_value}",
        )
        print(f"Healthy target source: (b) CLI → {target.value:.6f}")
        print(f"  detail: {target.detail}")
        return target
    target = fallback_healthy_from_metrics()
    print(f"Healthy target source: (c) documented CSV fallback → {target.value:.6f}")
    print(f"  detail: {target.detail}")
    return target


# ---------------------------------------------------------------------------
# Shader stub (ScreenShade loop.json-compatible)
# ---------------------------------------------------------------------------

@dataclass
class ShaderCommand:
    left_contrast_multiplier: float
    enabled: bool = True
    updated_at: str = ""

    def to_loop_json(self) -> dict:
        return {
            "leftContrastMultiplier": self.left_contrast_multiplier,
            "enabled": self.enabled,
            "updatedAt": self.updated_at
            or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }


class ShaderDriver:
    """Small seam for ScreenShade later: emit command + optional loop.json write."""

    def __init__(self, loop_path: Path | None, also_copy_under: Path | None = None):
        self.loop_path = loop_path
        self.also_copy_under = also_copy_under
        self.last: ShaderCommand | None = None

    def apply(self, left_contrast_multiplier: float, enabled: bool = True) -> ShaderCommand:
        cmd = ShaderCommand(
            left_contrast_multiplier=float(left_contrast_multiplier),
            enabled=enabled,
        )
        self.last = cmd
        payload = cmd.to_loop_json()
        text = json.dumps(payload, indent=2)
        print(
            f"Shader command: leftContrastMultiplier={cmd.left_contrast_multiplier:.2f} "
            f"enabled={cmd.enabled}"
        )
        if self.loop_path is not None:
            self.loop_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.loop_path.with_suffix(".tmp.json")
            tmp.write_text(text + "\n", encoding="utf-8")
            tmp.replace(self.loop_path)
        if self.also_copy_under is not None:
            copy = self.also_copy_under / "loop.json"
            copy.write_text(text + "\n", encoding="utf-8")
        return cmd


# ---------------------------------------------------------------------------
# EEG sources
# ---------------------------------------------------------------------------

@dataclass
class AnticipatoryWindow:
    raw_v: np.ndarray  # (8, samples) volts, BEFORE filtering
    cue_side: str
    shader_multiplier: float
    t0: float
    t1: float
    source: str  # live | mock


class EEGSource(Protocol):
    kind: str

    def anticipatory_window(
        self,
        cue_side: str,
        shader_multiplier: float,
        duration_s: float = WINDOW_SECONDS,
    ) -> AnticipatoryWindow:
        ...


class MockEEGSource:
    """Labeled synthetic Unicorn-like stream. Always reports kind='mock'.

    Documented demo bias (same curve as NeuroCompass MockEEGSource.swift):
        gain = exp(-((m - 1.15) / 0.22)^2)
    Stronger gain → larger attend-side alpha asymmetry → swing closer to the
    healthy target near ×1.15 when using the CSV fallback magnitude.
    """

    kind = "mock"

    def __init__(self, rng: np.random.Generator, target_swing: float):
        self.rng = rng
        self.target_swing = float(target_swing)
        print("EEG source: MOCK (labeled synthetic stream — not a headset)")

    @staticmethod
    def response_gain(multiplier: float) -> float:
        return math.exp(-(((multiplier - MOCK_OPTIMUM) / MOCK_GAIN_WIDTH) ** 2))

    def anticipatory_window(
        self,
        cue_side: str,
        shader_multiplier: float,
        duration_s: float = WINDOW_SECONDS,
    ) -> AnticipatoryWindow:
        n = max(8, int(round(duration_s * SFREQ)))
        t = np.arange(n, dtype=float) / SFREQ
        # Keep 8–13 Hz energy on PO7/PO8 only so avg-ref does not remix alpha
        # from other sites. Other channels use ~18 Hz + slow drift (still < PTP gate).
        alpha = 2.0 * np.pi * 10.0 * t
        other = 2.0 * np.pi * 18.0 * t
        data_uv = np.zeros((len(CHANNELS), n), dtype=float)
        gain = self.response_gain(shader_multiplier)
        # Desired Welch swing ≈ target_swing * gain → min-loss locks near ×1.15.
        # Amplitude→Welch index is compressive/expansive after avg-ref; AMP_SCALE
        # (0.40) is a documented mock calibration so peak swing ≈ healthy target.
        AMP_SCALE = 0.40
        half = 0.5 * self.target_swing * gain * AMP_SCALE
        base = 8.0
        if cue_side == "left":
            amp_index = -half
        elif cue_side == "right":
            amp_index = half
        else:
            raise ValueError(f"cue_side must be left|right, got {cue_side!r}")
        amp_index = float(np.clip(amp_index, -0.55, 0.55))
        po7_amp = base * (1.0 - amp_index)
        po8_amp = base * (1.0 + amp_index)

        for ch in range(len(CHANNELS)):
            data_uv[ch] = 4.0 * np.sin(other + 0.15 * ch)
            data_uv[ch] += 1.2 * np.sin(2.0 * np.pi * 3.0 * t + 0.1 * ch)
            data_uv[ch] += 0.35 * self.rng.normal(size=n)
        data_uv[PO7] = po7_amp * np.sin(alpha) + 0.25 * self.rng.normal(size=n)
        data_uv[PO8] = po8_amp * np.sin(alpha) + 0.25 * self.rng.normal(size=n)

        raw_v = data_uv * 1e-6
        now = time.time()
        return AnticipatoryWindow(
            raw_v=raw_v,
            cue_side=cue_side,
            shader_multiplier=float(shader_multiplier),
            t0=now,
            t1=now + duration_s,
            source="mock",
        )


class LSLEEGSource:
    """Reuse the LSL pull pattern from neurocompass_eeg_reference.live()."""

    kind = "live"

    def __init__(self, lsl_name: str | None, units: str):
        try:
            from pylsl import StreamInlet, resolve_byprop
        except ImportError as exc:
            raise RuntimeError(
                "Install pylsl for live EEG: python3 -m pip install pylsl"
            ) from exc
        if lsl_name:
            available = resolve_byprop("name", lsl_name, timeout=6)
        else:
            available = resolve_byprop("type", "EEG", timeout=6)
        if not available:
            raise RuntimeError(
                "No matching LSL stream. Start Unicorn LSL Interface, or pass --mock."
            )
        meta = available[0]
        print(
            f"EEG source: LIVE LSL {meta.name()} "
            f"({meta.channel_count()} ch, {meta.nominal_srate()} Hz)"
        )
        if meta.channel_count() < 8:
            raise ValueError("LSL stream contains fewer than 8 channels")
        if abs(meta.nominal_srate() - SFREQ) > 1:
            raise ValueError("This controller requires the Unicorn 250 Hz stream")
        print(f"Assumed incoming EEG units: {units}; VERIFY against device docs.")
        self.inlet = StreamInlet(meta)
        self.units = units
        self.buffer: deque[np.ndarray] = deque(maxlen=round(WINDOW_SECONDS * SFREQ))

    def _pull_until(self, n_samples: int, timeout_s: float) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while len(self.buffer) < n_samples and time.monotonic() < deadline:
            values, _ts = self.inlet.pull_chunk(timeout=0.3, max_samples=128)
            if values:
                array = np.asarray(values, dtype=float)
                for sample in array[:, :8]:
                    self.buffer.append(sample)
        if len(self.buffer) < n_samples:
            raise TimeoutError(
                f"Only collected {len(self.buffer)}/{n_samples} LSL samples"
            )
        live = np.asarray(list(self.buffer)[-n_samples:], dtype=float).T
        if self.units == "uv":
            live = live * 1e-6
        return live

    def anticipatory_window(
        self,
        cue_side: str,
        shader_multiplier: float,
        duration_s: float = WINDOW_SECONDS,
    ) -> AnticipatoryWindow:
        n = round(duration_s * SFREQ)
        print(
            f"  LIVE: collecting {duration_s:.1f}s anticipatory window "
            f"(cue={cue_side}, shader×{shader_multiplier:.2f})…"
        )
        t0 = time.time()
        raw_v = self._pull_until(n, timeout_s=duration_s + 8.0)
        t1 = time.time()
        return AnticipatoryWindow(
            raw_v=raw_v,
            cue_side=cue_side,
            shader_multiplier=float(shader_multiplier),
            t0=t0,
            t1=t1,
            source="live",
        )


# ---------------------------------------------------------------------------
# Trial + measurement
# ---------------------------------------------------------------------------

@dataclass
class TrialRecord:
    participant_id: str
    trial_id: int
    block: str
    mode: str
    shader_multiplier: float
    cue_side: str
    target_side: str
    target_present: bool
    cue_onset: float
    anticipatory_start: float
    anticipatory_end: float
    target_onset: float
    response_onset: float | None
    correct: bool | None
    reaction_time_ms: float | None
    artifact_flag: bool
    alpha_po7: float | None
    alpha_po8: float | None
    lateralization: float | None
    eeg_source: str
    quality_ok: bool


@dataclass
class SettingResult:
    shader_multiplier: float
    swing: float
    loss: float
    index_attend_left: float
    index_attend_right: float
    n_left_ok: int
    n_right_ok: int
    uncertain: bool
    reason: str
    block: str


@dataclass
class AdaptSession:
    participant_id: str
    eeg_source: str
    healthy_target: float
    healthy_target_source: str
    healthy_target_detail: str
    grid: list[float]
    calibration_order: list[float]
    calibration_results: list[SettingResult] = field(default_factory=list)
    locked_multiplier: float = 1.0
    locked_uncertain: bool = True
    lock_message: str = ""
    heldout_results: list[SettingResult] = field(default_factory=list)
    trials: list[TrialRecord] = field(default_factory=list)
    raw_windows: list[np.ndarray] = field(default_factory=list)
    raw_meta: list[dict] = field(default_factory=list)


def decode_window(raw_v: np.ndarray) -> tuple[bool, float | None, float | None, float | None]:
    """Gate with quality_ok; return alpha PO7/PO8 + index. Raw is untouched."""
    if not quality_ok(raw_v):
        return False, None, None, None
    a7, a8 = alpha_powers_uv2(raw_v, SFREQ)
    return True, a7, a8, posterior_alpha_index(a7, a8)


def run_cued_block(
    source: EEGSource,
    shader: ShaderDriver,
    session: AdaptSession,
    *,
    multiplier: float,
    trials_per_condition: int,
    block: str,
    mode: str,
    rng: random.Random,
) -> list[TrialRecord]:
    """Balanced attend-left / attend-right anticipatory trials under one setting."""
    shader.apply(multiplier)
    cues = (["left"] * trials_per_condition) + (["right"] * trials_per_condition)
    rng.shuffle(cues)
    block_trials: list[TrialRecord] = []
    for cue in cues:
        target_side = cue  # valid cue for this exploratory loop
        target_present = True
        cue_onset = time.time()
        window = source.anticipatory_window(cue, multiplier)
        ok, a7, a8, lat = decode_window(window.raw_v)
        session.raw_windows.append(window.raw_v.copy())
        session.raw_meta.append(
            {
                "trial_id": len(session.trials) + 1,
                "block": block,
                "cue_side": cue,
                "shader_multiplier": multiplier,
                "eeg_source": window.source,
                "t0": window.t0,
                "t1": window.t1,
                "quality_ok": ok,
            }
        )
        # Simulated behavioral response for CSV completeness (not used in loss).
        response_onset = window.t1 + 0.35 + rng.random() * 0.25
        rt_ms = (response_onset - window.t1) * 1000.0
        trial = TrialRecord(
            participant_id=session.participant_id,
            trial_id=len(session.trials) + 1,
            block=block,
            mode=mode,
            shader_multiplier=multiplier,
            cue_side=cue,
            target_side=target_side,
            target_present=target_present,
            cue_onset=cue_onset,
            anticipatory_start=window.t0,
            anticipatory_end=window.t1,
            target_onset=window.t1,
            response_onset=response_onset,
            correct=True if ok else None,
            reaction_time_ms=rt_ms if ok else None,
            artifact_flag=not ok,
            alpha_po7=a7,
            alpha_po8=a8,
            lateralization=lat,
            eeg_source=window.source,
            quality_ok=ok,
        )
        session.trials.append(trial)
        block_trials.append(trial)
        tag = "MOCK" if window.source == "mock" else "LIVE"
        status = "ok" if ok else "ARTIFACT"
        lat_s = f"{lat:+.4f}" if lat is not None else "n/a"
        print(
            f"  [{tag}] trial {trial.trial_id} cue={cue:5s} ×{multiplier:.2f} "
            f"{status} index={lat_s}"
        )
    return block_trials


def measure_swing_and_loss(
    source: EEGSource,
    shader: ShaderDriver,
    session: AdaptSession,
    shader_setting: float,
    *,
    trials_per_condition: int,
    block: str,
    mode: str,
    rng: random.Random,
) -> SettingResult:
    """Clean seam: apply shader → cued block → swing → loss vs healthy target."""
    trials = run_cued_block(
        source,
        shader,
        session,
        multiplier=shader_setting,
        trials_per_condition=trials_per_condition,
        block=block,
        mode=mode,
        rng=rng,
    )
    left = [t for t in trials if t.cue_side == "left" and t.quality_ok and t.lateralization is not None]
    right = [t for t in trials if t.cue_side == "right" and t.quality_ok and t.lateralization is not None]
    uncertain = len(left) == 0 or len(right) == 0
    if uncertain:
        result = SettingResult(
            shader_multiplier=shader_setting,
            swing=float("nan"),
            loss=float("inf"),
            index_attend_left=float("nan"),
            index_attend_right=float("nan"),
            n_left_ok=len(left),
            n_right_ok=len(right),
            uncertain=True,
            reason="Need ≥1 clean anticipatory window per cue side",
            block=block,
        )
    else:
        idx_l = float(np.mean([t.lateralization for t in left]))
        idx_r = float(np.mean([t.lateralization for t in right]))
        swing = swing_from_indices(idx_r, idx_l)
        loss = loss_vs_healthy(swing, session.healthy_target)
        result = SettingResult(
            shader_multiplier=shader_setting,
            swing=swing,
            loss=loss,
            index_attend_left=idx_l,
            index_attend_right=idx_r,
            n_left_ok=len(left),
            n_right_ok=len(right),
            uncertain=False,
            reason="ok",
            block=block,
        )
    print(
        f"  → swing={result.swing if not uncertain else float('nan'):+.4f} "
        f"loss={result.loss if math.isfinite(result.loss) else float('inf'):.6f} "
        f"(L_ok={result.n_left_ok} R_ok={result.n_right_ok})"
    )
    return result


def pick_lock(
    results: list[SettingResult],
    noise_eps: float,
) -> tuple[float, bool, str]:
    """Min-loss lock; if within noise of runner-up → neutral 1.00."""
    valid = [r for r in results if not r.uncertain and math.isfinite(r.loss)]
    if not valid:
        return (
            1.0,
            True,
            "No usable calibration settings — defaulting to neutral ×1.00.",
        )
    ranked = sorted(valid, key=lambda r: (r.loss, abs(r.shader_multiplier - 1.0)))
    best = ranked[0]
    if len(ranked) >= 2:
        second = ranked[1]
        delta = abs(best.loss - second.loss)
        if delta < noise_eps:
            return (
                1.0,
                True,
                (
                    f"Top settings nearly tied in loss "
                    f"(Δ={delta:.6f} < noise_eps={noise_eps:.6f}) — "
                    f"defaulting to neutral ×1.00 (not locking ×{best.shader_multiplier:.2f})."
                ),
            )
    return (
        best.shader_multiplier,
        False,
        (
            f"Locked left-contrast ×{best.shader_multiplier:.2f} "
            f"(swing={best.swing:+.4f}, loss={best.loss:.6f})."
        ),
    )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_trials_csv(path: Path, trials: list[TrialRecord]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_CSV_FIELDS)
        writer.writeheader()
        for trial in trials:
            row = asdict(trial)
            writer.writerow({k: row[k] for k in TRIAL_CSV_FIELDS})


def write_raw_npz(path: Path, session: AdaptSession) -> None:
    if not session.raw_windows:
        return
    stacked = np.stack(session.raw_windows, axis=0)  # trials × 8 × samples
    np.savez_compressed(
        path,
        raw_windows_v=stacked,
        channels=np.array(CHANNELS),
        sfreq=SFREQ,
        note="Raw anticipatory EEG in volts BEFORE average-ref / 1-35 Hz filter",
        meta_json=np.array(json.dumps(session.raw_meta)),
    )


def setting_to_dict(r: SettingResult) -> dict:
    return {
        "shader_multiplier": r.shader_multiplier,
        "swing": None if (isinstance(r.swing, float) and math.isnan(r.swing)) else r.swing,
        "loss": None if not math.isfinite(r.loss) else r.loss,
        "index_attend_left": None
        if (isinstance(r.index_attend_left, float) and math.isnan(r.index_attend_left))
        else r.index_attend_left,
        "index_attend_right": None
        if (isinstance(r.index_attend_right, float) and math.isnan(r.index_attend_right))
        else r.index_attend_right,
        "n_left_ok": r.n_left_ok,
        "n_right_ok": r.n_right_ok,
        "uncertain": r.uncertain,
        "reason": r.reason,
        "block": r.block,
    }


def latest_clean_maps(session: AdaptSession) -> tuple[np.ndarray | None, np.ndarray | None]:
    for trial, raw in zip(reversed(session.trials), reversed(session.raw_windows)):
        if trial.quality_ok:
            return window_features(raw, SFREQ)
    return None, None


def write_state_json(
    path: Path,
    session: AdaptSession,
    shader: ShaderDriver,
    status: str,
) -> None:
    powers, maps = latest_clean_maps(session)
    bandpowers = None
    relative = None
    distances = None
    if powers is not None and maps is not None:
        bandpowers = {
            band: dict(zip(CHANNELS, map(float, powers[i])))
            for i, band in enumerate(BANDS)
        }
        relative = {
            band: dict(zip(CHANNELS, map(float, maps[i])))
            for i, band in enumerate(BANDS)
        }
        # Optional descriptive distances if templates exist (never drive lock).
        try:
            from neurocompass_eeg_reference import get_templates

            templates = get_templates(FIGURES)
            distances = {}
            for i, group in enumerate(["CON", "N-", "N+"]):
                if np.all(np.isfinite(templates[i])):
                    distances[group] = round(
                        float(np.linalg.norm(maps - templates[i])), 4
                    )
                else:
                    distances[group] = None
        except (FileNotFoundError, ValueError, OSError):
            distances = None

    locked_cal = next(
        (
            r
            for r in session.calibration_results
            if abs(r.shader_multiplier - session.locked_multiplier) < 1e-9
            and not r.uncertain
        ),
        None,
    )
    state = {
        "status": status,
        "timestamp_unix": time.time(),
        "channels": CHANNELS,
        "units_absolute": "uV^2",
        "units_spatial": "dB relative to mean of eight channels",
        "bandpowers_uV2": bandpowers,
        "relative_maps_db": relative,
        "distances_descriptive_not_diagnostic": distances,
        "shader_command": None
        if shader.last is None
        else shader.last.to_loop_json(),
        "shader": {
            "left_contrast_multiplier": session.locked_multiplier,
            "locked_uncertain": session.locked_uncertain,
            "lock_message": session.lock_message,
            "grid": session.grid,
            "calibration_order_randomized": session.calibration_order,
        },
        "swing": None if locked_cal is None else locked_cal.swing,
        "loss": None if locked_cal is None else locked_cal.loss,
        "healthy_target": session.healthy_target,
        "healthy_target_source": session.healthy_target_source,
        "healthy_target_detail": session.healthy_target_detail,
        "eeg_source": session.eeg_source,
        "per_setting_results": {
            "calibration": [setting_to_dict(r) for r in session.calibration_results],
            "held_out": [setting_to_dict(r) for r in session.heldout_results],
        },
        "note": (
            "Exploratory closed-loop demo — not diagnostic, not a neglect "
            "classifier, no automated treatment decision. Calibration and "
            "held-out are separate; mock streams are always labeled."
        ),
        "mock": session.eeg_source == "mock",
    }
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_grid(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid must list at least one multiplier")
    return values


def default_loop_path() -> Path:
    support = Path.home() / "Library" / "Application Support" / "ScreenShade"
    return support / "loop.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use labeled synthetic EEG (no headset). Always tagged mock.",
    )
    parser.add_argument("--lsl-name", type=str, default=None, help="LSL stream name")
    parser.add_argument(
        "--units",
        choices=["uv", "v"],
        default="uv",
        help="Incoming LSL units (live only)",
    )
    parser.add_argument(
        "--trials-per-condition",
        type=int,
        default=8,
        help="Anticipatory trials per cue side per setting (default 8)",
    )
    parser.add_argument(
        "--held-out-size",
        type=int,
        default=6,
        help="Trials per cue side for each held-out setting (default 6)",
    )
    parser.add_argument(
        "--grid",
        type=parse_grid,
        default=list(DEFAULT_GRID),
        help="Comma-separated left-contrast multipliers (randomized order)",
    )
    parser.add_argument(
        "--healthy-baseline",
        type=Path,
        default=None,
        help="(a) JSON/CSV with swing from a healthy teammate on this rig",
    )
    parser.add_argument(
        "--healthy-target",
        type=float,
        default=None,
        help="(b) Explicit healthy swing target",
    )
    parser.add_argument(
        "--loss-noise-eps",
        type=float,
        default=LOSS_NOISE_EPS_DEFAULT,
        help="If best vs 2nd-best loss Δ < this → lock neutral 1.00",
    )
    parser.add_argument(
        "--comparison-multiplier",
        type=float,
        default=1.0,
        help="Held-out comparison setting alongside the lock (default 1.00)",
    )
    parser.add_argument(
        "--participant-id",
        type=str,
        default="pilot",
        help="participant_id column in trials.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for grid shuffle + mock noise",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Session output directory (default neurocompass_figures/adapt_<stamp>)",
    )
    parser.add_argument(
        "--write-loop-json",
        action="store_true",
        help="Also write ~/Library/Application Support/ScreenShade/loop.json",
    )
    parser.add_argument(
        "--skip-held-out",
        action="store_true",
        help="Calibration only (not recommended for claims)",
    )
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.trials_per_condition < 1:
        raise SystemExit("--trials-per-condition must be ≥ 1")
    if args.held_out_size < 1 and not args.skip_held_out:
        raise SystemExit("--held-out-size must be ≥ 1")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir or (FIGURES / f"adapt_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("NeuroCompass adapt — exploratory closed-loop controller")
    print("NOT diagnostic · NOT a neglect classifier · mock is always labeled")
    print("=" * 64)

    healthy = resolve_healthy_target(args.healthy_baseline, args.healthy_target)
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    if args.mock:
        source: EEGSource = MockEEGSource(np_rng, healthy.value)
        eeg_source = "mock"
        print(f"MOCK tag active — console + state JSON + CSV will say eeg_source=mock")
        print(
            f"MOCK bias: response gain peaks at ×{MOCK_OPTIMUM:.2f} "
            f"(gain=exp(-((m-{MOCK_OPTIMUM})/{MOCK_GAIN_WIDTH})^2))"
        )
    else:
        source = LSLEEGSource(args.lsl_name, args.units)
        eeg_source = "live"

    loop_path = default_loop_path() if args.write_loop_json else None
    shader = ShaderDriver(loop_path=loop_path, also_copy_under=out)

    order = list(args.grid)
    py_rng.shuffle(order)
    print(f"Grid (randomized): {[f'{x:.2f}' for x in order]}")

    session = AdaptSession(
        participant_id=args.participant_id,
        eeg_source=eeg_source,
        healthy_target=healthy.value,
        healthy_target_source=healthy.source,
        healthy_target_detail=healthy.detail,
        grid=[float(x) for x in args.grid],
        calibration_order=[float(x) for x in order],
    )

    # --- Calibration grid (never treated as validation) ---
    print("\n--- CALIBRATION (randomized grid; not held-out) ---")
    for setting in order:
        print(f"\nmeasure_swing_and_loss(shader_setting={setting:.2f})")
        result = measure_swing_and_loss(
            source,
            shader,
            session,
            setting,
            trials_per_condition=args.trials_per_condition,
            block="calibration",
            mode="adjusted",
            rng=py_rng,
        )
        session.calibration_results.append(result)
        write_state_json(out / "eeg_live_state.json", session, shader, status="calibrating")
        # Mirror into figures root for watchers expecting the classic path.
        write_state_json(FIGURES / "eeg_live_state.json", session, shader, status="calibrating")

    locked, uncertain, message = pick_lock(session.calibration_results, args.loss_noise_eps)
    session.locked_multiplier = locked
    session.locked_uncertain = uncertain
    session.lock_message = message
    print(f"\nLOCK: {message}")
    shader.apply(locked)

    # --- Held-out (separate block; never merge into calibration) ---
    if not args.skip_held_out:
        print("\n--- HELD-OUT (separate from calibration; not used to choose lock) ---")
        held_settings = []
        for m in (locked, args.comparison_multiplier):
            if all(abs(m - x) > 1e-9 for x in held_settings):
                held_settings.append(m)
        for setting in held_settings:
            mode = "locked" if abs(setting - locked) < 1e-9 else "comparison"
            print(f"\nheld-out measure_swing_and_loss(shader_setting={setting:.2f}, mode={mode})")
            result = measure_swing_and_loss(
                source,
                shader,
                session,
                setting,
                trials_per_condition=args.held_out_size,
                block="held_out",
                mode=mode,
                rng=py_rng,
            )
            session.heldout_results.append(result)

    status = "mock_complete" if eeg_source == "mock" else "live_complete"
    write_trials_csv(out / "trials.csv", session.trials)
    write_raw_npz(out / "raw_eeg_windows.npz", session)
    write_state_json(out / "eeg_live_state.json", session, shader, status=status)
    write_state_json(FIGURES / "eeg_live_state.json", session, shader, status=status)

    # Session summary sidecar
    summary = {
        "participant_id": session.participant_id,
        "eeg_source": session.eeg_source,
        "healthy_target": session.healthy_target,
        "healthy_target_source": session.healthy_target_source,
        "healthy_target_detail": session.healthy_target_detail,
        "locked_multiplier": session.locked_multiplier,
        "locked_uncertain": session.locked_uncertain,
        "lock_message": session.lock_message,
        "calibration_order": session.calibration_order,
        "output_dir": str(out),
        "trials_csv": str(out / "trials.csv"),
        "state_json": str(out / "eeg_live_state.json"),
        "raw_npz": str(out / "raw_eeg_windows.npz"),
        "caveat": (
            "Not diagnostic / not a neglect classifier / exploratory only. "
            "Held-out must not be presented as calibration."
        ),
    }
    (out / "session_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 64)
    print(f"eeg_source: {session.eeg_source}")
    print(f"healthy_target: {session.healthy_target:.6f} [{session.healthy_target_source}]")
    print(f"locked: ×{session.locked_multiplier:.2f} uncertain={session.locked_uncertain}")
    print(f"state JSON: {out / 'eeg_live_state.json'}")
    print(f"trials CSV: {out / 'trials.csv'}")
    print(f"raw EEG:    {out / 'raw_eeg_windows.npz'}")
    print("=" * 64)
    return out


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
