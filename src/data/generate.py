"""Synthetic job-card dataset generator.

Implements spec/project-specification.md section "Synthetic dataset" and
spec/architecture.md section 4 (`data/generate.py` interface). Target is
built from two grounded per-technician signal components (persistent skill
effect, experience-based learning curve), a modest additive main effect
from staffing_ratio on the log-mean, and multiplicative log-normal noise,
clipped to [OVERRUN_FLOOR, OVERRUN_CAP]. The staffing_ratio effect is a
plain main effect (linear in log-space), not an interaction between
features — the earlier interaction-term revision (e.g. learning-curve x
shift type) was proposed and then reversed; see spec.

Technicians are given staggered roster start dates and jobs are generated
in per-technician chronological order so that `rolling_efficiency_30d` and
`task_type_experience_count` can be computed causally, using only that
technician's own job history strictly prior to each job's date.
"""

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

from src import config as cfg

TASK_TYPES = ["engine", "avionics", "airframe", "hydraulics", "electrical"]
TASK_TYPE_BASE_HOURS = {
    "engine": 8.0,
    "avionics": 4.0,
    "airframe": 6.0,
    "hydraulics": 5.0,
    "electrical": 3.0,
}
TASK_CATEGORIES = ["scheduled", "unscheduled", "inspection"]
SHIFT_TYPES = ["day", "evening", "night"]

WINDOW_START = pd.Timestamp("2024-01-01")

# Redraw attempts per job before giving up and shrinking that technician's
# effective quota (see _generate_job_offsets). 100 is generous: for a gap
# to fail 100 independent redraws, the remaining budget has to already be
# under ~4.6% of avg_gap, i.e. this only ever bites on the last job or two
# of a technician's quota, not a systematic quota mismatch.
MAX_GAP_RETRIES = 100


def _generate_job_offsets(
    rng: np.random.Generator, quota: int, available_days: float, avg_gap: float
) -> list[float]:
    """Increasing offsets (days from a technician's start_date) via
    cumulative exponential gaps, redrawing any gap that would overshoot
    available_days instead of clipping it to the boundary — clipping would
    pin multiple jobs to the exact same fabricated date (see spec review:
    this previously produced a 52-record pileup on the single latest
    possible date, concentrated entirely in the test partition since it's
    the dataset's global maximum date).

    If a gap can't be redrawn to fit within MAX_GAP_RETRIES attempts, stops
    generating further jobs for this technician rather than fabricating a
    date or silently padding — this technician's effective job count ends
    up below their assigned quota, and a warning is raised so the shortfall
    is visible rather than silent.
    """
    offsets: list[float] = []
    current = 0.0
    for _ in range(quota):
        placed = False
        for _attempt in range(MAX_GAP_RETRIES):
            gap = rng.exponential(avg_gap)
            candidate = current + gap
            if candidate <= available_days:
                offsets.append(candidate)
                current = candidate
                placed = True
                break
        if not placed:
            break
    return offsets


def _build_technician_roster(rng: np.random.Generator) -> pd.DataFrame:
    """Persistent per-technician skill effect + staggered roster start date.

    Staggering start dates (rather than starting every technician on day
    one) is what makes the later causal rolling-feature computation
    meaningful instead of trivial — some technicians genuinely have little
    or no history for stretches of the window.
    """
    n = cfg.N_TECHNICIANS
    log_skill = rng.normal(0.0, 0.15, size=n)
    skill_effect = np.exp(log_skill)
    max_offset = max(int(cfg.JOB_DATE_WINDOW_DAYS * 0.6), 1)
    start_offset_days = rng.integers(0, max_offset, size=n)
    start_date = [WINDOW_START + timedelta(days=int(d)) for d in start_offset_days]
    return pd.DataFrame(
        {
            "technician_id": np.arange(n),
            "skill_effect": skill_effect,
            "start_date": start_date,
        }
    )


def _jobs_per_technician(n_records: int, rng: np.random.Generator) -> np.ndarray:
    """Distribute n_records exactly across technicians (base + remainder)."""
    n_technicians = cfg.N_TECHNICIANS
    base = n_records // n_technicians
    counts = np.full(n_technicians, base, dtype=int)
    remainder = n_records - base * n_technicians
    if remainder:
        bump_idx = rng.choice(n_technicians, size=remainder, replace=False)
        counts[bump_idx] += 1
    return counts


def generate_dataset(
    n_records: int = cfg.N_RECORDS,
    seed: int = cfg.SEED,
    return_debug: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the synthetic job-card dataset.

    Returns a DataFrame with job-level, technician-level, and contextual
    feature columns plus `record_id`, `job_date`, and the target
    `overrun_factor`. If `return_debug=True`, also returns a second
    DataFrame exposing the internal signal components per record
    (technician_id, skill_effect, learning_curve_multiplier,
    staffing_log_effect, noise, raw_overrun) — generation/validation
    scaffolding, not part of the modeling feature set.
    """
    rng = np.random.default_rng(seed)
    window_end = WINDOW_START + timedelta(days=cfg.JOB_DATE_WINDOW_DAYS)

    roster = _build_technician_roster(rng)
    job_counts = _jobs_per_technician(n_records, rng)

    records = []
    debug_records = []
    record_id = 0

    for tech, quota in zip(roster.itertuples(index=False), job_counts):
        if quota == 0:
            continue

        available_days = max((window_end - tech.start_date).days, 1)
        avg_gap = available_days / (quota + 1)
        offsets = _generate_job_offsets(rng, quota, available_days, avg_gap)
        if len(offsets) < quota:
            warnings.warn(
                f"Technician {tech.technician_id}: placed only "
                f"{len(offsets)}/{quota} jobs within their "
                f"{available_days}-day available window after "
                f"{MAX_GAP_RETRIES} redraw attempts per job; quota reduced "
                f"rather than fabricating a boundary date.",
                stacklevel=2,
            )
        job_dates = sorted(tech.start_date + timedelta(days=o) for o in offsets)

        history = []  # causal record of this technician's own prior jobs

        for job_date in job_dates:
            task_type = rng.choice(TASK_TYPES)
            task_category = rng.choice(TASK_CATEGORIES)
            shift_type = rng.choice(SHIFT_TYPES)
            day_of_week = job_date.day_name()

            base_hours = TASK_TYPE_BASE_HOURS[task_type]
            estimated_duration_hours = round(float(base_hours * rng.lognormal(0.0, 0.2)), 2)

            staffing_ratio = round(float(rng.uniform(0.6, 1.2)), 3)
            attendance_rate = round(float(rng.uniform(0.75, 1.0)), 3)

            # --- causal technician-level features: prior jobs only ---
            prior = [h for h in history if h["job_date"] < job_date]
            recent = [h for h in prior if h["job_date"] >= job_date - timedelta(days=30)]
            rolling_efficiency_30d = (
                round(float(np.mean([h["overrun_factor"] for h in recent])), 3)
                if recent
                else 1.0  # neutral default: no prior-30-day history yet
            )
            task_type_experience_count = sum(1 for h in prior if h["task_type"] == task_type)
            cumulative_experience = len(prior)
            learning_curve_index = round(float(np.log1p(cumulative_experience)), 3)

            # --- target: two technician signal components, a modest additive
            # staffing_ratio main effect on the log-mean, + log-normal noise ---
            learning_curve_multiplier = 1.0 + 0.5 * np.exp(-cumulative_experience / 20.0)
            staffing_log_effect = cfg.STAFFING_RATIO_LOG_COEF * (
                staffing_ratio - cfg.STAFFING_RATIO_REFERENCE
            )
            noise = float(np.exp(rng.normal(0.0, cfg.SIGMA_LOG)))
            raw_overrun = (
                tech.skill_effect
                * learning_curve_multiplier
                * np.exp(staffing_log_effect)
                * noise
            )
            overrun_factor = float(np.clip(raw_overrun, cfg.OVERRUN_FLOOR, cfg.OVERRUN_CAP))

            records.append(
                {
                    "record_id": record_id,
                    "job_date": job_date,
                    "task_type": task_type,
                    "estimated_duration_hours": estimated_duration_hours,
                    "task_category": task_category,
                    "rolling_efficiency_30d": rolling_efficiency_30d,
                    "task_type_experience_count": task_type_experience_count,
                    "learning_curve_index": learning_curve_index,
                    "shift_type": shift_type,
                    "day_of_week": day_of_week,
                    "staffing_ratio": staffing_ratio,
                    "attendance_rate": attendance_rate,
                    "overrun_factor": overrun_factor,
                }
            )
            if return_debug:
                debug_records.append(
                    {
                        "record_id": record_id,
                        "technician_id": tech.technician_id,
                        "skill_effect": tech.skill_effect,
                        "learning_curve_multiplier": learning_curve_multiplier,
                        "staffing_log_effect": staffing_log_effect,
                        "noise": noise,
                        "raw_overrun": raw_overrun,
                    }
                )

            history.append(
                {"job_date": job_date, "task_type": task_type, "overrun_factor": overrun_factor}
            )
            record_id += 1

    df = pd.DataFrame.from_records(records)
    if return_debug:
        return df, pd.DataFrame.from_records(debug_records)
    return df
