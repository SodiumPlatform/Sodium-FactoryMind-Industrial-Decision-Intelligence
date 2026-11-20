"""Deterministic, traceable maintenance rule engine (advisory only).

Every recommendation is derived from explicit, documented thresholds applied to
the actual input values; each trigger carries its evidence string so a judge
or engineer can trace exactly why it fired. No LLM, no randomness, no hidden
state. FactoryMind NEVER authorises shutdowns or physical intervention.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src import config


@dataclass
class Trigger:
    """A single fired rule with its evidence and suggested inspection."""
    code: str
    title: str
    severity: int          # 0 none, 1 watch, 2 act soon, 3 urgent
    evidence: str
    action: str


def _power_w(row: dict) -> float:
    """Estimated spindle power: Torque x angular velocity (W)."""
    torque = float(row["Torque [Nm]"])
    rpm = float(row["Rotational speed [rpm]"])
    return torque * rpm * 2 * np.pi / 60.0


def evaluate_rules(row: dict, pred: dict, bundle: dict) -> list:
    """Apply all deterministic rules to one scored machine record."""
    triggers = []
    thr = bundle["thresholds"]
    q95_torque = bundle["train_stats"]["healthy"]["Torque [Nm]"]["q95"]

    wear = float(row["Tool wear [min]"])
    torque = float(row["Torque [Nm]"])
    rpm = float(row["Rotational speed [rpm]"])
    delta = float(row["Process temperature [K]"]) - float(row["Air temperature [K]"])

    # ML-assessed risk (validation-informed thresholds)
    if pred["tier"] == "High":
        triggers.append(Trigger(
            "ML-1", "Model-assessed high failure risk", 3,
            f"Predicted failure probability {pred['probability']:.2f} >= High-risk "
            f"threshold {thr['high']:.2f} (95th percentile of validation scores).",
            "Ask a qualified technician to inspect the machine and review its "
            "recent operating trends before the next production run."))
    elif pred["tier"] == "Medium":
        triggers.append(Trigger(
            "ML-2", "Model-assessed elevated risk", 1,
            f"Predicted failure probability {pred['probability']:.2f} >= Medium-risk "
            f"threshold {thr['medium']:.2f} (80th percentile of validation scores).",
            "Add the machine to the next inspection round and watch for "
            "deteriorating trends."))
    if pred["alert"] and pred["tier"] != "High":
        triggers.append(Trigger(
            "ML-3", "Prediction crosses the alert operating point", 2,
            f"Predicted probability {pred['probability']:.2f} >= alert threshold "
            f"{thr['alert']:.2f} (F1-optimal on validation data).",
            "Review this prediction with the Explainable AI page and confirm "
            "with an engineer."))
    elif pred["alert"] and pred["tier"] == "High":
        triggers.append(Trigger(
            "ML-3", "Prediction crosses the alert operating point", 3,
            f"Predicted probability {pred['probability']:.2f} >= alert threshold "
            f"{thr['alert']:.2f} (F1-optimal on validation data).",
            "Include in the daily maintenance review meeting."))

    # Tool wear
    if wear >= config.WEAR_CRITICAL_MIN:
        triggers.append(Trigger(
            "TL-1", "Tool wear near end-of-life window", 3,
            f"Tool wear = {wear:.0f} min (critical threshold {config.WEAR_CRITICAL_MIN:.0f} min).",
            "Plan a tool replacement at the next safe stop; verify insert "
            "condition before further cutting."))
    elif wear >= config.WEAR_HIGH_MIN:
        triggers.append(Trigger(
            "TL-2", "High tool wear", 2,
            f"Tool wear = {wear:.0f} min (service threshold {config.WEAR_HIGH_MIN:.0f} min).",
            "Schedule tool service within the next maintenance window."))
    elif wear >= config.WEAR_WATCH_MIN:
        triggers.append(Trigger(
            "TL-3", "Elevated tool wear", 1,
            f"Tool wear = {wear:.0f} min (watch threshold {config.WEAR_WATCH_MIN:.0f} min).",
            "Monitor tool wear trend after each shift."))

    # Heat dissipation pattern (AI4I HDF mechanism)
    if delta < config.HDF_TEMP_DELTA_K and rpm < config.HDF_RPM_LIMIT:
        triggers.append(Trigger(
            "HD-1", "Heat-dissipation risk pattern", 2,
            f"Process-air temperature difference = {delta:.1f} K "
            f"(< {config.HDF_TEMP_DELTA_K} K) with rotational speed {rpm:.0f} rpm "
            f"(< {config.HDF_RPM_LIMIT:.0f} rpm).",
            "Inspect cooling system (fans, heat exchanger, coolant flow) and "
            "verify with a maintenance engineer."))

    # Power band (AI4I PWF mechanism)
    power = _power_w(row)
    if power < config.POWER_MIN_W or power > config.POWER_MAX_W:
        triggers.append(Trigger(
            "PW-1", "Spindle power outside normal band", 2,
            f"Estimated spindle power = {power:.0f} W "
            f"(normal band {config.POWER_MIN_W:.0f}-{config.POWER_MAX_W:.0f} W).",
            "Check drive electronics, belt tension and motor load profile."))

    # Overstrain pattern (AI4I OSF mechanism)
    osf_limit = config.OSF_WEAR_TORQUE.get(str(row.get("Type", "L")).upper(), 11000.0)
    if wear * torque > osf_limit:
        triggers.append(Trigger(
            "OS-1", "Overstrain pattern (wear x torque)", 2,
            f"Tool wear x Torque = {wear * torque:,.0f} min*Nm "
            f"(limit for type {row.get('Type', '?')}: {osf_limit:,.0f}).",
            "Inspect mechanical drive components for overstrain; review "
            "cutting parameters with the process engineer."))

    # Torque above the healthy operating range
    if torque > q95_torque:
        triggers.append(Trigger(
            "TQ-1", "Torque above typical operating range", 1,
            f"Torque = {torque:.1f} Nm (95th percentile of healthy training "
            f"readings = {q95_torque:.1f} Nm).",
            "Review cutting load and feed rate; monitor for bearing friction."))

    return triggers


def generate_recommendation(row: dict, pred: dict, bundle: dict) -> dict:
    """Produce a deterministic maintenance recommendation for one machine.

    Priority is the maximum severity across the ML assessment and all fired
    rules, so the output is fully traceable to actual input values.
    """
    triggers = evaluate_rules(row, pred, bundle)
    ml_severity = {"High": 3, "Medium": 1, "Low": 0}[pred["tier"]]
    severity = max([ml_severity] + [t.severity for t in triggers]) if triggers else ml_severity

    priority = {3: "P1", 2: "P2", 1: "P3"}.get(severity, "P4")
    priority_label = {
        "P1": "Critical - immediate human review required",
        "P2": "High - schedule inspection within 24 hours",
        "P3": "Medium - include in next inspection round",
        "P4": "Routine - normal monitoring",
    }[priority]

    if triggers:
        actions = []
        seen = set()
        for t in triggers:
            if t.code not in seen:
                seen.add(t.code)
                actions.append(f"[{t.code}] {t.action}")
    else:
        actions = [
            "No abnormal indicators detected; keep standard monitoring routine.",
        ]

    summary = (
        f"{len(triggers)} rule(s) triggered. Model risk: {pred['tier']} "
        f"(p={pred['probability']:.2f}). "
        + ("Highest severity signals: " + "; ".join(t.title for t in sorted(triggers, key=lambda x: -x.severity)[:3]) + "."
           if triggers else "All sensor readings within typical healthy ranges.")
    )

    return {
        "machine": row.get("Machine ID") or "unspecified",
        "priority": priority,
        "priority_label": priority_label,
        "severity": severity,
        "triggers": triggers,
        "actions": actions,
        "summary": summary,
        "approval": config.APPROVAL_TEXT,
    }


def recommendation_counts(recs: list) -> dict:
    """Count recommendations per priority level (P1..P4)."""
    counts = {k: 0 for k in ["P1", "P2", "P3", "P4"]}
    for r in recs:
        counts[r["priority"]] += 1
    return counts
