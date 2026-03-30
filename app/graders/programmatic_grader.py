"""
Programmatic graders for ClinTriageAI tasks 1-5.
All functions return (score: float, feedback: str).
Scores are always clamped to [0.0, 1.0].
"""
from typing import List, Dict, Tuple


def _parse_level(level_str: str) -> int:
    """Safely extract integer from 'LEVEL_X' string."""
    try:
        return int(str(level_str).upper().replace("LEVEL_", "").strip())
    except (ValueError, AttributeError):
        return 3  # default to middle if unparseable


def grade_task1(agent_decision: str, ground_truth_level: int) -> Tuple[float, str]:
    """
    Binary Triage: CRITICAL (1,2) vs NON-CRITICAL (3,4,5).
    1.0 = correct category
    0.3 = over-triage (non-critical flagged as critical — wastes resources but safe)
    0.0 = under-triage (critical flagged as non-critical — DANGEROUS)
    """
    critical = {1, 2}
    agent_level = _parse_level(agent_decision)
    agent_critical = agent_level in critical
    truth_critical = ground_truth_level in critical

    if agent_critical == truth_critical:
        return 1.0, "Correct triage category."
    elif truth_critical and not agent_critical:
        return 0.0, (
            f"DANGEROUS: Critical patient (LEVEL_{ground_truth_level}) "
            f"classified as non-critical (LEVEL_{agent_level}). Patient could die waiting."
        )
    else:
        return 0.3, (
            f"Over-triage: Non-critical patient (LEVEL_{ground_truth_level}) "
            f"sent as critical (LEVEL_{agent_level}). Wastes resources but patient is safe."
        )


def grade_task2(agent_ranking: List[str], patients: list) -> Tuple[float, str]:
    """
    Priority Ordering: compare agent ranking to ground truth.
    1.0 = perfect order
    0.6 = most critical patient correctly at #1
    0.3 = partial overlap
    0.0 = completely wrong
    """
    if not agent_ranking:
        return 0.0, "No ranking provided."

    ground_truth = sorted(patients, key=lambda x: x["ground_truth_level"])
    truth_ids = [p["patient_id"] for p in ground_truth]

    if agent_ranking == truth_ids:
        return 1.0, "Perfect priority order."

    if agent_ranking[0] == truth_ids[0]:
        return 0.6, (
            f"Most critical patient ({truth_ids[0]}) correctly identified as #1. "
            f"Full ground truth order: {truth_ids}"
        )

    # Check positional overlap
    overlap = sum(
        1 for i, pid in enumerate(agent_ranking)
        if i < len(truth_ids) and pid == truth_ids[i]
    )
    if overlap > 0:
        return 0.3, f"Partial correct ordering — {overlap} position(s) correct. Truth: {truth_ids}"

    return 0.0, f"Incorrect priority order. Ground truth: {truth_ids}"


def grade_task3(assignments: Dict[str, str], patients: list) -> Tuple[float, str]:
    """
    Multi-Patient Assignment: per-patient accuracy.
    score = correct_count / total_patients
    """
    if not assignments:
        return 0.0, "No assignments provided."

    correct = 0
    errors = []

    for p in patients:
        pid = p["patient_id"]
        truth = p["ground_truth_level"]
        agent_level = _parse_level(assignments.get(pid, "LEVEL_3"))

        if agent_level == truth:
            correct += 1
        else:
            errors.append(f"{pid}: assigned LEVEL_{agent_level}, truth LEVEL_{truth}")

    total = len(patients)
    score = round(correct / total, 2) if total > 0 else 0.0
    feedback = f"{correct}/{total} correct."
    if errors:
        feedback += " Errors: " + "; ".join(errors)

    return score, feedback


def grade_task4_programmatic(icu_selection: List[str], patients: list) -> Tuple[float, str]:
    """
    ICU Resource Allocation — programmatic half (max 0.5).
    Ground truth = 3 patients with lowest ground_truth_level.
    Score = (correct_count / 3) * 0.5
    """
    if not icu_selection:
        return 0.0, "No ICU selection provided."

    sorted_patients = sorted(patients, key=lambda x: x["ground_truth_level"])
    truth_icu = {p["patient_id"] for p in sorted_patients[:3]}
    agent_icu = set(icu_selection)

    correct_count = len(truth_icu & agent_icu)
    score = round((correct_count / 3) * 0.5, 2)

    return score, (
        f"{correct_count}/3 correct ICU patients selected. "
        f"Ground truth ICU: {sorted(truth_icu)}"
    )


def grade_task5(agent_decision: str, patient: dict, reasoning: str = "") -> Tuple[float, str]:
    """
    Edge Case Detection: did the agent catch the hidden condition?
    1.0 = LEVEL_1 or LEVEL_2 (correctly escalated)
    0.3 = LEVEL_3 with uncertainty expressed in reasoning
    0.0 = LEVEL_3/4/5 without uncertainty (patient could die)
    """
    agent_level = _parse_level(agent_decision)
    hidden = patient.get("hidden_diagnosis", "unknown serious condition")

    if agent_level <= 2:
        return 1.0, f"Correctly escalated to LEVEL_{agent_level}. Hidden condition: {hidden}"

    # Check if reasoning expresses uncertainty
    uncertainty_keywords = [
        "unsure", "uncertain", "not sure", "could be", "possible",
        "rule out", "cannot exclude", "differential", "concerned",
        "worried", "suspicious", "atypical", "monitor closely",
        "further evaluation", "escalate if"
    ]
    reasoning_lower = reasoning.lower() if reasoning else ""
    expressed_uncertainty = any(kw in reasoning_lower for kw in uncertainty_keywords)

    if agent_level == 3 and expressed_uncertainty:
        return 0.3, (
            f"Partial credit: assigned LEVEL_3 but expressed uncertainty. "
            f"Hidden condition: {hidden}"
        )

    return 0.0, (
        f"DANGEROUS: Patient with hidden critical condition assigned LEVEL_{agent_level}. "
        f"Hidden condition: {hidden}"
    )
