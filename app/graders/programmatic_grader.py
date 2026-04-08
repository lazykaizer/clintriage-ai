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


def grade_task1(agent_ranking: List[str], patients: list) -> Tuple[float, str]:
    """
    Binary Ordering: compare agent ranking of 2 patients to ground truth.
    1.0 = perfect order
    0.0 = completely wrong
    """
    if not agent_ranking or len(agent_ranking) != 2:
        return 0.0, "Invalid ranking provided. Must rank 2 patients."

    ground_truth = sorted(patients, key=lambda x: x["ground_truth_level"])
    truth_ids = [p["patient_id"] for p in ground_truth]

    if agent_ranking == truth_ids:
        return 1.0, "Perfect priority order for 2 patients."
    else:
        return 0.0, f"Incorrect priority order. Ground truth: {truth_ids}"


def grade_task2(agent_ranking: List[str], patients: list) -> Tuple[float, str]:
    """
    Priority Ordering: compare agent ranking to ground truth.
    Improved: Handle ties in ground_truth_level (equally critical patients).
    """
    if not agent_ranking or len(agent_ranking) != len(patients):
        return 0.0, f"Invalid ranking length. Expected {len(patients)}."

    # Map each patient ID to its ground truth level
    truth_map = {p["patient_id"]: p["ground_truth_level"] for p in patients}
    
    # Convert agent ranking IDs to their ground truth levels
    agent_levels = [truth_map.get(pid, 99) for pid in agent_ranking]
    
    # A perfect ranking means the levels are non-decreasing (1, 1, 2, 5... etc)
    is_sorted = all(agent_levels[i] <= agent_levels[i+1] for i in range(len(agent_levels)-1))
    
    if is_sorted:
        return 1.0, "Perfect clinical priority order (levels are correctly prioritized)."

    # Check if at least the first patient is one of the most critical ones
    min_level = min(truth_map.values())
    if truth_map.get(agent_ranking[0]) == min_level:
        return 0.6, f"Safely identified a most-critical patient (Level {min_level}) as #1."

    # Partial credit for positional correctness
    sorted_levels = sorted(truth_map.values())
    correct_positions = 0
    for i, lvl in enumerate(agent_levels):
        if lvl == sorted_levels[i]:
            correct_positions += 1
    
    if correct_positions > 0:
        return 0.3, f"Partial overlap — {correct_positions} clinical level(s) in correct position."

    return 0.0, "Incorrect priority order."


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


