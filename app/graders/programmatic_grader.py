"""
Programmatic graders for ClinTriageAI tasks 1-5.
All functions return (score: float, feedback: str).
Scores are always nudged to (0.01, 0.99) to satisfy strict validator rules.
"""
from typing import List, Dict, Tuple


def _nudge_score(score: float) -> float:
    """Ensure score is strictly between 0 and 0.99."""
    return max(0.01, min(0.99, score))


def _parse_level(level_str: str) -> int:
    """Safely extract integer level from any string format using regex."""
    import re
    try:
        # Extract the first digit found in the string
        match = re.search(r'\d', str(level_str))
        if match:
            return int(match.group())
        return 3 # Default to Level 3 if no digit found
    except Exception:
        return 3


def grade_task1(agent_ranking: List[str], patients: list) -> Tuple[float, str]:
    if not agent_ranking or len(agent_ranking) != 2:
        return 0.01, "Invalid ranking provided."

    truth_levels = {p["patient_id"]: p["ground_truth_level"] for p in patients}
    p1, p2 = agent_ranking[0], agent_ranking[1]
    
    if truth_levels[p1] <= truth_levels[p2]:
        return _nudge_score(1.0), "Correct clinical priority."
    
    diff = abs(truth_levels[p1] - truth_levels[p2])
    if diff <= 1:
        return _nudge_score(0.6), "Incorrect, but patients had similar clinical urgency (Near Miss)."
    return _nudge_score(0.1), "Incorrect. Misplaced a critical patient for a stable one (Far Miss)."


def grade_task2(agent_ranking: List[str], patients: list) -> Tuple[float, str]:
    """
    Priority Ordering (3 patients): uses granular distance rewards.
    """
    if not agent_ranking or len(agent_ranking) != len(patients):
        return 0.01, f"Expected {len(patients)} patients."

    truth_map = {p["patient_id"]: p["ground_truth_level"] for p in patients}
    agent_levels = [truth_map.get(pid, 5) for pid in agent_ranking]
    
    # Check if sorted non-decreasingly (The Gold Standard)
    correct_order = all(agent_levels[i] <= agent_levels[i+1] for i in range(len(agent_levels)-1))
    if correct_order:
        return _nudge_score(1.0), "Perfect priority order."

    # Distance Reward: How far are we from a perfect level sum at each position?
    sorted_truth = sorted(truth_map.values())
    reward = 0.0
    for i, lvl in enumerate(agent_levels):
        diff = abs(lvl - sorted_truth[i])
        if diff == 0:
            reward += 0.33
        elif diff == 1:
            reward += 0.20 # Close enough
        else:
            reward += 0.05
    
    # Bonus for getting the #1 most critical patient right
    if agent_levels[0] == sorted_truth[0]:
        reward += 0.15

    return _nudge_score(reward), "Partial credit for clinical proximity in ordering."


def grade_task3(assignments: Dict[str, str], patients: list) -> Tuple[float, str]:
    if not assignments:
        return 0.01, "No assignments provided."

    correct = 0
    total = len(patients)
    mismatches = []

    # Normalize keys to prevent minor mismatch errors
    norm_assignments = {str(k).strip().upper(): v for k, v in assignments.items()}

    for p in patients:
        pid = str(p["patient_id"]).strip().upper()
        truth = p["ground_truth_level"]
        agent_level = _parse_level(norm_assignments.get(pid, "LEVEL_3"))

        if agent_level == truth:
            correct += 1
        else:
            mismatches.append(f"{pid}: Should be LEVEL_{truth}")
        
    accuracy = correct / total
    
    if correct == total:
        score = 0.99
        feedback = f"Accuracy: {correct}/{total}. Perfect."
    else:
        score = max(0.1, accuracy * 0.9)
        feedback = f"Accuracy: {correct}/{total}. Errors: {', '.join(mismatches)}"
        
    return _nudge_score(score), feedback


def grade_task4_programmatic(icu_selection: List[str], patients: list) -> Tuple[float, str]:
    if not icu_selection or len(icu_selection) == 0:
        return 0.01, "No patients selected."

    sorted_patients = sorted(patients, key=lambda x: x["ground_truth_level"])
    target_sum = sum(p["ground_truth_level"] for p in sorted_patients[:3])
    
    patient_map = {p["patient_id"]: p for p in patients}
    agent_sum = sum(patient_map[pid]["ground_truth_level"] for pid in icu_selection if pid in patient_map)
    
    # Give full points (0.5) if agent sum is within reasonable medical range (+3)
    if agent_sum <= target_sum + 3 and len(icu_selection) == 3:
        return 0.5, "Clinically Sound selection."
        
    score = (target_sum / (agent_sum or 100)) * 0.5
    return _nudge_score(score), f"Quality: {round(score, 2)}"
