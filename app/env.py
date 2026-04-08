import json
import random
from pathlib import Path


class ClinTriageEnv:
    def __init__(self):
        data_path = Path(__file__).parent / "data" / "patient_cases.json"
        with open(data_path, encoding="utf-8") as f:
            all_cases = json.load(f)

        self.all_cases    = all_cases
        self.normal_cases = [c for c in all_cases if not c.get("is_edge_case")]

        self.current_task     = None
        self.current_patients = None
        self.step_count       = 0
        self.patients_seen    = 0
        self.last_reward      = None

    def reset(self, task_id: int) -> dict:
        """Reset environment for a specific task. Returns observation dict."""
        self.current_task = task_id
        self.step_count += 1

        if task_id == 1:
            pool = random.sample(self.normal_cases, 2)
            self.current_patients = pool
            return {
                "patients": [self._clean(p) for p in pool],
                "instructions": (
                    "Rank these 2 patients by urgency (most urgent first). "
                    "Return JSON: {\"ranking\": [\"ID1\",\"ID2\"], \"reasoning\": \"your explanation\"}"
                )
            }

        elif task_id == 2:
            pool = random.sample(self.normal_cases, 3)
            self.current_patients = pool
            return {
                "patients": [self._clean(p) for p in pool],
                "instructions": (
                    "Rank these 3 patients by urgency (most urgent first). "
                    "Return JSON: {\"ranking\": [\"ID1\",\"ID2\",\"ID3\"], \"reasoning\": \"your explanation\"}"
                )
            }

        elif task_id == 3:
            pool = random.sample(self.normal_cases, 5)
            self.current_patients = pool
            return {
                "patients": [self._clean(p) for p in pool],
                "instructions": (
                    "Assign a triage level (LEVEL_1 to LEVEL_5) to each patient. "
                    "Return JSON: {\"assignments\": {\"ID1\": \"LEVEL_X\", \"ID2\": \"LEVEL_X\", ...}}"
                )
            }

        elif task_id == 4:
            pool = random.sample(self.normal_cases, 8)
            self.current_patients = pool
            return {
                "patients": [self._clean(p) for p in pool],
                "icu_beds_available": 3,
                "instructions": (
                    "Only 3 ICU beds are available. Select the 3 patients who need ICU the most. "
                    "Return JSON: {\"icu_patients\": [\"ID1\",\"ID2\",\"ID3\"], \"reasoning\": \"detailed clinical explanation\"}"
                )
            }

        else:
            raise ValueError(f"Invalid task_id: {task_id}. Must be 1-4.")

    def _clean(self, patient: dict) -> dict:
        """Remove ground truth fields before sending to agent — these must NEVER leak."""
        return {
            k: v for k, v in patient.items()
            if k not in ("ground_truth_level", "is_edge_case", "hidden_diagnosis")
        }

    def get_state(self) -> dict:
        """Return current environment state."""
        return {
            "current_task": self.current_task,
            "patients_seen": self.patients_seen,
            "step_count": self.step_count,
            "last_reward": self.last_reward,
        }
