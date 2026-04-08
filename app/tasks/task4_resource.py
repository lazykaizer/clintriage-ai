"""
Task 4 — ICU Resource Allocation [HARD]
8 patients, only 3 ICU beds. Select best 3 with clinical reasoning.
Reward: 0.5 programmatic (correct IDs) + 0.5 LLM judge (reasoning quality)
"""

TASK_ID = 4
TASK_NAME = "icu_resource_allocation"
DIFFICULTY = "hard"
DESCRIPTION = "Select 3 ICU patients from 8 with clinical reasoning"
NUM_PATIENTS = 8
ICU_BEDS = 3
