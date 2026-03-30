from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class Vitals(BaseModel):
    heart_rate: int
    blood_pressure: str          # "120/80" format
    oxygen_saturation: int       # percentage, 0-100
    temperature: float           # celsius
    respiratory_rate: int


class Patient(BaseModel):
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    vitals: Vitals
    history: str
    arrival_mode: str            # walk-in | ambulance | police
    time_since_onset: str
    # ground_truth_level is in JSON but NEVER in API response to agent


class ResetRequest(BaseModel):
    task_id: int                 # 1 to 5


class ResetResponse(BaseModel):
    task_id: int
    observation: Any             # Patient or list of Patients
    instructions: str            # What the agent should do


class StepRequest(BaseModel):
    task_id: int
    triage_decision: Optional[str] = None      # LEVEL_1 to LEVEL_5
    reasoning: Optional[str] = None            # Agent's explanation
    ranking: Optional[List[str]] = None        # For task 2
    assignments: Optional[Dict[str, str]] = None  # For task 3
    icu_patients: Optional[List[str]] = None   # For task 4


class StepResponse(BaseModel):
    reward: float                # 0.0 to 1.0
    done: bool
    feedback: str                # Human-readable explanation
    info: Dict[str, Any]


class StateResponse(BaseModel):
    current_task: Optional[int] = None
    patients_seen: int = 0
    step_count: int = 0
    last_reward: Optional[float] = None
