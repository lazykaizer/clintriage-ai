from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class TicketStatus(str, Enum):
    OPEN = "open"
    PENDING = "pending"
    RESOLVED = "resolved"
    CLOSED = "closed"


class ActionTypeEnum(str, Enum):
    TRIAGE = "triage"
    ASK_VITALS = "ask_vitals"
    RESPOND = "respond"


class ConversationTurn(BaseModel):
    role: str           # "user" | "agent"
    content: str
    turn: int


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


class ResetRequest(BaseModel):
    task_id: Optional[int] = 1                 # 1 to 4


class Observation(BaseModel):
    task_id: int
    patients: List[Patient]
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    instructions: str
    status: TicketStatus = TicketStatus.OPEN
    turn_number: int = 0


class ResetResponse(BaseModel):
    session_id: str
    observation: Observation


class StepRequest(BaseModel):
    session_id: str
    task_id: int
    # Actions
    action_type: str = "triage"                # "triage" | "ask_vitals" | "respond"
    response_text: Optional[str] = None        # Dialogue with nurse/patient
    triage_decision: Optional[str] = None      # LEVEL_1 to LEVEL_5
    reasoning: Optional[str] = None            # Agent's explanation
    ranking: Optional[List[str]] = None        # For task 2
    assignments: Optional[Dict[str, str]] = None  # For task 3
    icu_patients: Optional[List[str]] = None   # For task 4


class StepResponse(BaseModel):
    observation: Observation
    reward: float                # 0.0 to 1.0 (clamped)
    done: bool
    feedback: str                # Human-readable explanation
    info: Dict[str, Any]


class StateResponse(BaseModel):
    current_task: Optional[int] = None
    patients_seen: int = 0
    step_count: int = 0
    last_reward: Optional[float] = None
    conversation_history: List[ConversationTurn] = []
