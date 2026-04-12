import json
import random
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from app.models import (
    Patient, Observation, ConversationTurn, TicketStatus,
    ActionTypeEnum, StepRequest, Vitals
)
from app.graders import programmatic_grader, llm_grader

# Reward Constants
R_EMPATHY = 0.05
R_LOOP_PENALTY = -0.15
R_CLINICAL_QUERY = 0.03
R_SAFETY_BONUS = 0.10

class ClinTriageEnv:
    """
    Advanced ClinTriageAI Environment.
    Supports multi-turn clinical dialogue, state persistence, and behavioral rewards.
    """
    def __init__(self):
        self.data_path = Path(__file__).parent / "data" / "patient_cases.json"
        self._load_data()
        
        self.current_task: Optional[int] = None
        self.current_patients: List[dict] = []
        self.turn_number = 0
        self.patients_seen = 0
        self.last_reward = 0.0
        self.conversation_history: List[ConversationTurn] = []
        self.last_agent_message: Optional[str] = None
        self.status = TicketStatus.OPEN

    def _load_data(self):
        with open(self.data_path, encoding="utf-8") as f:
            all_cases = json.load(f)
        self.all_cases = all_cases
        self.normal_cases = [c for c in all_cases if not c.get("is_edge_case")]

    def reset(self, task_id: int) -> Observation:
        self.current_task = task_id
        self.turn_number = 0
        self.status = TicketStatus.OPEN
        self.last_reward = 0.0
        self.last_agent_message = None
        
        # Select patient pool based on task difficulty
        patient_counts = {1: 2, 2: 3, 3: 5, 4: 8}
        count = patient_counts.get(task_id, 2)
        
        self.current_patients = random.sample(self.normal_cases, count)
        self.patients_seen += count
        
        # Initial Clinical Observation
        msg = f"Triage Nurse: A new batch of {count} patients has arrived. Vitals are uploaded. Please review and prioritize."
        self.conversation_history = [
            ConversationTurn(role="user", content=msg, turn=0)
        ]

        instr_map = {
            1: "Binary Triage: Compare 2 patients and rank by urgency.",
            2: "Priority Ordering: Rank 3 patients from most to least critical.",
            3: "Multi-Patient Assignment: Assign specific triage levels (LEVEL_1 to 5).",
            4: "ICU Allocation: Select 3 patients for limited ICU beds based on clinical risk."
        }
        
        return self._build_observation(instr_map.get(task_id, "Process triage."))

    def step(self, req: StepRequest) -> Tuple[Observation, float, bool, str]:
        self.turn_number += 1
        reward = 0.0
        feedback_parts = []
        done = False

        # 1. Update History
        agent_msg = req.response_text or f"[Action: {req.action_type.upper()}]"
        self.conversation_history.append(
            ConversationTurn(role="agent", content=agent_msg, turn=self.turn_number)
        )

        # 2. Behavioral Intelligence Rewards
        b_reward = self._check_behavior_quality(agent_msg)
        reward += b_reward
        if b_reward > 0: feedback_parts.append("Positive bedside manner detected.")

        # 3. Decision Logic
        if req.action_type == ActionTypeEnum.TRIAGE:
            main_reward, triage_feedback = self._grade_action(req)
            reward += main_reward
            feedback_parts.append(triage_feedback)
            self.status = TicketStatus.RESOLVED
            done = True
            
        elif req.action_type == ActionTypeEnum.ASK_VITALS:
            reward += R_CLINICAL_QUERY
            # Dynamic follow-up simulation
            follow_up = "Nurse: Patient vitals updated. BP shows slight stabilization, but respiratory distress remains."
            self.conversation_history.append(
                ConversationTurn(role="user", content=follow_up, turn=self.turn_number + 1)
            )
            feedback_parts.append("Deep clinical dive initiated.")
            self.status = TicketStatus.PENDING

        elif req.action_type == ActionTypeEnum.RESPOND:
            reward += 0.02
            ack = "Nurse: Copy that. Team is standing by."
            self.conversation_history.append(
                ConversationTurn(role="user", content=ack, turn=self.turn_number + 1)
            )
            feedback_parts.append("Team communication logged.")

        # Final Score Processing
        final_reward = round(max(0.01, min(0.99, float(reward))), 2)
        self.last_reward = final_reward
        self.last_agent_message = agent_msg

        return self._build_observation("Finalize triage or gather more info."), final_reward, done, " | ".join(feedback_parts)

    def _grade_action(self, req: StepRequest) -> Tuple[float, str]:
        if self.current_task == 1:
            return programmatic_grader.grade_task1(req.ranking, self.current_patients)
        elif self.current_task == 2:
            return programmatic_grader.grade_task2(req.ranking, self.current_patients)
        elif self.current_task == 3:
            return programmatic_grader.grade_task3(req.assignments, self.current_patients)
        elif self.current_task == 4:
            prog_s, prog_f = programmatic_grader.grade_task4_programmatic(req.icu_patients, self.current_patients)
            llm_s, llm_f = llm_grader.grade_reasoning(
                [self._clean(p) for p in self.current_patients[:3]],
                req.reasoning or "",
                self.current_patients[0].get("ground_truth_level", 3)
            )
            return (prog_s + llm_s), f"{prog_f} {llm_f}"
        return 0.01, "Invalid Task."

    def _check_behavior_quality(self, text: str) -> float:
        score = 0.0
        text_l = text.lower()
        # Empathy Detection
        if any(kw in text_l for kw in ["sorry", "apologize", "understand", "priority", "critical"]):
            score += R_EMPATHY
        # Loop Detection
        if self.last_agent_message and text.strip().lower() == self.last_agent_message.strip().lower():
            score += R_LOOP_PENALTY
        return score

    def _build_observation(self, instructions: str) -> Observation:
        return Observation(
            task_id=self.current_task,
            patients=[Patient(**self._clean(p)) for p in self.current_patients],
            conversation_history=self.conversation_history,
            instructions=instructions,
            status=self.status,
            turn_number=self.turn_number
        )

    def _clean(self, patient: dict) -> dict:
        return {k: v for k, v in patient.items() if k not in ("ground_truth_level", "hidden_diagnosis")}

    def get_state(self) -> dict:
        return {
            "task_id": self.current_task,
            "turn": self.turn_number,
            "status": self.status,
            "last_reward": self.last_reward,
            "history_len": len(self.conversation_history)
        }
