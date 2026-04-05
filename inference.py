"""
ClinTriageAI — Inference Script
MANDATORY
- Standardized logging format: [START], [STEP], [END]
- Participants must use OpenAI Client for all LLM calls
- Root directory placement
"""
import os
import json
import time
import requests
import textwrap
from typing import List, Optional
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
ENV_URL = "http://localhost:8000"
# Mandatory per sample: API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api-inference.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Meta-Llama-3-8B-Instruct"
BENCHMARK = "ClinTriageAI"

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ─── Logging Helpers ──────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Clean action string from newlines and quotes to ensure single line logging
    action_clean = str(action).replace("\n", " ").replace("\"", "'")
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM = """You are an expert emergency medicine physician in an Indian hospital.
Your job is to triage patients based on their vitals, symptoms, and medical history.

Triage Levels:
  LEVEL_1 → Immediate (life-threatening — seen NOW)
  LEVEL_2 → Emergency (seen within 15 minutes)
  LEVEL_3 → Urgent (seen within 30 minutes)
  LEVEL_4 → Semi-Urgent (seen within 60 minutes)
  LEVEL_5 → Non-Urgent (can wait 2+ hours)

CRITICAL Clinical Tips:
- Diabetic patients often have MI WITHOUT chest pain — check BP, diaphoresis
- BP below 90/60 = shock = LEVEL_1 regardless of complaint
- Snake bite with ANY swelling = LEVEL_2 minimum
- Post-surgical patients with SOB = rule out PE (LEVEL_1)
- Elderly with confusion + fever = rule out sepsis (LEVEL_1)

Respond ONLY in valid JSON. No extra text, no markdown, no explanation outside the JSON."""

# ─── Task-Specific Prompts ────────────────────────────────────────────────────
def build_prompt(task_id: int, observation: dict) -> str:
    if task_id == 1:
        patient = observation.get("patient", observation)
        return f"Triage this single patient: {json.dumps(patient, indent=2)}\nReturn JSON: {{\"triage_decision\": \"LEVEL_X\", \"reasoning\": \"explanation\"}}"
    elif task_id == 2:
        patients = observation.get("patients", [])
        return f"Rank these 3 patients by urgency: {json.dumps(patients, indent=2)}\nReturn JSON: {{\"ranking\": [\"ID1\", \"ID2\", \"ID3\"], \"reasoning\": \"explanation\"}}"
    elif task_id == 3:
        patients = observation.get("patients", [])
        return f"Assign triage levels to these 5 patients: {json.dumps(patients, indent=2)}\nReturn JSON: {{\"assignments\": {{\"ID1\": \"LEVEL_X\", ...}}}}"
    elif task_id == 4:
        patients = observation.get("patients", [])
        return f"Only 3 ICU beds available. Select top 3 patients: {json.dumps(patients, indent=2)}\nReturn JSON: {{\"icu_patients\": [\"ID1\", \"ID2\", \"ID3\"], \"reasoning\": \"explanation\"}}"
    elif task_id == 5:
        patient = observation.get("patient", observation)
        return f"Triage this patient carefully (watch for hidden conditions): {json.dumps(patient, indent=2)}\nReturn JSON: {{\"triage_decision\": \"LEVEL_X\", \"reasoning\": \"explanation\"}}"
    return ""

# ─── LLM Call ─────────────────────────────────────────────────────────────────
def call_llm(user_prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    text = response.choices[0].message.content.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()
    return json.loads(text)

# ─── Task Runner ───────────────────────────────────────────────────────────────
def run_task(task_id: int, task_name: str) -> float:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    success = False
    steps_taken = 0
    final_score = 0.0

    try:
        # Step 1: Reset
        reset_resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        reset_resp.raise_for_status()
        observation = reset_resp.json().get("observation", {})

        # Step 2: Agent Action
        prompt = build_prompt(task_id, observation)
        try:
            action = call_llm(prompt)
            error = None
        except Exception as e:
            action = {"triage_decision": "LEVEL_3", "reasoning": f"Error: {str(e)}"}
            error = str(e)

        # Step 3: Step
        action["task_id"] = task_id
        step_resp = requests.post(f"{ENV_URL}/step", json=action)
        step_resp.raise_for_status()
        result = step_resp.json()

        reward = result["reward"]
        done = result["done"]
        rewards.append(reward)
        steps_taken = 1
        final_score = reward
        success = reward >= 0.5

        log_step(step=1, action=json.dumps(action), reward=reward, done=done, error=error)

    except Exception as e:
        # Emit a step line even on failure if possible, or just fail to the finally block
        pass
    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
    
    return final_score

# ─── Main Evaluation Loop ────────────────────────────────────────────────────
if __name__ == "__main__":
    tasks = {
        1: "binary_triage",
        2: "priority_ordering",
        3: "multi_patient_assignment",
        4: "icu_resource_allocation",
        5: "edge_case_detection"
    }

    total_scores = []
    # No extra noise in stdout during automated grading, but keeping one header or using [DEBUG]
    # Small debug header
    print(f"[DEBUG] ClinTriageAI Evaluation Run — Model: {MODEL_NAME}", flush=True)

    for t_id, t_name in tasks.items():
        score = run_task(t_id, t_name)
        total_scores.append(score)

    avg_score = sum(total_scores) / len(total_scores)
    print(f"[DEBUG] Final Average Score: {avg_score:.2f}", flush=True)
