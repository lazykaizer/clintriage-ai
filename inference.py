"""
ClinTriageAI — Inference Script
Location: ROOT directory (MANDATORY — moving this file = disqualification)

Runs all 5 tasks against the local ClinTriageAI environment server.
ALL LLM calls use openai.OpenAI() as required.
Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN
"""
import os
import json
import time
import requests
from openai import OpenAI


# ─── Configuration ────────────────────────────────────────────────────────────
ENV_URL = "http://localhost:8000"
client  = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["HF_TOKEN"],
)
MODEL = os.environ["MODEL_NAME"]


# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM = """You are an expert emergency medicine physician in an Indian hospital.
Your job is to triage patients based on their vitals, symptoms, and medical history.

Triage Levels:
  LEVEL_1 → Immediate (life-threatening — seen NOW)
  LEVEL_2 → Emergency (seen within 15 minutes)
  LEVEL_3 → Urgent (seen within 30 minutes)
  LEVEL_4 → Semi-Urgent (seen within 60 minutes)
  LEVEL_5 → Non-Urgent (can wait 2+ hours)

CRITICAL Clinical Tips for Indian ER:
- Diabetic patients (especially women) can have MI WITHOUT chest pain — check BP, diaphoresis
- BP below 90/60 = shock = LEVEL_1 regardless of complaint
- Snake bite with ANY swelling = LEVEL_2 minimum
- Post-surgical patients with SOB = rule out PE (LEVEL_1)
- Elderly with confusion + fever = rule out sepsis (LEVEL_1)
- Always check vitals AND history together — complaints can be misleading
- Dengue with platelets <50k + bleeding = LEVEL_2
- Organophosphate poisoning (common in farmers) = LEVEL_1

Respond ONLY in valid JSON. No extra text, no markdown, no explanation outside the JSON."""


# ─── LLM Call ─────────────────────────────────────────────────────────────────
def call_llm(user_prompt: str) -> dict:
    """Call LLM and parse JSON response."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    text = response.choices[0].message.content.strip()

    # Clean markdown-wrapped JSON
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

    return json.loads(text)


# ─── Task-Specific Prompts ────────────────────────────────────────────────────
def build_prompt(task_id: int, observation: dict) -> str:
    """Build task-specific prompt from observation."""

    if task_id == 1:
        patient = observation.get("patient", observation)
        return (
            f"Triage this single patient. Classify as CRITICAL (LEVEL_1 or LEVEL_2) "
            f"or NON-CRITICAL (LEVEL_3, LEVEL_4, LEVEL_5).\n\n"
            f"Patient:\n{json.dumps(patient, indent=2)}\n\n"
            f"Return JSON: {{\"triage_decision\": \"LEVEL_X\", \"reasoning\": \"your clinical explanation\"}}"
        )

    elif task_id == 2:
        patients = observation.get("patients", [])
        ids = [p["patient_id"] for p in patients]
        return (
            f"Rank these 3 patients by urgency (most urgent FIRST):\n\n"
            f"{json.dumps(patients, indent=2)}\n\n"
            f"Patient IDs: {ids}\n"
            f"Return JSON: {{\"ranking\": [\"{ids[0]}\", ...], \"reasoning\": \"your clinical explanation\"}}"
        )

    elif task_id == 3:
        patients = observation.get("patients", [])
        ids = [p["patient_id"] for p in patients]
        return (
            f"Assign a triage level (LEVEL_1 through LEVEL_5) to each of these 5 patients:\n\n"
            f"{json.dumps(patients, indent=2)}\n\n"
            f"Patient IDs: {ids}\n"
            f"Return JSON: {{\"assignments\": {{\"{ids[0]}\": \"LEVEL_X\", \"{ids[1]}\": \"LEVEL_X\", ...}}}}"
        )

    elif task_id == 4:
        patients = observation.get("patients", [])
        ids = [p["patient_id"] for p in patients]
        return (
            f"RESOURCE CONSTRAINT: Only 3 ICU beds are available.\n"
            f"Choose the 3 patients who MOST need ICU admission:\n\n"
            f"{json.dumps(patients, indent=2)}\n\n"
            f"Patient IDs: {ids}\n"
            f"Return JSON: {{\"icu_patients\": [\"ID1\", \"ID2\", \"ID3\"], "
            f"\"reasoning\": \"detailed clinical explanation for each choice\"}}"
        )

    elif task_id == 5:
        patient = observation.get("patient", observation)
        return (
            f"CAREFULLY triage this patient. Be alert for hidden serious conditions "
            f"that may not be immediately obvious from the chief complaint.\n"
            f"Look for subtle vital sign abnormalities and risk factors in the history.\n\n"
            f"Patient:\n{json.dumps(patient, indent=2)}\n\n"
            f"Return JSON: {{\"triage_decision\": \"LEVEL_X\", \"reasoning\": \"your detailed clinical explanation\"}}"
        )

    return ""


# ─── Run Single Task ──────────────────────────────────────────────────────────
def run_task(task_id: int) -> float:
    """Run a single task: reset → LLM → step → return reward."""
    # Step 1: Reset
    reset_resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    reset_resp.raise_for_status()
    observation = reset_resp.json().get("observation", {})

    # Step 2: Build prompt and call LLM
    prompt = build_prompt(task_id, observation)
    try:
        action = call_llm(prompt)
    except Exception as e:
        print(f"  ⚠ LLM parse error: {e}. Using fallback action.")
        if task_id in (1, 5):
            action = {"triage_decision": "LEVEL_2", "reasoning": f"Fallback due to parse error: {e}"}
        elif task_id == 2:
            patients = observation.get("patients", [])
            action = {"ranking": [p["patient_id"] for p in patients], "reasoning": "Fallback order"}
        elif task_id == 3:
            patients = observation.get("patients", [])
            action = {"assignments": {p["patient_id"]: "LEVEL_3" for p in patients}}
        elif task_id == 4:
            patients = observation.get("patients", [])
            action = {"icu_patients": [p["patient_id"] for p in patients[:3]], "reasoning": "Fallback selection"}

    # Step 3: Send action to environment
    action["task_id"] = task_id
    step_resp = requests.post(f"{ENV_URL}/step", json=action)
    step_resp.raise_for_status()
    result = step_resp.json()

    print(f"  Reward: {result['reward']:.2f} | Feedback: {result['feedback'][:100]}")
    return result["reward"]


# ─── Main Evaluation Loop ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  ClinTriageAI — Evaluation Run")
    print("  Indian ER Triage RL Environment")
    print("=" * 55)

    task_names = {
        1: "Binary Triage",
        2: "Priority Ordering",
        3: "Multi-Patient Assignment",
        4: "ICU Resource Allocation",
        5: "Edge Case Detection",
    }

    scores = {}
    total_start = time.time()

    for task_id in range(1, 6):
        name = task_names[task_id]
        print(f"\n{'─'*55}")
        print(f"[Task {task_id}] {name}")
        print(f"{'─'*55}")

        t_start = time.time()
        try:
            score = run_task(task_id)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            score = 0.0
        elapsed = time.time() - t_start
        scores[task_id] = score
        print(f"  Time: {elapsed:.1f}s")

    total_time = time.time() - total_start
    avg_score = sum(scores.values()) / len(scores)

    print(f"\n{'='*55}")
    print("  FINAL RESULTS")
    print(f"{'='*55}")
    for task_id, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  Task {task_id}: {score:.2f}  {bar}  {task_names[task_id]}")
    print(f"\n  Average Score : {avg_score:.2f}")
    print(f"  Total Runtime : {total_time:.1f}s")
    print(f"{'='*55}")

    if avg_score >= 0.5:
        print("  ✓ GOOD — Score meets minimum threshold")
    else:
        print("  ⚠ LOW — Consider tuning prompts for higher scores")
    print(f"{'='*55}")
