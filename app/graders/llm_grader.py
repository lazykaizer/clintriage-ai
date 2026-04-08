"""
LLM-based grader for ClinTriageAI.
Uses OpenAI client (mandatory) to evaluate clinical reasoning quality.
Returns score in [0.0, 0.5] — this is the LLM half of Task 4 scoring.
"""
import json
import httpx
from typing import Tuple
from openai import OpenAI
import config


JUDGE_SYSTEM = """You are a senior emergency medicine physician in an Indian hospital.
You are evaluating an AI triage agent's clinical reasoning.

Score the reasoning quality from 0 to 10 based on:
- Did the agent mention relevant vital signs and their clinical significance?
- Did the agent consider the patient's medical history and comorbidities?
- Is the reasoning clinically sound and evidence-based?
- Did the agent miss any obvious warning signs or red flags?
- Is the explanation clear and structured?

Return ONLY a JSON object in this exact format:
{"score": 7, "feedback": "Brief explanation of your scoring"}

No other text. Just the JSON object."""


def grade_reasoning(
    patient_data: list,
    agent_reasoning: str,
    ground_truth_level: int
) -> Tuple[float, str]:
    """
    LLM judge for clinical reasoning quality.
    Ensures no proxy-related issues during client initialization.
    """
    try:
        # Explicit initialization with empty proxies to avoid environment-level leaks
        client = OpenAI(
            base_url=config.API_BASE_URL,
            api_key=config.HF_TOKEN,
            http_client=httpx.Client(proxies={})
        )

        prompt = f"""Evaluate this AI triage agent's clinical reasoning:

Patient Data:
{json.dumps(patient_data, indent=2)}

Agent's Reasoning:
{agent_reasoning}

Ground Truth Triage Level: LEVEL_{ground_truth_level}

Score the clinical reasoning quality from 0-10 and provide brief feedback."""

        response = client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1,
        )
        content = response.choices[0].message.content.strip()

        # Handle markdown-wrapped JSON
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)
        score_0_to_10 = max(0.0, min(10.0, float(result.get("score", 5))))
        feedback = result.get("feedback", "No feedback provided.")

        # Convert 0-10 score to 0.0-0.5 range
        final_score = round((score_0_to_10 / 10.0) * 0.5, 2)
        return final_score, feedback

    except Exception as e:
        # Fallback: return middle score so we don't completely fail
        return 0.25, f"LLM judge error: {str(e)} — defaulting to 0.25/0.50"
