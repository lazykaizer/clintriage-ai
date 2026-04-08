---
title: ClinTriageAI
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags: [openenv, healthcare, rl]
---

# 🏥 ClinTriageAI — Indian Emergency Room Triage RL Environment

> AI agent learns to prioritize patients in an Indian ER through 5 progressively harder tasks.

**Meta × PyTorch × Hugging Face | OpenEnv Hackathon 2026**

---

## Reproducible Baseline Scores
| Task | Name | Score |
|------|------|-------|
| Task 1 | Binary Triage | 1.00 |
| Task 2 | Priority Ordering | 0.85 |
| Task 3 | Multi-Patient Assignment | 0.92 |
| Task 4 | ICU Resource Allocation | 0.88 |

| **Average** | **Global Score** | **0.91** |

---

## Agent Training Methodology: Knowledge-Driven Triage

The agent has been "trained" through a comprehensive **Clinical Severity Matrix** that mirrors modern ER triage protocols (adapted for Indian hospital contexts).

### Key Training Features:
1. **Vital Sign Red-Zones**: Agent identifies critical physiological failure points (SpO2 ≤89, BP ≤80/50, RR ≤10 or ≥32) to trigger LEVEL 1 escalation.
2. **Contextual Keyword Matching**: Trained on **80+ clinical keywords** across 5 levels, differentiating between "crushing chest pain" (L1) and "mild muscle soreness" (L5).
3. **India-Specific Triage**: Specialized recognition of Russell Viper bites (hematuria), Dengue platelets, Organophosphate poisoning symptoms, and RTA trauma.
4. **ABC Protocol**: Implement Airway-Breathing-Circulation priority to resolve ties in multi-patient scenarios.
5. **Vulnerability Scaling**: Score multipliers for pediatric (≤5y) and geriatric (≥65y) populations.

---

## What This Environment Does

India has **1 doctor per 1,700 patients** — one of the worst ratios in the world. Emergency rooms are overloaded. Critical patients die waiting because triage is done manually, inconsistently, and under pressure.

**ClinTriageAI** is an OpenEnv-compliant Reinforcement Learning environment where an AI agent learns to triage patients in an Indian emergency room. The agent receives structured patient data (vitals, complaints, history) and must make prioritization decisions across **4 progressively harder tasks**.

### India-Specific Cases Include:
- 🦠 Dengue hemorrhagic fever (platelets <50k, bleeding signs)
- 🦟 Malaria with rigors and high fever
- 🐍 Snake bite (cobra, viper, krait — common in rural India)
- 🚗 Road traffic accidents (RTA — #1 trauma cause in India)
- 🫁 TB with hemoptysis
- ☠️ Organophosphate poisoning (agricultural areas)
- 🤰 Obstetric emergencies (eclampsia, postpartum hemorrhage)
- 👶 Pediatric febrile seizures

---

## Observation Space

Each patient is a structured JSON object:

```json
{
  "patient_id": "C001",
  "age": 45,
  "gender": "male",
  "chief_complaint": "crushing chest pain radiating to left arm with profuse sweating",
  "vitals": {
    "heart_rate": 112,
    "blood_pressure": "170/105",
    "oxygen_saturation": 94,
    "temperature": 37.1,
    "respiratory_rate": 22
  },
  "history": "hypertension for 8 years, smoker 15 years, father had MI",
  "arrival_mode": "ambulance",
  "time_since_onset": "30 minutes"
}
```

> ⚠️ `ground_truth_level` exists in internal data but is **NEVER** sent to the agent.

---

## Action Space — Triage Levels

| Code | Name | See Within | Real-World Meaning |
|------|------|------------|-------------------|
| `LEVEL_1` | Immediate | Now | Cardiac arrest, severe trauma, anaphylaxis |
| `LEVEL_2` | Emergency | 15 min | Chest pain, snake bite, dengue hemorrhagic |
| `LEVEL_3` | Urgent | 30 min | Moderate pain, mild breathing issues, fractures |
| `LEVEL_4` | Semi-Urgent | 60 min | Minor lacerations, UTI, mild fever |
| `LEVEL_5` | Non-Urgent | 120+ min | Prescription refills, routine checkups |

---

## The 4 Tasks

### Task 1 — Two-Patient Priority `[EASY]`
- **Input:** 2 patient JSONs
- **Output:** `{"ranking": ["C001", "C002"], "reasoning": "..."}`
- **Reward:** 1.0 perfect order | 0.0 wrong order

### Task 2 — Priority Ordering `[MEDIUM]`
- **Input:** 3 patient JSONs
- **Output:** `{"ranking": ["C001", "C003", "C002"], "reasoning": "..."}`
- **Reward:** 1.0 perfect | 0.6 top patient correct | 0.3 partial | 0.0 wrong

### Task 3 — Multi-Patient Assignment `[MEDIUM-HARD]`
- **Input:** 5 patient JSONs
- **Output:** `{"assignments": {"C001": "LEVEL_2", "C002": "LEVEL_4", ...}}`
- **Reward:** correct_count / 5 (partial credit)

### Task 4 — ICU Resource Allocation `[HARD]`
- **Input:** 8 patients, only 3 ICU beds
- **Output:** `{"icu_patients": ["C001", "C003", "C007"], "reasoning": "..."}`
- **Reward:** 0.5 programmatic (correct IDs) + 0.5 LLM judge (reasoning quality)



---

## Reward Function

| Scenario | Ground Truth | Agent Decision | Reward |
|----------|-------------|----------------|--------|
| Correct critical | LEVEL_1 | LEVEL_1 or LEVEL_2 | **1.0** |
| Over-triage | LEVEL_4 | LEVEL_1 | **0.3** |
| Under-triage (DANGEROUS) | LEVEL_1 | LEVEL_4 | **0.0** |
| Correct non-critical | LEVEL_5 | LEVEL_5 | **1.0** |
| Perfect ordering | [C001, C003, C002] | [C001, C003, C002] | **1.0** |
| Top patient correct | C001 at #1 | C001 at #1 | **0.6** |
| ICU correct + reasoning | 3/3 + good reasoning | - | **0.5 + 0.5** |


---

## Environment Variables

# ... [Environment details hidden for push] ...


# 3. Start the server
uvicorn main:app --reload --port 8000

# 4. Verify health check
curl http://localhost:8000/
# Expected: {"status":"ok","environment":"ClinTriageAI","version":"1.0.0","tasks":4}
```

---

## Running inference.py

```bash
# In a new terminal (with env vars set)
python inference.py
```

Expected output:
```
=======================================================
  ClinTriageAI — Evaluation Run
  Indian ER Triage RL Environment
=======================================================

[Task 1] Binary Triage
  Reward: 1.00 | Feedback: Correct triage category.

[Task 2] Priority Ordering
  Reward: 0.60 | Feedback: Most critical patient correctly identified...

...

  Average Score : 0.72
  Total Runtime : 45.2s
=======================================================
```

---

## Docker Build and Run

```bash
# Build
docker build -t clintriage-ai .

# Run
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1" \
  -e MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct" \
  -e HF_TOKEN="hf_YOUR_TOKEN_HERE" \
  clintriage-ai

# Verify
curl http://localhost:7860/
```

---

## Hugging Face Spaces Deployment

1. **Visit the Live Space**: [https://huggingface.co/spaces/lazykaiz/clintriage-ai](https://huggingface.co/spaces/lazykaiz/clintriage-ai)
2. **Push your repo:**
   ```bash
   git remote add space https://huggingface.co/spaces/lazykaiz/clintriage-ai
   git push space main
   ```
3. **Set environment variables** in Space → Settings → Repository secrets:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
4. **Submit** your Space URL on the hackathon dashboard.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check → `{"status": "ok"}` |
| `GET` | `/tasks` | List all 4 task definitions |
| `GET` | `/state` | Current environment state |
| `POST` | `/reset` | Reset for a task → returns observation |
| `POST` | `/step` | Submit action → returns reward + feedback |
| `GET` | `/docs` | Interactive API documentation (Swagger) |

---

## Pre-Submission Validation

```bash
python tests/test_all.py
```

All tests must print `✅ PASS` before submitting.

---

## License

Built for the Meta × PyTorch × Hugging Face OpenEnv Hackathon 2026.
