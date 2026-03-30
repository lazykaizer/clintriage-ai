"""
ClinTriageAI API Router — All endpoints.
GET  /       → health check
GET  /tasks  → list of 5 task definitions
GET  /state  → current environment state
POST /reset  → reset environment for a task
POST /step   → submit agent action, get reward
"""
from fastapi import APIRouter, HTTPException
from app.models import ResetRequest, StepRequest, StepResponse
from app.env import ClinTriageEnv
from app.graders import programmatic_grader, llm_grader

router = APIRouter()
env = ClinTriageEnv()


@router.get("/")
def health_check():
    """Health check endpoint. Judges ping this first."""
    return {
        "status": "ok",
        "environment": "ClinTriageAI",
        "version": "1.0.0",
        "tasks": 5,
    }


@router.get("/tasks")
def list_tasks():
    """Return definitions for all 5 tasks."""
    return [
        {
            "id": 1,
            "name": "binary_triage",
            "difficulty": "easy",
            "description": "Classify single patient as critical or non-critical",
        },
        {
            "id": 2,
            "name": "priority_ordering",
            "difficulty": "medium",
            "description": "Rank 3 patients by treatment urgency",
        },
        {
            "id": 3,
            "name": "multi_patient_assignment",
            "difficulty": "medium",
            "description": "Assign triage levels to 5 simultaneous patients",
        },
        {
            "id": 4,
            "name": "icu_resource_allocation",
            "difficulty": "hard",
            "description": "Select 3 ICU patients from 8 with clinical reasoning",
        },
        {
            "id": 5,
            "name": "edge_case_detection",
            "difficulty": "hard",
            "description": "Identify life-threatening condition from misleading symptoms",
        },
    ]


@router.get("/state")
def get_state():
    """Return the current environment state."""
    return env.get_state()


@router.post("/reset")
def reset(req: ResetRequest):
    """Reset environment for a specific task. Returns observation."""
    if req.task_id not in range(1, 6):
        raise HTTPException(status_code=400, detail="task_id must be 1-5")

    observation = env.reset(req.task_id)
    return {"task_id": req.task_id, "observation": observation}


@router.post("/step")
def step(req: StepRequest):
    """
    Process agent action and return reward + feedback.
    Must call /reset before /step.
    """
    if env.current_patients is None:
        raise HTTPException(
            status_code=400,
            detail="No active task. Call POST /reset first.",
        )

    patients = env.current_patients
    task_id = req.task_id

    try:
        if task_id == 1:
            if not req.triage_decision:
                raise HTTPException(400, "Task 1 requires triage_decision")
            reward, feedback = programmatic_grader.grade_task1(
                req.triage_decision,
                patients[0]["ground_truth_level"],
            )

        elif task_id == 2:
            if not req.ranking:
                raise HTTPException(400, "Task 2 requires ranking")
            reward, feedback = programmatic_grader.grade_task2(
                req.ranking, patients
            )

        elif task_id == 3:
            if not req.assignments:
                raise HTTPException(400, "Task 3 requires assignments")
            reward, feedback = programmatic_grader.grade_task3(
                req.assignments, patients
            )

        elif task_id == 4:
            if not req.icu_patients:
                raise HTTPException(400, "Task 4 requires icu_patients")

            # Programmatic score (0.0 - 0.5)
            prog_score, prog_fb = programmatic_grader.grade_task4_programmatic(
                req.icu_patients, patients
            )

            # LLM judge score (0.0 - 0.5)
            llm_score, llm_fb = llm_grader.grade_reasoning(
                [env._clean(p) for p in patients[:3]],
                req.reasoning or "",
                patients[0]["ground_truth_level"],
            )

            reward = round(min(1.0, prog_score + llm_score), 2)
            feedback = f"Programmatic: {prog_fb} | LLM Judge: {llm_fb}"

        elif task_id == 5:
            if not req.triage_decision:
                raise HTTPException(400, "Task 5 requires triage_decision")
            reward, feedback = programmatic_grader.grade_task5(
                req.triage_decision,
                patients[0],
                req.reasoning or "",
            )

        else:
            raise HTTPException(400, "Invalid task_id. Must be 1-5.")

    except HTTPException:
        raise
    except Exception as e:
        # Safety net — never return rewards outside [0.0, 1.0]
        reward = 0.0
        feedback = f"Grading error: {str(e)}"

    # Clamp reward to [0.0, 1.0] — mandatory rule
    reward = round(max(0.0, min(1.0, float(reward))), 2)

    env.last_reward = reward
    env.patients_seen += len(patients)

    return StepResponse(
        reward=reward,
        done=True,
        feedback=feedback,
        info={
            "task_id": task_id,
            "patients_seen": env.patients_seen,
            "step_count": env.step_count,
        },
    )
