"""
ClinTriageAI API Router — All endpoints.
"""
import os
import uuid
import pickle
import json
from fastapi import APIRouter, HTTPException, Request
from app.models import (
    ResetRequest, StepRequest, StepResponse, ResetResponse,
    Observation, StateResponse
)
from app.env import ClinTriageEnv

router = APIRouter()

SESSION_DIR = "/tmp/clinical_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

_memory_cache = {}


def _session_path(session_id: str) -> str:
    safe = session_id.replace("-", "")[:64]
    return os.path.join(SESSION_DIR, f"{safe}.pkl")


def save_session(session_id: str, env: ClinTriageEnv):
    _memory_cache[session_id] = env
    try:
        with open(_session_path(session_id), "wb") as f:
            pickle.dump(env, f)
    except Exception:
        pass


def load_session(session_id: str) -> ClinTriageEnv:
    if session_id in _memory_cache:
        return _memory_cache[session_id]
    try:
        path = _session_path(session_id)
        if os.path.exists(path):
            with open(path, "rb") as f:
                env = pickle.load(f)
            _memory_cache[session_id] = env
            return env
    except Exception:
        pass
    return None


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "environment": "ClinTriageAI",
        "version": "1.0.0",
    }


@router.get("/tasks")
def list_tasks():
    return [
        {"id": 1, "name": "binary_triage", "difficulty": "easy"},
        {"id": 2, "name": "priority_ordering", "difficulty": "medium"},
        {"id": 3, "name": "multi_patient_assignment", "difficulty": "medium"},
        {"id": 4, "name": "icu_resource_allocation", "difficulty": "hard"},
    ]


@router.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest = None):
    task_id = req.task_id if req and req.task_id else 1
    session_id = str(uuid.uuid4())
    
    env = ClinTriageEnv()
    obs = env.reset(task_id)
    save_session(session_id, env)

    return ResetResponse(session_id=session_id, observation=obs)


@router.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    env = load_session(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    obs, reward, done, feedback = env.step(req)
    
    if not done:
        save_session(req.session_id, env)
    
    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        feedback=feedback,
        info=env.get_state()
    )


@router.get("/state/{session_id}")
async def get_state(session_id: str):
    env = load_session(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.get_state()
