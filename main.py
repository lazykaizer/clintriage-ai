"""
ClinTriageAI — FastAPI Entry Point
Indian ER Triage RL Environment — OpenEnv Hackathon 2026
"""
from fastapi import FastAPI
from app.router import router

app = FastAPI(
    title="ClinTriageAI",
    description=(
        "Indian Emergency Room Triage RL Environment. "
        "AI agent learns to prioritize patients across 5 progressively harder tasks."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(router)
