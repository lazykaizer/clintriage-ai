"""
ClinTriageAI — FastAPI Entry Point
Indian ER Triage RL Environment — OpenEnv Hackathon 2026
"""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# ── API Routes ──
app.include_router(router)

# ── Static Files & Dashboard ──
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    async def serve_dashboard():
        return FileResponse(os.path.join(static_dir, "index.html"))
