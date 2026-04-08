"""
ClinTriageAI — Pre-Submission Validator
Run this before submitting. ALL tests must pass.

Usage:
    1. Start server: uvicorn main:app --port 8000
    2. Run tests:    python tests/test_all.py
"""
import sys
import requests

BASE = "http://localhost:8000"
PASS_COUNT = 0
FAIL_COUNT = 0


def check(name: str, condition: bool, detail: str = ""):
    """Record a test result."""
    global PASS_COUNT, FAIL_COUNT
    if condition:
        print(f"  ✅ PASS — {name}")
        PASS_COUNT += 1
    else:
        print(f"  ❌ FAIL — {name}: {detail}")
        FAIL_COUNT += 1


def main():
    global PASS_COUNT, FAIL_COUNT

    print("\n" + "=" * 55)
    print("  ClinTriageAI — Pre-Submission Validator")
    print("=" * 55)

    # ─── Health Check ─────────────────────────────────────
    print("\n[1] Health Check (GET /health)")
    try:
        r = requests.get(f"{BASE}/health", timeout=10)
        check("Returns 200", r.status_code == 200, f"Got {r.status_code}")
        data = r.json()
        check("Has status:ok", data.get("status") == "ok", f"Got: {data}")
        check("Has environment name", "environment" in data, "Missing 'environment' key")
    except Exception as e:
        check("Server reachable", False, str(e))

    # ─── Tasks List ───────────────────────────────────────
    print("\n[2] Tasks List (GET /tasks)")
    try:
        r = requests.get(f"{BASE}/tasks", timeout=10)
        check("Returns 200", r.status_code == 200, f"Got {r.status_code}")
        tasks = r.json()
        check("Returns 4 tasks", len(tasks) == 4, f"Got {len(tasks)} tasks")
        check("Each task has id", all("id" in t for t in tasks), "Missing 'id' in some tasks")
    except Exception as e:
        check("Tasks endpoint reachable", False, str(e))

    # ─── State ────────────────────────────────────────────
    print("\n[3] State (GET /state)")
    try:
        r = requests.get(f"{BASE}/state", timeout=10)
        check("Returns 200", r.status_code == 200, f"Got {r.status_code}")
        state = r.json()
        check("Has current_task key", "current_task" in state, f"Keys: {list(state.keys())}")
        check("Has step_count key", "step_count" in state, f"Keys: {list(state.keys())}")
    except Exception as e:
        check("State endpoint reachable", False, str(e))

    # ─── Reset All Tasks ──────────────────────────────────
    print("\n[4] Reset All Tasks (POST /reset)")
    for task_id in range(1, 5):
        try:
            r = requests.post(f"{BASE}/reset", json={"task_id": task_id}, timeout=10)
            check(f"Task {task_id} reset returns 200", r.status_code == 200, f"Got {r.status_code}")
            data = r.json()
            check(
                f"Task {task_id} has observation",
                "observation" in data,
                f"Keys: {list(data.keys())}",
            )
        except Exception as e:
            check(f"Task {task_id} reset", False, str(e))

    # ─── Step: Task 1 (Binary Ordering) ─────────────────────
    print("\n[5] Step — Task 1 (POST /step)")
    try:
        reset_r = requests.post(f"{BASE}/reset", json={"task_id": 1}, timeout=10)
        patients = reset_r.json()["observation"]["patients"]
        ids = [p["patient_id"] for p in patients]
        r = requests.post(
            f"{BASE}/step",
            json={"task_id": 1, "ranking": ids, "reasoning": "Test ranking"},
            timeout=10,
        )
        check("Returns 200", r.status_code == 200, f"Got {r.status_code}")
        data = r.json()
        check("Has reward", "reward" in data, f"Keys: {list(data.keys())}")
        check(
            "Reward is float [0,1]",
            isinstance(data.get("reward"), (int, float)) and 0.0 <= data["reward"] <= 1.0,
            f"Reward: {data.get('reward')}",
        )
        check("Has done", "done" in data, "Missing 'done'")
        check("Has feedback", "feedback" in data, "Missing 'feedback'")
    except Exception as e:
        check("Step Task 1", False, str(e))

    # ─── Step: Task 2 (Priority Ordering) ─────────────────
    print("\n[6] Step — Task 2 (POST /step)")
    try:
        reset_r = requests.post(f"{BASE}/reset", json={"task_id": 2}, timeout=10)
        patients = reset_r.json()["observation"]["patients"]
        ids = [p["patient_id"] for p in patients]
        r = requests.post(
            f"{BASE}/step",
            json={"task_id": 2, "ranking": ids, "reasoning": "Test ranking"},
            timeout=10,
        )
        check("Returns 200", r.status_code == 200, f"Got {r.status_code}")
        data = r.json()
        check(
            "Reward in [0,1]",
            0.0 <= data.get("reward", -1) <= 1.0,
            f"Reward: {data.get('reward')}",
        )
    except Exception as e:
        check("Step Task 2", False, str(e))

    # ─── Step: Task 3 (Multi-Patient) ─────────────────────
    print("\n[7] Step — Task 3 (POST /step)")
    try:
        reset_r = requests.post(f"{BASE}/reset", json={"task_id": 3}, timeout=10)
        patients = reset_r.json()["observation"]["patients"]
        assignments = {p["patient_id"]: "LEVEL_3" for p in patients}
        r = requests.post(
            f"{BASE}/step",
            json={"task_id": 3, "assignments": assignments},
            timeout=10,
        )
        check("Returns 200", r.status_code == 200, f"Got {r.status_code}")
        data = r.json()
        check(
            "Reward in [0,1]",
            0.0 <= data.get("reward", -1) <= 1.0,
            f"Reward: {data.get('reward')}",
        )
    except Exception as e:
        check("Step Task 3", False, str(e))

    # ─── Step: Task 4 (ICU — no LLM, just programmatic) ──
    print("\n[8] Step — Task 4 (POST /step)")
    try:
        reset_r = requests.post(f"{BASE}/reset", json={"task_id": 4}, timeout=10)
        patients = reset_r.json()["observation"]["patients"]
        icu_ids = [p["patient_id"] for p in patients[:3]]
        r = requests.post(
            f"{BASE}/step",
            json={
                "task_id": 4,
                "icu_patients": icu_ids,
                "reasoning": "Selected first three patients for ICU based on clinical severity",
            },
            timeout=30,
        )
        check("Returns 200", r.status_code == 200, f"Got {r.status_code}")
        data = r.json()
        check(
            "Reward in [0,1]",
            0.0 <= data.get("reward", -1) <= 1.0,
            f"Reward: {data.get('reward')}",
        )
    except Exception as e:
        check("Step Task 4", False, str(e))



    # ─── Summary ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Results: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print(f"{'='*55}")

    if FAIL_COUNT == 0:
        print("  🎉 ALL TESTS PASSED — Safe to submit!")
    else:
        print("  ⚠️  Fix failures before submitting!")

    print(f"{'='*55}\n")
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
