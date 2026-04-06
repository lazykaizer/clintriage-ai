/* ═══════════════════════════════════════════════════════════════
   ClinTriageAI — Dashboard Application Logic (Agent-Driven)
   The AI Agent runs automatically. User watches the process.
   ═══════════════════════════════════════════════════════════════ */

const API = '';  // Same origin

// ─── State ───────────────────────────────────────────────────
let isRunning = false;

// ─── DOM Refs ────────────────────────────────────────────────
const $pool         = document.getElementById('patient-cards');
const $emptyPool    = document.getElementById('empty-pool');
const $patientCount = document.getElementById('patient-count');
const $btnRun       = document.getElementById('btn-run-agent');
const $dropZones    = document.querySelectorAll('.drop-zone');
const $rewardDisp   = document.getElementById('reward-display');
const $emptyResults = document.getElementById('empty-results');
const $rewardCircle = document.getElementById('reward-circle');
const $rewardValue  = document.getElementById('reward-value');
const $feedbackText = document.getElementById('feedback-text');
const $infoJson     = document.getElementById('info-json');
const $status       = document.getElementById('server-status');
const $toasts       = document.getElementById('toast-container');
const $agentLog     = document.getElementById('agent-log');
const $agentStatus  = document.getElementById('agent-status-text');
const $taskProgress = document.getElementById('task-progress-bar');

// ─── Toast System ────────────────────────────────────────────
function toast(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    $toasts.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 3500);
}

// ─── Agent Log ───────────────────────────────────────────────
function agentLog(msg, type = 'info') {
    const line = document.createElement('div');
    line.className = `log-line log-${type}`;
    
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    line.innerHTML = `<span class="log-time">${time}</span> ${msg}`;
    $agentLog.appendChild(line);
    $agentLog.scrollTop = $agentLog.scrollHeight;
}

function clearLog() {
    $agentLog.innerHTML = '';
}

// ─── Server Health Check ─────────────────────────────────────
async function checkServer() {
    try {
        const res = await fetch(`${API}/`);
        if (res.ok) {
            $status.className = 'status-badge online';
            $status.querySelector('.status-text').textContent = 'Server Online';
            return true;
        }
    } catch (e) { /* fall through */ }
    $status.className = 'status-badge offline';
    $status.querySelector('.status-text').textContent = 'Server Offline';
    return false;
}

// ─── Vital Color Logic ───────────────────────────────────────
function vitalClass(type, val) {
    if (type === 'hr') {
        if (val > 110 || val < 50) return 'danger';
        if (val > 100 || val < 60) return 'warning';
        return 'ok';
    }
    if (type === 'o2') {
        if (val < 90) return 'danger';
        if (val < 95) return 'warning';
        return 'ok';
    }
    return '';
}

// ─── Render a Patient Card ───────────────────────────────────
function createCard(p, rankPosition = null) {
    const card = document.createElement('div');
    card.className = 'patient-card';
    if (rankPosition !== null) card.classList.add('ranked');
    card.dataset.patientId = p.patient_id;

    const hr = p.vitals?.heart_rate ?? '—';
    const o2 = p.vitals?.oxygen_saturation ?? '—';
    const bp = p.vitals?.blood_pressure ?? '—';

    card.innerHTML = `
        <div class="card-top">
            <span class="card-id">${p.patient_id}</span>
            <span class="card-age-gender">${p.age}${p.gender === 'male' ? 'M' : 'F'}</span>
        </div>
        <div class="card-complaint">${p.chief_complaint || 'No complaint recorded'}</div>
        <div class="card-vitals">
            <div class="vital-item">
                <div class="vital-label">HR</div>
                <div class="vital-value ${vitalClass('hr', hr)}">${hr}</div>
            </div>
            <div class="vital-item">
                <div class="vital-label">SpO₂</div>
                <div class="vital-value ${vitalClass('o2', o2)}">${o2}%</div>
            </div>
            <div class="vital-item">
                <div class="vital-label">BP</div>
                <div class="vital-value">${bp}</div>
            </div>
        </div>
        <div class="card-arrival">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            <span>${p.time_since_onset || 'Unknown'} · ${p.arrival_mode || 'walk-in'}</span>
        </div>
    `;
    return card;
}

// ─── Animate card into a drop zone ───────────────────────────
function placeCardInSlot(patient, slotIndex) {
    const zone = $dropZones[slotIndex];
    zone.innerHTML = '';
    zone.appendChild(createCard(patient, slotIndex + 1));
    zone.classList.add('filled');
    zone.classList.add('animate-in');
    setTimeout(() => zone.classList.remove('animate-in'), 500);
}

// ─── Show Reward ─────────────────────────────────────────────
function showReward(reward, feedback, info) {
    $emptyResults.classList.add('hidden');
    $rewardDisp.classList.remove('hidden');

    $rewardValue.textContent = reward.toFixed(2);
    $rewardCircle.classList.remove('high', 'mid', 'low');
    if (reward >= 0.7) $rewardCircle.classList.add('high');
    else if (reward >= 0.4) $rewardCircle.classList.add('mid');
    else $rewardCircle.classList.add('low');

    $feedbackText.textContent = feedback || 'No feedback.';
    $infoJson.textContent = JSON.stringify(info || {}, null, 2);
}

// ─── Delay Helper ────────────────────────────────────────────
const wait = (ms) => new Promise(r => setTimeout(r, ms));

// ─── Update Progress ────────────────────────────────────────
function setProgress(phase, percent) {
    $agentStatus.textContent = phase;
    $taskProgress.style.width = percent + '%';
}

// ─── The Agent's Simple Triage Logic ─────────────────────────
// This mimics what inference.py does — a rule-based heuristic
// so the dashboard can run WITHOUT an LLM connection
function localTriageScore(patient) {
    let score = 0;
    const v = patient.vitals || {};
    
    // Heart rate
    if (v.heart_rate > 120 || v.heart_rate < 50) score += 40;
    else if (v.heart_rate > 100) score += 20;
    
    // Oxygen
    if (v.oxygen_saturation < 90) score += 40;
    else if (v.oxygen_saturation < 95) score += 20;
    
    // BP — parse systolic
    if (v.blood_pressure) {
        const sys = parseInt(v.blood_pressure.split('/')[0]);
        if (sys < 90 || sys > 180) score += 35;
        else if (sys < 100) score += 15;
    }
    
    // Respiratory rate
    if (v.respiratory_rate > 24) score += 25;
    else if (v.respiratory_rate > 20) score += 10;
    
    // Arrival mode
    if (patient.arrival_mode === 'ambulance') score += 15;
    
    // Keywords in complaint
    const complaint = (patient.chief_complaint || '').toLowerCase();
    const criticalWords = ['hemorrhage', 'bleeding', 'unconscious', 'seizure', 'chest pain', 'stroke', 'snake', 'poisoning', 'anaphylaxis', 'cardiac'];
    const urgentWords = ['fracture', 'pregnant', 'vomiting blood', 'difficulty breathing', 'high fever'];
    
    criticalWords.forEach(w => { if (complaint.includes(w)) score += 30; });
    urgentWords.forEach(w => { if (complaint.includes(w)) score += 15; });
    
    return score;
}

// ─── Main Agent Run ──────────────────────────────────────────
async function runAgent() {
    if (isRunning) return;
    isRunning = true;
    $btnRun.disabled = true;
    $btnRun.innerHTML = `<span class="btn-spinner"></span> Agent Running…`;

    // Reset UI
    clearLog();
    $rewardDisp.classList.add('hidden');
    $emptyResults.classList.remove('hidden');
    $dropZones.forEach(z => { z.innerHTML = '<p class="drop-hint">Waiting…</p>'; z.classList.remove('filled'); });
    $pool.innerHTML = '';

    const taskNames = { 
        1: 'Binary Triage', 
        2: 'Priority Ordering', 
        3: 'Multi-Patient Assignment', 
        4: 'ICU Resource Allocation', 
        5: 'Edge Case Detection' 
    };
    const currentTaskName = taskNames[selectedTaskId];
    
    agentLog(`<span class="log-tag start">[START]</span> task=${selectedTaskId} (${currentTaskName}) env=ClinTriageAI`, 'start');

    setProgress('Resetting environment…', 10);

    await wait(800);

    // ── Step 1: Reset ──
    try {
        agentLog(`Calling <span class="log-code">POST /reset</span> with task_id=${selectedTaskId} …`, 'info');
        const resetRes = await fetch(`${API}/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: selectedTaskId })
        });

        const resetData = await resetRes.json();
        const patients = resetData.observation?.patients || [];

        setProgress('Patients received.', 30);
        agentLog(`Received <strong>${patients.length} patients</strong> from environment`, 'success');

        await wait(600);

        // Show patients in pool
        $pool.innerHTML = '';
        $patientCount.textContent = patients.length;
        patients.forEach((p, i) => {
            setTimeout(() => {
                $pool.appendChild(createCard(p));
            }, i * 300);
        });

        await wait(patients.length * 300 + 500);

        // ── Step 2: Agent Analyzes ──
        setProgress('Analyzing vitals & symptoms…', 50);
        agentLog('Agent analyzing patient vitals, symptoms, and medical history…', 'info');
        await wait(1500);

        // Score each patient
        const scored = patients.map(p => ({
            patient: p,
            score: localTriageScore(p)
        }));

        // Log analysis
        scored.forEach(({ patient: p, score: s }) => {
            const severity = s > 60 ? '🔴 CRITICAL' : s > 30 ? '🟡 URGENT' : '🟢 STABLE';
            agentLog(`  ${p.patient_id}: urgency_score=${s} → ${severity}`, s > 60 ? 'critical' : s > 30 ? 'warning' : 'info');
        });

        await wait(1000);

        // Sort by score descending (most urgent first)
        scored.sort((a, b) => b.score - a.score);
        const ranking = scored.map(s => s.patient.patient_id);

        agentLog(`Agent decision: <strong>[${ranking.join(' → ')}]</strong>`, 'success');
        setProgress('Placing patients in ranking…', 70);

        await wait(800);

        // ── Step 3: Animate Placement ──
        for (let i = 0; i < Math.min(scored.length, 3); i++) {
            placeCardInSlot(scored[i].patient, i);
            agentLog(`<span class="log-tag step">[PLACE]</span> ${scored[i].patient.patient_id} → Rank #${i + 1}`, 'step');
            await wait(700);
        }

        // Remove placed patients from pool
        $pool.innerHTML = '';
        const msg = document.createElement('div');
        msg.className = 'empty-state';
        msg.innerHTML = '<p>All patients ranked by agent ✅</p>';
        $pool.appendChild(msg);

        setProgress('Submitting ranking to grader…', 85);
        await wait(800);

        // ── Step 4: Submit ──
        agentLog(`Calling <span class="log-code">POST /step</span> with ranking=[${ranking.join(', ')}]`, 'info');

        const stepRes = await fetch(`${API}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_id: selectedTaskId,
                ranking: (selectedTaskId === 2) ? ranking : undefined,
                triage_decision: (selectedTaskId === 1 || selectedTaskId === 5) ? (scored[0]?.patient?.score > 50 ? 'LEVEL_1' : 'LEVEL_3') : undefined,
                assignments: (selectedTaskId === 3) ? Object.fromEntries(scored.map(s => [s.patient.patient_id, s.score > 60 ? 'LEVEL_1' : 'LEVEL_3'])) : undefined,
                icu_patients: (selectedTaskId === 4) ? ranking.slice(0, 3) : undefined,
                reasoning: `Agent evaluated patients based on clinical features for ${currentTaskName}.`

            })
        });
        const stepData = await stepRes.json();

        await wait(500);

        const reward = stepData.reward ?? 0;
        const done = stepData.done ?? true;

        agentLog(`<span class="log-tag step">[STEP]</span> step=1 action=rank(${ranking.join(',')}) reward=<strong>${reward.toFixed(2)}</strong> done=${done}`, 'step');

        // Show reward
        showReward(reward, stepData.feedback, stepData.info);

        setProgress('Evaluation complete!', 100);

        const verdict = reward >= 0.8 ? '🎉 EXCELLENT' : reward >= 0.5 ? '⚡ GOOD' : reward >= 0.3 ? '⚠️ PARTIAL' : '❌ POOR';
        agentLog(`<span class="log-tag end">[END]</span> success=${reward >= 0.5} steps=1 score=<strong>${reward.toFixed(2)}</strong> — ${verdict}`, reward >= 0.5 ? 'success' : 'error');

        toast(`Agent scored ${reward.toFixed(2)} — ${verdict}`, reward >= 0.5 ? 'success' : 'error');

    } catch (e) {
        agentLog(`<span class="log-tag error">[ERROR]</span> ${e.message}`, 'error');
        toast('Agent run failed — is the server running?', 'error');
        setProgress('Error!', 0);
    } finally {
        isRunning = false;
        $btnRun.disabled = false;
        $btnRun.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Run Agent Again
        `;
    }
}

// ─── Event Listeners ─────────────────────────────────────────
$btnRun.addEventListener('click', runAgent);

let selectedTaskId = 2; // Default to Task 2

// Task buttons
document.querySelectorAll('.task-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (isRunning) return;
        document.querySelectorAll('.task-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedTaskId = parseInt(btn.dataset.task);
        toast(`Task ${selectedTaskId}: ${btn.innerText.trim()} selected`, 'info');
        agentLog(`Switched to <strong>Task ${selectedTaskId}</strong>: ${btn.innerText.trim()}`, 'info');
    });
});


// ─── Init ────────────────────────────────────────────────────
(async function init() {
    const online = await checkServer();
    if (online) {
        toast('Connected to ClinTriageAI server', 'success');
        agentLog('Server connection established. Ready to run agent.', 'success');
    } else {
        toast('Server not running — start with: uvicorn main:app --port 8000', 'error');
        agentLog('⚠ Server offline. Start the FastAPI server first.', 'error');
    }
    setInterval(checkServer, 15000);
})();
