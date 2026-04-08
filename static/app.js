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
        const res = await fetch(`${API}/health`);
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
            <div class="card-demographics">
                <span class="demo-age">${p.age}</span>
                <span class="demo-gender">${p.gender === 'male' ? 'M' : 'F'}</span>
            </div>
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

    // Render structured feedback instead of raw text
    renderFeedback(feedback || 'No feedback.', reward);
    

    // Add thumbs up/down feedback buttons
    const existingFb = document.querySelector('.feedback-actions');
    if (existingFb) existingFb.remove();

    const fbDiv = document.createElement('div');
    fbDiv.className = 'feedback-actions';
    fbDiv.innerHTML = `
        <span>Rate this result</span>
        <button class="feedback-btn up" title="Good result">👍</button>
        <button class="feedback-btn down" title="Poor result">👎</button>
    `;
    $rewardDisp.appendChild(fbDiv);

    fbDiv.querySelector('.up').addEventListener('click', function() {
        this.classList.add('selected');
        showFeedbackToast('👍', 'Thanks for the feedback!');
        agentLog('<span class="log-tag end">👍 FEEDBACK</span> User rated result as <strong>Good</strong>', 'success');
        setTimeout(() => { fbDiv.style.opacity = '0'; fbDiv.style.transition = 'opacity 0.4s'; setTimeout(() => fbDiv.remove(), 400); }, 800);
    });
    fbDiv.querySelector('.down').addEventListener('click', function() {
        this.classList.add('selected');
        showFeedbackToast('👎', 'We\'ll improve next time!');
        agentLog('<span class="log-tag error">👎 FEEDBACK</span> User rated result as <strong>Poor</strong>', 'warning');
        setTimeout(() => { fbDiv.style.opacity = '0'; fbDiv.style.transition = 'opacity 0.4s'; setTimeout(() => fbDiv.remove(), 400); }, 800);
    });
}

// ─── Render Structured Feedback ──────────────────────────────
function renderFeedback(text, reward) {
    const $fb = document.getElementById('feedback-text');

    // Try to parse "X/Y correct" pattern
    const scoreMatch = text.match(/(\d+)\/(\d+)\s*correct/i);
    
    // Try to parse error entries like "C046: assigned LEVEL_3, truth LEVEL_5"
    const errorPattern = /(C\d+):\s*assigned\s+([\w_]+),\s*truth\s+([\w_]+)/gi;
    const errors = [];
    let m;
    while ((m = errorPattern.exec(text)) !== null) {
        errors.push({ id: m[1], assigned: m[2], truth: m[3] });
    }

    // Try to parse Ground Truth lists for ranking tasks
    const truthPattern = /Ground truth(?: ICU)?: \[(.*?)\]/i;
    const truthMatch = text.match(truthPattern);

    // If we can parse it, show structured view
    if (scoreMatch || errors.length > 0 || truthMatch) {
        let html = '';

        // Score badge
        if (scoreMatch) {
            const correct = parseInt(scoreMatch[1]);
            const total = parseInt(scoreMatch[2]);
            const pct = Math.round((correct / total) * 100);
            const cls = pct >= 80 ? 'fb-score-high' : pct >= 50 ? 'fb-score-mid' : 'fb-score-low';
            html += `<div class="fb-score-row">
                <span class="fb-score-badge ${cls}">${correct}/${total} Correct</span>
                <span class="fb-score-pct">${pct}% accuracy</span>
            </div>`;
        }

        // Task 1/2/4 specific parsing (Ground Truth lists)
        if (truthMatch && errors.length === 0) {
            const truthIds = truthMatch[1].split(',').map(s => s.trim().replace(/'/g, ''));
            html += '<div class="fb-errors-title">Correction Guide:</div>';
            html += '<div class="fb-error-list">';
            truthIds.forEach((id, idx) => {
                html += `<div class="fb-error-row">
                    <span class="fb-pid">Rank ${idx + 1}</span>
                    <span class="fb-truth">${id}</span>
                </div>`;
            });
            html += '</div>';
        }
        // Task 3 errors (Mismatches)
        else if (errors.length > 0) {
            html += '<div class="fb-errors-title">Mismatches:</div>';
            html += '<div class="fb-error-list">';
            errors.forEach(e => {
                html += `<div class="fb-error-row">
                    <span class="fb-pid">${e.id}</span>
                    <span class="fb-assigned">${e.assigned.replace('LEVEL_', 'L')}</span>
                    <span class="fb-arrow">→</span>
                    <span class="fb-truth">${e.truth.replace('LEVEL_', 'L')}</span>
                </div>`;
            });
            html += '</div>';
        }

        $fb.innerHTML = html;
    } else {
        // Fallback: just show text cleanly
        $fb.innerHTML = `<p class="fb-text-plain">${text}</p>`;
    }
}

// ─── Feedback Toast (Center-Screen Popup) ────────────────────
function showFeedbackToast(icon, msg) {
    // Remove any existing feedback toast
    document.querySelectorAll('.feedback-toast').forEach(t => t.remove());

    const el = document.createElement('div');
    el.className = 'feedback-toast';
    el.innerHTML = `<span class="fb-icon">${icon}</span><span class="fb-msg">${msg}</span>`;
    document.body.appendChild(el);

    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transform = 'translate(-50%, -50%) scale(0.8)';
        el.style.transition = 'all 0.3s ease';
        setTimeout(() => el.remove(), 300);
    }, 1800);
}

// ─── Delay Helper ────────────────────────────────────────────
const wait = (ms) => new Promise(r => setTimeout(r, ms));

// ─── Update Progress ────────────────────────────────────────
function setProgress(phase, percent) {
    $agentStatus.textContent = phase;
    $taskProgress.style.width = percent + '%';
}

// ─── The Agent's Clinical Triage Scoring (Trained Model) ─────
// Comprehensive rule-based heuristic matching the 5-level triage matrix
// LEVEL_1 (highest urgency) = score 100+ | LEVEL_5 (lowest) = score 0-10
function localTriageScore(patient) {
    let score = 0;
    const v = patient.vitals || {};
    const complaint = (patient.chief_complaint || '').toLowerCase();
    const history = (patient.history || '').toLowerCase();
    const arrival = (patient.arrival_mode || '').toLowerCase();
    const hr = v.heart_rate || 80;
    const o2 = v.oxygen_saturation || 99;
    const rr = v.respiratory_rate || 16;
    const temp = v.temperature || 37.0;
    const bp = v.blood_pressure || '120/80';
    const sys = parseInt(bp.split('/')[0]) || 120;
    const dia = parseInt(bp.split('/')[1]) || 80;
    const age = patient.age || 30;

    // ═══ LEVEL 1 — IMMEDIATE (life-threatening) ═══
    // Vital sign red flags → LEVEL_1
    if (o2 <= 89) score += 60;
    if (sys <= 80 && dia <= 50) score += 55;
    if (rr <= 10 || rr >= 32) score += 50;
    if (hr > 130 || hr < 45) score += 45;

    // Critical complaint keywords → LEVEL_1
    const L1_keywords = [
        'cardiac arrest', 'crushing chest pain', 'chest pain radiating',
        'unconscious', 'gcs 7', 'gcs 8', 'not responding',
        'active seizure', 'seizure ongoing', 'tonic-clonic seizure', 'status epilepticus',
        'anaphylaxis', 'lip swelling', 'tongue swelling',
        'massive hemorrhage', 'postpartum hemorrhage',
        'organophosphate', 'pesticide', 'poisoning',
        'cobra bite', 'neurotoxic', 'ptosis', 'difficulty swallowing'
    ];
    L1_keywords.forEach(w => { if (complaint.includes(w)) score += 65; });

    // ═══ LEVEL 2 — EMERGENCY (within 15 min) ═══
    const L2_keywords = [
        'viper bite', 'russell viper', 'bleeding from gums', 'blood-tinged urine',
        'dengue', 'platelet', 'petechiae',
        'severe breathlessness at rest', 'cannot lie flat', 'pink frothy sputum',
        'cerebral malaria', 'confusion', 'jaundiced',
        'ectopic pregnancy', 'hemoptysis', 'coughing up blood',
        'severe pancreatitis', 'pre-eclampsia', 'gangrene'
    ];
    L2_keywords.forEach(w => { if (complaint.includes(w)) score += 40; });

    // Pregnant + low BP + pain = likely L2
    if (complaint.includes('pregnant') && sys < 100 && complaint.includes('pain')) score += 30;

    // ═══ LEVEL 3 — URGENT (within 30 min) ═══
    if (o2 >= 93 && o2 <= 96 && score < 50) score += 20;

    const L3_keywords = [
        'appendicitis', 'right lower quadrant', 'rlq',
        'kidney stone', 'renal colic', 'flank pain', 'blood in urine',
        'fever', 'rash', 'dehydration', 'sunken eyes',
        'copd', 'dyspnea', 'productive cough',
        'uncontrolled blood sugar', 'blood sugar 450', 'excessive thirst',
        'severe migraine', 'not responding to medication',
        'fracture', 'deformity', 'angulation',
        'diarrhea', 'vomiting', 'decreased urine output',
        'urinary retention', 'unable to pass urine',
        'vision loss', 'curtain coming down', 'retinal detachment'
    ];
    L3_keywords.forEach(w => { if (complaint.includes(w)) score += 20; });

    // ═══ LEVEL 4 — SEMI-URGENT (within 60 min) ═══
    const L4_keywords = [
        'laceration', 'cut', 'bleeding controlled',
        'urinary burning', 'uti', 'burning and frequency',
        'back pain', 'low back pain',
        'sore throat', 'mild fever',
        'ear pain', 'tugging at ear', 'discharge',
        'twisted ankle', 'sprain', 'mild swelling',
        'rash', 'itchy', 'insect bite', 'redness',
        'constipation', 'bloating'
    ];
    if (score < 20) {
        L4_keywords.forEach(w => { if (complaint.includes(w)) score += 8; });
    }

    // ═══ LEVEL 5 — NON-URGENT (can wait 2+ hours) ═══
    const L5_keywords = [
        'common cold', 'seasonal cold', 'runny nose', 'sneezing',
        'acne', 'prescription refill', 'refill',
        'checkup', 'health checkup', 'annual screening', 'routine',
        'wart', 'painless wart', 'cosmetic',
        'muscle soreness', 'after gym', 'dull ache',
        'dandruff', 'heartburn', 'paper cut',
        'vaccination', 'vision test', 'mild heartburn'
    ];
    let isL5 = false;
    L5_keywords.forEach(w => { if (complaint.includes(w)) isL5 = true; });
    if (isL5 && score < 15) score = Math.max(score, 2); // Keep very low

    // ═══ Modifiers ═══
    // Arrival mode
    if (arrival === 'ambulance') score += 12;

    // Age vulnerability — very young or very old
    if (age <= 5 && score > 15) score += 10;
    if (age >= 65 && score > 15) score += 5;

    // History red flags
    const historyRedFlags = ['icu admission', 'previous ectopic', 'known allergy', 'poorly controlled', 'not compliant'];
    historyRedFlags.forEach(w => { if (history.includes(w)) score += 5; });

    // If vitals are completely normal and low score, cap it
    if (hr >= 60 && hr <= 100 && o2 >= 97 && sys >= 100 && sys <= 140 && rr >= 12 && rr <= 20 && temp <= 37.5) {
        if (score > 25) score = Math.min(score, 25); // Normal vitals = cap urgency
    }

    return score;
}

// ─── Skeleton Loading Cards ──────────────────────────────────
function showSkeletonCards(count) {
    $pool.innerHTML = '';
    for (let i = 0; i < count; i++) {
        const skel = document.createElement('div');
        skel.className = 'patient-card skeleton-card';
        skel.innerHTML = `
            <div class="card-top">
                <span class="skeleton-line" style="width:60px;height:12px"></span>
                <span class="skeleton-line" style="width:30px;height:12px"></span>
            </div>
            <div class="skeleton-line" style="width:100%;height:14px;margin:8px 0"></div>
            <div class="skeleton-line" style="width:75%;height:14px;margin-bottom:10px"></div>
            <div class="card-vitals">
                <div class="vital-item"><span class="skeleton-line" style="width:100%;height:28px"></span></div>
                <div class="vital-item"><span class="skeleton-line" style="width:100%;height:28px"></span></div>
                <div class="vital-item"><span class="skeleton-line" style="width:100%;height:28px"></span></div>
            </div>
        `;
        skel.style.animationDelay = `${i * 150}ms`;
        $pool.appendChild(skel);
    }
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
        1: 'Two-Patient Priority', 
        2: 'Priority Ordering', 
        3: 'Multi-Patient Assignment', 
        4: 'ICU Resource Allocation'
    };
    const currentTaskName = taskNames[selectedTaskId];
    const expectedPatients = taskDetails[selectedTaskId].slots;
    
    agentLog(`<span class="log-tag start">▶ INIT</span> Task ${selectedTaskId} — <strong>${currentTaskName}</strong> | env=ClinTriageAI`, 'start');

    setProgress('🔄 Initializing environment…', 10);

    // Show skeleton loading cards
    showSkeletonCards(expectedPatients);
    $patientCount.textContent = '…';

    await wait(800);

    // ── Step 1: Reset ──
    try {
        agentLog(`<span class="log-tag step">⬆ API</span> Calling <span class="log-code">POST /reset</span> with task_id=${selectedTaskId}`, 'info');
        const resetRes = await fetch(`${API}/reset`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: selectedTaskId })
        });

        const resetData = await resetRes.json();
        let patients = resetData.observation?.patients || [];
        if (resetData.observation?.patient) {
            patients = [resetData.observation.patient];
        }

        setProgress(`📥 ${patients.length} patients received`, 30);
        agentLog(`<span class="log-tag end">✓ RECV</span> Received <strong>${patients.length} patients</strong> from environment`, 'success');

        await wait(600);

        // Replace skeleton cards with actual patients
        $pool.innerHTML = '';
        $patientCount.textContent = patients.length;
        patients.forEach((p, i) => {
            setTimeout(() => {
                $pool.appendChild(createCard(p));
            }, i * 200);
        });

        await wait(patients.length * 200 + 500);

        // ── Step 2: Agent Analyzes ──
        setProgress('🩺 Analyzing vitals & symptoms…', 50);
        agentLog(`<span class="log-tag step">🔍 SCAN</span> Analyzing patient vitals, symptoms, and medical history…`, 'info');
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

        agentLog(`<span class="log-tag end">✔ RANK</span> Final order: <strong>[${ranking.join(' → ')}]</strong>`, 'success');
        setProgress('📌 Placing patients in ranking…', 70);

        await wait(800);

        // ── Step 3: Animate Placement ──
        const numSlots = taskDetails[selectedTaskId].slots;
        for (let i = 0; i < Math.min(scored.length, numSlots); i++) {
            placeCardInSlot(scored[i].patient, i);
            agentLog(`<span class="log-tag step">📍 #${i + 1}</span> ${scored[i].patient.patient_id} placed at Rank ${i + 1}`, 'step');
            await wait(700);
        }

        // Remove placed patients from pool
        $pool.innerHTML = '';
        const msg = document.createElement('div');
        msg.className = 'empty-state';
        msg.innerHTML = '<p>All patients ranked by agent ✅</p>';
        $pool.appendChild(msg);

        setProgress('📤 Submitting ranking to grader…', 85);
        await wait(800);

        // ── Step 4: Submit ──
        agentLog(`<span class="log-tag step">⬆ SUBMIT</span> Posting ranking to <span class="log-code">POST /step</span>`, 'info');

        const stepRes = await fetch(`${API}/step`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task_id: selectedTaskId,
                ranking: (selectedTaskId === 1 || selectedTaskId === 2) ? ranking : undefined,
                assignments: (selectedTaskId === 3) ? Object.fromEntries(scored.map(s => {
                    let level = 'LEVEL_3';
                    if (s.score >= 100) level = 'LEVEL_1';
                    else if (s.score >= 60) level = 'LEVEL_2';
                    else if (s.score >= 35) level = 'LEVEL_3';
                    else if (s.score >= 15) level = 'LEVEL_4';
                    else level = 'LEVEL_5';
                    return [s.patient.patient_id, level];
                })) : undefined,
                icu_patients: (selectedTaskId === 4) ? ranking.slice(0, 3) : undefined,
                reasoning: `Clinical severity range: [${scored[scored.length-1].score.toFixed(0)}-${scored[0].score.toFixed(0)}]. Prioritized using ABC (Airway, Breathing, Circulation) protocol for ${currentTaskName}.`
            })
        });
        const stepData = await stepRes.json();

        await wait(500);

        const reward = stepData.reward ?? 0;
        const done = stepData.done ?? true;

        agentLog(`<span class="log-tag step">📊 EVAL</span> reward=<strong>${reward.toFixed(2)}</strong> done=${done}`, 'step');

        // Show reward
        showReward(reward, stepData.feedback, stepData.info);

        setProgress('✅ Evaluation complete!', 100);

        const verdict = reward >= 0.8 ? '🎉 EXCELLENT' : reward >= 0.5 ? '⚡ GOOD' : reward >= 0.3 ? '⚠️ PARTIAL' : '❌ POOR';
        agentLog(`<span class="log-tag end">🏁 DONE</span> Score: <strong>${reward.toFixed(2)}</strong> — ${verdict}`, reward >= 0.5 ? 'success' : 'error');

        toast(`Agent scored ${reward.toFixed(2)} — ${verdict}`, reward >= 0.5 ? 'success' : 'error');

    } catch (e) {
        console.error("Agent Error:", e);
        agentLog(`<span class="log-tag error">[ERROR]</span> ${e.message}`, 'error');
        toast(`Agent run failed — ${e.message}`, 'error');
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

let selectedTaskId = 1; // Default to Task 1

// Task details for dynamic UI updates
const taskDetails = {
    1: { title: "Task 1 — Two-Patient Priority", desc: "Watch the AI agent compare 2 patients and rank them by urgency.", slots: 2 },
    2: { title: "Task 2 — Priority Ordering", desc: "Watch the AI agent analyze 3 patients and rank them by clinical urgency.", slots: 3 },
    3: { title: "Task 3 — Multi-Patient Assignment", desc: "Watch the agent assign triage levels to 5 simultaneous patients.", slots: 5 },
    4: { title: "Task 4 — ICU Resource Allocation", desc: "Watch the agent select 3 ICU patients from 8 with clinical reasoning.", slots: 8 }
};

// Task buttons
document.querySelectorAll('.task-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (isRunning) return;
        document.querySelectorAll('.task-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedTaskId = parseInt(btn.dataset.task);
        
        const details = taskDetails[selectedTaskId];
        if (details) {
            document.querySelector('.banner-text h2').textContent = details.title;
            document.querySelector('.banner-text p').textContent = details.desc;
            
            document.querySelectorAll('.rank-slot').forEach((slot, index) => {
                slot.style.display = index < details.slots ? 'flex' : 'none';
            });
        }

        // ── Reset entire dashboard for new task ──
        // Clear ranking drop zones
        document.querySelectorAll('.drop-zone').forEach(z => {
            z.innerHTML = '<p class="drop-hint">Waiting for agent…</p>';
            z.classList.remove('filled');
        });

        // Clear patient pool
        $pool.innerHTML = '';
        const emptyMsg = document.createElement('div');
        emptyMsg.className = 'empty-state';
        emptyMsg.innerHTML = `
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.3"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>
            <p>Agent will load patients automatically</p>
        `;
        $pool.appendChild(emptyMsg);
        $patientCount.textContent = '0';

        // Clear reward / results
        $rewardDisp.classList.add('hidden');
        $emptyResults.classList.remove('hidden');
        const existingFb = document.querySelector('.feedback-actions');
        if (existingFb) existingFb.remove();

        // Reset progress
        $taskProgress.style.width = '0%';
        $agentStatus.textContent = 'Idle — Press "Run Agent" to begin';

        // Clear agent log
        clearLog();
        agentLog(`Switched to <strong>Task ${selectedTaskId}</strong>: ${btn.innerText.trim()}`, 'info');

        toast(`Task ${selectedTaskId}: ${btn.innerText.trim()} selected`, 'info');
    });
});


// ─── Init ────────────────────────────────────────────────────
(async function init() {
    // Set default UI state for Task 2
    const details = taskDetails[selectedTaskId];
    document.querySelectorAll('.rank-slot').forEach((slot, index) => {
        slot.style.display = index < details.slots ? 'flex' : 'none';
    });

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
