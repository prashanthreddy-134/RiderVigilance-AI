// Real-Time Dashboard JavaScript Engine for RiderVigilance AI

let chart = null;
let maxChartPoints = 40;

// Web Audio API Continuous Alarm State Trackers
let audioCtx = null;
let alarmOscillator = null;
let alarmGainNode = null;
let alarmSirenInterval = null;
let isAlarmSoundPlaying = false;
let isAudioMuted = false;

document.addEventListener('DOMContentLoaded', () => {
    initChart();
    startTelemetryLoop();
    startLogsLoop();

    // Auto-unlock Web Audio API AudioContext on ANY user gesture
    const unlockEvents = ['click', 'keydown', 'touchstart', 'pointerdown', 'mousedown'];
    unlockEvents.forEach(evt => {
        window.addEventListener(evt, unlockAudioContext, { passive: true, once: false });
    });
});

// Auto-unlock Web Audio API AudioContext safely
function unlockAudioContext() {
    try {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        const btnText = document.getElementById('audio-btn-text');
        if (btnText && audioCtx.state === 'running') {
            btnText.innerText = "Alarm Sound Active";
        }
        const btn = document.getElementById('enable-audio-btn');
        if (btn && audioCtx.state === 'running') {
            btn.classList.add('btn-success');
        }
    } catch (e) {
        console.error("[AudioAlarm] AudioContext unlock error:", e);
    }
}

// Starts LOUD continuous siren alarm sound loop until stopped
function startContinuousAlarm() {
    if (isAlarmSoundPlaying || isAudioMuted) return;

    unlockAudioContext();

    try {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        alarmOscillator = audioCtx.createOscillator();
        alarmGainNode = audioCtx.createGain();

        alarmOscillator.type = 'sawtooth';
        // High volume continuous siren sound (0.65 gain)
        alarmGainNode.gain.setValueAtTime(0.65, audioCtx.currentTime);

        alarmOscillator.connect(alarmGainNode);
        alarmGainNode.connect(audioCtx.destination);

        alarmOscillator.start();
        isAlarmSoundPlaying = true;

        // Continuous high-pitch siren sweep: 950Hz <-> 620Hz every 200ms
        let highTone = true;
        alarmSirenInterval = setInterval(() => {
            if (!isAlarmSoundPlaying || !alarmOscillator || !audioCtx) return;
            const now = audioCtx.currentTime;
            const targetFreq = highTone ? 950 : 620;
            alarmOscillator.frequency.cancelScheduledValues(now);
            alarmOscillator.frequency.setValueAtTime(targetFreq, now);
            highTone = !highTone;
        }, 200);

        console.log("[AudioAlarm] Continuous loud alarm sound STARTED!");
    } catch (e) {
        console.error("[AudioAlarm] Error starting continuous alarm sound:", e);
    }
}

// Stops and resets continuous alarm sound immediately when rider becomes alert
function stopContinuousAlarm() {
    if (alarmSirenInterval) {
        clearInterval(alarmSirenInterval);
        alarmSirenInterval = null;
    }

    if (alarmOscillator) {
        try {
            const now = audioCtx ? audioCtx.currentTime : 0;
            if (alarmGainNode && audioCtx) {
                alarmGainNode.gain.cancelScheduledValues(now);
                alarmGainNode.gain.setValueAtTime(0.001, now);
            }
            alarmOscillator.stop();
            alarmOscillator.disconnect();
        } catch (e) {}
        alarmOscillator = null;
        alarmGainNode = null;
    }
    isAlarmSoundPlaying = false;
    console.log("[AudioAlarm] Continuous alarm sound STOPPED and reset.");
}

// Reusable Alarm System Handler
function handleAlarmSystem(data) {
    const banner = document.getElementById('alarm-banner');
    const bannerText = document.getElementById('alarm-banner-text');

    const isAlertState = (
        data.alarm_triggered || 
        data.mobile_detected ||
        data.sleepy_confirmed ||
        data.state_code === 5 ||
        data.state_code === 4 || 
        data.state_code === 3 || 
        data.state_code === 2 || 
        data.closure_duration >= 2.0 || 
        data.yawn_confirmed
    );

    if (isAlertState) {
        if (banner) {
            banner.classList.remove('hidden');
            if (data.alert_message) {
                bannerText.innerText = data.alert_message;
            }
        }
        // PLAY LOUD CONTINUOUS ALARM SOUND IMMEDIATELY UNTIL RIDER IS ALERT AGAIN
        startContinuousAlarm();
    } else {
        if (banner) {
            banner.classList.add('hidden');
        }
        // STOP & RESET ALARM SOUND IMMEDIATELY WHEN RIDER RETURNS TO ALERT STATE
        stopContinuousAlarm();
    }
}

// Toggle mute/unmute audio alarm sound
function toggleMuteAudio() {
    isAudioMuted = !isAudioMuted;
    const btn = document.getElementById('mute-alarm-btn');
    if (isAudioMuted) {
        stopContinuousAlarm();
        if (btn) btn.innerHTML = '<i class="fa-solid fa-volume-high"></i> Unmute';
    } else {
        if (btn) btn.innerHTML = '<i class="fa-solid fa-volume-xmark"></i> Mute';
    }
}

// Initialize Chart.js Telemetry Graph
function initChart() {
    const ctx = document.getElementById('telemetryChart').getContext('2d');
    
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Average EAR',
                    data: [],
                    borderColor: '#60A5FA',
                    backgroundColor: 'rgba(96, 165, 250, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                },
                {
                    label: 'Mouth Aspect Ratio (MAR)',
                    data: [],
                    borderColor: '#FBBF24',
                    backgroundColor: 'rgba(251, 191, 36, 0.05)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: { display: false },
                y: {
                    min: 0.0,
                    max: 0.8,
                    grid: { color: 'rgba(255, 255, 255, 0.06)' },
                    ticks: { color: '#9CA3AF' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#F3F4F6', font: { family: 'Inter', size: 12 } }
                }
            }
        }
    });
}

// Start Telemetry Polling Loop (Every 250ms)
function startTelemetryLoop() {
    setInterval(fetchTelemetry, 250);
}

// Start Logs & Summary Polling Loop (Every 2.5s)
function startLogsLoop() {
    fetchLogs();
    fetchSummary();
    setInterval(() => {
        fetchLogs();
        fetchSummary();
    }, 2500);
}

// Fetch Real-time Telemetry Data
async function fetchTelemetry() {
    try {
        const resp = await fetch('/api/telemetry');
        if (!resp.ok) return;
        const data = await resp.json();
        if (!data.state_name) return;

        updateUI(data);
        updateChart(data.avg_ear || data.ear, data.mar);
        handleAlarmSystem(data);
    } catch (e) {
        console.error("Telemetry fetch error:", e);
    }
}

// Update Frontend UI & All Debugging Metrics
function updateUI(data) {
    const avgEar = (data.avg_ear !== undefined) ? data.avg_ear : data.ear;
    const lEar = data.left_ear !== undefined ? data.left_ear : avgEar;
    const rEar = data.right_ear !== undefined ? data.right_ear : avgEar;

    // Eye Metrics Debugging Panel
    document.getElementById('ear-val').innerText = avgEar.toFixed(3);
    document.getElementById('ear-l-r').innerText = `L: ${lEar.toFixed(3)} | R: ${rEar.toFixed(3)}`;
    document.getElementById('ear-thresh').innerText = `Thresh: < ${data.ear_threshold.toFixed(2)} | Closed: ${data.is_eye_closed}`;
    document.getElementById('closure-duration-subtext').innerText = `Closure Duration: ${data.closure_duration.toFixed(1)}s`;
    document.getElementById('closure-val').innerHTML = `${data.closure_duration.toFixed(1)} <small>sec</small>`;

    const shortSub = document.getElementById('short-closures-subtext');
    if (shortSub) {
        shortSub.innerText = `Short Closures (1-2s): ${data.short_closure_count || 0} / ${data.sleepiness_event_count || 3}`;
    }

    // Mouth Metrics Debugging Panel
    document.getElementById('mar-val').innerText = data.mar.toFixed(3);
    document.getElementById('yawn-thresh').innerText = `Thresh: > ${data.yawn_threshold.toFixed(2)} | Open: ${data.is_mouth_open}`;
    document.getElementById('yawn-duration-subtext').innerText = `Yawn Duration: ${data.yawn_duration.toFixed(1)}s`;
    document.getElementById('yawn-confirmed-subtext').innerText = `Yawn Confirmed: ${data.yawn_confirmed}`;

    // YOLO Mobile Phone Detector Card
    const mobileStatus = document.getElementById('mobile-status-val');
    const mobileConf = document.getElementById('mobile-conf-subtext');
    const mobileAlert = document.getElementById('mobile-alert-subtext');
    const mobileCard = document.getElementById('mobile-card');
    const mobileIcon = document.getElementById('mobile-icon');

    if (data.mobile_detected) {
        if (mobileStatus) mobileStatus.innerHTML = '<span class="text-danger">MOBILE DETECTED</span>';
        if (mobileConf) mobileConf.innerText = `Confidence: ${Math.round((data.mobile_confidence || 0.95) * 100)}%`;
        if (mobileAlert) mobileAlert.innerText = '⚠️ DO NOT USE MOBILE!';
        if (mobileCard) mobileCard.style.borderColor = '#EF4444';
        if (mobileIcon) mobileIcon.style.color = '#EF4444';
    } else {
        if (mobileStatus) mobileStatus.innerHTML = '<span class="text-success">NO MOBILE</span>';
        if (mobileConf) mobileConf.innerText = 'Confidence: 0%';
        if (mobileAlert) mobileAlert.innerText = 'YOLOv8 Active';
        if (mobileCard) mobileCard.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        if (mobileIcon) mobileIcon.style.color = '#10B981';
    }

    // Blink & Alarm Subtext
    document.getElementById('blink-val').innerHTML = `${data.blink_rate.toFixed(1)} <small>/min</small>`;
    document.getElementById('alarm-state-subtext').innerText = `Alarm Triggered: ${data.alarm_triggered}`;

    // Confidence & Score
    document.getElementById('confidence-val').innerText = `${Math.round(data.confidence * 100)}%`;
    document.getElementById('fatigue-score-val').innerText = data.fatigue_score.toFixed(2);
    
    const fillBar = document.getElementById('fatigue-score-bar');
    fillBar.style.width = `${Math.min(100, Math.max(0, data.fatigue_score * 100))}%`;

    // State Colors
    const stateColors = {
        0: '#10B981', // Alert Green
        1: '#F59E0B', // Mild Yellow
        2: '#F97316', // Severe Orange
        3: '#EF4444', // Drowsy Red
        4: '#EF4444', // Mobile Danger Red
        5: '#F97316'  // Feeling Sleepy Orange
    };
    
    const color = stateColors[data.state_code] || '#10B981';
    
    // Top State Badge & Overlay Dot
    const stateText = data.state_name.toUpperCase();
    document.getElementById('state-text').innerText = stateText;
    document.getElementById('state-dot').style.backgroundColor = color;
    
    // Main Classifier Display Box
    const box = document.getElementById('state-display-box');
    box.style.borderColor = color;
    
    const icon = document.getElementById('state-icon');
    const title = document.getElementById('current-state-title');
    const desc = document.getElementById('current-state-desc');
    
    icon.style.color = color;
    title.style.color = color;
    title.innerText = stateText;

    if (data.state_code === 0) {
        icon.innerHTML = '<i class="fa-solid fa-face-smile"></i>';
        desc.innerText = "Rider is attentive with steady blink rate and eye openness.";
    } else if (data.state_code === 1) {
        icon.innerHTML = '<i class="fa-solid fa-face-meh"></i>';
        desc.innerText = "Early fatigue signs detected. Variations in blink patterns observed.";
    } else if (data.state_code === 2) {
        icon.innerHTML = '<i class="fa-solid fa-face-tired"></i>';
        desc.innerText = "YAWN DETECTED! Sustained wide mouth opening detected. Take a break!";
    } else if (data.state_code === 3) {
        icon.innerHTML = '<i class="fa-solid fa-triangle-exclamation pulse-icon"></i>';
        desc.innerText = "DROWSINESS DETECTED! Continuous eye closure >= 2.0s detected! WAKE UP IMMEDIATELY!";
    } else if (data.state_code === 4 || data.mobile_detected) {
        icon.innerHTML = '<i class="fa-solid fa-mobile-screen-button pulse-icon"></i>';
        desc.innerText = "⚠️ DO NOT USE MOBILE WHILE DRIVING! Mobile phone detected in use by rider!";
    } else if (data.state_code === 5 || data.sleepy_confirmed) {
        icon.innerHTML = '<i class="fa-solid fa-eye-slash pulse-icon"></i>';
        desc.innerText = "⚠️ YOU ARE FEELING SLEEPY! 3+ repeated short eye closures (1-2s) detected! Please do face wash / take a short break!";
    }

    // Feed Mode Badge & Button Text
    const badge = document.getElementById('feed-mode-badge');
    const btnText = document.getElementById('source-btn-text');
    if (data.use_simulation) {
        badge.innerText = `SIMULATED: ${data.sim_scenario.toUpperCase()}`;
        btnText.innerText = "Switch to Live Webcam";
    } else {
        badge.innerText = "LIVE WEBCAM STREAM";
        btnText.innerText = "Switch to Simulated Feed";
    }
}

// Update Real-Time Telemetry Chart
function updateChart(ear, mar) {
    if (!chart) return;
    
    const now = new Date().toLocaleTimeString();
    
    chart.data.labels.push(now);
    chart.data.datasets[0].data.push(ear);
    chart.data.datasets[1].data.push(mar);

    if (chart.data.labels.length > maxChartPoints) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
        chart.data.datasets[1].data.shift();
    }
    chart.update('none');
}

// Switch Scenario
async function setScenario(scenario, btnElement) {
    const btns = document.querySelectorAll('.btn-scenario');
    btns.forEach(b => b.classList.remove('active'));
    btnElement.classList.add('active');

    try {
        await fetch('/api/set_scenario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario })
        });
        const vImg = document.getElementById('video-stream');
        if (vImg) {
            vImg.src = `/video_feed?t=${Date.now()}`;
        }
    } catch (e) {
        console.error("Failed setting scenario:", e);
    }
}

// Toggle Camera Source (Webcam <-> Simulated Feed)
async function toggleFeedSource() {
    const btnText = document.getElementById('source-btn-text');
    const oldText = btnText.innerText;
    btnText.innerText = "Switching Feed...";
    
    try {
        const resp = await fetch('/api/toggle_source', { method: 'POST' });
        const data = await resp.json();
        
        const vImg = document.getElementById('video-stream');
        if (vImg) {
            vImg.src = `/video_feed?t=${Date.now()}`;
        }
        
        if (data.use_simulation) {
            btnText.innerText = "Switch to Live Webcam";
        } else {
            btnText.innerText = "Switch to Simulated Feed";
        }
    } catch (e) {
        console.error("Failed toggling source:", e);
        btnText.innerText = oldText;
    }
}

// Fetch Session Summary
async function fetchSummary() {
    try {
        const resp = await fetch('/api/summary');
        if (!resp.ok) return;
        const data = await resp.json();

        document.getElementById('summary-frames').innerText = data.total_frames.toLocaleString();
        document.getElementById('summary-duration').innerText = `${data.duration_seconds}s`;
        document.getElementById('summary-size').innerText = `${data.log_file_size_kb} KB`;
    } catch (e) {
        console.error("Summary fetch error:", e);
    }
}

// Fetch Recent Logs Table
async function fetchLogs() {
    try {
        const resp = await fetch('/api/logs');
        if (!resp.ok) return;
        const data = await resp.json();
        
        const tbody = document.getElementById('logs-table-body');
        if (!data.recent_rows || data.recent_rows.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center">No telemetry entries yet.</td></tr>';
            return;
        }

        const rowsHtml = data.recent_rows.slice(-15).reverse().map(r => `
            <tr>
                <td>${r.iso_time ? r.iso_time.split(' ')[1] : ''}</td>
                <td>#${r.frame_id}</td>
                <td>${parseFloat(r.ear).toFixed(3)}</td>
                <td>${parseFloat(r.mar).toFixed(3)}</td>
                <td>${parseFloat(r.blink_rate).toFixed(1)}</td>
                <td>${parseFloat(r.closure_duration).toFixed(1)}s</td>
                <td><strong>${parseFloat(r.fatigue_score).toFixed(2)}</strong></td>
                <td><span class="pill-state state-${r.state_code}">${r.state_name}</span></td>
            </tr>
        `).join('');
        
        tbody.innerHTML = rowsHtml;
    } catch (e) {
        console.error("Logs fetch error:", e);
    }
}

// Export Active CSV Log
async function exportActiveLog() {
    try {
        const resp = await fetch('/api/logs');
        const data = await resp.json();
        if (data.current_log) {
            window.location.href = `/api/download_log/${data.current_log}`;
        }
    } catch (e) {
        alert("Failed to export log file.");
    }
}

// Trigger Model Retrain
async function retrainModel() {
    if (!confirm("Retrain Random Forest model on telemetry data?")) return;
    try {
        const resp = await fetch('/api/retrain', { method: 'POST' });
        const data = await resp.json();
        alert(data.message || data.error);
    } catch (e) {
        alert("Retrain failed.");
    }
}
