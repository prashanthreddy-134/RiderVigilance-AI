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

// Browser webcam streaming for hosted deployments (Render).
// Frames are captured locally and POSTed to Flask.
let browserCameraStream = null;
let browserCameraVideo = null;
let browserCameraCanvas = null;
let browserCameraTimer = null;
let browserCameraStarting = false;
let browserFrameInFlight = false;
let hostedBrowserCamera = false;

const BROWSER_FRAME_INTERVAL_MS = 200;
const BROWSER_FRAME_WIDTH = 640;


// ============================================================
// BROWSER WEBCAM FUNCTIONS
// ============================================================

function ensureBrowserCameraElements() {
    if (!browserCameraVideo) {
        browserCameraVideo = document.createElement('video');
        browserCameraVideo.autoplay = true;
        browserCameraVideo.muted = true;
        browserCameraVideo.playsInline = true;
        browserCameraVideo.style.display = 'none';
        document.body.appendChild(browserCameraVideo);
    }

    if (!browserCameraCanvas) {
        browserCameraCanvas = document.createElement('canvas');
        browserCameraCanvas.style.display = 'none';
        document.body.appendChild(browserCameraCanvas);
    }
}


async function startBrowserCamera() {
    if (
        !hostedBrowserCamera ||
        browserCameraStream ||
        browserCameraStarting
    ) {
        return;
    }

    browserCameraStarting = true;
    ensureBrowserCameraElements();

    try {
        if (
            !navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia
        ) {
            throw new Error(
                'Browser webcam access requires HTTPS and a supported browser.'
            );
        }

        browserCameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user',
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 10, max: 15 }
            },
            audio: false
        });

        browserCameraVideo.srcObject = browserCameraStream;

        await browserCameraVideo.play();

        const videoFeed = document.getElementById('video-stream');

        if (videoFeed) {
            videoFeed.src = `/video_feed?t=${Date.now()}`;
        }

        browserCameraTimer = setInterval(
            sendBrowserCameraFrame,
            BROWSER_FRAME_INTERVAL_MS
        );

        console.log(
            '[BrowserCamera] Webcam started; sending frames to server.'
        );

    } catch (error) {
        console.error(
            '[BrowserCamera] Could not start webcam:',
            error
        );

        stopBrowserCamera();

        alert(
            `Could not access your webcam: ${error.message}\n` +
            'Allow camera permission and make sure this page is opened over HTTPS.'
        );

        try {
            await fetch('/api/toggle_source', {
                method: 'POST'
            });
        } catch (toggleError) {
            console.error(
                '[BrowserCamera] Could not restore simulation:',
                toggleError
            );
        }

    } finally {
        browserCameraStarting = false;
    }
}


function sendBrowserCameraFrame() {
    if (
        !browserCameraStream ||
        !browserCameraVideo ||
        browserCameraVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
        browserFrameInFlight
    ) {
        return;
    }

    const videoWidth = browserCameraVideo.videoWidth;
    const videoHeight = browserCameraVideo.videoHeight;

    if (!videoWidth || !videoHeight) {
        return;
    }

    const scale = Math.min(
        1,
        BROWSER_FRAME_WIDTH / videoWidth
    );

    const width = Math.round(videoWidth * scale);
    const height = Math.round(videoHeight * scale);

    browserCameraCanvas.width = width;
    browserCameraCanvas.height = height;

    const context = browserCameraCanvas.getContext('2d');

    context.drawImage(
        browserCameraVideo,
        0,
        0,
        width,
        height
    );

    browserFrameInFlight = true;

    browserCameraCanvas.toBlob(async (blob) => {

        if (!blob || !browserCameraStream) {
            browserFrameInFlight = false;
            return;
        }

        try {
            const response = await fetch('/api/frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'image/jpeg'
                },
                body: blob
            });

            if (!response.ok) {
                const result = await response.json().catch(() => ({}));

                console.error(
                    '[BrowserCamera] Frame rejected:',
                    result.error || response.statusText
                );
            }

        } catch (error) {
            console.error(
                '[BrowserCamera] Frame upload failed:',
                error
            );

        } finally {
            browserFrameInFlight = false;
        }

    }, 'image/jpeg', 0.75);
}


function stopBrowserCamera() {

    if (browserCameraTimer) {
        clearInterval(browserCameraTimer);
        browserCameraTimer = null;
    }

    if (browserCameraStream) {
        browserCameraStream
            .getTracks()
            .forEach(track => track.stop());

        browserCameraStream = null;
    }

    if (browserCameraVideo) {
        browserCameraVideo.pause();
        browserCameraVideo.srcObject = null;
    }

    browserFrameInFlight = false;
}


window.addEventListener('beforeunload', stopBrowserCamera);


// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', () => {

    initChart();
    startTelemetryLoop();
    startLogsLoop();

    const unlockEvents = [
        'click',
        'keydown',
        'touchstart',
        'pointerdown',
        'mousedown'
    ];

    unlockEvents.forEach(evt => {
        window.addEventListener(
            evt,
            unlockAudioContext,
            {
                passive: true,
                once: false
            }
        );
    });

});


// ============================================================
// AUDIO SYSTEM
// ============================================================

function unlockAudioContext() {

    try {
        if (!audioCtx) {
            audioCtx = new (
                window.AudioContext ||
                window.webkitAudioContext
            )();
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
        console.error(
            "[AudioAlarm] AudioContext unlock error:",
            e
        );
    }
}


function startContinuousAlarm() {

    if (isAlarmSoundPlaying || isAudioMuted) {
        return;
    }

    unlockAudioContext();

    try {
        if (!audioCtx) {
            audioCtx = new (
                window.AudioContext ||
                window.webkitAudioContext
            )();
        }

        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        alarmOscillator = audioCtx.createOscillator();
        alarmGainNode = audioCtx.createGain();

        alarmOscillator.type = 'sawtooth';

        alarmGainNode.gain.setValueAtTime(
            0.65,
            audioCtx.currentTime
        );

        alarmOscillator.connect(alarmGainNode);
        alarmGainNode.connect(audioCtx.destination);

        alarmOscillator.start();

        isAlarmSoundPlaying = true;

        let highTone = true;

        alarmSirenInterval = setInterval(() => {

            if (
                !isAlarmSoundPlaying ||
                !alarmOscillator ||
                !audioCtx
            ) {
                return;
            }

            const now = audioCtx.currentTime;
            const targetFreq = highTone ? 950 : 620;

            alarmOscillator.frequency.cancelScheduledValues(now);

            alarmOscillator.frequency.setValueAtTime(
                targetFreq,
                now
            );

            highTone = !highTone;

        }, 200);

        console.log(
            "[AudioAlarm] Continuous alarm sound STARTED!"
        );

    } catch (e) {
        console.error(
            "[AudioAlarm] Error starting alarm:",
            e
        );
    }
}


function stopContinuousAlarm() {

    if (alarmSirenInterval) {
        clearInterval(alarmSirenInterval);
        alarmSirenInterval = null;
    }

    if (alarmOscillator) {

        try {
            const now = audioCtx
                ? audioCtx.currentTime
                : 0;

            if (alarmGainNode && audioCtx) {
                alarmGainNode.gain.cancelScheduledValues(now);

                alarmGainNode.gain.setValueAtTime(
                    0.001,
                    now
                );
            }

            alarmOscillator.stop();
            alarmOscillator.disconnect();

        } catch (e) {}

        alarmOscillator = null;
        alarmGainNode = null;
    }

    isAlarmSoundPlaying = false;

    console.log(
        "[AudioAlarm] Continuous alarm sound STOPPED."
    );
}


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

            if (bannerText && data.alert_message) {
                bannerText.innerText = data.alert_message;
            }
        }

        startContinuousAlarm();

    } else {

        if (banner) {
            banner.classList.add('hidden');
        }

        stopContinuousAlarm();
    }
}


function toggleMuteAudio() {

    isAudioMuted = !isAudioMuted;

    const btn = document.getElementById('mute-alarm-btn');

    if (isAudioMuted) {

        stopContinuousAlarm();

        if (btn) {
            btn.innerHTML =
                '<i class="fa-solid fa-volume-high"></i> Unmute';
        }

    } else {

        if (btn) {
            btn.innerHTML =
                '<i class="fa-solid fa-volume-xmark"></i> Mute';
        }
    }
}
// ============================================================
// CHART INITIALIZATION
// ============================================================

function initChart() {

    const canvas = document.getElementById('fatigueChart');

    if (!canvas) {
        console.warn('[Chart] fatigueChart canvas not found.');
        return;
    }

    const ctx = canvas.getContext('2d');

    if (typeof Chart === 'undefined') {
        console.warn('[Chart] Chart.js is not loaded.');
        return;
    }

    chart = new Chart(ctx, {
        type: 'line',

        data: {
            labels: [],

            datasets: [
                {
                    label: 'Fatigue Score',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.12)',
                    borderWidth: 2,
                    tension: 0.35,
                    fill: true,
                    pointRadius: 0
                },
                {
                    label: 'EAR',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.08)',
                    borderWidth: 2,
                    tension: 0.35,
                    fill: false,
                    pointRadius: 0
                }
            ]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,

            scales: {
                x: {
                    display: true,
                    ticks: {
                        maxTicksLimit: 8,
                        color: '#94a3b8'
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)'
                    }
                },

                y: {
                    min: 0,
                    max: 1,
                    ticks: {
                        color: '#94a3b8'
                    },
                    grid: {
                        color: 'rgba(148, 163, 184, 0.1)'
                    }
                }
            },

            plugins: {
                legend: {
                    labels: {
                        color: '#94a3b8'
                    }
                }
            }
        }
    });
}


function updateChart(data) {

    if (!chart || !data) {
        return;
    }

    const now = new Date().toLocaleTimeString();

    const fatigueScore = Number(
        data.fatigue_score ?? 0
    );

    const ear = Number(
        data.avg_ear ?? data.ear ?? 0
    );

    chart.data.labels.push(now);

    chart.data.datasets[0].data.push(
        Math.max(0, Math.min(1, fatigueScore))
    );

    chart.data.datasets[1].data.push(
        Math.max(0, Math.min(1, ear))
    );

    if (chart.data.labels.length > maxChartPoints) {

        chart.data.labels.shift();

        chart.data.datasets.forEach(dataset => {
            dataset.data.shift();
        });
    }

    chart.update('none');
}


// ============================================================
// SAFE DOM UPDATE HELPERS
// ============================================================

function setText(id, value) {

    const element = document.getElementById(id);

    if (element && value !== undefined && value !== null) {
        element.innerText = value;
    }
}


function setHTML(id, value) {

    const element = document.getElementById(id);

    if (element && value !== undefined && value !== null) {
        element.innerHTML = value;
    }
}


function setClass(id, className) {

    const element = document.getElementById(id);

    if (element) {
        element.className = className;
    }
}


function setWidth(id, percentage) {

    const element = document.getElementById(id);

    if (element) {
        const safeValue = Math.max(
            0,
            Math.min(100, Number(percentage) || 0)
        );

        element.style.width = `${safeValue}%`;
    }
}


function formatNumber(value, decimals = 2) {

    const number = Number(value);

    if (!Number.isFinite(number)) {
        return '0.00';
    }

    return number.toFixed(decimals);
}


function getFatigueLabel(data) {

    if (data.state_name) {
        return String(data.state_name).toUpperCase();
    }

    if (data.status) {
        return String(data.status).toUpperCase();
    }

    return 'INITIALIZING';
}


function normalizeTelemetry(rawData) {

    const data = rawData || {};

    return {
        ...data,

        state_name:
            data.state_name ??
            data.status ??
            'INITIALIZING',

        status:
            data.status ??
            data.state_name ??
            'INITIALIZING',

        avg_ear:
            Number(data.avg_ear ?? data.ear ?? 0),

        ear:
            Number(data.ear ?? data.avg_ear ?? 0),

        mar:
            Number(data.mar ?? 0),

        fatigue_score:
            Number(data.fatigue_score ?? 0),

        is_eye_closed:
            Boolean(data.is_eye_closed ?? data.eye_closed ?? false),

        eye_closed:
            Boolean(data.eye_closed ?? data.is_eye_closed ?? false),

        is_mouth_open:
            Boolean(data.is_mouth_open ?? false),

        yawn_confirmed:
            Boolean(data.yawn_confirmed ?? data.yawn_detected ?? false),

        yawn_detected:
            Boolean(data.yawn_detected ?? data.yawn_confirmed ?? false),

        sleepy_confirmed:
            Boolean(data.sleepy_confirmed ?? false),

        mobile_detected:
            Boolean(data.mobile_detected ?? data.phone_detected ?? false),

        phone_detected:
            Boolean(data.phone_detected ?? data.mobile_detected ?? false),

        alarm_triggered:
            Boolean(data.alarm_triggered ?? false),

        face_detected:
            Boolean(data.face_detected ?? false),

        fps:
            Number(data.fps ?? 0),

        closure_duration:
            Number(data.closure_duration ?? 0),

        yawn_duration:
            Number(data.yawn_duration ?? 0),

        mobile_confidence:
            Number(data.mobile_confidence ?? 0),

        blink_rate:
            Number(data.blink_rate ?? 0),

        yawn_rate:
            Number(data.yawn_rate ?? 0)
    };
}


// ============================================================
// TELEMETRY POLLING
// ============================================================

let telemetryRequestInProgress = false;

function startTelemetryLoop() {

    fetchTelemetry();

    setInterval(() => {
        fetchTelemetry();
    }, 500);
}


async function fetchTelemetry() {

    if (telemetryRequestInProgress) {
        return;
    }

    telemetryRequestInProgress = true;

    try {

        const response = await fetch(
            `/api/telemetry?t=${Date.now()}`,
            {
                method: 'GET',
                cache: 'no-store'
            }
        );

        if (!response.ok) {
            throw new Error(
                `Telemetry HTTP error: ${response.status}`
            );
        }

        const rawData = await response.json();

        const data = normalizeTelemetry(rawData);

        updateDashboard(data);
        updateChart(data);
        handleAlarmSystem(data);

        hostedBrowserCamera = Boolean(
            data.hosted_browser_camera
        );

        updateSourceStatus(data);

    } catch (error) {

        console.error(
            '[Telemetry] Failed to fetch telemetry:',
            error
        );

        setText(
            'system-status',
            'CONNECTION ERROR'
        );

    } finally {

        telemetryRequestInProgress = false;
    }
}


// ============================================================
// DASHBOARD METRICS
// ============================================================

function updateDashboard(data) {

    if (!data) {
        return;
    }

    // EAR
    setText(
        'ear-value',
        formatNumber(data.avg_ear, 3)
    );

    setText(
        'avg-ear',
        formatNumber(data.avg_ear, 3)
    );

    // MAR
    setText(
        'mar-value',
        formatNumber(data.mar, 3)
    );

    setText(
        'mar',
        formatNumber(data.mar, 3)
    );

    // Fatigue score
    const fatigue = Math.max(
        0,
        Math.min(1, Number(data.fatigue_score) || 0)
    );

    setText(
        'fatigue-score',
        `${(fatigue * 100).toFixed(1)}%`
    );

    setText(
        'fatigue-value',
        `${(fatigue * 100).toFixed(1)}%`
    );

    setWidth(
        'fatigue-progress',
        fatigue * 100
    );

    // FPS
    setText(
        'fps-value',
        formatNumber(data.fps, 1)
    );

    setText(
        'fps',
        formatNumber(data.fps, 1)
    );

    // Eye closure
    setText(
        'closure-duration',
        `${formatNumber(data.closure_duration, 2)} s`
    );

    // Yawn duration
    setText(
        'yawn-duration',
        `${formatNumber(data.yawn_duration, 2)} s`
    );

    // Blink rate
    setText(
        'blink-rate',
        formatNumber(data.blink_rate, 1)
    );

    // Yawn rate
    setText(
        'yawn-rate',
        formatNumber(data.yawn_rate, 1)
    );

    // Phone confidence
    setText(
        'phone-confidence',
        `${(data.mobile_confidence * 100).toFixed(1)}%`
    );

    // Face status
    setText(
        'face-status',
        data.face_detected
            ? 'FACE DETECTED'
            : 'NO FACE DETECTED'
    );

    // Eye status
    setText(
        'eye-status',
        data.is_eye_closed
            ? 'EYES CLOSED'
            : 'EYES OPEN'
    );

    // Yawn status
    setText(
        'yawn-status',
        data.yawn_confirmed
            ? 'YAWN DETECTED'
            : 'NORMAL'
    );

    // Phone status
    setText(
        'phone-status',
        data.mobile_detected
            ? 'PHONE DETECTED'
            : 'NO PHONE'
    );

    // Main state
    const state = getFatigueLabel(data);

    setText(
        'state-name',
        state
    );

    setText(
        'fatigue-state',
        state
    );

    setText(
        'system-status',
        data.status || state
    );

    // Alert message
    if (data.alert_message) {
        setText(
            'alert-message',
            data.alert_message
        );
    }

    updateDetectionIndicators(data);
}


function updateDetectionIndicators(data) {

    const faceIndicator = document.getElementById(
        'face-indicator'
    );

    if (faceIndicator) {
        faceIndicator.classList.toggle(
            'active',
            Boolean(data.face_detected)
        );
    }

    const eyeIndicator = document.getElementById(
        'eye-indicator'
    );

    if (eyeIndicator) {
        eyeIndicator.classList.toggle(
            'warning',
            Boolean(data.is_eye_closed)
        );
    }

    const phoneIndicator = document.getElementById(
        'phone-indicator'
    );

    if (phoneIndicator) {
        phoneIndicator.classList.toggle(
            'warning',
            Boolean(data.mobile_detected)
        );
    }

    const yawnIndicator = document.getElementById(
        'yawn-indicator'
    );

    if (yawnIndicator) {
        yawnIndicator.classList.toggle(
            'warning',
            Boolean(data.yawn_confirmed)
        );
    }
}


// ============================================================
// CAMERA SOURCE STATUS
// ============================================================

function updateSourceStatus(data) {

    const source = data.source || 'unknown';

    setText(
        'camera-source',
        source.toUpperCase()
    );

    setText(
        'source-label',
        source.toUpperCase()
    );

    const badge = document.getElementById(
        'source-badge'
    );

    if (badge) {
        badge.innerText = source.toUpperCase();

        badge.classList.toggle(
            'source-live',
            source === 'local' || source === 'browser'
        );

        badge.classList.toggle(
            'source-simulation',
            source === 'simulation'
        );
    }

    const videoFeed = document.getElementById(
        'video-stream'
    );

    if (videoFeed && !hostedBrowserCamera && !videoFeed.src) {
        videoFeed.src = `/video_feed?t=${Date.now()}`;
    }

    if (hostedBrowserCamera && !browserCameraStream) {
        startBrowserCamera();
    }

    if (!hostedBrowserCamera && browserCameraStream) {
        stopBrowserCamera();
    }
}
// ============================================================
// SIMULATION SCENARIO CONTROL
// ============================================================

async function setScenario(scenario, btnElement) {

    stopBrowserCamera();

    const buttons = document.querySelectorAll('.btn-scenario');

    buttons.forEach(button => button.classList.remove('active'));

    if (btnElement) {
        btnElement.classList.add('active');
    }

    try {
        const response = await fetch('/api/set_scenario', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ scenario })
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(data.error || 'Could not change simulation scenario.');
        }

        const videoImage = document.getElementById('video-stream');

        if (videoImage) {
            videoImage.src = `/video_feed?t=${Date.now()}`;
        }

    } catch (error) {
        console.error('[Scenario] Failed:', error);
        alert(error.message);
    }
}


// ============================================================
// CAMERA SOURCE CONTROL
// ============================================================

async function toggleFeedSource() {

    const buttonText = document.getElementById('source-btn-text');
    const previousText = buttonText ? buttonText.innerText : '';

    if (buttonText) {
        buttonText.innerText = 'Switching feed...';
    }

    try {
        const response = await fetch('/api/toggle_source', {
            method: 'POST'
        });

        const data = await response.json();

        if (!response.ok || data.success === false) {
            throw new Error(data.error || 'Could not switch camera source.');
        }

        hostedBrowserCamera = Boolean(
            data.browser_camera_required ||
            data.hosted_browser_camera
        );

        const videoImage = document.getElementById('video-stream');

        if (videoImage) {
            videoImage.src = `/video_feed?t=${Date.now()}`;
        }

        if (data.use_simulation) {

            stopBrowserCamera();

            if (buttonText) {
                buttonText.innerText = 'Switch to Live Webcam';
            }

        } else if (hostedBrowserCamera) {

            if (buttonText) {
                buttonText.innerText = 'Starting browser webcam...';
            }

            await startBrowserCamera();

            if (buttonText) {
                buttonText.innerText = browserCameraStream
                    ? 'Switch to Simulated Feed'
                    : 'Switch to Live Webcam';
            }

        } else {

            stopBrowserCamera();

            if (buttonText) {
                buttonText.innerText = 'Switch to Simulated Feed';
            }
        }

    } catch (error) {
        console.error('[Camera] Source switch failed:', error);

        if (buttonText) {
            buttonText.innerText = previousText;
        }

        alert(error.message || 'Failed to switch camera source.');
    }
}


// ============================================================
// SESSION SUMMARY
// ============================================================

async function fetchSummary() {

    try {
        const response = await fetch('/api/summary', {
            cache: 'no-store'
        });

        if (!response.ok) {
            return;
        }

        const data = await response.json();

        const totalFrames = Number(
            data.total_frames ?? data.total_frames_logged ?? 0
        );

        const duration = Number(data.duration_seconds ?? 0);
        const sizeKb = Number(data.log_file_size_kb ?? 0);

        setText('summary-frames', totalFrames.toLocaleString());
        setText('summary-duration', `${duration.toFixed(1)}s`);
        setText('summary-size', `${sizeKb.toFixed(1)} KB`);

    } catch (error) {
        console.error('[Summary] Fetch error:', error);
    }
}


// ============================================================
// RECENT CSV LOGS
// ============================================================

async function fetchLogs() {

    try {
        const response = await fetch('/api/logs', {
            cache: 'no-store'
        });

        if (!response.ok) {
            return;
        }

        const data = await response.json();

        const tbody = document.getElementById('logs-table-body');

        if (!tbody) {
            return;
        }

        const rows = Array.isArray(data)
            ? data
            : (data.recent_rows || []);

        if (rows.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center">
                        No telemetry entries yet.
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = rows.slice(-15).reverse().map(row => {

            const ear = Number(row.ear ?? row.avg_ear ?? 0);
            const mar = Number(row.mar ?? 0);
            const blinkRate = Number(row.blink_rate ?? 0);
            const closure = Number(row.closure_duration ?? 0);
            const fatigue = Number(row.fatigue_score ?? 0);

            return `
                <tr>
                    <td>${row.iso_time ? String(row.iso_time).split(' ')[1] : ''}</td>
                    <td>#${row.frame_id ?? ''}</td>
                    <td>${ear.toFixed(3)}</td>
                    <td>${mar.toFixed(3)}</td>
                    <td>${blinkRate.toFixed(1)}</td>
                    <td>${closure.toFixed(1)}s</td>
                    <td><strong>${fatigue.toFixed(2)}</strong></td>
                    <td>
                        <span class="pill-state state-${row.state_code ?? 0}">
                            ${row.state_name ?? 'UNKNOWN'}
                        </span>
                    </td>
                </tr>
            `;
        }).join('');

    } catch (error) {
        console.error('[Logs] Fetch error:', error);
    }
}


let logsRequestInProgress = false;

function startLogsLoop() {

    fetchLogs();
    fetchSummary();

    setInterval(async () => {

        if (logsRequestInProgress) {
            return;
        }

        logsRequestInProgress = true;

        try {
            await fetchLogs();
            await fetchSummary();
        } finally {
            logsRequestInProgress = false;
        }

    }, 5000);
}


// ============================================================
// EXPORT ACTIVE CSV LOG
// ============================================================

async function exportActiveLog() {

    try {
        const response = await fetch('/api/logs', {
            cache: 'no-store'
        });

        if (!response.ok) {
            throw new Error('Could not retrieve log information.');
        }

        const data = await response.json();

        if (data.current_log) {
            window.location.href =
                `/api/download_log/${encodeURIComponent(data.current_log)}`;
        } else {
            alert('No active log file is available yet.');
        }

    } catch (error) {
        console.error('[Export] Failed:', error);
        alert('Failed to export log file.');
    }
}


// ============================================================
// MODEL RETRAINING
// ============================================================

async function retrainModel() {

    const confirmed = confirm(
        'Retrain the Random Forest model using telemetry data?'
    );

    if (!confirmed) {
        return;
    }

    try {
        const response = await fetch('/api/retrain', {
            method: 'POST'
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(data.error || 'Model retraining failed.');
        }

        alert(data.message || 'Model retraining request completed.');

    } catch (error) {
        console.error('[Retraining] Failed:', error);
        alert(error.message || 'Retraining failed.');
    }
}


// ============================================================
// END OF RIDERVIGILANCE DASHBOARD SCRIPT
// ============================================================
