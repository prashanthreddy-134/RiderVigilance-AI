// ============================================================
// RIDERVIGILANCE AI - COMPLETE DASHBOARD JAVASCRIPT
// Browser camera, telemetry, charts, alarms, logs and controls
// ============================================================

let chart = null;
const maxChartPoints = 40;

// Audio state
let audioCtx = null;
let alarmOscillator = null;
let alarmGainNode = null;
let alarmSirenInterval = null;
let isAlarmSoundPlaying = false;
let isAudioMuted = false;

// Browser webcam state
let browserCameraStream = null;
let browserCameraVideo = null;
let browserCameraCanvas = null;
let browserCameraTimer = null;
let browserCameraStarting = false;
let browserFrameInFlight = false;
let hostedBrowserCamera = false;

const BROWSER_FRAME_INTERVAL_MS = 200;
const BROWSER_FRAME_WIDTH = 640;

// Polling state
let telemetryRequestInProgress = false;
let telemetryIntervalId = null;
let logsRequestInProgress = false;
let logsIntervalId = null;

// ============================================================
// BROWSER WEBCAM
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
                'Webcam access requires HTTPS and a supported browser.'
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

        browserCameraTimer = window.setInterval(
            sendBrowserCameraFrame,
            BROWSER_FRAME_INTERVAL_MS
        );

        console.log('[BrowserCamera] Webcam started.');

    } catch (error) {
        console.error('[BrowserCamera] Webcam error:', error);

        stopBrowserCamera();

        alert(
            `Could not access your webcam: ${error.message}\n` +
            'Allow camera permission and use HTTPS.'
        );

        // Restore simulation mode through the backend.
        try {
            const response = await fetch('/api/toggle_source', {
                method: 'POST',
                cache: 'no-store'
            });

            if (!response.ok) {
                console.error(
                    '[BrowserCamera] Could not restore simulation mode.'
                );
            }
        } catch (restoreError) {
            console.error(
                '[BrowserCamera] Restore error:',
                restoreError
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
        browserCameraVideo.readyState <
            HTMLMediaElement.HAVE_CURRENT_DATA ||
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

    if (!context) {
        return;
    }

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
                body: blob,
                cache: 'no-store'
            });

            if (!response.ok) {
                const message = await response.text();

                console.error(
                    '[BrowserCamera] Frame rejected:',
                    response.status,
                    message.slice(0, 200)
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
    if (browserCameraTimer !== null) {
        clearInterval(browserCameraTimer);
        browserCameraTimer = null;
    }

    if (browserCameraStream) {
        browserCameraStream.getTracks().forEach(track => {
            track.stop();
        });

        browserCameraStream = null;
    }

    if (browserCameraVideo) {
        browserCameraVideo.pause();
        browserCameraVideo.srcObject = null;
    }

    browserFrameInFlight = false;

    console.log('[BrowserCamera] Webcam stopped.');
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

    unlockEvents.forEach(eventName => {
        window.addEventListener(
            eventName,
            unlockAudioContext,
            {
                passive: true
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
        const button = document.getElementById('enable-audio-btn');

        if (btnText && audioCtx.state === 'running') {
            btnText.innerText = 'Alarm Sound Active';
        }

        if (button && audioCtx.state === 'running') {
            button.classList.add('btn-success');
        }

    } catch (error) {
        console.error('[AudioAlarm] Audio unlock error:', error);
    }
}

function enableAudioContext() {
    unlockAudioContext();
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
            0.35,
            audioCtx.currentTime
        );

        alarmOscillator.connect(alarmGainNode);
        alarmGainNode.connect(audioCtx.destination);

        alarmOscillator.frequency.setValueAtTime(
            950,
            audioCtx.currentTime
        );

        alarmOscillator.start();
        isAlarmSoundPlaying = true;

        let highTone = true;

        alarmSirenInterval = window.setInterval(() => {
            if (
                !isAlarmSoundPlaying ||
                !alarmOscillator ||
                !audioCtx
            ) {
                return;
            }

            const now = audioCtx.currentTime;

            alarmOscillator.frequency.setValueAtTime(
                highTone ? 950 : 620,
                now
            );

            highTone = !highTone;
        }, 200);

        console.log('[AudioAlarm] Alarm started.');

    } catch (error) {
        console.error('[AudioAlarm] Alarm start error:', error);
    }
}

function stopContinuousAlarm() {
    if (alarmSirenInterval !== null) {
        clearInterval(alarmSirenInterval);
        alarmSirenInterval = null;
    }

    if (alarmOscillator) {
        try {
            if (alarmGainNode && audioCtx) {
                alarmGainNode.gain.setValueAtTime(
                    0.001,
                    audioCtx.currentTime
                );
            }

            alarmOscillator.stop();
            alarmOscillator.disconnect();

        } catch (error) {
            // Oscillator may already be stopped.
        }

        alarmOscillator = null;
        alarmGainNode = null;
    }

    isAlarmSoundPlaying = false;
}

function handleAlarmSystem(data) {
    const banner = document.getElementById('alarm-banner');
    const bannerText = document.getElementById('alarm-banner-text');

    const isAlertState = Boolean(
        data.alarm_triggered ||
        data.mobile_detected ||
        data.sleepy_confirmed ||
        Number(data.state_code) >= 2 ||
        Number(data.closure_duration) >= 2.0 ||
        data.yawn_confirmed
    );

    if (isAlertState) {
        if (banner) {
            banner.classList.remove('hidden');
        }

        if (bannerText && data.alert_message) {
            bannerText.innerText = data.alert_message;
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

    const button = document.getElementById('mute-alarm-btn');

    if (isAudioMuted) {
        stopContinuousAlarm();

        if (button) {
            button.innerHTML =
                '<i class="fa-solid fa-volume-high"></i> Unmute';
        }

    } else {
        if (button) {
            button.innerHTML =
                '<i class="fa-solid fa-volume-xmark"></i> Mute';
        }
    }
}

// ============================================================
// CHART INITIALIZATION
// ============================================================

function initChart() {
    const canvas = document.getElementById('telemetryChart');

    if (!canvas) {
        console.warn('[Chart] telemetryChart canvas not found.');
        return;
    }

    if (typeof Chart === 'undefined') {
        console.warn('[Chart] Chart.js is not loaded.');
        return;
    }

    const context = canvas.getContext('2d');

    if (!context) {
        console.warn('[Chart] Canvas context unavailable.');
        return;
    }

    chart = new Chart(context, {
        type: 'line',

        data: {
            labels: [],
            datasets: [
                {
                    label: 'EAR',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.08)',
                    borderWidth: 2,
                    tension: 0.35,
                    fill: false,
                    pointRadius: 0
                },
                {
                    label: 'MAR',
                    data: [],
                    borderColor: '#a855f7',
                    backgroundColor: 'rgba(168, 85, 247, 0.08)',
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

    const ear = Number(data.avg_ear ?? data.ear ?? 0);
    const mar = Number(data.mar ?? 0);

    chart.data.labels.push(now);

    chart.data.datasets[0].data.push(
        Math.max(0, Math.min(1, ear))
    );

    chart.data.datasets[1].data.push(
        Math.max(0, Math.min(1, mar))
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
// DOM HELPERS
// ============================================================

function setText(id, value) {
    const element = document.getElementById(id);

    if (
        element &&
        value !== undefined &&
        value !== null
    ) {
        element.innerText = value;
    }
}

function setHTML(id, value) {
    const element = document.getElementById(id);

    if (
        element &&
        value !== undefined &&
        value !== null
    ) {
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
        return (0).toFixed(decimals);
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

// ============================================================
// TELEMETRY NORMALIZATION
// ============================================================

function normalizeTelemetry(rawData) {
    const data = rawData || {};

    return {
        ...data,

        state_name: data.state_name ??
            data.status ?? 'INITIALIZING',

        status: data.status ??
            data.state_name ?? 'INITIALIZING',

        avg_ear: Number(data.avg_ear ?? data.ear ?? 0),
        ear: Number(data.ear ?? data.avg_ear ?? 0),
        mar: Number(data.mar ?? 0),

        fatigue_score: Number(data.fatigue_score ?? 0),

        is_eye_closed: Boolean(
            data.is_eye_closed ?? data.eye_closed ?? false
        ),

        eye_closed: Boolean(
            data.eye_closed ?? data.is_eye_closed ?? false
        ),

        is_mouth_open: Boolean(data.is_mouth_open ?? false),

        yawn_confirmed: Boolean(
            data.yawn_confirmed ?? data.yawn_detected ?? false
        ),

        yawn_detected: Boolean(
            data.yawn_detected ?? data.yawn_confirmed ?? false
        ),

        sleepy_confirmed: Boolean(data.sleepy_confirmed ?? false),

        mobile_detected: Boolean(
            data.mobile_detected ?? data.phone_detected ?? false
        ),

        phone_detected: Boolean(
            data.phone_detected ?? data.mobile_detected ?? false
        ),

        alarm_triggered: Boolean(data.alarm_triggered ?? false),
        face_detected: Boolean(data.face_detected ?? false),

        fps: Number(data.fps ?? 0),
        closure_duration: Number(data.closure_duration ?? 0),
        yawn_duration: Number(data.yawn_duration ?? 0),
        mobile_confidence: Number(data.mobile_confidence ?? 0),
        blink_rate: Number(data.blink_rate ?? 0),
        yawn_rate: Number(data.yawn_rate ?? 0)
    };
}

// ============================================================
// TELEMETRY POLLING
// ============================================================

function startTelemetryLoop() {
    if (telemetryIntervalId !== null) {
        return;
    }

    fetchTelemetry();

    telemetryIntervalId = window.setInterval(
        fetchTelemetry,
        1000
    );
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
                cache: 'no-store',
                headers: {
                    'Accept': 'application/json'
                }
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
        console.error('[Telemetry] Fetch failed:', error);
        setText('system-status-text', 'CONNECTION ERROR');

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

    const ear = Number(data.avg_ear ?? data.ear ?? 0);
    const mar = Number(data.mar ?? 0);

    const fatigue = Math.max(
        0,
        Math.min(1, Number(data.fatigue_score ?? 0))
    );

    const state = getFatigueLabel(data);

    const leftEar = Number(
        data.left_ear ?? data.ear_left ?? ear
    );

    const rightEar = Number(
        data.right_ear ?? data.ear_right ?? ear
    );

    const confidence = Number(
        data.confidence ??
        data.classification_confidence ?? 0
    );

    setText('ear-val', formatNumber(ear, 3));

    setText(
        'ear-l-r',
        `L: ${formatNumber(leftEar, 3)} | R: ${formatNumber(rightEar, 3)}`
    );

    setText(
        'ear-thresh',
        `Thresh: < 0.26 | Closed: ${Boolean(data.is_eye_closed ?? data.eye_closed)}`
    );

    setText(
        'closure-duration-subtext',
        `Closure Duration: ${formatNumber(data.closure_duration, 1)}s`
    );

    setText('mar-val', formatNumber(mar, 3));

    setText(
        'yawn-thresh',
        `Thresh: > 0.50 | Open: ${Boolean(data.is_mouth_open)}`
    );

    setText(
        'yawn-duration-subtext',
        `Yawn Duration: ${formatNumber(data.yawn_duration, 1)}s`
    );

    setText(
        'yawn-confirmed-subtext',
        `Yawn Confirmed: ${Boolean(data.yawn_confirmed ?? data.yawn_detected)}`
    );

    setText('fatigue-score-val', fatigue.toFixed(2));
    setWidth('fatigue-score-bar', fatigue * 100);

    setText(
        'confidence-val',
        confidence > 0
            ? `${(confidence <= 1 ? confidence * 100 : confidence).toFixed(0)}%`
            : '—'
    );

    setText(
        'mobile-status-val',
        data.mobile_detected ? 'PHONE DETECTED' : 'NO MOBILE'
    );

    setText(
        'mobile-conf-subtext',
        `Confidence: ${(Number(data.mobile_confidence ?? 0) * 100).toFixed(1)}%`
    );

    setText(
        'mobile-alert-subtext',
        data.mobile_detected
            ? 'Potential mobile use detected'
            : 'YOLOv8 Active'
    );

    setText(
        'blink-val',
        `${formatNumber(data.blink_rate, 1)} /min`
    );

    setText(
        'alarm-state-subtext',
        `Alarm Triggered: ${Boolean(data.alarm_triggered)}`
    );

    setText(
        'closure-val',
        `${formatNumber(data.closure_duration, 1)} sec`
    );

    setText(
        'short-closures-subtext',
        `Short Closures (1-2s): ${Number(data.short_closures_count ?? data.short_closures ?? 0)} / 3`
    );

    setText('current-state-title', state);
    setText('state-text', state);

    setText(
        'current-state-desc',
        data.state_description ||
        data.alert_message ||
        (
            data.face_detected
                ? 'Facial landmarks are being monitored.'
                : 'Position your face clearly in the camera.'
        )
    );

    setText(
        'system-status-text',
        data.face_detected
            ? 'SYSTEM ACTIVE'
            : 'WAITING FOR FACE'
    );

    const stateDot = document.getElementById('state-dot');

    if (stateDot) {
        stateDot.classList.toggle(
            'warning',
            Number(data.state_code) >= 2 ||
            Boolean(data.alarm_triggered)
        );
    }

    const stateBox = document.getElementById('state-display-box');

    if (stateBox) {
        stateBox.dataset.state = String(
            data.state_name ?? data.status ?? 'unknown'
        ).toLowerCase();
    }

    const mobileCard = document.getElementById('mobile-card');

    if (mobileCard) {
        mobileCard.classList.toggle(
            'warning',
            Boolean(data.mobile_detected)
        );
    }

    const bannerText = document.getElementById('alarm-banner-text');

    if (bannerText && data.alert_message) {
        bannerText.innerText = data.alert_message;
    }

    updateDetectionIndicators(data);
}

function updateDetectionIndicators(data) {
    const dot = document.getElementById('state-dot');

    if (dot) {
        dot.classList.toggle(
            'warning',
            Boolean(
                data.is_eye_closed ||
                data.yawn_confirmed ||
                data.mobile_detected ||
                data.alarm_triggered
            )
        );
    }

    const icon = document.getElementById('state-icon');

    if (icon) {
        const iconClass = data.mobile_detected
            ? 'fa-solid fa-mobile-screen-button'
            : data.sleepy_confirmed
                ? 'fa-solid fa-triangle-exclamation'
                : data.yawn_confirmed
                    ? 'fa-solid fa-face-grimace'
                    : 'fa-solid fa-face-smile';

        icon.innerHTML = `<i class="${iconClass}"></i>`;
    }
}

// ============================================================
// CAMERA SOURCE STATUS
// ============================================================

function updateSourceStatus(data) {
    const source = String(
        data.source || 'unknown'
    ).toLowerCase();

    const badge = document.getElementById('feed-mode-badge');

    if (badge) {
        badge.innerText =
            source === 'browser' || source === 'local'
                ? 'LIVE CAMERA'
                : source === 'simulation'
                    ? 'SIMULATED FEED'
                    : source.toUpperCase();
    }

    const buttonText = document.getElementById('source-btn-text');
    const button = document.getElementById('toggle-source-btn');

    if (buttonText && !button?.disabled) {
        buttonText.innerText = source === 'simulation'
            ? 'Switch to Live Webcam'
            : 'Switch to Simulated Feed';
    }

    if (
        source === 'browser' &&
        hostedBrowserCamera &&
        !browserCameraStream &&
        !browserCameraStarting
    ) {
        startBrowserCamera();

    } else if (
        source !== 'browser' &&
        browserCameraStream
    ) {
        stopBrowserCamera();
    }
}

// ============================================================
// SIMULATION SCENARIO CONTROL
// ============================================================

async function setScenario(scenario, btnElement) {
    const buttons = document.querySelectorAll('.btn-scenario');

    buttons.forEach(button => {
        button.classList.remove('active');
    });

    if (btnElement) {
        btnElement.classList.add('active');
    }

    try {
        const response = await fetch('/api/set_scenario', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ scenario }),
            cache: 'no-store'
        });

        const rawBody = await response.text();
        let data = {};

        try {
            data = rawBody ? JSON.parse(rawBody) : {};
        } catch {
            throw new Error(
                `Invalid server response (HTTP ${response.status}).`
            );
        }

        if (!response.ok) {
            throw new Error(
                data.error || 'Could not change simulation scenario.'
            );
        }

        stopBrowserCamera();

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
// CAMERA SOURCE SWITCHING
// ============================================================

async function toggleFeedSource() {
    const button = document.getElementById('toggle-source-btn');
    const buttonText = document.getElementById('source-btn-text');

    if (!button || button.disabled) {
        return;
    }

    const previousText = buttonText?.innerText || '';

    button.disabled = true;

    if (buttonText) {
        buttonText.innerText = 'Switching feed...';
    }

    try {
        const response = await fetch('/api/toggle_source', {
            method: 'POST',
            headers: {
                'Accept': 'application/json'
            },
            cache: 'no-store'
        });

        const rawBody = await response.text();
        let data = {};

        try {
            data = rawBody ? JSON.parse(rawBody) : {};
        } catch {
            throw new Error(
                `Server returned invalid JSON (HTTP ${response.status}): ${rawBody.slice(0, 200)}`
            );
        }

        if (!response.ok || !data.success) {
            throw new Error(
                data.error ||
                `Source switch failed (HTTP ${response.status})`
            );
        }

        hostedBrowserCamera = Boolean(
            data.hosted_browser_camera
        );

        const videoImage = document.getElementById('video-stream');

        if (videoImage) {
            videoImage.src = `/video_feed?t=${Date.now()}`;
        }

        if (data.source === 'simulation') {
            stopBrowserCamera();

            if (buttonText) {
                buttonText.innerText = 'Switch to Live Webcam';
            }

        } else if (
            data.source === 'browser' &&
            data.browser_camera_required
        ) {
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

        console.log('[Camera] Active source:', data.source);

    } catch (error) {
        console.error('[Camera] Source switch failed:', error);

        if (buttonText) {
            buttonText.innerText = previousText;
        }

        alert(error.message || 'Failed to switch camera source.');

    } finally {
        button.disabled = false;
    }
}

// ============================================================
// SESSION SUMMARY
// ============================================================

async function fetchSummary() {
    try {
        const response = await fetch('/api/summary', {
            cache: 'no-store',
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`Summary HTTP error: ${response.status}`);
        }

        const data = await response.json();

        const totalFrames = Number(
            data.total_frames ??
            data.total_frames_logged ?? 0
        );

        const duration = Number(
            data.duration_seconds ?? 0
        );

        const sizeKb = Number(
            data.log_file_size_kb ?? 0
        );

        setText(
            'summary-frames',
            totalFrames.toLocaleString()
        );

        setText(
            'summary-duration',
            `${duration.toFixed(1)}s`
        );

        setText(
            'summary-size',
            `${sizeKb.toFixed(1)} KB`
        );

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
            cache: 'no-store',
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`Logs HTTP error: ${response.status}`);
        }

        const data = await response.json();

        const tbody = document.getElementById('logs-table-body');

        if (!tbody) {
            return;
        }

        const rows = Array.isArray(data)
            ? data
            : (data.recent_rows || []);

        if (!rows.length) {
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
                        <span class="pill-state state-${Number(row.state_code ?? 0)}">
                            ${String(row.state_name ?? 'UNKNOWN')}
                        </span>
                    </td>
                </tr>
            `;
        }).join('');

    } catch (error) {
        console.error('[Logs] Fetch error:', error);
    }
}

function startLogsLoop() {
    if (logsIntervalId !== null) {
        return;
    }

    fetchLogs();
    fetchSummary();

    logsIntervalId = window.setInterval(async () => {
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
// EXPORT CSV LOG
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
            method: 'POST',
            headers: {
                'Accept': 'application/json'
            }
        });

        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(
                data.error || 'Model retraining failed.'
            );
        }

        alert(
            data.message ||
            'Model retraining request completed.'
        );

    } catch (error) {
        console.error('[Retraining] Failed:', error);
        alert(error.message || 'Retraining failed.');
    }
}

// ============================================================
// END OF RIDERVIGILANCE AI JAVASCRIPT
// ============================================================