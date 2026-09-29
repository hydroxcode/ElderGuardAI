"""
ElderGuard AI - Edge AI-Based Elderly Fall Detection
PS-06: Navonmesh 26 in Collaboration with Qualcomm
Author: ElderGuard AI Development Team
"""
import time
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# Local core modules
from config import sys_config, fall_config
from core.database import EventDatabase
from core.pose_detector import PoseDetector, PoseResult
from core.fall_detector import FallDetector, FallAnalysisResult
from core.state_machine import EmergencyStateMachine, SystemState
from core.camera import VideoCaptureManager
from utils.visualizer import EdgeVisualizer
from utils.demo_scenarios import ScenarioGenerator
from utils.audio_alert import SoundAlertManager

# Page Configuration
st.set_page_config(
    page_title="ElderGuard AI - Edge Fall Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    /* Dark Futuristic Edge Theme */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Top Navigation Banner */
    .hero-banner {
        background: linear-gradient(135deg, #131b2e 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-tagline {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 4px;
        margin-bottom: 0;
    }

    /* System Status Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 12px;
    }
    .edge-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-green {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-blue {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-purple {
        background-color: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    
    /* Emergency Alert Banner */
    .alert-emergency {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        color: white;
        animation: pulseRed 1.5s infinite;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.6);
    }
    @keyframes pulseRed {
        0% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 35px rgba(239, 68, 68, 0.8); }
        100% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.4); }
    }

    /* Verifying Banner */
    .alert-verifying {
        background: linear-gradient(135deg, #78350f 0%, #d97706 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
        color: white;
        animation: pulseYellow 1.2s infinite;
        box-shadow: 0 0 20px rgba(245, 158, 11, 0.5);
    }
    @keyframes pulseYellow {
        0% { box-shadow: 0 0 10px rgba(245, 158, 11, 0.3); }
        50% { box-shadow: 0 0 25px rgba(245, 158, 11, 0.7); }
        100% { box-shadow: 0 0 10px rgba(245, 158, 11, 0.3); }
    }

    /* Metric Cards */
    .telemetry-card {
        background-color: #161f30;
        border: 1px solid #283548;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .telemetry-label {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .telemetry-value {
        font-size: 22px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }

    /* Privacy Banner */
    .privacy-notice {
        background: #0f172a;
        border-left: 4px solid #10b981;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 12px;
        color: #cbd5e1;
        margin-top: 10px;
    }

    /* Video Container Styling */
    [data-testid="stImage"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #334155;
        background-color: #05070a;
    }
    [data-testid="stImage"] img {
        border-radius: 8px;
        display: block;
        width: 100%;
        object-fit: cover;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State Singletons
if "db" not in st.session_state:
    st.session_state.db = EventDatabase(sys_config.DB_PATH)
    st.session_state.db.seed_demo_history_if_empty()

if "pose_detector" not in st.session_state:
    st.session_state.pose_detector = PoseDetector()

if "fall_detector" not in st.session_state:
    st.session_state.fall_detector = FallDetector(fall_config)

if "state_machine" not in st.session_state:
    st.session_state.state_machine = EmergencyStateMachine(
        st.session_state.db, sys_config, fall_config
    )

if "visualizer" not in st.session_state:
    st.session_state.visualizer = EdgeVisualizer()

if "scenario_gen" not in st.session_state:
    st.session_state.scenario_gen = ScenarioGenerator()

if "audio_mgr" not in st.session_state:
    st.session_state.audio_mgr = SoundAlertManager()

if "camera_mgr" not in st.session_state:
    st.session_state.camera_mgr = VideoCaptureManager(
        camera_index=sys_config.CAMERA_INDEX,
        width=sys_config.CAMERA_WIDTH,
        height=sys_config.CAMERA_HEIGHT
    )

if "sim_step" not in st.session_state:
    st.session_state.sim_step = 0

if "is_monitoring" not in st.session_state:
    st.session_state.is_monitoring = True

if "current_source" not in st.session_state:
    st.session_state.current_source = "Live Laptop Webcam"

# Top Header Hero
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">🛡️ ELDERGUARD AI</h1>
            <p class="hero-tagline">Privacy-Preserving Edge AI for Elderly Fall Detection | Autonomous Continuous Monitoring</p>
            <div class="badge-container">
                <span class="edge-badge badge-green">🟢 SYSTEM: ACTIVE</span>
                <span class="edge-badge badge-blue">🔒 PRIVACY: ON (LOCAL EDGE ONLY)</span>
                <span class="edge-badge badge-purple">⚡ TARGET: QUALCOMM SNAPDRAGON / QCS NPU</span>
                <span class="edge-badge badge-green">☁️ CONTINUOUS CLOUD STREAMING: OFF</span>
            </div>
        </div>
        <div style="text-align: right; margin-top: 5px;">
            <span style="font-size: 12px; color: #94a3b8;">Navonmesh 26 • Problem Statement PS-06</span><br>
            <span style="font-size: 11px; color: #64748b;">In Collaboration with Qualcomm</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration & Controls
with st.sidebar:
    st.header("🎛️ Edge Station Controls")
    
    input_source = st.selectbox(
        "Input Video Stream Source",
        options=[
            "Live Laptop Webcam",
            "Scenario 1: Person Standing (SAFE)",
            "Scenario 2: Person Walking (SAFE)",
            "Scenario 3: Person Sitting Down (SAFE)",
            "Scenario 4: Intentional Lie Down (SAFE - False Positive Check)",
            "Scenario 5: Sudden Slip & Fall (POSSIBLE FALL)",
            "Scenario 6: Fall with Recovery (RECOVERED)",
            "Scenario 7: Fall without Recovery (10s Countdown -> EMERGENCY)"
        ],
        index=0
    )

    # Detect input source switch
    if input_source != st.session_state.current_source:
        st.session_state.current_source = input_source
        st.session_state.sim_step = 0
        st.session_state.state_machine.resolve_emergency()
        st.session_state.fall_detector.reset()

    # Monitoring and Audio Toggles
    monitoring_toggle = st.toggle("🔴 Real-Time Edge Monitoring", value=st.session_state.is_monitoring)
    st.session_state.is_monitoring = monitoring_toggle

    audio_toggle = st.toggle("🔊 Acoustic Alarm Siren & Chimes", value=not st.session_state.audio_mgr.muted)
    st.session_state.audio_mgr.muted = not audio_toggle

    st.markdown("---")
    st.subheader("🔒 Privacy Shield Settings")
    privacy_mode = st.toggle("Enable Privacy Mode", value=sys_config.PRIVACY_MODE)
    anonymize_face = st.toggle("Anonymize / Blur Facial Region", value=sys_config.ANONYMIZE_FACE, disabled=not privacy_mode)
    skeleton_only = st.toggle("Blackout Silhouette (Skeleton Only)", value=sys_config.SKELETON_ONLY_VIEW)

    st.markdown("""
    <div class="privacy-notice">
        <b>Zero Surveillance Leakage:</b><br>
        Raw RGB video frames never leave the device. No cloud upload, no biometric face recognition, 100% on-device processing.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⚡ Fall Biomechanics & Debounce")
    fall_vel_thresh = st.slider("Drop Velocity Threshold (m/s)", 0.15, 0.70, float(fall_config.FALL_VELOCITY_THRESHOLD), 0.02)
    fall_angle_thresh = st.slider("Torso Fall Angle Threshold (°)", 40, 80, int(fall_config.FALL_ANGLE_THRESHOLD))
    req_susp_frames = st.slider("Debounce Consecutive Frames", 3, 12, int(fall_config.REQUIRED_SUSPICIOUS_FRAMES))
    cooldown_dur = st.slider("False Alarm Cooldown (sec)", 2.0, 10.0, float(fall_config.COOLDOWN_DURATION_SEC), 0.5)
    countdown_duration = st.slider("Verification Countdown (sec)", 5, 20, int(fall_config.COUNTDOWN_DURATION_SEC))
    
    # Update config dynamically
    fall_config.FALL_VELOCITY_THRESHOLD = float(fall_vel_thresh)
    fall_config.FALL_ANGLE_THRESHOLD = float(fall_angle_thresh)
    fall_config.REQUIRED_SUSPICIOUS_FRAMES = int(req_susp_frames)
    fall_config.COOLDOWN_DURATION_SEC = float(cooldown_dur)
    fall_config.COUNTDOWN_DURATION_SEC = int(countdown_duration)
    st.session_state.state_machine.fall_cfg.COUNTDOWN_DURATION_SEC = int(countdown_duration)
    st.session_state.state_machine.fall_cfg.COOLDOWN_DURATION_SEC = float(cooldown_dur)

    st.markdown("---")
    st.subheader("🚨 Manual Demonstration Triggers")
    col_trig1, col_trig2 = st.columns(2)
    with col_trig1:
        if st.button("Simulate Fall", width="stretch"):
            st.session_state.state_machine.current_state = SystemState.POSSIBLE_FALL
            st.session_state.state_machine.fall_detected_time = time.time()
            st.session_state.state_machine.possible_fall_frames = fall_config.POSSIBLE_FALL_CONFIRM_FRAMES
            st.session_state.state_machine.last_confidence = 94.5
            st.session_state.state_machine.cooldown_until = 0.0
            st.rerun()
    with col_trig2:
        if st.button("Reset / Safe", width="stretch"):
            st.session_state.state_machine.resolve_emergency()
            st.rerun()

# Dynamic Top Alert Banner Placeholder (Updates in-place without page reload)
alert_banner_placeholder = st.empty()

# Layout Columns: Video Stream (Left) + Telemetry & Profile (Right)
col_video, col_telemetry = st.columns([1.6, 1.0])

with col_video:
    st.subheader("📹 Live Edge Camera Feed")
    video_placeholder = st.empty()

    # Contextual Action Bar (Mounted once, always responsive)
    col_act1, col_act2, col_act3 = st.columns(3)
    with col_act1:
        btn_im_ok = st.button("✅ I'M OK", width="stretch", help="Cancel countdown / confirm safety")
    with col_act2:
        btn_sos = st.button("🚨 TRIGGER SOS", width="stretch", help="Immediately dispatch emergency distress signal")
    with col_act3:
        btn_reset = st.button("🔄 RESET / SAFE", width="stretch", help="Reset system state back to SAFE")

    if btn_im_ok:
        st.session_state.state_machine.trigger_user_im_ok()
        st.toast("Resident marked SAFE. Countdown cancelled.", icon="✅")
    if btn_sos:
        st.session_state.state_machine.current_state = SystemState.EMERGENCY_CONFIRMED
        st.session_state.state_machine.last_confidence = 98.0
        st.session_state.state_machine.state_message = "MANUAL SOS TRIGGERED"
        st.session_state.db.log_event(
            event_type="FALL_MANUAL_SOS",
            confidence=98.0,
            location=sys_config.DEFAULT_LOCATION,
            metrics_summary="Manual emergency distress signal triggered by caregiver/resident",
            alert_status="TRIGGERED"
        )
        st.toast("🚨 EMERGENCY ALARM TRIGGERED!", icon="🚨")
    if btn_reset:
        st.session_state.state_machine.resolve_emergency()
        st.toast("System status reset to SAFE.", icon="🟢")

with col_telemetry:
    st.subheader("📊 Live Telemetry & Biomechanics")
    telemetry_state_placeholder = st.empty()
    telemetry_metrics_placeholder = st.empty()

    # Resident Monitoring Profile Card (Rendered once, fixed)
    st.markdown(f"""
    <div class="telemetry-card">
        <div class="telemetry-label">RESIDENT MONITORING PROFILE</div>
        <div style="margin-top: 6px; font-size: 13px;">
            <b>Resident:</b> {sys_config.DEFAULT_RESIDENT}<br>
            <b>Assigned Room:</b> {sys_config.DEFAULT_LOCATION}<br>
            <b>Edge AI Hardware:</b> On-Device Hexagon NPU Architecture<br>
            <b>Cloud Streaming:</b> <span style="color: #4ade80;">Disabled (100% Local Inference)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Tabs below for History, Qualcomm Architecture, and Biomechanical Diagnostics
st.markdown("---")
tab_events, tab_qualcomm, tab_tech = st.tabs([
    "📜 Caregiver Event History (Edge DB)",
    "⚡ Qualcomm Edge Architecture",
    "🔬 Biomechanical Diagnostics & Verification"
])

with tab_events:
    col_hist_head, col_hist_btn = st.columns([4, 1])
    with col_hist_head:
        st.subheader("Local Event Audit Trail (Privacy-Preserving SQLite)")
    with col_hist_btn:
        if st.button("🔄 Refresh Logs"):
            st.rerun()

    events = st.session_state.db.get_recent_events(limit=20)
    if events:
        df = pd.DataFrame(events)
        df_display = df[[
            "event_id", "timestamp", "event_type", "confidence",
            "location", "response_status", "alert_status", "metrics_summary"
        ]].copy()
        
        df_display.rename(columns={
            "event_id": "Event ID",
            "timestamp": "Timestamp",
            "event_type": "Event Type",
            "confidence": "Confidence (%)",
            "location": "Location",
            "response_status": "Resident Response",
            "alert_status": "Status",
            "metrics_summary": "Biomechanical Summary"
        }, inplace=True)

        st.dataframe(df_display, width="stretch", height=280)
    else:
        st.info("No recorded events yet in local Edge database.")

    stats = st.session_state.db.get_statistics()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Events Logged", stats["total_events"])
    s2.metric("Confirmed Emergencies", stats["confirmed_emergencies"])
    s3.metric("Self-Recovered Incidents", stats["recovered_events"])
    s4.metric("False Alarms Cancelled", stats["false_alarms_cancelled"])

with tab_qualcomm:
    st.subheader("⚡ Target Qualcomm Edge AI Architecture")
    st.markdown("""
    ### Why Edge AI for Elderly Fall Detection?
    Traditional fall detection systems either force the elderly to wear uncomfortable pendant buttons (which are frequently forgotten or discarded) or stream 24/7 video into cloud servers (violating privacy and exposing sensitive domestic spaces).

    **ElderGuard AI utilizes an on-device Edge AI architecture designed for Qualcomm Snapdragon platforms:**
    """)
    
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        st.markdown("""
        #### 1. On-Device NPU Acceleration
        - **Target:** Qualcomm Hexagon NPU / Snapdragon X Elite / QCS610 IoT SoC
        - **Optimization:** INT8 post-training quantization via Qualcomm AI Hub
        - **Throughput:** 60+ FPS inference at < 2.5W thermal envelope
        """)
    with qc2:
        st.markdown("""
        #### 2. Absolute Privacy by Design
        - **Local Processing:** Raw RGB pixels terminate at on-device ISP memory
        - **Zero Cloud Upload:** 0 bytes of surveillance video ever transmitted
        - **Metadata Only:** Only encrypted event triggers (e.g. `FALL_CONFIRMED`) leave device
        """)
    with qc3:
        st.markdown("""
        #### 3. Zero-Latency & Reliability
        - **Offline Immunity:** Continues detecting falls even during internet outages
        - **Instant Verification:** Temporal kinematic verification runs at 33ms frame latency
        - **Direct SOS:** Cellular/Matter/BLE emergency beacon trigger
        """)

    st.markdown("""
    ```
    [ Fixed Smart Camera ]
             │ (Local MIPI CSI / USB)
             ▼
    ┌────────────────────────────────────────────────────────┐
    │          QUALCOMM SNAPDRAGON EDGE PLATFORM             │
    │                                                        │
    │  Qualcomm Spectra ISP  ──>  Qualcomm Hexagon NPU       │
    │    (Zero-Copy Frames)         (INT8 Pose Estimation)   │
    │                                    │                   │
    │                                    ▼                   │
    │                        Biomechanical Fall Engine       │
    │                        (Angular, Velocity, Floor)      │
    │                                    │                   │
    │                                    ▼                   │
    │                        10s Verification Countdown      │
    └────────────────────────────────────────────────────────┘
             │ (Encrypted Emergency Event Metadata ONLY)
             ▼
    [ Caregiver Dashboard / Local Acoustic Siren / Matter SOS ]
    ```
    """)

with tab_tech:
    st.subheader("🔬 Fall Kinematics & False Positive Suppression")
    st.markdown(r"""
    ElderGuard AI avoids naive "person lying down" traps by evaluating multidimensional biomechanical signals over a continuous temporal sliding window:
    """)
    
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(r"""
        **1. Torso Inclination Angle ($\theta_{\text{torso}}$):**
        - Measured from the vertical axis using the spinal vector connecting the hip midpoint to the shoulder midpoint.
        - Standing / Walking: $0^\circ \le \theta \le 35^\circ$
        - Sitting: $0^\circ \le \theta \le 42^\circ$
        - Acute Fall / Prone: $\theta > 65^\circ$

        **2. Aspect Ratio ($AR = H/W$):**
        - Standing silhouette: $AR > 1.3$
        - Fallen silhouette: $AR < 0.85$ (rapid horizontal broadening)
        """)
    with t2:
        st.markdown(r"""
        **3. Vertical Drop Velocity ($V_y$):**
        - Rate of hip downward displacement over temporal window $\Delta t$.
        - Intentional Lying Down: $V_y < 0.20$ frame_height/s (slow, controlled transition).
        - Sudden Uncontrolled Fall: $V_y > 0.35$ frame_height/s (acute downward acceleration).

        **4. Post-Impact Immobility & Recovery:**
        - Verifies whether the resident regains vertical posture within the 10-second window.
        - Automatic self-cancellation if upright posture is restored.
        """)

# Helper: Update Alert Banner In-Place
def update_alert_banner(current_state, remaining_sec, confidence):
    if current_state == SystemState.EMERGENCY_CONFIRMED:
        st.session_state.audio_mgr.play_emergency_siren()
        alert_banner_placeholder.markdown(f"""
        <div class="alert-emergency">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0; font-size: 24px; font-weight: 800;">🚨 EMERGENCY ALERT: FALL CONFIRMED</h2>
                    <p style="margin: 6px 0 0 0; font-size: 15px;">
                        <b>Resident:</b> {sys_config.DEFAULT_RESIDENT} &nbsp;|&nbsp; 
                        <b>Location:</b> {sys_config.DEFAULT_LOCATION} &nbsp;|&nbsp; 
                        <b>Time:</b> {datetime.now().strftime('%H:%M:%S')} &nbsp;|&nbsp; 
                        <b>Status:</b> No response received after countdown timeout.
                    </p>
                </div>
                <div style="font-size: 32px; font-weight: 800;">
                    {confidence:.0f}%
                    <div style="font-size: 11px; text-transform: uppercase;">Confidence</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif current_state == SystemState.VERIFYING:
        st.session_state.audio_mgr.play_warning_chime()
        alert_banner_placeholder.markdown(f"""
        <div class="alert-verifying">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0; font-size: 22px; font-weight: 800;">⚠ POSSIBLE FALL DETECTED — ARE YOU OK?</h2>
                    <p style="margin: 4px 0 0 0; font-size: 14px;">
                        Verifying safety. Resident can stand up or press <b>[ I'M OK ]</b> below to cancel.
                    </p>
                </div>
                <div style="font-size: 38px; font-weight: 900; background: rgba(0,0,0,0.3); padding: 4px 16px; border-radius: 8px;">
                    {remaining_sec}s
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        alert_banner_placeholder.empty()

# Helper: Update Live Telemetry In-Place
def render_telemetry(new_state, analysis_result, pose_result):
    state_color_map = {
        SystemState.SAFE: ("#22c55e", "🟢 SAFE"),
        SystemState.POSSIBLE_FALL: ("#f59e0b", "🟡 POSSIBLE FALL"),
        SystemState.VERIFYING: ("#f59e0b", "🟠 VERIFYING (ARE YOU OK?)"),
        SystemState.RECOVERED: ("#38bdf8", "🔵 RECOVERED"),
        SystemState.EMERGENCY_CONFIRMED: ("#ef4444", "🚨 EMERGENCY CONFIRMED")
    }
    s_col, s_txt = state_color_map.get(new_state, ("#94a3b8", "UNKNOWN"))

    telemetry_state_placeholder.markdown(f"""
    <div class="telemetry-card" style="border-left: 5px solid {s_col};">
        <div class="telemetry-label">CURRENT SYSTEM STATE</div>
        <div class="telemetry-value" style="color: {s_col};">{s_txt}</div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">{st.session_state.state_machine.state_message}</div>
    </div>
    """, unsafe_allow_html=True)

    raw_act = getattr(analysis_result, "raw_activity", analysis_result.activity)
    track_score = int(pose_result.confidence * 100) if pose_result.detected else 0
    q_label = getattr(pose_result, "quality", "TRACKING")
    drop_cnt = getattr(pose_result, "dropout_frames", 0)
    drop_txt = f"{drop_cnt} held" if drop_cnt > 0 else "ACTIVE"
    drop_col = "#f59e0b" if drop_cnt > 0 else "#4ade80"
    cooldown_left = max(0.0, st.session_state.state_machine.cooldown_until - time.time())
    cd_txt = f"{cooldown_left:.1f}s" if cooldown_left > 0 else "IDLE"
    cd_col = "#38bdf8" if cooldown_left > 0 else "#4ade80"

    telemetry_metrics_placeholder.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
        <div class="telemetry-card">
            <div class="telemetry-label">FALL CONFIDENCE</div>
            <div class="telemetry-value">{analysis_result.confidence:.1f}%</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-label">STABLE ACTIVITY</div>
            <div class="telemetry-value" style="font-size: 16px; margin-top: 6px;">
                {analysis_result.activity} <span style="font-size: 11px; color:#94a3b8;">({raw_act})</span>
            </div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-label">TRACKING & QUALITY</div>
            <div class="telemetry-value" style="font-size: 16px; margin-top: 6px;">
                {track_score}% <span style="font-size: 11px; color:#94a3b8;">({q_label})</span>
            </div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-label">TORSO INCLINATION</div>
            <div class="telemetry-value">{analysis_result.torso_angle:.1f}°</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-label">DROPOUT BUFFER</div>
            <div class="telemetry-value" style="color: {drop_col}; font-size: 18px;">{drop_txt}</div>
        </div>
        <div class="telemetry-card">
            <div class="telemetry-label">FALSE-ALARM COOLDOWN</div>
            <div class="telemetry-value" style="color: {cd_col}; font-size: 18px;">{cd_txt}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# In-Place Real-Time Streaming Loop (Zero-Flicker WebSocket Architecture)
# ==============================================================================
if not st.session_state.is_monitoring:
    video_placeholder.info("⏸️ Edge Monitoring Paused. Toggle 'Real-Time Edge Monitoring' in the sidebar to resume.")
    # Release camera when paused
    if st.session_state.camera_mgr.is_running:
        st.session_state.camera_mgr.stop()
else:
    last_banner_state = None
    last_banner_sec = -1
    frame_counter = 0

    while st.session_state.is_monitoring:
        loop_start = time.time()

        # 1. Acquire Frame & Pose Detection
        if input_source == "Live Laptop Webcam":
            if not st.session_state.camera_mgr.is_running:
                st.session_state.camera_mgr.start()
            success, raw_frame = st.session_state.camera_mgr.read_frame()
            if raw_frame is not None:
                pose_result = st.session_state.pose_detector.process_frame(raw_frame)
            else:
                raw_frame = np.zeros((sys_config.CAMERA_HEIGHT, sys_config.CAMERA_WIDTH, 3), dtype=np.uint8)
                cv2.putText(raw_frame, "WEBCAM CONNECTING...", (130, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                pose_result = PoseResult(detected=False)
        else:
            # Release live webcam hardware if running scenario
            if st.session_state.camera_mgr.is_running:
                st.session_state.camera_mgr.stop()

            scenario_key_map = {
                "Scenario 1": "1_standing",
                "Scenario 2": "2_walking",
                "Scenario 3": "3_sitting",
                "Scenario 4": "4_lying_down",
                "Scenario 5": "5_slip_fall",
                "Scenario 6": "6_fall_recovery",
                "Scenario 7": "7_fall_emergency"
            }
            matched_key = "1_standing"
            for prefix, key in scenario_key_map.items():
                if input_source.startswith(prefix):
                    matched_key = key
                    break

            st.session_state.sim_step += 1
            raw_frame, pose_result = st.session_state.scenario_gen.generate_frame(
                matched_key, st.session_state.sim_step, total_steps=90
            )

        # 2. Fall Kinematics Biomechanical Analysis
        analysis_result = st.session_state.fall_detector.analyze(pose_result)

        # 3. Emergency State Machine Transition
        new_state = st.session_state.state_machine.update(analysis_result)

        # 4. Render Frame Overlay with Privacy & Biomechanical HUD
        annotated_frame = st.session_state.visualizer.render(
            frame=raw_frame,
            pose=pose_result,
            analysis=analysis_result,
            state=new_state,
            countdown_sec=st.session_state.state_machine.countdown_seconds_remaining,
            privacy_mode=privacy_mode,
            anonymize_face=anonymize_face,
            skeleton_only=skeleton_only
        )

        # 5. Display Frame in Streamlit (In-Place WebSocket Delta, Zero Flicker)
        rgb_display = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_display, channels="RGB", output_format="JPEG", width="stretch")

        # 6. Update Alert Banner (Only when state or countdown changes)
        rem_sec = st.session_state.state_machine.countdown_seconds_remaining
        if (new_state != last_banner_state) or (rem_sec != last_banner_sec):
            update_alert_banner(new_state, rem_sec, st.session_state.state_machine.last_confidence)
            last_banner_state = new_state
            last_banner_sec = rem_sec

        # 7. Update Telemetry Cards (Every 2 frames for silky smooth metrics without DOM churn)
        frame_counter += 1
        if frame_counter % 2 == 0:
            render_telemetry(new_state, analysis_result, pose_result)

        # 8. Adaptive FPS Pacing (~30 FPS target)
        elapsed = time.time() - loop_start
        sleep_time = max(0.005, 0.033 - elapsed)
        time.sleep(sleep_time)
