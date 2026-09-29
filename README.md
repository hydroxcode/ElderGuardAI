# 🛡️ ElderGuard AI: Privacy-Preserving Edge AI for Elderly Fall Detection

> **Navonmesh 26 — National Level Hackathon**  
> **Problem Statement PS-06:** Edge AI-Based Elderly Fall Detection  
> *In Collaboration with Qualcomm*  
> **Tagline:** *"Privacy-preserving Edge AI for Elderly Fall Detection — Autonomous, Zero-Cloud, Continuous On-Device Protection."*

---

## 🌟 Executive Summary

Falls are the leading cause of fatal and non-fatal injuries among elderly individuals worldwide. According to the WHO, over 684,000 fatal falls occur globally each year, with individuals over 60 suffering the highest morbidity. Traditional solutions suffer from two fatal flaws:
1. **Wearable Pendants / Smartwatches:** Require charging, remembering to wear, and are frequently taken off before sleep, bathing, or resting—the exact times falls occur most frequently.
2. **Cloud-Connected Cameras:** Continuously stream domestic surveillance video into third-party cloud servers, posing severe privacy violations, high bandwidth costs, latency delays, and vulnerability to network outages.

**ElderGuard AI** solves this dilemma through **Zero-Cloud Edge AI**. A fixed smart camera monitors the domestic environment entirely locally on an edge device (optimized for Qualcomm Snapdragon / Hexagon NPU architecture). The system performs local pose estimation, biomechanical kinematic analysis, and temporal verification to reliably detect falls, filter false positives (such as sitting or intentional lying down), initiate an autonomous "ARE YOU OK?" verification countdown, and emit emergency caregiver alerts—**without uploading a single frame of video to the cloud**.

---

## 🚀 Key Features

* **🔒 Absolute Privacy by Design:** 
  - Zero continuous video streaming to the cloud (0 KB video egress).
  - On-device local inference: frames are analyzed in memory and immediately discarded.
  - Built-in Privacy Face Shield & Wireframe Blackout Mode (anonymizes resident identity).
* **🔬 Advanced Biomechanical Fall Kinematics:**
  - Evaluates multi-axial torso angle ($\theta_{\text{torso}}$), bounding box aspect ratio collapse ($AR = H/W$), instantaneous and peak downward drop velocities ($V_y$), and floor proximity ($Y_{\text{floor}}$).
  - True temporal sliding window analysis (not naive "person lying down" detection).
* **🛡️ False-Positive Reduction:**
  - Intelligently differentiates:
    * **Standing & Walking:** Upright spinal alignment ($<35^\circ$).
    * **Controlled Sitting:** Stable vertical torso with mid-height hip seating.
    * **Bending Forward:** High hip center, controlled descent.
    * **Slow Intentional Lying Down:** Gradual descent velocity below critical impact threshold.
    * **Sudden Slip & Fall:** Acute downward drop velocity spike ($>0.35$ m/s) + horizontal spinal tilt ($>65^\circ$) + floor proximity impact.
* **⏱️ Autonomous 10-Second Verification & Self-Recovery:**
  - When a fall is suspected, ElderGuard triggers an audible and visual **"ARE YOU OK?"** 10-second countdown.
  - **Self-Recovery Detection:** If the resident stands back up during the countdown, the system detects upright recovery and automatically cancels the emergency alert.
  - **Tactile Resident Override:** Big `[ I'M OK ]` button cancels false alarms instantly.
* **🚨 Rapid Emergency Alert Dispatch:**
  - If the countdown reaches 0 without response, the system confirms an emergency, initiates local acoustic sirens, logs the incident with complete telemetry into a local SQLite database, and alerts the caregiver dashboard.
* **📊 Professional Caregiver Dashboard:**
  - Real-time video view with pose overlay, privacy masking, live telemetry meters (fall confidence gauge, torso inclination, drop velocity, activity classifier).
  - SQLite local audit trail with incident history, event duration, recovery status, and system statistics.
* **🎭 7-Scenario Interactive Demo Engine:**
  - Built-in simulation generator to guarantee 100% reliable hackathon presentations even in low-light or constrained presentation booths.

---

## 🏗️ System Architecture

```
                            ROOM CAMERA (Fixed Sensor)
                                       │ (Local Video Stream)
                                       ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │                       ELDERGUARD EDGE DEVICE                           │
    │                                                                        │
    │   ┌────────────────────────────────────────────────────────────────┐   │
    │   │  Local Frame Capture & Privacy Buffer                          │   │
    │   │  (Raw RGB kept in ephemeral memory only; NEVER uploaded)       │   │
    │   └──────────────────────────────┬─────────────────────────────────┘   │
    │                                  ▼                                     │
    │   ┌────────────────────────────────────────────────────────────────┐   │
    │   │  Edge Pose Estimation Engine (MediaPipe / OpenCV DNN)          │   │
    │   │  • 33 Keypoint Landmark Coordinates                            │   │
    │   │  • Hip Center & Shoulder Center Vector Extraction              │   │
    │   └──────────────────────────────┬─────────────────────────────────┘   │
    │                                  ▼                                     │
    │   ┌────────────────────────────────────────────────────────────────┐   │
    │   │  Biomechanical Kinematic & Temporal Fall Engine                │   │
    │   │  • Torso Inclination Angle: θ = atan2(|dx|, |dy|)               │   │
    │   │  • Aspect Ratio: AR = Height / Width                           │   │
    │   │  • Peak Downward Velocity: Vy = ΔY_hip / Δt                    │   │
    │   │  • False Positive Filter: Intentional Lie Down vs Acute Fall   │   │
    │   └──────────────────────────────┬─────────────────────────────────┘   │
    │                                  ▼                                     │
    │   ┌────────────────────────────────────────────────────────────────┐   │
    │   │  State Machine Verification Pipeline                           │   │
    │   │                                                                │   │
    │   │      [ SAFE ] ──(Sudden Drop)──> [ POSSIBLE FALL ]             │   │
    │   │                                          │                     │   │
    │   │                                          ▼                     │   │
    │   │                                   [ VERIFYING ]                │   │
    │   │                               ("ARE YOU OK? [10s]")            │   │
    │   │                                ┌─────────┴─────────┐           │   │
    │   │                                ▼                   ▼           │   │
    │   │                     (Upright / [I'M OK])    (Timeout / 0s)     │   │
    │   │                                │                   │           │   │
    │   │                                ▼                   ▼           │   │
    │   │                          [ RECOVERED ]   [ EMERGENCY CONFIRMED ]│
    │   │                         (Cancel Alert)     (Siren + Dispatch)  │   │
    │   └────────────────────────────────────────────────────────────────┘   │
    │                                  │                                     │
    │                                  ▼                                     │
    │   ┌────────────────────────────────────────────────────────────────┐   │
    │   │  Local SQLite Audit Trail (elderguard.db)                      │   │
    │   │  • Encrypted event metadata: ID, Timestamp, Confidence, Status │   │
    │   └────────────────────────────────────────────────────────────────┘   │
    └──────────────────────────────────┬─────────────────────────────────────┘
                                       │ (Zero video; Event Metadata ONLY)
                                       ▼
                       CAREGIVER DASHBOARD & LOCAL ALERT
```

---

## ⚡ Qualcomm / Snapdragon Edge AI Connection

ElderGuard AI is designed specifically around Qualcomm's Edge Computing vision:

### 1. Current Prototype vs Target Edge Production
* **Current Hackathon Prototype:** Demonstrates the end-to-end edge pipeline running locally on a standard laptop with built-in webcam. Inference, temporal kinematics, verification, and SQLite database run 100% on-device.
* **Target Qualcomm Production Hardware:**
  - **SoC:** Qualcomm Snapdragon X Elite, Snapdragon Neural Processing IoT Platforms (e.g., QCS610 / QCS6490 / RB5 Robotics Platform).
  - **NPU Acceleration:** Qualcomm Hexagon NPU using the Qualcomm AI Engine Direct (QNN) SDK.
  - **ISP Integration:** Qualcomm Spectra ISP with hardware-accelerated direct memory access (zero-copy RGB feeding directly into NPU).
  - **Thermal & Power Envelope:** 24/7 continuous smart-room deployment running at `< 2.5W` average power dissipation.

### 2. Edge AI vs Cloud AI Comparison

| Parameter | Cloud-Based Systems | ElderGuard AI (Qualcomm Edge) |
|---|---|---|
| **Privacy** | ❌ Continuous video uploaded to cloud | ✅ **100% On-Device (0 bytes video uploaded)** |
| **Network Outage Immunity** | ❌ Fails if Wi-Fi / broadband goes down | ✅ **Autonomous Local Protection 24/7** |
| **Detection Latency** | ❌ 500ms – 3000ms network roundtrip | ✅ **< 33ms (Real-time 30+ FPS)** |
| **Bandwidth Cost** | ❌ Massive continuous GBs upload daily | ✅ **Zero video bandwidth required** |
| **Hardware Efficiency** | ❌ Expensive GPU cloud servers | ✅ **Ultra-low power Qualcomm Hexagon NPU** |

---

## 🔬 How Fall Detection & False Positive Reduction Works

ElderGuard AI does not use a simplistic static threshold. It processes human movement across four simultaneous kinematic dimensions:

### 1. Torso Inclination Angle ($\theta_{\text{torso}}$)
Calculated from the directional vector between the hip midpoint $H = (x_h, y_h)$ and the shoulder midpoint $S = (x_s, y_s)$:
$$\theta_{\text{torso}} = \arctan2(|x_s - x_h|, |y_s - y_h|) \times \frac{180^\circ}{\pi}$$
* Normal upright stance: $\theta_{\text{torso}} \in [0^\circ, 35^\circ]$
* Controlled sitting: $\theta_{\text{torso}} \in [0^\circ, 42^\circ]$
* Acute Fall / Prone: $\theta_{\text{torso}} > 60^\circ - 85^\circ$

### 2. Silhouette Aspect Ratio Collapse ($AR$)
$$AR = \frac{\text{Bounding Box Height}}{\text{Bounding Box Width}}$$
* Upright: $AR > 1.3$ (typically $1.6 - 2.4$)
* Horizontal Fall: $AR < 0.85$ (typically $0.4 - 0.7$)

### 3. Vertical Drop Velocity ($V_y$)
Calculated as the derivative of hip displacement over a temporal window $\Delta t$:
$$V_y = \frac{Y_{\text{hip}}(t) - Y_{\text{hip}}(t - \Delta t)}{\Delta t}$$
* **Slow Intentional Lying Down:** $V_y < 0.20$ frame\_height/s (transition takes $> 1.5$ seconds) $\to$ **SAFE / FALSE POSITIVE SUPPRESSED**.
* **Sudden Slip & Fall:** $V_y > 0.35$ frame\_height/s (rapid downward drop in $< 0.3$ seconds) $\to$ **FALL CANDIDATE TRIGGERED**.

### 4. Dynamic Verification & Recovery Check
* Upon impact, if a fall is identified, the system initiates the **10-second "ARE YOU OK?"** verification countdown.
* If the resident regains upright posture ($\theta_{\text{torso}} \le 38^\circ$ and $AR \ge 1.20$), the system records an automatic self-recovery and cancels the alert.
* If the resident remains down until 0s, an **EMERGENCY ALERT** is confirmed.

---

## 💻 Tech Stack

* **Language:** Python 3.11+ / Python 3.13
* **Edge Computer Vision:** OpenCV (`opencv-python`), MediaPipe Pose
* **Biomechanical Processing:** NumPy, SciPy Kinematics
* **Caregiver Dashboard:** Streamlit (Custom Responsive Dark Theme)
* **Local Storage:** SQLite 3 (Zero-Cloud Embedded DB)
* **Acoustic Warning:** Native OS Frequency Synthesizer (`winsound`)

---

## 📦 Installation & Setup

### Prerequisites
* Windows 10/11, macOS, or Linux.
* Python 3.10, 3.11, 3.12, or 3.13.
* Standard built-in webcam or external USB camera.

### Step 1: Clone or Navigate to Directory
```bash
cd c:\Users\Abhishek Nandan\Desktop\ElderGuard-AI
```

### Step 2: Install Dependencies
```bash
py -m pip install -r requirements.txt
```

---

## 🏃 How to Run the Application

### Option A: One-Click Launch (Windows)
Double-click `run_app.bat` or execute in Command Prompt/PowerShell:
```bash
.\run_app.bat
```

### Option B: Terminal Command
```bash
py -m streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🎭 Hackathon Demonstration Guide (For Judges)

ElderGuard AI includes a **7-Scenario Interactive Simulator** built right into the sidebar so you can demonstrate every scenario reliably without needing physical gymnastics during a live presentation:

1. **Scenario 1: Person Standing (SAFE)**
   - Select `Scenario 1` in the sidebar.
   - Observe: Torso angle ~ $4^\circ$, Status: `🟢 SAFE`, Fall Confidence: $5\%$.
2. **Scenario 2: Person Walking (SAFE)**
   - Select `Scenario 2`.
   - Observe: Person strides across the room. Status remains `🟢 SAFE`.
3. **Scenario 3: Person Sitting Down (SAFE - Controlled Motion)**
   - Select `Scenario 3`.
   - Observe: Smooth sitting transition. Aspect ratio adjusts to sitting (~$1.15$), but velocity stays low. System recognizes sitting and remains `🟢 SAFE`.
4. **Scenario 4: Intentional Lie Down (SAFE - False Positive Suppression)**
   - Select `Scenario 4`.
   - Observe: Person lies down slowly. Even though angle reaches $82^\circ$, descent velocity is below threshold. System classifies `INTENTIONAL_LIE` and **DOES NOT** trigger an emergency!
5. **Scenario 5: Sudden Slip & Fall (POSSIBLE FALL)**
   - Select `Scenario 5`.
   - Observe: Sudden downward drop spike! Torso tilts rapidly to $85^\circ$. System instantly triggers `⚠ POSSIBLE FALL DETECTED`.
6. **Scenario 6: Fall with Recovery (RECOVERED)**
   - Select `Scenario 6`.
   - Observe: Person slips and falls $\to$ Verification Countdown begins $\to$ Person stands back up $\to$ System detects recovery $\to$ `🔵 RECOVERED` banner displayed $\to$ Emergency cancelled and logged in audit history!
7. **Scenario 7: Fall without Recovery (10s Countdown -> EMERGENCY)**
   - Select `Scenario 7`.
   - Observe: Sudden fall occurs $\to$ Warning chime sounds $\to$ "ARE YOU OK? [10s]" countdown counts down $10, 9, 8... 0$ $\to$ `🚨 EMERGENCY ALERT: FALL CONFIRMED` triggers with acoustic siren $\to$ Event saved to SQLite database.
8. **Live Webcam Test:**
   - Switch dropdown to `Live Laptop Webcam`.
   - Stand in front of camera, lean or sit to view real-time angle and velocity tracking.
   - Click `[ Simulate Fall ]` to trigger immediate verification flow on live feed.
   - Click `[ I'M OK ]` to test resident tactile cancellation.
   - Click `[ Call Caregiver ]` to test simulated emergency dispatch.

---

## 🔒 Privacy Guarantee

ElderGuard AI adheres strictly to the **Principle of Least Data**:
1. **No Cloud Surveillance:** Raw camera frames are processed directly in RAM and never written to disk or transmitted over the network.
2. **No Facial Biometrics:** The system does not perform facial recognition, facial landmark extraction, or identify who the resident is from facial features.
3. **Face Privacy Mask:** When Privacy Mode is enabled, the facial region is automatically blurred or pixelated.
4. **Silhouette Blackout View:** An optional privacy mode blacks out the entire background, displaying only the anonymous biomechanical stick skeleton.
5. **Encrypted Event Metadata Only:** When an emergency occurs, only an event package (`event_id`, `timestamp`, `confidence`, `location`) is logged or transmitted.

---

## ⚠️ Limitations & Future Scope

### Current Prototype Limitations
* **Single Resident Focus:** Optimized for primary resident monitoring in an individual room. Occlusion by a second person walking in front can temporarily disrupt landmark tracking.
* **Camera Field of View:** Fixed webcam angle requires the resident to be within camera sightlines.

### Production Roadmap with Qualcomm
* **Qualcomm AI Hub Model Export:** Convert pose estimation pipeline to ONNX and quantize to INT8 for Qualcomm Hexagon NPU using Qualcomm AI Engine Direct SDK.
* **Multi-Camera Smart Home Mesh:** Connect multiple ceiling/wall Qualcomm QCS610 smart camera nodes across rooms via Matter/Thread protocol.
* **Thermal / IR Sensor Fusion:** Integrate low-resolution thermal arrays (e.g. 32x24) for privacy-absolute darkness monitoring (nighttime falls near the bed).
* **Direct Cellular SOS Module:** Integrate Snapdragon X35 5G RedCap modem for independent emergency calling during power and broadband outages.

---

## 👥 Authors & Acknowledgments

* **Project:** ElderGuard AI
* **Hackathon:** Navonmesh 26 (In Collaboration with Qualcomm)
* **Problem Statement:** PS-06 (Edge AI-Based Elderly Fall Detection)
* Developed with ❤️ for elderly safety, dignity, and independence.
