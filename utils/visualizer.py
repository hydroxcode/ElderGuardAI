"""
ElderGuard AI - Edge Visualization and Privacy Rendering Engine
Renders pose skeletons, biomechanical telemetry, privacy masks, and emergency HUD overlays.
"""
import cv2
import numpy as np
import time
from typing import Tuple, Dict, Any, Optional
from core.pose_detector import PoseResult
from core.fall_detector import FallAnalysisResult
from core.state_machine import SystemState

# Color Palette (BGR)
COLOR_SAFE = (76, 217, 100)        # Bright Green
COLOR_WARNING = (30, 165, 255)     # Amber / Orange
COLOR_EMERGENCY = (40, 40, 255)    # High-intensity Red
COLOR_RECOVERED = (255, 180, 50)   # Blue / Cyan
COLOR_DARK_BG = (24, 24, 27)       # Near Black
COLOR_CYAN = (240, 200, 0)         # Edge Cyan

# Skeletal Connection pairs
POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

class EdgeVisualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.pulse_phase = 0.0

    def render(self, frame: np.ndarray, pose: PoseResult,
               analysis: FallAnalysisResult, state: SystemState,
               countdown_sec: int = 10, privacy_mode: bool = True,
               anonymize_face: bool = True, skeleton_only: bool = False) -> np.ndarray:
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        canvas = frame.copy()

        # Update pulsing animation
        self.pulse_phase = (self.pulse_phase + 0.15) % (2 * np.pi)
        pulse_alpha = 0.5 + 0.5 * np.sin(self.pulse_phase)

        # 1. Privacy Anonymization
        if skeleton_only:
            # Complete blackout privacy mode: only show biomechanical wireframe
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            # Add subtle grid lines for tech edge look
            cv2.line(canvas, (0, int(h * 0.75)), (w, int(h * 0.75)), (40, 40, 40), 1)
        elif privacy_mode and anonymize_face and pose.detected:
            canvas = self._apply_face_privacy_blur(canvas, pose)

        # 2. Draw Skeletal Wireframe
        if pose.detected:
            canvas = self._draw_skeleton(canvas, pose, state)

        # 3. Draw State-Specific Alerts & HUD
        if state == SystemState.VERIFYING:
            canvas = self._draw_verifying_overlay(canvas, countdown_sec, pulse_alpha)
        elif state == SystemState.EMERGENCY_CONFIRMED:
            canvas = self._draw_emergency_overlay(canvas, analysis, pulse_alpha)
        elif state == SystemState.RECOVERED:
            canvas = self._draw_recovered_overlay(canvas)

        # 4. Draw Edge Telemetry Card at Top Left
        canvas = self._draw_telemetry_hud(canvas, pose, analysis, state, privacy_mode)

        return canvas

    def _apply_face_privacy_blur(self, canvas: np.ndarray, pose: PoseResult) -> np.ndarray:
        """Applies a privacy blur/mosaic or oval mask over the resident's head."""
        h, w = canvas.shape[:2]
        
        # Determine head region
        if "nose" in pose.landmarks_px:
            nx, ny = pose.landmarks_px["nose"]
            r = int(w * 0.08) # 8% width radius
        elif "shoulder_center" in pose.landmarks_px:
            sx, sy = pose.landmarks_px["shoulder_center"]
            nx, ny = sx, max(10, sy - int(h * 0.15))
            r = int(w * 0.08)
        else:
            return canvas

        x1, y1 = max(0, nx - r), max(0, ny - r)
        x2, y2 = min(w, nx + r), min(h, ny + r)

        if (x2 - x1) > 10 and (y2 - y1) > 10:
            roi = canvas[y1:y2, x1:x2]
            # Heavy Gaussian Blur to remove facial features completely
            blurred = cv2.GaussianBlur(roi, (51, 51), 30)
            canvas[y1:y2, x1:x2] = blurred

            # Draw privacy shield icon/circle
            cv2.circle(canvas, (nx, ny), r, (80, 80, 80), 2)
            cv2.putText(canvas, "PRIVACY MASK", (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return canvas

    def _draw_skeleton(self, canvas: np.ndarray, pose: PoseResult, state: SystemState) -> np.ndarray:
        """Draws skeletal links and joint nodes."""
        lms = pose.landmarks_px
        
        # Color based on state
        if state == SystemState.EMERGENCY_CONFIRMED:
            limb_color = COLOR_EMERGENCY
            joint_color = (0, 0, 255)
        elif state == SystemState.VERIFYING:
            limb_color = COLOR_WARNING
            joint_color = (0, 215, 255)
        elif state == SystemState.RECOVERED:
            limb_color = COLOR_RECOVERED
            joint_color = (255, 255, 100)
        else:
            limb_color = COLOR_SAFE
            joint_color = (100, 255, 100)

        # Draw connecting bones
        for p1_name, p2_name in POSE_CONNECTIONS:
            if p1_name in lms and p2_name in lms:
                pt1 = lms[p1_name]
                pt2 = lms[p2_name]
                if pt1 != (0, 0) and pt2 != (0, 0):
                    cv2.line(canvas, pt1, pt2, limb_color, 3, cv2.LINE_AA)

        # Draw spine vector (Torso orientation)
        if "shoulder_center" in lms and "hip_center" in lms:
            sc = lms["shoulder_center"]
            hc = lms["hip_center"]
            cv2.line(canvas, hc, sc, COLOR_CYAN, 4, cv2.LINE_AA)
            cv2.circle(canvas, sc, 6, (0, 255, 255), -1)
            cv2.circle(canvas, hc, 6, (0, 255, 255), -1)

        # Draw joints
        for name, pt in lms.items():
            if pt != (0, 0):
                cv2.circle(canvas, pt, 4, joint_color, -1, cv2.LINE_AA)

        return canvas

    def _draw_telemetry_hud(self, canvas: np.ndarray, pose: PoseResult,
                            analysis: FallAnalysisResult, state: SystemState,
                            privacy_mode: bool) -> np.ndarray:
        """Draws modern translucent HUD card with kinematics and edge badges."""
        h, w = canvas.shape[:2]
        
        # HUD background box
        card_w, card_h = 360, 225
        overlay = canvas.copy()
        cv2.rectangle(overlay, (15, 15), (15 + card_w, 15 + card_h), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (15, 15), (15 + card_w, 15 + card_h), (60, 60, 70), 1)

        # Status badge color
        badge_color = COLOR_SAFE
        status_text = "SAFE"
        if state == SystemState.POSSIBLE_FALL:
            badge_color = COLOR_WARNING
            status_text = "POSSIBLE FALL (EVALUATING)"
        elif state == SystemState.VERIFYING:
            badge_color = COLOR_WARNING
            status_text = "VERIFYING (ARE YOU OK?)"
        elif state == SystemState.EMERGENCY_CONFIRMED:
            badge_color = COLOR_EMERGENCY
            status_text = "EMERGENCY: FALL DETECTED"
        elif state == SystemState.RECOVERED:
            badge_color = COLOR_RECOVERED
            status_text = "RECOVERED / SAFE"

        # Header Title
        cv2.putText(canvas, "ELDERGUARD AI", (25, 38), self.font, 0.65, (255, 255, 255), 2)
        
        # Status Pill
        cv2.circle(canvas, (25, 60), 6, badge_color, -1)
        cv2.putText(canvas, f"STATUS: {status_text}", (38, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_color, 1)

        # Detailed Biomechanical Debug Metrics
        y_off = 86
        raw_act = getattr(analysis, "raw_activity", analysis.activity)
        cv2.putText(canvas, f"Stable: {analysis.activity}  (Raw: {raw_act})", (25, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)
        
        track_pct = int(pose.confidence * 100) if pose.detected else 0
        quality_str = getattr(pose, "quality", "TRACKING")
        drop_frames = getattr(pose, "dropout_frames", 0)
        track_info = f"Pose Track: {track_pct}% ({quality_str}) | Dropout: {drop_frames}f"
        cv2.putText(canvas, track_info, (25, y_off + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        cv2.putText(canvas, f"Torso Angle: {analysis.torso_angle:.1f} deg", (25, y_off + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        cv2.putText(canvas, f"Drop Vel: {analysis.vertical_velocity:.2f} m/s (Peak: {analysis.max_recent_drop_velocity:.2f})", (25, y_off + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        cv2.putText(canvas, f"Floor Prox: {analysis.floor_proximity:.2f}  |  Aspect Ratio: {analysis.aspect_ratio:.2f}", (25, y_off + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1)

        susp_color = (0, 215, 255) if analysis.consecutive_suspicious_frames > 0 else (180, 180, 180)
        cv2.putText(canvas, f"Suspicious Frames: {analysis.consecutive_suspicious_frames} / 8", (25, y_off + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, susp_color, 1)

        conf_color = (0, 100, 255) if analysis.confidence >= 65 else (100, 255, 100)
        cv2.putText(canvas, f"Fall Confidence: {analysis.confidence:.1f}%", (25, y_off + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, conf_color, 1)

        # Top Right Badges: Qualcomm Edge + Privacy Shield
        priv_text = "PRIVACY: ON (LOCAL EDGE)" if privacy_mode else "PRIVACY: STANDARD"
        cv2.putText(canvas, priv_text, (w - 230, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 255, 100), 1)
        cv2.putText(canvas, "QUALCOMM ARCH: ACTIVE", (w - 230, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 200, 0), 1)

        return canvas

    def _draw_verifying_overlay(self, canvas: np.ndarray, countdown_sec: int, pulse_alpha: float) -> np.ndarray:
        """Renders the 'ARE YOU OK?' countdown overlay."""
        h, w = canvas.shape[:2]
        
        # Pulsing warning border
        border_thickness = int(6 + 4 * pulse_alpha)
        cv2.rectangle(canvas, (0, 0), (w, h), COLOR_WARNING, border_thickness)

        # Center banner
        bw, bh = 460, 110
        bx = (w - bw) // 2
        by = (h - bh) // 2

        overlay = canvas.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (15, 20, 35), -1)
        cv2.addWeighted(overlay, 0.9, canvas, 0.1, 0, canvas)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), COLOR_WARNING, 2)

        # Text banner
        cv2.putText(canvas, "POSSIBLE FALL DETECTED", (bx + 40, by + 35),
                    self.font, 0.7, (0, 215, 255), 2)
        
        msg = f"ARE YOU OK? [{countdown_sec}s]"
        cv2.putText(canvas, msg, (bx + 85, by + 75),
                    self.font, 0.9, (255, 255, 255), 2)

        cv2.putText(canvas, "Stand up or press [I'M OK] on dashboard to cancel",
                    (bx + 35, by + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        return canvas

    def _draw_emergency_overlay(self, canvas: np.ndarray, analysis: FallAnalysisResult, pulse_alpha: float) -> np.ndarray:
        """Renders full-screen emergency alert HUD."""
        h, w = canvas.shape[:2]
        
        # High-intensity red pulsing border
        thickness = int(8 + 6 * pulse_alpha)
        cv2.rectangle(canvas, (0, 0), (w, h), COLOR_EMERGENCY, thickness)

        # Top Emergency Banner
        banner_h = 75
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 180), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        cv2.putText(canvas, "EMERGENCY ALERT: FALL CONFIRMED", (w // 2 - 240, 48),
                    self.font, 0.85, (255, 255, 255), 2)

        # Bottom info bar
        cv2.rectangle(canvas, (0, h - 45), (w, h), (18, 18, 24), -1)
        info_str = f"CONFIDENCE: {analysis.confidence:.1f}% | NO RESPONSE TIMEOUT | DISPATCHING LOCAL ALERT"
        cv2.putText(canvas, info_str, (30, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 255), 1)

        return canvas

    def _draw_recovered_overlay(self, canvas: np.ndarray) -> np.ndarray:
        """Renders green/blue 'RECOVERED' banner."""
        h, w = canvas.shape[:2]
        bw, bh = 380, 60
        bx = (w - bw) // 2
        by = (h - bh) // 2

        overlay = canvas.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (20, 40, 20), -1)
        cv2.addWeighted(overlay, 0.9, canvas, 0.1, 0, canvas)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), COLOR_RECOVERED, 2)

        cv2.putText(canvas, "RECOVERY DETECTED", (bx + 55, by + 38),
                    self.font, 0.75, (255, 255, 255), 2)
        return canvas
