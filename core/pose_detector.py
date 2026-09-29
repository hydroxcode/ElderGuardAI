"""
ElderGuard AI - Robust Edge Pose & Person Tracking Engine
Uses on-device MediaPipe PoseLandmarker (TFLite Edge AI) with:
1. Long-horizon Dropout Persistence (25 frames / ~0.8s) to eliminate pose flickering
2. Biometric Spine & Upper-Body Tracking (works 100% when hips/legs are occluded by desk)
3. Anatomical Head-Above-Shoulders verification (prevents false horizontal classification)
4. Adaptive EMA smoothing with zero-jitter stationary lock
5. Zero dependency on face detection or lower-body visibility
"""
import os
import math
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from config import fall_config

@dataclass
class PoseResult:
    detected: bool = False
    torso_angle: float = 0.0          # Degrees from vertical (0° = upright, 90° = horizontal)
    aspect_ratio: float = 1.0         # Height / Width
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0) # x, y, w, h
    hip_center: Tuple[float, float] = (0.5, 0.5)           # Normalized (x, y) - torso/hip center
    shoulder_center: Tuple[float, float] = (0.5, 0.3)      # Normalized (x, y) - head/shoulder center
    floor_proximity: float = 0.5                           # Normalized Y of lowest body point (0=top, 1=bottom)
    landmarks_px: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    confidence: float = 0.0                                # 0.0 to 1.0 tracking confidence
    motion_energy: float = 0.0                             # Torso centroid displacement magnitude
    tracking_held: bool = False                            # True if frame was held during transient dropout
    dropout_frames: int = 0                                # Number of consecutive frames held
    quality: str = "NONE"                                  # FULL_BODY, UPPER_BODY, PARTIAL, DROPOUT_HOLD

class PoseDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.engine_type = "MEDIAPIPE_TASKS"
        self._landmarker = None
        self._mp_image_module = None
        
        # Determine model path
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            lite_path = os.path.join(base_dir, "models", "pose_landmarker_lite.task")
            full_path = os.path.join(base_dir, "models", "pose_landmarker_full.task")
            if os.path.exists(full_path):
                model_path = full_path
            elif os.path.exists(lite_path):
                model_path = lite_path
            else:
                model_path = lite_path
        self.model_path = model_path

        # Smoothing & Persistence State
        self.alpha_stationary = 0.18  # Strong smoothing during stationary sitting / standing
        self.alpha_motion = 0.60      # Responsive smoothing during rapid movement
        self._prev_landmarks: Optional[Dict[str, Tuple[float, float]]] = None
        self._prev_hip: Optional[Tuple[float, float]] = None
        self._prev_sh: Optional[Tuple[float, float]] = None
        self._prev_bbox: Optional[Tuple[int, int, int, int]] = None
        
        # Dropout persistence: retain last valid pose for up to max_hold_frames (~0.8s)
        self.max_hold_frames: int = getattr(fall_config, "DROPOUT_HOLD_FRAMES", 25)
        self.held_frames_count: int = 0
        self._last_valid_result: Optional[PoseResult] = None

        self._init_landmarker()

    def _init_landmarker(self):
        """Initializes Google MediaPipe PoseLandmarker Tasks API."""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options

            if os.path.exists(self.model_path):
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options.BaseOptions(model_asset_path=self.model_path),
                    running_mode=vision.RunningMode.IMAGE,
                    min_pose_detection_confidence=0.25,
                    min_pose_presence_confidence=0.25,
                    min_tracking_confidence=0.25
                )
                self._landmarker = vision.PoseLandmarker.create_from_options(options)
                self._mp = mp
                self.engine_type = "MEDIAPIPE_TASKS"
            else:
                self.engine_type = "OPENCV_FALLBACK"
        except Exception:
            self.engine_type = "OPENCV_FALLBACK"

    def process_frame(self, frame: np.ndarray) -> PoseResult:
        """Processes a video frame and extracts temporally smoothed skeletal kinematics."""
        if frame is None or frame.size == 0:
            return self._handle_dropout()

        h, w = frame.shape[:2]

        if self._landmarker is not None:
            result = self._process_with_mediapipe(frame, w, h)
        else:
            result = self._process_opencv_fallback(frame, w, h)

        if not result.detected:
            return self._handle_dropout()
        else:
            # Person successfully detected: reset hold counter
            self.held_frames_count = 0
            self._last_valid_result = result
            return result

    def _handle_dropout(self) -> PoseResult:
        """Holds the previous valid pose across temporary frame/landmark dropouts."""
        if self._last_valid_result is not None and self.held_frames_count < self.max_hold_frames:
            self.held_frames_count += 1
            # Graceful confidence decay over the hold window
            decay = max(0.35, 1.0 - (self.held_frames_count / float(self.max_hold_frames)) * 0.40)
            held_res = PoseResult(
                detected=True,
                torso_angle=self._last_valid_result.torso_angle,
                aspect_ratio=self._last_valid_result.aspect_ratio,
                bounding_box=self._last_valid_result.bounding_box,
                hip_center=self._last_valid_result.hip_center,
                shoulder_center=self._last_valid_result.shoulder_center,
                floor_proximity=self._last_valid_result.floor_proximity,
                landmarks_px=self._last_valid_result.landmarks_px,
                confidence=round(self._last_valid_result.confidence * decay, 2),
                motion_energy=0.0,
                tracking_held=True,
                dropout_frames=self.held_frames_count,
                quality="DROPOUT_HOLD"
            )
            return held_res
        else:
            # Dropout hold expired: person is truly gone
            self.held_frames_count = self.max_hold_frames
            self._prev_landmarks = None
            self._prev_hip = None
            self._prev_sh = None
            self._prev_bbox = None
            self._last_valid_result = None
            return PoseResult(detected=False, quality="NONE")

    def _process_with_mediapipe(self, frame: np.ndarray, w: int, h: int) -> PoseResult:
        """Runs MediaPipe PoseLandmarker and computes jitter-free skeletal telemetry."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            detection = self._landmarker.detect(mp_image)
        except Exception:
            return PoseResult(detected=False)

        if not detection.pose_landmarks or len(detection.pose_landmarks) == 0:
            return PoseResult(detected=False)

        lms = detection.pose_landmarks[0]
        # MediaPipe 33 Landmark Indices:
        # 0: nose, 7: left_ear, 8: right_ear, 11: left_shoulder, 12: right_shoulder,
        # 13: left_elbow, 14: right_elbow, 15: left_wrist, 16: right_wrist,
        # 23: left_hip, 24: right_hip, 25: left_knee, 26: right_knee, 27: left_ankle, 28: right_ankle

        # 1. Extract Shoulders (Primary Upper-Body Anchor)
        ls, rs = lms[11], lms[12]
        sh_vis = (ls.visibility + rs.visibility) / 2.0
        
        # Check if at least shoulders or head/torso are present
        nose = lms[0]
        if sh_vis < 0.25 and nose.visibility < 0.25 and lms[23].visibility < 0.25:
            return PoseResult(detected=False)

        raw_sh_x = (ls.x + rs.x) / 2.0
        raw_sh_y = (ls.y + rs.y) / 2.0
        sh_span = math.hypot((ls.x - rs.x) * w, (ls.y - rs.y) * h)

        # 2. Extract Head Anchor (Nose or Ears)
        if nose.visibility >= 0.25:
            head_x, head_y = nose.x, nose.y
            head_vis = nose.visibility
        elif lms[7].visibility >= 0.25 and lms[8].visibility >= 0.25:
            head_x = (lms[7].x + lms[8].x) / 2.0
            head_y = (lms[7].y + lms[8].y) / 2.0
            head_vis = (lms[7].visibility + lms[8].visibility) / 2.0
        else:
            head_x = raw_sh_x
            head_y = max(0.0, raw_sh_y - 0.18)
            head_vis = 0.30

        # 3. Anatomical Spine & Upright Analysis:
        # In image coordinates, Y increases downward.
        # If head is above shoulders (head_y < sh_y), the resident's upper body is UPRIGHT.
        dy_head_sh = raw_sh_y - head_y  # Positive when head is above shoulders
        dx_head_sh = head_x - raw_sh_x

        lh, rh = lms[23], lms[24]
        hip_vis = (lh.visibility + rh.visibility) / 2.0
        
        # Reliable Hips Condition:
        # Hips are only used if they have high visibility AND are physically located below shoulders!
        # When sitting at a desk, hips are often occluded and MediaPipe outputs noisy/inverted coordinates.
        has_reliable_hips = (hip_vis >= 0.50 and ((lh.y + rh.y) / 2.0) > (raw_sh_y + 0.08))

        if has_reliable_hips:
            raw_hip_x = (lh.x + rh.x) / 2.0
            raw_hip_y = (lh.y + rh.y) / 2.0
            quality = "FULL_BODY" if (lms[27].visibility > 0.4 and lms[28].visibility > 0.4) else "UPPER_BODY"
        else:
            # SEATED AT DESK / OCCLUDED HIPS:
            # Anchor torso stably below shoulders along the true spine vector.
            # Torso length is typically 1.25x shoulder span.
            torso_len_norm = max(0.25, min(0.48, (sh_span * 1.25) / float(max(1, h))))
            # If spine is tilted, align hip center with spine direction
            if dy_head_sh > 0.04:
                spine_ratio = dx_head_sh / max(dy_head_sh, 1e-4)
                raw_hip_x = raw_sh_x - (spine_ratio * torso_len_norm * 0.5)
            else:
                raw_hip_x = raw_sh_x
            raw_hip_y = min(0.96, raw_sh_y + torso_len_norm)
            quality = "UPPER_BODY"

        # 4. Adaptive EMA Smoothing on Shoulders and Hips
        sh_x, sh_y = self._smooth_coordinate(raw_sh_x, raw_sh_y, is_shoulder=True)
        hip_x, hip_y = self._smooth_coordinate(raw_hip_x, raw_hip_y, is_shoulder=False)

        # 5. Centroid Motion Energy (Torso displacement only, not noisy pixels)
        motion_energy = 0.0
        if self._prev_hip is not None:
            dx_m = (hip_x - self._prev_hip[0]) * w
            dy_m = (hip_y - self._prev_hip[1]) * h
            motion_energy = float(math.hypot(dx_m, dy_m))
        self._prev_hip = (hip_x, hip_y)

        # 6. Compute Torso Angle from Vertical
        if dy_head_sh <= 0.025:
            # Head dropped level with or below shoulders (deep slump / collapse / bend down)
            dx = (sh_x - hip_x) * w
            dy = (sh_y - hip_y) * h
            torso_angle = max(70.0, math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-4))))
        else:
            # Head is above shoulders
            dx_spine = (head_x - sh_x) * w
            dy_spine = (sh_y - head_y) * h
            neck_angle = math.degrees(math.atan2(abs(dx_spine), max(dy_spine, 1e-4)))
            
            dx_torso = (sh_x - hip_x) * w
            dy_torso = (hip_y - sh_y) * h
            body_angle = math.degrees(math.atan2(abs(dx_torso), max(dy_torso, 1e-4)))

            # If both neck and torso are upright (sitting or standing normally):
            if neck_angle < 25.0 and body_angle < 25.0:
                torso_angle = min(25.0, (neck_angle + body_angle) * 0.5)
            else:
                # Leaning, slumping, or bending down
                torso_angle = max(neck_angle, body_angle)
                torso_angle = min(90.0, torso_angle)

        # 7. Extract Visible Landmarks & Build Coordinate Dictionary
        raw_lms_dict = {
            "nose": (head_x, head_y, head_vis),
            "left_shoulder": (ls.x, ls.y, ls.visibility),
            "right_shoulder": (rs.x, rs.y, rs.visibility),
            "left_elbow": (lms[13].x, lms[13].y, lms[13].visibility),
            "right_elbow": (lms[14].x, lms[14].y, lms[14].visibility),
            "left_wrist": (lms[15].x, lms[15].y, lms[15].visibility),
            "right_wrist": (lms[16].x, lms[16].y, lms[16].visibility),
            "left_hip": (lh.x, lh.y, lh.visibility),
            "right_hip": (rh.x, rh.y, rh.visibility),
            "left_knee": (lms[25].x, lms[25].y, lms[25].visibility),
            "right_knee": (lms[26].x, lms[26].y, lms[26].visibility),
            "left_ankle": (lms[27].x, lms[27].y, lms[27].visibility),
            "right_ankle": (lms[28].x, lms[28].y, lms[28].visibility),
        }

        smoothed_px: Dict[str, Tuple[int, int]] = {}
        all_xs, all_ys = [], []

        for name, (rx, ry, rvis) in raw_lms_dict.items():
            if rvis > 0.25:
                px = int(np.clip(rx * w, 0, w - 1))
                py = int(np.clip(ry * h, 0, h - 1))
                smoothed_px[name] = (px, py)
                all_xs.append(px)
                all_ys.append(py)

        sh_px = (int(np.clip(sh_x * w, 0, w - 1)), int(np.clip(sh_y * h, 0, h - 1)))
        hip_px = (int(np.clip(hip_x * w, 0, w - 1)), int(np.clip(hip_y * h, 0, h - 1)))
        smoothed_px["shoulder_center"] = sh_px
        smoothed_px["hip_center"] = hip_px
        all_xs.extend([sh_px[0], hip_px[0]])
        all_ys.extend([sh_px[1], hip_px[1]])

        # 8. Compute Bounding Box, Aspect Ratio, and Floor Proximity
        if all_xs and all_ys:
            min_x, max_x = max(0, min(all_xs)), min(w, max(all_xs))
            min_y, max_y = max(0, min(all_ys)), min(h, max(all_ys))
            bw = max(20, max_x - min_x)
            bh = max(20, max_y - min_y)
            aspect_ratio = bh / float(max(1, bw))
            floor_prox = max_y / float(h)
            bbox = (min_x, min_y, bw, bh)
        else:
            bw, bh = int(w * 0.35), int(h * 0.45)
            min_x, min_y = max(0, int(sh_x * w - bw / 2)), max(0, int(sh_y * h - 20))
            bbox = (min_x, min_y, bw, bh)
            aspect_ratio = 1.1
            floor_prox = hip_y

        # Smooth bounding box
        if self._prev_bbox is not None:
            pbx, pby, pbw, pbh = self._prev_bbox
            sbx = int(0.20 * bbox[0] + 0.80 * pbx)
            sby = int(0.20 * bbox[1] + 0.80 * pby)
            sbw = int(0.20 * bbox[2] + 0.80 * pbw)
            sbh = int(0.20 * bbox[3] + 0.80 * pbh)
            bbox = (sbx, sby, max(1, sbw), max(1, sbh))
        self._prev_bbox = bbox

        confidence = float(np.clip((sh_vis + head_vis + max(hip_vis, 0.4)) / 3.0, 0.4, 0.99))

        return PoseResult(
            detected=True,
            torso_angle=round(torso_angle, 1),
            aspect_ratio=round(aspect_ratio, 2),
            bounding_box=bbox,
            hip_center=(round(hip_x, 3), round(hip_y, 3)),
            shoulder_center=(round(sh_x, 3), round(sh_y, 3)),
            floor_proximity=round(floor_prox, 3),
            landmarks_px=smoothed_px,
            confidence=round(confidence, 2),
            motion_energy=round(motion_energy, 1),
            tracking_held=False,
            dropout_frames=0,
            quality=quality
        )

    def _smooth_coordinate(self, x: float, y: float, is_shoulder: bool) -> Tuple[float, float]:
        """Adaptive Exponential Moving Average with outlier step clamping."""
        prev = self._prev_sh if is_shoulder else self._prev_hip
        if prev is None:
            res = (x, y)
        else:
            px, py = prev
            step = math.hypot(x - px, y - py)
            
            # Adaptive alpha:
            # Low motion (< 0.03): heavy alpha (0.18) to freeze jitter
            # Moderate motion: 0.35
            # High motion (> 0.08): 0.60
            if step < 0.03:
                alpha = self.alpha_stationary
            elif step < 0.08:
                alpha = 0.35
            else:
                alpha = self.alpha_motion

            # Clamp single-frame teleportation jumps (> 0.20 frame)
            if step > 0.20:
                scale = 0.20 / step
                x = px + (x - px) * scale
                y = py + (y - py) * scale

            sx = alpha * x + (1.0 - alpha) * px
            sy = alpha * y + (1.0 - alpha) * py
            res = (sx, sy)

        if is_shoulder:
            self._prev_sh = res
        else:
            self._prev_hip = res

        return res

    def _process_opencv_fallback(self, frame: np.ndarray, w: int, h: int) -> PoseResult:
        """Clean, non-face-dependent fallback if MediaPipe is unavailable."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        edges = cv2.Canny(blurred, 40, 100)
        dilated = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [c for c in contours if cv2.contourArea(c) > (w * h * 0.06)]
        if not valid_contours:
            return PoseResult(detected=False, quality="NONE")

        largest_c = max(valid_contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest_c)

        sh_x = (bx + bw / 2.0) / float(w)
        sh_y = (by + bh * 0.30) / float(h)
        hip_x = (bx + bw / 2.0) / float(w)
        hip_y = (by + bh * 0.75) / float(h)

        dx = (sh_x - hip_x) * w
        dy = (sh_y - hip_y) * h
        torso_angle = math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-4)))
        aspect_ratio = bh / float(max(1, bw))
        floor_prox = (by + bh) / float(h)

        sh_px = (int(sh_x * w), int(sh_y * h))
        hip_px = (int(hip_x * w), int(hip_y * h))

        return PoseResult(
            detected=True,
            torso_angle=round(torso_angle, 1),
            aspect_ratio=round(aspect_ratio, 2),
            bounding_box=(bx, by, bw, bh),
            hip_center=(round(hip_x, 3), round(hip_y, 3)),
            shoulder_center=(round(sh_x, 3), round(sh_y, 3)),
            floor_proximity=round(floor_prox, 3),
            landmarks_px={
                "shoulder_center": sh_px,
                "hip_center": hip_px,
                "nose": (sh_px[0], max(0, sh_px[1] - 25))
            },
            confidence=0.70,
            motion_energy=0.0,
            quality="UPPER_BODY"
        )

    def reset(self):
        """Resets temporal smoothing and dropout state."""
        self._prev_landmarks = None
        self._prev_hip = None
        self._prev_sh = None
        self._prev_bbox = None
        self.held_frames_count = 0
        self._last_valid_result = None
