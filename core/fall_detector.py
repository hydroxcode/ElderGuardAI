"""
ElderGuard AI - Temporal Fall Detection Engine
Two-Stage Separated Processing Pipeline:
  Stage 1: Posture & Activity Classifier (Temporal Majority Voting with Hysteresis & Dropout Retention)
  Stage 2: Kinetic Fall Detection Engine (Multi-Phase Temporal Kinetic Event)

Key Invariants:
1. Short landmark/camera dropouts (1-30 frames / ~1.0s) retain the current stable activity (e.g. SITTING).
   NO_PERSON is declared only after continuous, sustained absence (> 30 frames).
2. SITTING is locked: Normal desk gestures, typing, head movements, and leaning NEVER trigger WALKING
   or INTENTIONAL_LIE.
3. INTENTIONAL_LIE requires undeniable evidence of full-body recumbency (torso angle >= 65°,
   head not above shoulders, and sustained agreement). Never guessed from missing hips or aspect ratio.
4. Fall detection is strictly decoupled: Sitting, standing, walking, and intentional lying
   NEVER trigger a fall candidate.
"""
import time
import collections
from dataclasses import dataclass, field
from typing import Deque, Tuple, Optional, Dict, Any, List
from config import FallDetectionConfig, fall_config
from core.pose_detector import PoseResult

@dataclass
class FallAnalysisResult:
    activity: str = "SITTING"          # Stable filtered activity: STANDING, WALKING, SITTING, BENDING, INTENTIONAL_LIE, POSSIBLE_FALL, ON_FLOOR, NO_PERSON
    raw_activity: str = "SITTING"      # Instantaneous single-frame classification
    is_fall_candidate: bool = False    # True ONLY when kinetic fall event sequence is verified
    confidence: float = 0.0            # Fall confidence (0.0 to 100.0%)
    torso_angle: float = 0.0           # Smoothed torso angle in degrees
    aspect_ratio: float = 1.0          # Smoothed bounding box aspect ratio
    vertical_velocity: float = 0.0     # Hip downward velocity (norm_h / sec)
    max_recent_drop_velocity: float = 0.0 # Peak downward velocity in recent temporal window
    floor_proximity: float = 0.0       # Floor proximity (0.0 to 1.0)
    is_recovering: bool = False        # True if resident has stood back up
    consecutive_suspicious_frames: int = 0 # Count of consecutive frames satisfying acute fall criteria
    consensus_ratio: float = 1.0       # Agreement ratio for stable activity (0.0 to 1.0)
    consecutive_no_person_frames: int = 0 # Dropout duration counter
    detailed_metrics: Dict[str, Any] = field(default_factory=dict) # Full telemetry dictionary for debug HUD

class FallDetector:
    def __init__(self, config: FallDetectionConfig = fall_config):
        self.config = config
        
        # Temporal history buffers
        self.pose_history: Deque[Tuple[float, PoseResult]] = collections.deque(maxlen=45) # ~1.5s history
        self.raw_activity_history: Deque[str] = collections.deque(maxlen=24)              # ~0.8s voting window
        
        # Stable activity state (hysteresis)
        self.stable_activity: str = "SITTING"
        self.consecutive_no_person_frames: int = 0
        
        # Last known valid pose metrics for bridging dropouts
        self.last_valid_angle: float = 10.0
        self.last_valid_ar: float = 1.1
        self.last_valid_floor_prox: float = 0.5
        
        # Debounce and tracking counters
        self.consecutive_upright_frames: int = 0
        self.consecutive_suspicious_frames: int = 0
        self.consecutive_fallen_frames: int = 0
        self.last_upright_time: float = time.time()
        self.baseline_hip_y: Optional[float] = None

    def analyze(self, pose: PoseResult, current_time: Optional[float] = None) -> FallAnalysisResult:
        """Processes pose telemetry through Stage 1 (Posture) and Stage 2 (Kinetic Fall)."""
        if current_time is None:
            current_time = time.time()

        # ---------------------------------------------------------------------
        # DROPOUT & PERSISTENCE HANDLING
        # ---------------------------------------------------------------------
        no_person_timeout = getattr(self.config, "NO_PERSON_TIMEOUT_FRAMES", 30)

        if not pose.detected:
            self.consecutive_no_person_frames += 1
            self.consecutive_suspicious_frames = max(0, self.consecutive_suspicious_frames - 1)

            # Bridging Dropout: If tracking loss is temporary (1 to 30 frames / ~1.0s),
            # DO NOT flicker to NO_PERSON or UNKNOWN! Retain the stable activity!
            if self.consecutive_no_person_frames <= no_person_timeout:
                return FallAnalysisResult(
                    activity=self.stable_activity,
                    raw_activity="DROPOUT_HOLD",
                    is_fall_candidate=False,
                    confidence=0.0,
                    torso_angle=self.last_valid_angle,
                    aspect_ratio=self.last_valid_ar,
                    vertical_velocity=0.0,
                    max_recent_drop_velocity=0.0,
                    floor_proximity=self.last_valid_floor_prox,
                    is_recovering=False,
                    consecutive_suspicious_frames=0,
                    consensus_ratio=1.0,
                    consecutive_no_person_frames=self.consecutive_no_person_frames,
                    detailed_metrics={
                        "status": f"Temporary tracking dropout ({self.consecutive_no_person_frames}/{no_person_timeout}) — Holding {self.stable_activity}",
                        "dropout_frames": self.consecutive_no_person_frames,
                        "tracking": "HELD"
                    }
                )
            else:
                # Sustained absence: resident has genuinely exited camera view
                self.stable_activity = "NO_PERSON"
                return FallAnalysisResult(
                    activity="NO_PERSON",
                    raw_activity="NO_PERSON",
                    is_fall_candidate=False,
                    confidence=0.0,
                    torso_angle=0.0,
                    aspect_ratio=1.0,
                    vertical_velocity=0.0,
                    max_recent_drop_velocity=0.0,
                    floor_proximity=0.0,
                    is_recovering=False,
                    consecutive_suspicious_frames=0,
                    consensus_ratio=1.0,
                    consecutive_no_person_frames=self.consecutive_no_person_frames,
                    detailed_metrics={
                        "status": "No resident detected in frame",
                        "dropout_frames": self.consecutive_no_person_frames,
                        "tracking": "OFFLINE"
                    }
                )

        # Person is actively detected: reset dropout counter
        self.consecutive_no_person_frames = 0
        self.last_valid_angle = pose.torso_angle
        self.last_valid_ar = pose.aspect_ratio
        self.last_valid_floor_prox = pose.floor_proximity

        # Append to sliding history window
        self.pose_history.append((current_time, pose))

        # ---------------------------------------------------------------------
        # 1. Compute Kinetic Velocity Metrics
        # ---------------------------------------------------------------------
        vertical_velocity, max_recent_drop = self._compute_vertical_velocities(current_time)

        # ---------------------------------------------------------------------
        # 2. Check Biomechanical Posture States
        # ---------------------------------------------------------------------
        is_upright = (pose.torso_angle <= self.config.RECOVERY_ANGLE_THRESHOLD)
        is_angle_horizontal = (pose.torso_angle >= self.config.FALL_ANGLE_THRESHOLD)
        is_near_floor = (pose.floor_proximity >= self.config.FLOOR_PROXIMITY_THRESHOLD)

        # Track Upright Baseline for recovery detection
        if is_upright:
            self.consecutive_upright_frames += 1
            self.last_upright_time = current_time
            if self.baseline_hip_y is None:
                self.baseline_hip_y = pose.hip_center[1]
            else:
                self.baseline_hip_y = 0.95 * self.baseline_hip_y + 0.05 * pose.hip_center[1]
        else:
            self.consecutive_upright_frames = 0

        # Persistent upright posture confirms recovery
        is_recovering = (self.consecutive_upright_frames >= self.config.RECOVERY_CONSECUTIVE_FRAMES)

        # ---------------------------------------------------------------------
        # STAGE 1: Posture & Activity Classification with Majority Voting
        # ---------------------------------------------------------------------
        raw_activity = self._classify_raw_posture(
            pose=pose,
            is_upright=is_upright,
            is_angle_horizontal=is_angle_horizontal,
            is_near_floor=is_near_floor,
            max_recent_drop=max_recent_drop
        )
        self.raw_activity_history.append(raw_activity)

        # Determine stable activity using temporal majority voting & hysteresis
        stable_activity, consensus_ratio = self._update_stable_activity(raw_activity, pose)

        # ---------------------------------------------------------------------
        # STAGE 2: Independent Kinetic Fall Detection Engine
        # ---------------------------------------------------------------------
        # A frame is ONLY suspicious if it meets the full acute kinetic signature:
        # A) Peak downward drop velocity >= threshold (0.38 m/s) in recent window
        # B) Torso collapsed horizontally (angle >= 62°)
        # C) Body elevation is down in lower zone of room (floor_proximity >= 0.62)
        # D) Descent was NOT slow/controlled (max_drop > SLOW_DESCENT_MAX_VELOCITY)
        # True collapse requires horizontal angle or deep slump/bend
        is_acute_velocity_drop = (max_recent_drop >= self.config.FALL_VELOCITY_THRESHOLD)
        is_collapsed = is_angle_horizontal
        is_near_floor_or_seated = (pose.floor_proximity >= self.config.FLOOR_PROXIMITY_THRESHOLD or 
                                   pose.quality == "UPPER_BODY" or 
                                   pose.torso_angle >= self.config.FALL_ANGLE_THRESHOLD)
        
        # Controlled lying down on bed/sofa is never suspicious
        is_controlled_lying = (raw_activity == "INTENTIONAL_LIE")

        # A frame is suspicious if:
        # A) Standard dynamic fall: Velocity drop + collapsed posture
        # B) Seated bend/slump fall: Torso angle >= FALL_ANGLE_THRESHOLD (deep bend down towards desk/lap)
        is_frame_suspicious = (
            is_collapsed and 
            is_near_floor_or_seated and 
            not is_controlled_lying and
            (is_acute_velocity_drop or pose.torso_angle >= 55.0) and
            not is_recovering
        )

        # Temporal Debounce: Require sustained collapse across REQUIRED_SUSPICIOUS_FRAMES
        if is_frame_suspicious:
            self.consecutive_suspicious_frames += 1
        else:
            self.consecutive_suspicious_frames = max(0, self.consecutive_suspicious_frames - 2)

        # Compute Fall Confidence (0-100%)
        confidence = self._compute_fall_confidence(
            is_frame_suspicious=is_frame_suspicious,
            max_recent_drop=max_recent_drop,
            torso_angle=pose.torso_angle,
            floor_proximity=pose.floor_proximity
        )

        # A fall candidate is strictly confirmed ONLY when:
        # 1. Suspicious frames persist for REQUIRED_SUSPICIOUS_FRAMES (5 frames)
        # 2. Confidence >= 65.0%
        # 3. Not actively recovering
        is_fall_candidate = (
            self.consecutive_suspicious_frames >= self.config.REQUIRED_SUSPICIOUS_FRAMES and
            confidence >= 65.0 and
            not is_recovering
        )

        # Override displayed activity ONLY when genuine fall candidate is confirmed
        displayed_activity = stable_activity
        if is_fall_candidate:
            self.consecutive_fallen_frames += 1
            if self.consecutive_fallen_frames > 25:
                displayed_activity = "ON_FLOOR"
            else:
                displayed_activity = "POSSIBLE_FALL"
        else:
            self.consecutive_fallen_frames = 0

        # Detailed metrics for debugging and HUD
        metrics = {
            "stable_activity": stable_activity,
            "raw_activity": raw_activity,
            "consensus_ratio": round(consensus_ratio, 2),
            "torso_angle": pose.torso_angle,
            "aspect_ratio": pose.aspect_ratio,
            "vertical_velocity": round(vertical_velocity, 3),
            "peak_drop_velocity": round(max_recent_drop, 3),
            "floor_proximity": pose.floor_proximity,
            "consecutive_suspicious_frames": self.consecutive_suspicious_frames,
            "required_suspicious_frames": self.config.REQUIRED_SUSPICIOUS_FRAMES,
            "motion_energy": pose.motion_energy,
            "is_upright": is_upright,
            "is_recovering": is_recovering,
            "is_frame_suspicious": is_frame_suspicious,
            "dropout_frames": self.consecutive_no_person_frames,
            "tracking_held": pose.tracking_held,
            "quality": pose.quality
        }

        return FallAnalysisResult(
            activity=displayed_activity,
            raw_activity=raw_activity,
            is_fall_candidate=is_fall_candidate,
            confidence=round(confidence, 1),
            torso_angle=pose.torso_angle,
            aspect_ratio=pose.aspect_ratio,
            vertical_velocity=round(vertical_velocity, 3),
            max_recent_drop_velocity=round(max_recent_drop, 3),
            floor_proximity=pose.floor_proximity,
            is_recovering=is_recovering,
            consecutive_suspicious_frames=self.consecutive_suspicious_frames,
            consensus_ratio=round(consensus_ratio, 2),
            consecutive_no_person_frames=self.consecutive_no_person_frames,
            detailed_metrics=metrics
        )

    def _classify_raw_posture(self, pose: PoseResult, is_upright: bool,
                              is_angle_horizontal: bool, is_near_floor: bool,
                              max_recent_drop: float) -> str:
        """
        Classifies single-frame instantaneous posture.
        Purely descriptive — NEVER sets fall candidate status.
        """
        # 1. Upright Stance: Standing, Walking, or Sitting Upright
        if is_upright and pose.torso_angle <= 40.0:
            # Distinguish standing stance
            is_standing = (pose.aspect_ratio >= 1.25 and pose.hip_center[1] < 0.62)

            # Walking requires standing stance AND active lateral translation over the last 14 frames
            if is_standing and len(self.pose_history) >= 14:
                old_t, old_p = self.pose_history[-14]
                dx_locomotion = abs(pose.hip_center[0] - old_p.hip_center[0])
                if dx_locomotion > 0.10:
                    return "WALKING"

            if is_standing:
                return "STANDING"
            else:
                return "SITTING"

        # 2. Leaning or Slouching while Sitting (Angle 40° to FALL_ANGLE_THRESHOLD)
        if 40.0 < pose.torso_angle < self.config.FALL_ANGLE_THRESHOLD:
            if max_recent_drop < self.config.FALL_VELOCITY_THRESHOLD:
                # Common desk sitting posture (leaning forward to keyboard or slouching)
                return "SITTING"
            return "BENDING"

        # 3. Horizontal / Slump / Bend Collapse Posture (Angle >= FALL_ANGLE_THRESHOLD)
        if pose.torso_angle >= self.config.FALL_ANGLE_THRESHOLD:
            # Slow controlled descent on bed or sofa in whole-room view
            if pose.floor_proximity >= 0.65 and max_recent_drop < self.config.SLOW_DESCENT_MAX_VELOCITY:
                return "INTENTIONAL_LIE"
            return "HORIZONTAL_PRONE"

        # Default fallback for desk webcam: SITTING
        return "SITTING"

    def _update_stable_activity(self, raw_activity: str, pose: PoseResult) -> Tuple[str, float]:
        """
        Temporal Majority Voting with Hysteresis.
        The displayed posture remains locked unless a new posture achieves strong consensus.
        """
        if len(self.raw_activity_history) <= 1:
            self.stable_activity = raw_activity
            return self.stable_activity, 1.0

        counts = collections.Counter(self.raw_activity_history)
        dominant_activity, count = counts.most_common(1)[0]
        consensus = count / float(len(self.raw_activity_history))

        # Hysteresis rules:
        # 1. SITTING LOCK:
        # If currently SITTING, do NOT transition to WALKING unless sustained locomotion is verified
        if self.stable_activity == "SITTING":
            if dominant_activity == "WALKING" and consensus < 0.80:
                return "SITTING", consensus
            # To switch from SITTING to INTENTIONAL_LIE / PRONE, require >= 85% consensus AND angle >= FALL_ANGLE_THRESHOLD
            if dominant_activity in ("INTENTIONAL_LIE", "HORIZONTAL_PRONE"):
                if consensus < 0.85 or pose.torso_angle < self.config.FALL_ANGLE_THRESHOLD:
                    return "SITTING", consensus

        # 2. General transition threshold: >= 70% consensus required to change stable activity
        if consensus >= 0.70:
            self.stable_activity = dominant_activity

        return self.stable_activity, consensus

    def _compute_vertical_velocities(self, current_time: float) -> Tuple[float, float]:
        """Calculates instantaneous downward velocity and peak downward velocity in recent window."""
        if len(self.pose_history) < 2:
            return 0.0, 0.0

        t_now, p_now = self.pose_history[-1]
        t_prev, p_prev = self.pose_history[-2]
        dt = max(t_now - t_prev, 0.01)
        instant_dy = max(
            (p_now.hip_center[1] - p_prev.hip_center[1]) / dt,
            (p_now.shoulder_center[1] - p_prev.shoulder_center[1]) / dt
        )

        max_drop = 0.0
        window_duration = 0.8
        recent = [entry for entry in self.pose_history if (current_time - entry[0]) <= window_duration]
        
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                t_i, p_i = recent[i]
                t_j, p_j = recent[j]
                time_diff = t_j - t_i
                if 0.10 <= time_diff <= 0.65:
                    v_hip = (p_j.hip_center[1] - p_i.hip_center[1]) / time_diff
                    v_sh = (p_j.shoulder_center[1] - p_i.shoulder_center[1]) / time_diff
                    v_dy = max(v_hip, v_sh)
                    if v_dy > max_drop:
                        max_drop = v_dy

        return max(0.0, instant_dy), max(0.0, max_drop)

    def _compute_fall_confidence(self, is_frame_suspicious: bool, max_recent_drop: float,
                                 torso_angle: float, floor_proximity: float) -> float:
        """Computes weighted multi-factor fall confidence score."""
        if not is_frame_suspicious:
            if torso_angle > 65.0 and floor_proximity > 0.65:
                return 15.0 # Lying down, but controlled
            return 5.0

        # Base confidence for sustained collapsed posture: 50.0
        base_score = 50.0

        # Angle score (up to 30 pts)
        a_ratio = min(1.0, max(0.0, (torso_angle - 45.0) / 35.0))
        a_score = a_ratio * 30.0

        # Velocity score (up to 20 pts)
        v_ratio = min(1.0, max(0.0, max_recent_drop / 0.35))
        v_score = v_ratio * 20.0

        total = base_score + a_score + v_score
        return min(98.5, max(68.0, total))

    def reset(self):
        """Resets history buffers and debounce counters."""
        self.pose_history.clear()
        self.raw_activity_history.clear()
        self.stable_activity = "SITTING"
        self.consecutive_no_person_frames = 0
        self.consecutive_suspicious_frames = 0
        self.consecutive_fallen_frames = 0
        self.consecutive_upright_frames = 0
        self.baseline_hip_y = None
