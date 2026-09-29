"""
ElderGuard AI - Robust Emergency State Machine
Coordinates stable state transitions with multi-frame confirmation,
automatic false-positive de-escalation, and post-recovery cooldown periods.
"""
import time
import uuid
from enum import Enum
from typing import Optional, Dict, Any
from core.fall_detector import FallAnalysisResult
from core.database import EventDatabase
from config import SystemConfig, sys_config, FallDetectionConfig, fall_config

class SystemState(Enum):
    SAFE = "SAFE"
    POSSIBLE_FALL = "POSSIBLE_FALL"
    VERIFYING = "VERIFYING"
    RECOVERED = "RECOVERED"
    EMERGENCY_CONFIRMED = "EMERGENCY_CONFIRMED"

class EmergencyStateMachine:
    def __init__(self, db: EventDatabase, sys_cfg: SystemConfig = sys_config,
                 fall_cfg: FallDetectionConfig = fall_config):
        self.db = db
        self.sys_cfg = sys_cfg
        self.fall_cfg = fall_cfg

        self.current_state = SystemState.SAFE
        self.current_event_id: Optional[str] = None
        self.fall_detected_time: Optional[float] = None
        self.countdown_start_time: Optional[float] = None
        self.countdown_seconds_remaining: int = fall_cfg.COUNTDOWN_DURATION_SEC
        self.last_confidence: float = 0.0
        self.last_analysis: Optional[FallAnalysisResult] = None
        self.state_message: str = "Monitoring Active — Resident Safe"
        self.last_status_change_time: float = time.time()
        self.acknowledged_by_caregiver: bool = False

        # Temporal Debounce & Cooldown Timers
        self.possible_fall_frames: int = 0
        self.cooldown_until: float = 0.0

    def update(self, analysis: FallAnalysisResult, current_time: Optional[float] = None) -> SystemState:
        if current_time is None:
            current_time = time.time()

        self.last_analysis = analysis
        self.last_confidence = analysis.confidence
        is_in_cooldown = current_time < self.cooldown_until

        # -------------------------------------------------------------
        # STATE 1: SAFE (Normal Continuous Monitoring)
        # -------------------------------------------------------------
        if self.current_state == SystemState.SAFE:
            self.possible_fall_frames = 0
            if is_in_cooldown:
                cooldown_left = max(0.0, self.cooldown_until - current_time)
                self.state_message = f"Monitoring Active — Stabilizing ({cooldown_left:.1f}s cooldown)"
            else:
                self.state_message = "Monitoring Active — Resident Safe"

            # Transition to POSSIBLE_FALL only if strict multi-frame candidate is met AND not in cooldown
            if analysis.is_fall_candidate and not is_in_cooldown:
                self.current_state = SystemState.POSSIBLE_FALL
                self.fall_detected_time = current_time
                self.possible_fall_frames = 1
                self.state_message = "⚠ Suspicious posture detected. Verifying stability..."
                self.last_status_change_time = current_time

        # -------------------------------------------------------------
        # STATE 2: POSSIBLE_FALL (Pre-Verification Confirmation)
        # Allows SAFE -> POSSIBLE_FALL -> SAFE without sounding alarms!
        # -------------------------------------------------------------
        elif self.current_state == SystemState.POSSIBLE_FALL:
            if analysis.is_fall_candidate:
                self.possible_fall_frames += 1
                # If fall signature persists for confirmation window, escalate to VERIFYING
                if self.possible_fall_frames >= self.fall_cfg.POSSIBLE_FALL_CONFIRM_FRAMES:
                    self.current_state = SystemState.VERIFYING
                    self.countdown_start_time = current_time
                    self.countdown_seconds_remaining = self.fall_cfg.COUNTDOWN_DURATION_SEC
                    self.current_event_id = f"EVT-{str(uuid.uuid4())[:8].upper()}"
                    self.state_message = "ARE YOU OK? Please confirm safety or stand up."
                    self.last_status_change_time = current_time
            else:
                # De-escalate immediately back to SAFE (glitch or posture cleared)
                self.current_state = SystemState.SAFE
                self.possible_fall_frames = 0
                self.state_message = "Monitoring Active — Resident Safe (False trigger cleared)"
                self.last_status_change_time = current_time

        # -------------------------------------------------------------
        # STATE 3: VERIFYING ("ARE YOU OK?" 10-Second Countdown)
        # -------------------------------------------------------------
        elif self.current_state == SystemState.VERIFYING:
            elapsed = current_time - (self.countdown_start_time or current_time)
            remaining = max(0, int(self.fall_cfg.COUNTDOWN_DURATION_SEC - elapsed))
            self.countdown_seconds_remaining = remaining

            # Branch A: Automatic Recovery Detected (Person stood up)
            if analysis.is_recovering:
                self._handle_recovery(source="AUTO_STANDING_RECOVERY", current_time=current_time)
                return self.current_state

            # Branch B: Countdown Reached 0 (No response from resident)
            if remaining <= 0:
                self._confirm_emergency(current_time=current_time)
                return self.current_state

            self.state_message = f"ARE YOU OK? Countdown: {remaining}s remaining"

        # -------------------------------------------------------------
        # STATE 4: RECOVERED (Temporary Confirmation Banner)
        # -------------------------------------------------------------
        elif self.current_state == SystemState.RECOVERED:
            # Display confirmation for 2.5 seconds, then return to SAFE with cooldown
            if (current_time - self.last_status_change_time) > 2.5:
                self.current_state = SystemState.SAFE
                self.state_message = "Monitoring Active — Resident Safe"
                self.last_status_change_time = current_time
                self.current_event_id = None
                self.cooldown_until = current_time + self.fall_cfg.COOLDOWN_DURATION_SEC

        # -------------------------------------------------------------
        # STATE 5: EMERGENCY_CONFIRMED
        # -------------------------------------------------------------
        elif self.current_state == SystemState.EMERGENCY_CONFIRMED:
            if analysis.is_recovering and not self.acknowledged_by_caregiver:
                self.state_message = "🚨 Resident stood up after emergency alert. Reviewing status."

        return self.current_state

    def trigger_user_im_ok(self, current_time: Optional[float] = None):
        """Called when resident or caregiver clicks [I'M OK] button."""
        if current_time is None:
            current_time = time.time()
        if self.current_state in (SystemState.POSSIBLE_FALL, SystemState.VERIFYING):
            self._handle_recovery(source="MANUAL_RESIDENT_CONFIRMATION", current_time=current_time)
        elif self.current_state == SystemState.EMERGENCY_CONFIRMED:
            self.acknowledge_emergency(note="False alarm cleared by resident (I'M OK)")
            self.resolve_emergency(current_time=current_time)

    def acknowledge_emergency(self, note: str = "Acknowledged by Caregiver"):
        """Called when caregiver clicks [ACKNOWLEDGE]."""
        self.acknowledged_by_caregiver = True
        self.state_message = f"Alert Acknowledged: {note}"
        if self.current_event_id:
            self.db.update_event_status(self.current_event_id, "ACKNOWLEDGED", note)

    def resolve_emergency(self, current_time: Optional[float] = None):
        """Reset emergency back to SAFE state with cooldown."""
        if current_time is None:
            current_time = time.time()
        if self.current_event_id:
            self.db.update_event_status(self.current_event_id, "RESOLVED", "Resident attended and safe")
        self.current_state = SystemState.SAFE
        self.state_message = "Monitoring Active — Resident Safe"
        self.current_event_id = None
        self.countdown_seconds_remaining = self.fall_cfg.COUNTDOWN_DURATION_SEC
        self.acknowledged_by_caregiver = False
        self.possible_fall_frames = 0
        self.cooldown_until = current_time + self.fall_cfg.COOLDOWN_DURATION_SEC

    def _handle_recovery(self, source: str, current_time: float):
        duration = current_time - (self.fall_detected_time or current_time)
        self.current_state = SystemState.RECOVERED
        self.last_status_change_time = current_time
        self.possible_fall_frames = 0
        self.cooldown_until = current_time + self.fall_cfg.COOLDOWN_DURATION_SEC
        self.state_message = f"Incident Cancelled: Recovery verified ({source})"
        
        # Log to local Edge Database
        if self.current_event_id:
            summary = (f"Recovery via {source}. Posture verified upright "
                       f"(Angle: {self.last_analysis.torso_angle if self.last_analysis else 'N/A'}°)")
            self.db.log_event(
                event_id=self.current_event_id,
                event_type="RECOVERED",
                confidence=self.last_confidence,
                resident_name=self.sys_cfg.DEFAULT_RESIDENT,
                location=self.sys_cfg.DEFAULT_LOCATION,
                response_status=f"CANCELLED_{source}",
                duration_seconds=round(duration, 1),
                metrics_summary=summary,
                alert_status="RESOLVED"
            )

    def _confirm_emergency(self, current_time: float):
        self.current_state = SystemState.EMERGENCY_CONFIRMED
        self.last_status_change_time = current_time
        self.acknowledged_by_caregiver = False
        duration = current_time - (self.fall_detected_time or current_time)
        self.state_message = "🚨 EMERGENCY: Fall confirmed! Immediate assistance needed."

        metrics_desc = (f"Fall confirmed with {self.last_confidence}% confidence. "
                        f"Torso angle: {self.last_analysis.torso_angle if self.last_analysis else 'N/A'}°, "
                        f"Peak drop: {self.last_analysis.max_recent_drop_velocity if self.last_analysis else 'N/A'} m/s, "
                        f"Response: No response within {self.fall_cfg.COUNTDOWN_DURATION_SEC}s.")
        
        self.db.log_event(
            event_id=self.current_event_id or f"EVT-{str(uuid.uuid4())[:8].upper()}",
            event_type="FALL_CONFIRMED",
            confidence=self.last_confidence,
            resident_name=self.sys_cfg.DEFAULT_RESIDENT,
            location=self.sys_cfg.DEFAULT_LOCATION,
            response_status="NO_RESPONSE_TIMEOUT",
            duration_seconds=round(duration, 1),
            metrics_summary=metrics_desc,
            alert_status="ACTIVE"
        )
