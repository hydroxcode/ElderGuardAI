"""
ElderGuard AI - Configuration
Privacy-preserving Edge AI for Elderly Fall Detection
Target Architecture: Qualcomm Snapdragon Edge / On-Device NPU
"""
from dataclasses import dataclass

@dataclass
class FallDetectionConfig:
    # Biomechanical & Kinematic Thresholds
    FALL_ANGLE_THRESHOLD: float = 52.0        # Deg from vertical (standing ~0-25°, sitting ~0-35°, fallen/slumped >52°)
    CRITICAL_ANGLE_THRESHOLD: float = 75.0    # Flat horizontal position
    FALL_VELOCITY_THRESHOLD: float = 0.32     # Minimum drop velocity spike required for a sudden fall (norm_h / sec)
    SLOW_DESCENT_MAX_VELOCITY: float = 0.28   # Threshold for controlled sitting/lying (below this is never a sudden fall)
    ASPECT_RATIO_FALL_THRESHOLD: float = 0.80 # H/W ratio (standing >1.3, sitting 0.8-1.4, fallen <0.80)
    FLOOR_PROXIMITY_THRESHOLD: float = 0.58   # Person down in lower region of room/frame (0.0=top, 1.0=bottom)
    
    # Temporal Smoothing & Debouncing (False Positive Prevention)
    REQUIRED_SUSPICIOUS_FRAMES: int = 5       # Consecutive frames meeting fall criteria before entering POSSIBLE_FALL (~0.18s)
    POSSIBLE_FALL_CONFIRM_FRAMES: int = 5     # Consecutive frames in POSSIBLE_FALL before triggering VERIFYING countdown
    COOLDOWN_DURATION_SEC: float = 5.0        # Cooldown period after false alarm / recovery to prevent retrigger loops
    SMOOTHING_ALPHA: float = 0.30             # Exponential moving average filter factor (0=freeze, 1=no smoothing)
    DROPOUT_HOLD_FRAMES: int = 25             # Retain last known skeleton across temporary landmark dropouts (~0.8s)
    NO_PERSON_TIMEOUT_FRAMES: int = 30        # Require continuous absence of person for 30 frames (~1.0s) before declaring NO_PERSON
    
    # Verification & Countdown
    COUNTDOWN_DURATION_SEC: int = 10          # 10s "ARE YOU OK?" countdown
    
    # Recovery Criteria (Self-recovery / False positive cancellation)
    RECOVERY_ANGLE_THRESHOLD: float = 40.0    # Returned to upright posture (angle <= 40°)
    RECOVERY_ASPECT_RATIO: float = 1.15       # Upright aspect ratio restored
    RECOVERY_CONSECUTIVE_FRAMES: int = 8      # Consecutive upright frames to confirm recovery

@dataclass
class SystemConfig:
    DEVICE_NAME: str = "ElderGuard Edge-Station (Qualcomm QCS / Snapdragon Architecture)"
    DEFAULT_RESIDENT: str = "Eleanor Vance (Age 78)"
    DEFAULT_LOCATION: str = "Living Room - Zone A"
    CAMERA_INDEX: int = 0
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    TARGET_FPS: int = 30
    
    # Privacy Defaults
    PRIVACY_MODE: bool = True
    ANONYMIZE_FACE: bool = True
    SKELETON_ONLY_VIEW: bool = False
    CLOUD_UPLOAD_ENABLED: bool = False  # Strictly False: Zero continuous cloud video streaming
    
    # Storage
    DB_PATH: str = "elderguard.db"

# Global default instances
fall_config = FallDetectionConfig()
sys_config = SystemConfig()
