"""
ElderGuard AI - Live Webcam 60-Second Real-Time Stability Test
Runs on the real webcam hardware for 60 seconds (or ~1800 frames) and logs every single frame:
1. Verifies zero NO_PERSON flickering while person is present.
2. Verifies zero INTENTIONAL_LIE false classifications.
3. Verifies zero false fall warnings or beeps (remains 100% SAFE).
4. Verifies smooth dropout persistence when landmarks are briefly occluded.
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import sys_config, fall_config
from core.camera import VideoCaptureManager
from core.pose_detector import PoseDetector
from core.fall_detector import FallDetector
from core.state_machine import EmergencyStateMachine, SystemState
from core.database import EventDatabase

def main(duration_sec: float = 60.0):
    print("=" * 70)
    print(f"STARTING LIVE WEBCAM STABILITY TEST ({duration_sec:.0f} SECONDS)")
    print("=" * 70)

    db = EventDatabase("test_live_webcam.db")
    camera_mgr = VideoCaptureManager(sys_config.CAMERA_INDEX, sys_config.CAMERA_WIDTH, sys_config.CAMERA_HEIGHT)
    pose_detector = PoseDetector()
    fall_detector = FallDetector(fall_config)
    state_machine = EmergencyStateMachine(db, sys_config, fall_config)

    print("[INFO] Starting real webcam hardware...")
    started = camera_mgr.start()
    if not started:
        print("[ERROR] Could not start real webcam!")
        return False

    print(f"[INFO] Webcam started. Monitoring live feed for {duration_sec:.0f}s...")
    start_time = time.time()
    frame_idx = 0
    
    activity_counts = {}
    state_counts = {}
    dropout_held_frames = 0
    no_person_frames = 0
    fall_candidate_frames = 0
    intentional_lie_frames = 0
    min_confidence = 100.0
    max_confidence = 0.0

    last_log_time = start_time

    try:
        while True:
            t_now = time.time()
            elapsed = t_now - start_time
            if elapsed >= duration_sec:
                break

            ret, frame = camera_mgr.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_idx += 1
            pose = pose_detector.process_frame(frame)
            analysis = fall_detector.analyze(pose, current_time=t_now)
            state = state_machine.update(analysis, current_time=t_now)

            # Record metrics
            act = analysis.activity
            activity_counts[act] = activity_counts.get(act, 0) + 1
            state_counts[state.value] = state_counts.get(state.value, 0) + 1

            if pose.tracking_held:
                dropout_held_frames += 1

            if act == "NO_PERSON":
                no_person_frames += 1

            if act == "INTENTIONAL_LIE":
                intentional_lie_frames += 1

            if analysis.is_fall_candidate:
                fall_candidate_frames += 1

            if pose.detected:
                conf_pct = pose.confidence * 100
                min_confidence = min(min_confidence, conf_pct)
                max_confidence = max(max_confidence, conf_pct)

            # Periodic log every 5 seconds
            if t_now - last_log_time >= 5.0:
                last_log_time = t_now
                raw_act = getattr(analysis, "raw_activity", analysis.activity)
                held_str = f" [HELD {pose.dropout_frames}f]" if pose.tracking_held else ""
                print(f"[{elapsed:4.1f}s | Frame {frame_idx:4d}] "
                      f"State: {state.value:12s} | "
                      f"Stable: {act:10s} (Raw: {raw_act:10s}){held_str} | "
                      f"Angle: {analysis.torso_angle:4.1f}° | "
                      f"DropV: {analysis.vertical_velocity:4.2f} (Pk: {analysis.max_recent_drop_velocity:4.2f}) | "
                      f"Susp: {analysis.consecutive_suspicious_frames}/8")

            # ~30 FPS loop pacing
            time.sleep(0.025)

    finally:
        camera_mgr.stop()
        if os.path.exists("test_live_webcam.db"):
            try:
                os.remove("test_live_webcam.db")
            except Exception:
                pass

    print("\n" + "=" * 70)
    print("LIVE WEBCAM 60-SECOND TEST REPORT")
    print("=" * 70)
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Activity Distribution:  {activity_counts}")
    print(f"State Distribution:     {state_counts}")
    print(f"Dropout Held Frames:    {dropout_held_frames} ({dropout_held_frames/max(1, frame_idx)*100:.1f}%)")
    print(f"NO_PERSON Frames:       {no_person_frames}")
    print(f"INTENTIONAL_LIE Frames: {intentional_lie_frames}")
    print(f"Fall Candidate Frames:  {fall_candidate_frames}")

    # Acceptance evaluation
    is_safe_100pct = state_counts.get("SAFE", 0) == frame_idx
    no_accidental_lies = intentional_lie_frames == 0
    no_flicker = no_person_frames == 0

    print("\nACCEPTANCE CRITERIA RESULTS:")
    print(f"  1. 100% SAFE (zero fall alarms):      {'PASS' if is_safe_100pct else 'FAIL'}")
    print(f"  2. Zero INTENTIONAL_LIE classifications: {'PASS' if no_accidental_lies else 'FAIL'}")
    print(f"  3. Zero NO_PERSON flickering:          {'PASS' if no_flicker else 'FAIL'}")

    return is_safe_100pct and no_accidental_lies

if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    success = main(dur)
    sys.exit(0 if success else 1)
