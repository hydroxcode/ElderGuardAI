"""
ElderGuard AI - Native Edge Station (OpenCV Desktop Runtime)
Runs directly on device with ultra-low latency (< 15ms), 30+ FPS HUD,
keyboard triggers, and SQLite logging.
Ideal for fixed room cameras, edge devices, and immediate presentation.
"""
import cv2
import time
import sys
import numpy as np
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

def main():
    print("=" * 70)
    print("      ELDERGUARD AI — PRIVACY-PRESERVING EDGE STATION")
    print("   PS-06: Edge AI-Based Elderly Fall Detection (Navonmesh 26)")
    print("                  In Collaboration with Qualcomm")
    print("=" * 70)
    print("\n[INFO] Initializing On-Device Modules...")

    db = EventDatabase(sys_config.DB_PATH)
    db.seed_demo_history_if_empty()
    pose_detector = PoseDetector()
    fall_detector = FallDetector(fall_config)
    state_machine = EmergencyStateMachine(db, sys_config, fall_config)
    visualizer = EdgeVisualizer()
    scenario_gen = ScenarioGenerator(sys_config.CAMERA_WIDTH, sys_config.CAMERA_HEIGHT)
    audio_mgr = SoundAlertManager()
    camera_mgr = VideoCaptureManager(
        camera_index=sys_config.CAMERA_INDEX,
        width=sys_config.CAMERA_WIDTH,
        height=sys_config.CAMERA_HEIGHT
    )

    # Try starting camera
    print("[INFO] Starting Edge Camera Feed...")
    cam_started = camera_mgr.start()
    
    current_source = "LIVE" if cam_started else "1_standing"
    if not cam_started:
        print(f"[WARN] Webcam not detected or busy. Initializing in Demo Scenario Mode.")
    else:
        print("[SUCCESS] Live Webcam initialized successfully.")

    privacy_mode = True
    anonymize_face = True
    skeleton_only = False
    sim_step = 0

    window_name = "ElderGuard AI - Edge Fall Detection Station [Qualcomm Edge Target]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    print("\n" + "-" * 70)
    print("INTERACTIVE DEMO KEYBOARD CONTROLS:")
    print("  [SPACE] : Resident 'I'M OK' / Cancel Verification Alert")
    print("  [F]     : Simulate Sudden Fall Trigger")
    print("  [R]     : Reset Emergency / Return to SAFE")
    print("  [P]     : Toggle Face Privacy Mask (Blur)")
    print("  [B]     : Toggle Blackout Silhouette (Skeleton Only)")
    print("  [L]     : Switch to Live Laptop Webcam")
    print("  [1]     : Scenario 1: Person Standing (SAFE)")
    print("  [2]     : Scenario 2: Person Walking (SAFE)")
    print("  [3]     : Scenario 3: Person Sitting Down (SAFE - Controlled)")
    print("  [4]     : Scenario 4: Intentional Lie Down (SAFE - False Positive Test)")
    print("  [5]     : Scenario 5: Sudden Slip & Fall (POSSIBLE FALL)")
    print("  [6]     : Scenario 6: Fall with Recovery (RECOVERED)")
    print("  [7]     : Scenario 7: Fall without Recovery (10s Countdown -> EMERGENCY)")
    print("  [Q/ESC] : Quit Application")
    print("-" * 70 + "\n")

    scenario_map = {
        ord('1'): "1_standing",
        ord('2'): "2_walking",
        ord('3'): "3_sitting",
        ord('4'): "4_lying_down",
        ord('5'): "5_slip_fall",
        ord('6'): "6_fall_recovery",
        ord('7'): "7_fall_emergency"
    }

    last_time = time.time()
    fps_display = 30.0

    try:
        while True:
            t_now = time.time()
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(1e-4, t_now - last_time))
            last_time = t_now

            # Fetch Frame
            if current_source == "LIVE":
                success, frame = camera_mgr.read_frame()
                if not success or frame is None:
                    current_source = "1_standing"
                    frame, pose = scenario_gen.generate_frame("1_standing", sim_step)
                else:
                    pose = pose_detector.process_frame(frame)
            else:
                sim_step += 1
                frame, pose = scenario_gen.generate_frame(current_source, sim_step, total_steps=90)

            # Analyze Fall Kinematics
            analysis = fall_detector.analyze(pose, current_time=t_now)

            # Update State Machine
            state = state_machine.update(analysis, current_time=t_now)

            # Audio warnings
            if state == SystemState.VERIFYING and state_machine.countdown_seconds_remaining in (10, 7, 4, 2):
                audio_mgr.play_warning_chime()
            elif state == SystemState.EMERGENCY_CONFIRMED and int(t_now) % 3 == 0:
                audio_mgr.play_emergency_siren()

            # Render Overlay HUD
            display_frame = visualizer.render(
                frame=frame,
                pose=pose,
                analysis=analysis,
                state=state,
                countdown_sec=state_machine.countdown_seconds_remaining,
                privacy_mode=privacy_mode,
                anonymize_face=anonymize_face,
                skeleton_only=skeleton_only
            )

            # Bottom Status Bar with Keyboard Legend
            dh, dw = display_frame.shape[:2]
            bar_overlay = display_frame.copy()
            cv2.rectangle(bar_overlay, (0, dh - 32), (dw, dh), (15, 15, 20), -1)
            cv2.addWeighted(bar_overlay, 0.9, display_frame, 0.1, 0, display_frame)
            
            legend_text = (f"SOURCE: {current_source} | FPS: {fps_display:.1f} | "
                           f"[SPACE] I'M OK  [F] Fall  [R] Reset  [P] Privacy  [1-7] Scenarios  [Q] Quit")
            cv2.putText(display_frame, legend_text, (15, dh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 240), 1, cv2.LINE_AA)

            # Display Window
            cv2.imshow(window_name, display_frame)

            # Process Key Commands
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), ord('Q'), 27): # Quit
                break
            elif key == 32: # SPACE: I'M OK
                print("[EVENT] Resident pressed [I'M OK]. Cancelling alert...")
                state_machine.trigger_user_im_ok()
            elif key in (ord('f'), ord('F')): # Fall trigger
                print("[EVENT] Manual Fall simulation triggered.")
                state_machine.current_state = SystemState.POSSIBLE_FALL
                state_machine.fall_detected_time = time.time()
                state_machine.last_confidence = 94.0
            elif key in (ord('r'), ord('R')): # Reset
                print("[EVENT] Reset to SAFE monitoring.")
                state_machine.resolve_emergency()
            elif key in (ord('p'), ord('P')): # Privacy Face Mask
                anonymize_face = not anonymize_face
                print(f"[CONFIG] Privacy Face Mask: {'ON' if anonymize_face else 'OFF'}")
            elif key in (ord('b'), ord('B')): # Blackout silhouette
                skeleton_only = not skeleton_only
                print(f"[CONFIG] Silhouette Blackout Mode: {'ON' if skeleton_only else 'OFF'}")
            elif key in (ord('l'), ord('L')): # Live webcam
                if not camera_mgr.is_running:
                    camera_mgr.start()
                current_source = "LIVE"
                print("[SOURCE] Switched to Live Laptop Webcam.")
            elif key in scenario_map:
                current_source = scenario_map[key]
                sim_step = 0
                state_machine.resolve_emergency()
                print(f"[SOURCE] Switched to {current_source}")

    finally:
        camera_mgr.stop()
        cv2.destroyAllWindows()
        print("\n[INFO] ElderGuard Edge Station cleanly terminated.")

if __name__ == "__main__":
    main()
