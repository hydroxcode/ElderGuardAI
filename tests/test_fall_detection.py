"""
ElderGuard AI - Comprehensive Automated Test Suite
Verifies end-to-end functionality of pose detection, fall kinematics,
temporal verification, false positive reduction, state machine, debouncing, and SQLite database.
"""
import sys
import os
import time
import unittest
import numpy as np

# Ensure root path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import sys_config, fall_config
from core.database import EventDatabase
from core.pose_detector import PoseDetector, PoseResult
from core.fall_detector import FallDetector, FallAnalysisResult
from core.state_machine import EmergencyStateMachine, SystemState
from utils.demo_scenarios import ScenarioGenerator
from utils.visualizer import EdgeVisualizer

class TestElderGuardAI(unittest.TestCase):
    def setUp(self):
        # Use an isolated test database with unique filename per test
        self.test_db_path = f"test_elderguard_{self._testMethodName}_{int(time.time()*1000)}.db"
        self.db = EventDatabase(self.test_db_path)
        self.pose_detector = PoseDetector()
        self.fall_detector = FallDetector(fall_config)
        self.state_machine = EmergencyStateMachine(self.db, sys_config, fall_config)
        self.scenario_gen = ScenarioGenerator()
        self.visualizer = EdgeVisualizer()

    def tearDown(self):
        if hasattr(self, 'test_db_path') and os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def test_database_logging_and_retrieval(self):
        """Test SQLite event logging, status updates, and statistics aggregation."""
        event_id = "EVT-TEST-001"
        self.db.log_event(
            event_id=event_id,
            event_type="FALL_CONFIRMED",
            confidence=92.5,
            resident_name="Test Resident",
            location="Room 101",
            response_status="NO_RESPONSE_TIMEOUT",
            duration_seconds=10.0,
            metrics_summary="Angle: 85 deg, Drop: 0.42 m/s"
        )
        
        events = self.db.get_recent_events(limit=5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_id"], event_id)
        self.assertEqual(events[0]["confidence"], 92.5)
        self.assertEqual(events[0]["alert_status"], "ACTIVE")

        # Test status update
        self.db.update_event_status(event_id, "RESOLVED", "Caregiver attended")
        updated = self.db.get_recent_events(limit=5)
        self.assertEqual(updated[0]["alert_status"], "RESOLVED")
        self.assertEqual(updated[0]["response_status"], "Caregiver attended")

        # Test statistics
        stats = self.db.get_statistics()
        self.assertEqual(stats["total_events"], 1)
        self.assertEqual(stats["confirmed_emergencies"], 1)

    def test_scenario_generator_all_seven(self):
        """Verifies that all 7 scenarios synthesize valid frames and biomechanical landmarks."""
        for key in ScenarioGenerator.SCENARIOS.keys():
            frame, pose = self.scenario_gen.generate_frame(key, step=15)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (480, 640, 3))
            self.assertTrue(pose.detected, f"Pose should be detected for scenario {key}")
            self.assertGreater(pose.confidence, 0.0)

    def test_standing_and_walking_are_safe(self):
        """Verifies that standing and walking postures produce SAFE status with low fall confidence."""
        # Scenario 1: Standing
        _, pose_stand = self.scenario_gen.generate_frame("1_standing", step=10)
        res_stand = self.fall_detector.analyze(pose_stand)
        self.assertIn(res_stand.activity, ("STANDING", "SAFE_TRANSITION"))
        self.assertFalse(res_stand.is_fall_candidate)
        self.assertLess(res_stand.confidence, 30.0)

        # Scenario 2: Walking
        _, pose_walk = self.scenario_gen.generate_frame("2_walking", step=10)
        res_walk = self.fall_detector.analyze(pose_walk)
        self.assertIn(res_walk.activity, ("WALKING", "STANDING", "SAFE_TRANSITION"))
        self.assertFalse(res_walk.is_fall_candidate)

    def test_sitting_extended_period_remains_strictly_safe(self):
        """
        CRITICAL TEST: Verifies that sitting normally for 60 consecutive frames
        never triggers a fall candidate, never plays warning beeps, and remains 100% SAFE.
        """
        detector = FallDetector(fall_config)
        sm = EmergencyStateMachine(self.db, sys_config, fall_config)
        t = time.time()
        
        for step in range(60):
            # Seated posture in front of webcam: angle ~ 15°, aspect ratio ~ 1.0, low velocity
            _, pose = self.scenario_gen.generate_frame("3_sitting", step=45, total_steps=60)
            res = detector.analyze(pose, current_time=t + step * 0.033)
            state = sm.update(res, current_time=t + step * 0.033)

            self.assertFalse(res.is_fall_candidate, f"Frame {step} sitting wrongly triggered fall candidate!")
            self.assertEqual(state, SystemState.SAFE, f"Frame {step} sitting changed state to {state}!")
            self.assertLess(res.confidence, 35.0, f"Frame {step} sitting has high confidence {res.confidence}%")
            self.assertEqual(res.consecutive_suspicious_frames, 0)

    def test_slowly_sitting_down_is_safe(self):
        """Verifies that the transition of sitting down slowly does not trigger fall alerts."""
        detector = FallDetector(fall_config)
        sm = EmergencyStateMachine(self.db, sys_config, fall_config)
        t = time.time()

        for step in range(60):
            _, pose = self.scenario_gen.generate_frame("3_sitting", step=step, total_steps=60)
            res = detector.analyze(pose, current_time=t + step * 0.033)
            state = sm.update(res, current_time=t + step * 0.033)
            self.assertFalse(res.is_fall_candidate, f"Slow sitting step {step} triggered false alarm: {res.activity}")
            self.assertEqual(state, SystemState.SAFE)

    def test_intentional_lying_down_suppresses_false_positive(self):
        """Verifies that slow intentional lying down does not trigger acute emergency falls."""
        detector = FallDetector(fall_config)
        sm = EmergencyStateMachine(self.db, sys_config, fall_config)
        t = time.time()

        for step in range(60):
            _, pose = self.scenario_gen.generate_frame("4_lying_down", step=step, total_steps=60)
            res = detector.analyze(pose, current_time=t + step * 0.033)
            state = sm.update(res, current_time=t + step * 0.033)
            
            # Must remain SAFE, cannot be fall candidate
            self.assertFalse(res.is_fall_candidate, f"Lying down step {step} triggered fall candidate!")
            self.assertEqual(state, SystemState.SAFE)
            self.assertLess(res.confidence, 40.0)

    def test_transient_noise_clears_without_entering_verifying(self):
        """
        Tests debouncing: if 1 or 2 suspicious frames appear due to noise,
        it does NOT enter VERIFYING, does NOT beep, and returns to SAFE.
        """
        sm = EmergencyStateMachine(self.db, sys_config, fall_config)
        now = time.time()

        # Transient glitch frame (1 frame only)
        glitch_analysis = FallAnalysisResult(
            activity="POSSIBLE_FALL", is_fall_candidate=True, confidence=80.0,
            torso_angle=75.0, aspect_ratio=0.6, vertical_velocity=0.45,
            max_recent_drop_velocity=0.45, floor_proximity=0.75, is_recovering=False,
            consecutive_suspicious_frames=8, detailed_metrics={}
        )
        s1 = sm.update(glitch_analysis, current_time=now)
        self.assertEqual(s1, SystemState.POSSIBLE_FALL)

        # Next frame is clean again (noise cleared)
        clean_analysis = FallAnalysisResult(
            activity="SITTING", is_fall_candidate=False, confidence=8.0,
            torso_angle=12.0, aspect_ratio=1.1, vertical_velocity=0.0,
            max_recent_drop_velocity=0.0, floor_proximity=0.55, is_recovering=False,
            consecutive_suspicious_frames=0, detailed_metrics={}
        )
        s2 = sm.update(clean_analysis, current_time=now + 0.05)
        # MUST de-escalate cleanly: SAFE -> POSSIBLE_FALL -> SAFE without entering VERIFYING!
        self.assertEqual(s2, SystemState.SAFE)

    def test_fall_and_countdown_emergency_flow(self):
        """
        Tests the complete emergency pipeline:
        Persistent fall occurs -> POSSIBLE_FALL -> VERIFYING (10s countdown) -> Timeout -> EMERGENCY_CONFIRMED.
        """
        sm = EmergencyStateMachine(self.db, sys_config, fall_config)
        now = time.time()

        # Step 1: Normal safe monitoring
        safe_analysis = FallAnalysisResult(
            activity="STANDING", is_fall_candidate=False, confidence=5.0,
            torso_angle=5.0, aspect_ratio=2.1, vertical_velocity=0.0,
            max_recent_drop_velocity=0.0, floor_proximity=0.88, is_recovering=False,
            consecutive_suspicious_frames=0, detailed_metrics={}
        )
        self.assertEqual(sm.update(safe_analysis, current_time=now), SystemState.SAFE)

        # Step 2: Persistent Sudden Fall Candidate (Feeds confirmation frames)
        fall_analysis = FallAnalysisResult(
            activity="POSSIBLE_FALL", is_fall_candidate=True, confidence=92.0,
            torso_angle=82.0, aspect_ratio=0.5, vertical_velocity=0.45,
            max_recent_drop_velocity=0.48, floor_proximity=0.88, is_recovering=False,
            consecutive_suspicious_frames=8, detailed_metrics={}
        )
        
        # Frame 1: enters POSSIBLE_FALL
        sm.update(fall_analysis, current_time=now + 0.05)
        self.assertEqual(sm.current_state, SystemState.POSSIBLE_FALL)
        
        # Frames 2..7: confirms fall signature over confirmation window
        for f in range(fall_config.POSSIBLE_FALL_CONFIRM_FRAMES):
            sm.update(fall_analysis, current_time=now + 0.1 + f * 0.033)

        # Must have escalated to VERIFYING
        self.assertEqual(sm.current_state, SystemState.VERIFYING)
        self.assertIn(sm.countdown_seconds_remaining, (fall_config.COUNTDOWN_DURATION_SEC, fall_config.COUNTDOWN_DURATION_SEC - 1))

        # Step 3: Fast-forward time past 10 seconds without recovery
        timeout_time = now + 0.5 + fall_config.COUNTDOWN_DURATION_SEC + 1.0
        final_state = sm.update(fall_analysis, current_time=timeout_time)
        self.assertEqual(final_state, SystemState.EMERGENCY_CONFIRMED)
        self.assertEqual(sm.countdown_seconds_remaining, 0)

        # Verify emergency was written to SQLite database
        events = self.db.get_recent_events(limit=5)
        self.assertGreater(len(events), 0)
        self.assertEqual(events[0]["event_type"], "FALL_CONFIRMED")
        self.assertEqual(events[0]["response_status"], "NO_RESPONSE_TIMEOUT")

    def test_fall_with_automatic_recovery_and_cooldown(self):
        """
        Tests fall recovery pipeline and verifies that cooldown prevents immediate retrigger.
        """
        sm = EmergencyStateMachine(self.db, sys_config, fall_config)
        now = time.time()

        fall_analysis = FallAnalysisResult(
            activity="POSSIBLE_FALL", is_fall_candidate=True, confidence=90.0,
            torso_angle=82.0, aspect_ratio=0.5, vertical_velocity=0.45,
            max_recent_drop_velocity=0.48, floor_proximity=0.88, is_recovering=False,
            consecutive_suspicious_frames=8, detailed_metrics={}
        )
        # Advance into VERIFYING
        for f in range(fall_config.POSSIBLE_FALL_CONFIRM_FRAMES + 2):
            sm.update(fall_analysis, current_time=now + f * 0.033)
        self.assertEqual(sm.current_state, SystemState.VERIFYING)

        # Person stands up (Recovery signaled)
        recovery_analysis = FallAnalysisResult(
            activity="STANDING", is_fall_candidate=False, confidence=10.0,
            torso_angle=12.0, aspect_ratio=2.0, vertical_velocity=0.0,
            max_recent_drop_velocity=0.0, floor_proximity=0.88, is_recovering=True,
            consecutive_suspicious_frames=0, detailed_metrics={}
        )
        state_after_recovery = sm.update(recovery_analysis, current_time=now + 2.0)
        self.assertEqual(state_after_recovery, SystemState.RECOVERED)

        # Verify cooldown is active
        self.assertGreater(sm.cooldown_until, now + 2.0)

        # Transition back to SAFE after 3 seconds
        state_safe = sm.update(recovery_analysis, current_time=now + 5.5)
        self.assertEqual(state_safe, SystemState.SAFE)

        # While in cooldown, even a fall candidate cannot re-trigger immediately
        state_during_cooldown = sm.update(fall_analysis, current_time=now + 6.0)
        self.assertEqual(state_during_cooldown, SystemState.SAFE)

    def test_resident_manual_im_ok_button(self):
        """Tests that pressing [I'M OK] immediately cancels the emergency countdown."""
        sm = EmergencyStateMachine(self.db, sys_config, fall_config)
        now = time.time()

        fall_analysis = FallAnalysisResult(
            activity="POSSIBLE_FALL", is_fall_candidate=True, confidence=90.0,
            torso_angle=82.0, aspect_ratio=0.5, vertical_velocity=0.45,
            max_recent_drop_velocity=0.48, floor_proximity=0.88, is_recovering=False,
            consecutive_suspicious_frames=8, detailed_metrics={}
        )
        for f in range(fall_config.POSSIBLE_FALL_CONFIRM_FRAMES + 2):
            sm.update(fall_analysis, current_time=now + f * 0.033)
        self.assertEqual(sm.current_state, SystemState.VERIFYING)

        # Resident presses [I'M OK]
        sm.trigger_user_im_ok(current_time=now + 2.0)
        self.assertEqual(sm.current_state, SystemState.RECOVERED)

    def test_visualizer_rendering_all_states(self):
        """Verifies that EdgeVisualizer renders without error across all states and privacy modes."""
        frame, pose = self.scenario_gen.generate_frame("1_standing", step=5)
        analysis = FallAnalysisResult(
            activity="STANDING", is_fall_candidate=False, confidence=5.0,
            torso_angle=5.0, aspect_ratio=2.1, vertical_velocity=0.0,
            max_recent_drop_velocity=0.0, floor_proximity=0.88, is_recovering=False,
            consecutive_suspicious_frames=0, detailed_metrics={}
        )

        # Test normal view
        out1 = self.visualizer.render(frame, pose, analysis, SystemState.SAFE)
        self.assertEqual(out1.shape, (480, 640, 3))

        # Test privacy face blur
        out2 = self.visualizer.render(frame, pose, analysis, SystemState.SAFE, privacy_mode=True, anonymize_face=True)
        self.assertEqual(out2.shape, (480, 640, 3))

        # Test skeleton blackout mode
        out3 = self.visualizer.render(frame, pose, analysis, SystemState.SAFE, skeleton_only=True)
        self.assertEqual(out3.shape, (480, 640, 3))

        # Test emergency overlay
        out4 = self.visualizer.render(frame, pose, analysis, SystemState.EMERGENCY_CONFIRMED)
        self.assertEqual(out4.shape, (480, 640, 3))

if __name__ == "__main__":
    unittest.main()
