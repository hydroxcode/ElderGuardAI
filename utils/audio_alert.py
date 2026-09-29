"""
ElderGuard AI - Acoustic Alert Module
Emits local audible warning chimes during verification countdown and emergency sirens.
Uses native Windows sound with strict rate-limiting to prevent continuous beeping loops.
"""
import threading
import time
import sys

class SoundAlertManager:
    def __init__(self):
        self.muted = False
        self._is_beeping = False
        self._last_warning_time = 0.0
        self._last_emergency_time = 0.0

    def play_warning_chime(self, min_interval: float = 3.0):
        """Short double chirp during 'ARE YOU OK?' countdown (rate-limited)."""
        now = time.time()
        if self.muted or self._is_beeping or (now - self._last_warning_time) < min_interval:
            return
        self._last_warning_time = now
        threading.Thread(target=self._beep_worker, args=([(880, 80), (1174, 120)],), daemon=True).start()

    def play_emergency_siren(self, min_interval: float = 4.0):
        """Urgent alternating alarm during confirmed emergency (rate-limited)."""
        now = time.time()
        if self.muted or self._is_beeping or (now - self._last_emergency_time) < min_interval:
            return
        self._last_emergency_time = now
        pattern = [(1200, 150), (800, 150), (1200, 150), (800, 150)]
        threading.Thread(target=self._beep_worker, args=(pattern,), daemon=True).start()

    def _beep_worker(self, notes):
        self._is_beeping = True
        try:
            if sys.platform == "win32":
                import winsound
                for freq, dur in notes:
                    if self.muted:
                        break
                    winsound.Beep(freq, dur)
                    time.sleep(0.04)
        except Exception:
            pass
        finally:
            self._is_beeping = False
