#!/usr/bin/env python3
"""test_tuner_safety.py - the tuner's failure detection must catch the layouts that break the engine.

Grew out of a real failure: `--host-budget-gib 8` beside a helper on a 16 GB primary made the driver refuse part of
the page-lock (`cudaHostRegister refused chunks`) and then broke a later device allocation (`the R4 hit path could
not allocate its device buffers`).  The engine kept running, so the bench saw a working run - the tuner must not
treat that as a successful layout, and must not write back a budget the machine never actually held.

Run: python tools/test_tuner_safety.py
"""
import unittest
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent / "cascade_bench"))
import tune_cascade as tune


# The exact lines a bad pin produced on the reference rig (Windows, 16 GB primary + 12 GB helper, --host-budget-gib 8).
BAD_PIN_LOG = """\
strata generate: expert cache auto: 9.61 GiB free, 1000 MiB reserved (+73 MiB for the draft head) -> 3945 slots
strata generate: only 0 MiB free once the slots are written (reserve 1000 MiB); shrinking the expert cache
strata generate: only 661 MiB free once the slots are written (reserve 1000 MiB); shrinking the expert cache
strata generate: CUDA1: 6250 additional experts, 10.11 GiB; results return through pinned host rows
strata generate: tiers: VRAM 10629 experts (17.24 GiB, 6250 held by another GPU, left to the standby list) | PINNED 4925 (8.00 GiB, 8 runs, 8.00 GiB locked) | COLD 9022 (14.73 GiB, paged from SSD) | budget 8.00 GiB | 4.9 s; cudaHostRegister refused chunks: out of memory
strata generate: the R4 hit path could not allocate its device buffers
"""

# A healthy run: pinned exactly as asked, no failure lines.
GOOD_LOG = """\
strata generate: CUDA1: 6250 additional experts, 10.11 GiB; results return through pinned host rows
strata generate: tiers: VRAM 10672 experts (17.31 GiB, 6250 held by another GPU, left to the standby list) | PINNED 3662 (6.00 GiB, 7 runs, 6.00 GiB locked) | COLD 10242 (16.66 GiB, paged from SSD) | budget 6.00 GiB | 4.2 s
strata serve: decode expert cache hit rate: 86.3% (1687679 hits / 1954583 lookups)
"""

# Linux demotes a refused run to COLD and reports a smaller PINNED than requested (no crash line).
LINUX_CLAMPED_LOG = """\
strata generate: CUDA1: 6250 additional experts, 10.11 GiB; results return through pinned host rows
strata generate: tiers: VRAM 10629 experts (17.24 GiB, 6250 held by another GPU, left to the standby list) | PINNED 2800 (4.55 GiB, 6 runs, 4.55 GiB locked) | COLD 9032 (14.73 GiB, paged from SSD) | budget 6.00 GiB | 4.9 s; cudaHostRegister refused runs: operation not supported
"""


class OomReason(unittest.TestCase):
    def _reason(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8") as f:
            f.write(text)
            p = f.name
        try:
            return tune.oom_reason(p)
        finally:
            Path(p).unlink(missing_ok=True)

    def test_catches_hostregister_refused(self):
        # the pin could not be fully locked: the engine carried on partially pinned
        self.assertIsNotNone(self._reason(BAD_PIN_LOG))

    def test_catches_device_buffer_alloc_failure(self):
        r = self._reason("strata generate: the R4 hit path could not allocate its device buffers\n")
        self.assertIsNotNone(r)

    def test_catches_shrinking_cache(self):
        self.assertIsNotNone(self._reason("strata generate: only 0 MiB free once the slots are written\n"))

    def test_catches_out_of_memory(self):
        self.assertIsNotNone(self._reason("strata generate: CUDA error: out of memory\n"))

    def test_catches_linux_anonymous_fill(self):
        self.assertIsNotNone(self._reason("strata: the anonymous fill failed\n"))

    def test_clean_log_is_not_flagged(self):
        self.assertIsNone(self._reason(GOOD_LOG))


class PinClamped(unittest.TestCase):
    def _clamped(self, text, requested):
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8") as f:
            f.write(text)
            p = f.name
        try:
            return tune.pin_clamped(p, requested)
        finally:
            Path(p).unlink(missing_ok=True)

    def test_linux_clamped_pin_is_flagged(self):
        # asked for 6, the engine only pinned 4.55 (a refused run demoted to COLD) -> flag it
        self.assertIsNotNone(self._clamped(LINUX_CLAMPED_LOG, 6.0))

    def test_achieved_matches_request_is_ok(self):
        self.assertIsNone(self._clamped(GOOD_LOG, 6.0))

    def test_small_rounding_gap_is_tolerated(self):
        # blob padding can land the pin a little under budget; that is not a failure
        self.assertIsNone(self._clamped(GOOD_LOG, 6.3))

    def test_no_tiers_line_is_ok(self):
        self.assertIsNone(self._clamped("nothing here\n", 6.0))

    def test_none_request_is_ok(self):
        self.assertIsNone(self._clamped(GOOD_LOG, None))


class EndToEndRowFlagging(unittest.TestCase):
    """One layout's row must be zeroed and flagged when the log shows either failure."""

    def test_bad_pin_log_would_be_flagged(self):
        # both detectors fire on the real failure, so one_layout's guard (oom first, then clamped) cannot miss it
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8") as f:
            f.write(BAD_PIN_LOG)
            p = f.name
        try:
            self.assertIsNotNone(tune.oom_reason(p), "the 8 GiB failure must be caught")
        finally:
            Path(p).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
