"""Tests for setup.py's Strata-Cascade integration (--low-ram tiered): engine detection, the auto default, the
helper-cache sizing, and the guard that upstream (non-ported) engines keep their exact behaviour.  Mocked - no GPU,
no downloads.

    python -m tools.test_setup_tiered
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class EngineDetection(unittest.TestCase):
    def tearDown(self):
        setup._TIERED_PROBE.clear()

    def test_no_engine_is_not_ported(self):
        self.assertFalse(setup.tiered_engine(None))

    def test_probe_reads_tiered_experts_from_help(self):
        with mock.patch.object(setup.Path, "exists", return_value=True), \
             mock.patch.object(setup.subprocess, "run") as run:
            run.return_value = mock.Mock(stdout="  --tiered-experts     the VRAM / pinned RAM / SSD tiers\n",
                                         stderr="")
            self.assertTrue(setup.tiered_engine(ROOT / "engine"))
            self.assertEqual(run.call_count, 1)
            # the probe is cached: a second call must not run the binary again
            self.assertTrue(setup.tiered_engine(ROOT / "engine"))
            self.assertEqual(run.call_count, 1)

    def test_an_upstream_engine_is_not_ported(self):
        with mock.patch.object(setup.Path, "exists", return_value=True), \
             mock.patch.object(setup.subprocess, "run") as run:
            run.return_value = mock.Mock(stdout="  --resident-experts   keep the experts in RAM\n", stderr="")
            self.assertFalse(setup.tiered_engine(ROOT / "engine"))

    def test_a_failing_probe_is_not_ported(self):
        with mock.patch.object(setup.Path, "exists", return_value=True), \
             mock.patch.object(setup.subprocess, "run", side_effect=OSError("boom")):
            self.assertFalse(setup.tiered_engine(ROOT / "engine"))


class ShouldDefault(unittest.TestCase):
    def card(self, i, vram, arch="120"):
        return {"index": i, "name": f"NVIDIA GeForce RTX {i}", "vram_gb": vram, "arch": arch, "driver": "580.97"}

    def test_low_ram_two_gpus_defaults_to_tiered(self):
        self.assertTrue(setup.tiered_should_default("IQ3_XXS", 32, [self.card(0, 16.0), self.card(1, 12.0, "86")]))

    def test_one_gpu_does_not(self):
        self.assertFalse(setup.tiered_should_default("IQ3_XXS", 32, [self.card(0, 16.0)]))

    def test_enough_ram_does_not(self):
        self.assertFalse(setup.tiered_should_default("IQ3_XXS", 256, [self.card(0, 16.0), self.card(1, 12.0, "86")]))


class HelperCounts(unittest.TestCase):
    def test_one_count_per_card_after_the_first(self):
        counts = setup.tiered_helper_counts("IQ3_XXS", [{"vram_gb": 16.0}, {"vram_gb": 11.9}])
        self.assertEqual(len(counts), 1)                       # only GPU 1 is a helper
        self.assertGreater(counts[0], 1000)                    # ~6k for a 12 GB card, not a nonsense number

    def test_bigger_quant_needs_more_bytes_per_expert(self):
        small = setup.tiered_helper_counts("IQ2_XS", [{"vram_gb": 16.0}, {"vram_gb": 12.0}])[0]
        big = setup.tiered_helper_counts("IQ3_S", [{"vram_gb": 16.0}, {"vram_gb": 12.0}])[0]
        self.assertGreater(small, big)                         # smaller quant -> more experts in the same VRAM

    def test_no_second_card_gives_no_counts(self):
        self.assertEqual(setup.tiered_helper_counts("IQ3_XXS", [{"vram_gb": 16.0}]), [])

    def test_helper_uses_real_free_vram_when_known(self):
        # hybrid: a card with little free VRAM (a display card, another app) gets a smaller cache than its total
        full = setup.tiered_helper_counts("IQ3_XXS", [{"vram_gb": 16.0}, {"vram_gb": 12.0, "vram_free_gb": 11.0}])[0]
        busy = setup.tiered_helper_counts("IQ3_XXS", [{"vram_gb": 16.0}, {"vram_gb": 12.0, "vram_free_gb": 3.0}])[0]
        self.assertLess(busy, full)                            # the free figure lowers it
        self.assertGreater(busy, 0)                            # but it stays a usable cache

    def test_helper_caps_at_82_percent_even_when_free_is_high(self):
        # free can read higher than what the engine accepts (CUDA context, staging); the 82% cap must hold
        capped = setup.tiered_helper_counts("IQ3_XXS", [{"vram_gb": 16.0}, {"vram_gb": 12.0, "vram_free_gb": 12.0}])[0]
        tot = setup.tiered_helper_counts("IQ3_XXS", [{"vram_gb": 16.0}, {"vram_gb": 12.0}])[0]
        self.assertEqual(capped, tot)                          # no more than the no-free 82% result

    def test_helper_without_free_falls_back_to_82_percent(self):
        # Windows AMD reports no free figure: the total-based 82% cap is used
        counts = setup.tiered_helper_counts("IQ3_XXS", [{"vram_gb": 16.0}, {"vram_gb": 12.0, "vram_free_gb": None}])
        self.assertGreater(counts[0], 1000)

    def test_budget_is_capped_beside_a_helper(self):
        # a helper shares the machine with CUDA0's cache: the pinned budget is capped, not left at auto
        self.assertEqual(setup.cascade_host_budget([6250]),
                         "%g" % setup.CASCADE_BUDGET_CAP_GIB)
        self.assertNotEqual(setup.cascade_host_budget([6250]), "auto")

    def test_budget_scales_down_on_a_small_pc(self):
        # 16 GB of RAM with ~13 free and an 8 GiB reserve leaves ~5: a fixed 6 GiB would over-pin
        small = float(setup.cascade_host_budget([7000], avail_gb=13, primary_gib=16))
        self.assertLess(small, setup.CASCADE_BUDGET_CAP_GIB)
        self.assertGreaterEqual(small, 2.0)
        # a big PC with plenty free keeps the measured 6 GiB
        self.assertEqual(float(setup.cascade_host_budget([6250], avail_gb=26, primary_gib=16)),
                         setup.CASCADE_BUDGET_CAP_GIB)

    def test_budget_uses_real_available_ram(self):
        # the same PC, busier: less free RAM -> a smaller pin (the engine's own auto rule: avail - reserve)
        idle = float(setup.cascade_host_budget([7000], avail_gb=20, primary_gib=16))
        busy = float(setup.cascade_host_budget([7000], avail_gb=12, primary_gib=16))
        self.assertLess(busy, idle)

    def test_budget_never_below_a_useful_pin(self):
        # even a very small PC keeps at least 2 GiB, and the value is always parseable by the engine
        val = setup.cascade_host_budget([7000], avail_gb=5, primary_gib=8)
        self.assertGreaterEqual(float(val), 2.0)

    def test_budget_is_auto_on_one_card(self):
        # no helper: `auto` lets the engine size the pin from MemAvailable, and more pinned is better
        self.assertEqual(setup.cascade_host_budget([]), "auto")
        self.assertEqual(setup.cascade_host_budget([0]), "auto")
        # and still `auto` on one card even with the RAM/VRAM arguments
        self.assertEqual(setup.cascade_host_budget([0], avail_gb=8, primary_gib=8), "auto")

    def test_cap_is_below_the_engine_pin_cap(self):
        # the engine's own page-lock cap is 8 GiB; the cascade default sits under it
        self.assertLess(setup.CASCADE_BUDGET_CAP_GIB, 8.0)


class Guards(unittest.TestCase):
    def test_split_mmap_leaves_a_tiered_config_alone(self):
        cfg = {"args": ["--tiered-experts", "--host-budget-gib", "auto"]}
        self.assertFalse(setup.split_mmap(cfg))
        self.assertIn("--tiered-experts", cfg["args"])
        self.assertNotIn("--mmap-experts", cfg["args"])

    def test_split_mmap_still_rewrites_a_resident_config(self):
        cfg = {"args": ["--resident-experts"]}
        self.assertTrue(setup.split_mmap(cfg))
        self.assertEqual(cfg["args"], ["--mmap-experts"])


class EndToEnd(unittest.TestCase):
    """setup.main() on a mocked 32 GB / two-GPU PC: the ported engine gets --tiered-experts, an upstream one does
    not.  Reuses test_setup_golden's harness so the outside effects are mocked (no GPU, no downloads)."""

    def test_ported_engine_writes_a_tiered_config(self):
        import tools.test_setup_golden as golden
        ram, found = golden.PROFILES["32GB-2x24GB"]
        with mock.patch.object(setup, "tiered_engine", lambda eng: True):
            code, out, cfg, asked = golden.install(ram, found, golden.argv_for("qwen", "IQ3_XXS"))
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("--tiered-experts", cfg["args"])
        self.assertIn("--host-budget-gib", cfg["args"])
        # a helper cache for the second card is present
        self.assertTrue(any(a.startswith("--expert-cache-device") for a in cfg["args"]))
        # never both the cascade and the resident/mmap flags (the engine refuses the pair)
        self.assertNotIn("--resident-experts", cfg["args"])
        self.assertNotIn("--mmap-experts", cfg["args"])

    def test_install_offers_the_cascade_tuner(self):
        """START-HERE (interactive, no --yes/--no-start) asks, and answering yes runs the tuner with the free RAM and
        the second card's usable VRAM."""
        import tools.test_setup_golden as golden
        ram, found = golden.PROFILES["32GB-2x24GB"]
        tuned = []

        def fake_tune(cfg_path, cfg, free_gib, helper_gibs):
            tuned.append((free_gib, helper_gibs))
            return True

        with mock.patch.object(setup, "tiered_engine", lambda eng: True), \
             mock.patch.object(setup, "tune_cascade_config", fake_tune), \
             mock.patch.object(setup, "start", lambda *a, **k: 0):
            code, out, cfg, asked = golden.install(
                ram, found, ["--family", "qwen", "--model", "IQ3_XXS"],
                answers={"Tune Strata": "n", "Tune the cascade layout": "y"})
        self.assertEqual(code, 0, out[-3000:])
        self.assertTrue(any("Tune the cascade layout" in q for q in asked), asked)
        self.assertEqual(len(tuned), 1)
        self.assertGreater(tuned[0][0], 0)                 # the free RAM was passed through
        self.assertTrue(tuned[0][1])                       # a helper GiB for the second card

    def test_upstream_engine_keeps_the_low_ram_mode(self):
        import tools.test_setup_golden as golden
        ram, found = golden.PROFILES["32GB-2x24GB"]
        # tiered_engine not patched: the mocked engine folder has no strata.exe, so the probe says "not ported"
        code, out, cfg, asked = golden.install(ram, found, golden.argv_for("qwen", "IQ3_XXS"))
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("--tiered-experts", cfg["args"])
        self.assertTrue("--resident-experts" in cfg["args"] or "--mmap-experts" in cfg["args"])

    def test_explicit_tiered_on_upstream_falls_back(self):
        import tools.test_setup_golden as golden
        ram, found = golden.PROFILES["32GB-2x24GB"]
        code, out, cfg, asked = golden.install(
            ram, found, golden.argv_for("qwen", "IQ3_XXS") + ["--low-ram", "tiered"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("--tiered-experts", cfg["args"])
        self.assertIn("needs the Strata-Cascade engine", out)

    def test_one_card_low_ram_asks_and_defaults_to_the_cascade(self):
        """One card whose RAM does not hold the model: the wizard asks, recommending the cascade (this fork is the
        cascade fork).  Enter takes the default -> --tiered-experts; no helper cache (there is no second card)."""
        import tools.test_setup_golden as golden
        ram, found = 31.9, [golden.card(0, "NVIDIA GeForce RTX 3090", 24.0, "86")]
        with mock.patch.object(setup, "tiered_engine", lambda eng: True):
            code, out, cfg, asked = golden.install(
                ram, found, golden.argv_for("qwen", "IQ3_XXS"),
                answers={"Use the cascade on this card": ""})    # Enter -> the default (yes)
        self.assertEqual(code, 0, out[-3000:])
        self.assertTrue(any("Use the cascade on this card" in q for q in asked), asked)
        self.assertIn("--tiered-experts", cfg["args"])
        self.assertIn("--host-budget-gib", cfg["args"])
        self.assertEqual(cfg["args"][cfg["args"].index("--host-budget-gib") + 1], "auto")   # one card -> auto
        self.assertFalse(any(a.startswith("--expert-cache-device") for a in cfg["args"]))
        self.assertNotIn("--resident-experts", cfg["args"])

    def test_one_card_low_ram_saying_no_keeps_the_resident_mode(self):
        import tools.test_setup_golden as golden
        ram, found = 31.9, [golden.card(0, "NVIDIA GeForce RTX 3090", 24.0, "86")]
        with mock.patch.object(setup, "tiered_engine", lambda eng: True):
            code, out, cfg, asked = golden.install(
                ram, found, golden.argv_for("qwen", "IQ3_XXS"),
                answers={"Use the cascade on this card": "n"})
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("--tiered-experts", cfg["args"])
        self.assertTrue("--resident-experts" in cfg["args"] or "--mmap-experts" in cfg["args"])

    def test_high_ram_one_card_never_asks_or_enables_the_cascade(self):
        """A 64 GB PC with a 24 GB card holds the model: no low-RAM mode, so no cascade and no question."""
        import tools.test_setup_golden as golden
        ram, found = 63.7, [golden.card(0, "NVIDIA GeForce RTX 5090", 31.8, "120")]
        with mock.patch.object(setup, "tiered_engine", lambda eng: True):
            code, out, cfg, asked = golden.install(
                ram, found, golden.argv_for("qwen", "IQ3_XXS"))
        self.assertEqual(code, 0, out[-3000:])
        self.assertFalse(any("Use the cascade on this card" in q for q in asked), asked)
        self.assertNotIn("--tiered-experts", cfg["args"])


if __name__ == "__main__":
    unittest.main()
