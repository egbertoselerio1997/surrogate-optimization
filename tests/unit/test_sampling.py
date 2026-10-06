"""Tests for the manuscript-defined deterministic Latin-hypercube designs."""

from __future__ import annotations
from surrogate_optimization.paths import REPOSITORY_ROOT
import json
import unittest
from surrogate_optimization.data.random_design import SplitMix64


class SplitMix64Tests(unittest.TestCase):
    def test_seed_42_matches_published_splitmix64_words(self) -> None:
        stream = SplitMix64(42)
        expected = (
            13679457532755275413,
            2949826092126892291,
            5139283748462763858,
            6349198060258255764,
            701532786141963250,
        )
        self.assertEqual(tuple((stream.next_uint64() for _ in expected)), expected)
        self.assertEqual(stream.draw_count, len(expected))
        self.assertEqual(
            stream.state, 42 + len(expected) * 11400714819323198485 & (1 << 64) - 1
        )

    def test_stream_restarts_exactly_and_float_is_half_open(self) -> None:
        left, right = (SplitMix64(314159), SplitMix64(314159))
        left_values = [left.random_float53() for _ in range(100)]
        right_values = [right.random_float53() for _ in range(100)]
        self.assertEqual(left_values, right_values)
        self.assertTrue(all((0.0 <= value < 1.0 for value in left_values)))

    def test_open_float52_excludes_both_faces_and_replays(self) -> None:
        left, right = (SplitMix64(200043), SplitMix64(200043))
        left_values = [left.random_open_float52() for _ in range(100)]
        right_values = [right.random_open_float52() for _ in range(100)]
        self.assertEqual(left_values, right_values)
        self.assertTrue(all((0.0 < value < 1.0 for value in left_values)))

        class EndpointStream(SplitMix64):
            def __init__(self, word: int) -> None:
                self.word = word

            def next_uint64(self) -> int:
                return self.word

        self.assertEqual(EndpointStream(0).random_open_float52(), 2.0 ** (-53))
        self.assertEqual(
            EndpointStream((1 << 64) - 1).random_open_float52(), 1.0 - 2.0 ** (-53)
        )

    def test_invalid_unsigned_seed_or_bound_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SplitMix64(-1)
        with self.assertRaises(ValueError):
            SplitMix64(1 << 64)
        stream = SplitMix64(0)
        with self.assertRaises(ValueError):
            stream.randbelow(0)

    def test_randbelow_rejects_the_modulo_bias_tail(self) -> None:

        class ScriptedStream(SplitMix64):
            def __init__(self) -> None:
                self.words = iter(((1 << 64) - 1, 5))

            def next_uint64(self) -> int:
                return next(self.words)

        self.assertEqual(ScriptedStream().randbelow(3), 2)


class UnifiedProductionProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = REPOSITORY_ROOT / "config" / "parameters.json"
        cls.config = json.loads(path.read_text(encoding="utf-8"))

    def test_production_profile_freezes_independent_candidate_streams(self) -> None:
        self.assertEqual(self.config["schema_version"], 6)
        self.assertEqual(self.config["execution"]["default_profile"], "production")
        profile = self.config["profiles"]["production"]
        self.assertEqual(
            (
                profile["development_candidate_count"],
                profile["holdout_candidate_count"],
            ),
            (8000, 2000),
        )
        self.assertEqual(
            (profile["development_seed"], profile["holdout_seed"]), (100042, 100043)
        )
        self.assertTrue(profile["counts_are_candidate_rows"])
        self.assertFalse(profile["replace_rejected_mechanistic_candidates"])

    def test_removed_guardrails_and_primary_time_are_explicit(self) -> None:
        engineering = self.config["engineering"]
        for key in ("srt_min_d", "srt_max_d", "sor_max_m_d", "slr_max_kg_m2_d"):
            self.assertNotIn(key, engineering)
        self.assertEqual(
            engineering["descriptive_quantities"], ["srt_d", "sor_m_d", "slr_kg_m2_d"]
        )
        reporting = self.config["reporting"]
        self.assertEqual(reporting["timing_metric"], "Time")
        self.assertEqual(reporting["timing_unit"], "s")
        self.assertEqual(reporting["timing_protocol"], "primary_route_time")
