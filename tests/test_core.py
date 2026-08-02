from __future__ import annotations

import unittest

from stc import (
    Candidate,
    EnergyContractError,
    Status,
    control_topk,
    rank_candidates,
)


def candidate(
    identifier: str,
    score: float,
    status: Status,
    energy: float,
    index: int,
) -> Candidate:
    return Candidate(identifier, score, status, energy, index)


class OptQTests(unittest.TestCase):
    def test_zero_pressure_when_quota_already_visible(self) -> None:
        window = [
            candidate("a", 0.9, Status.ADMISSIBLE, 0.0, 0),
            candidate("b", 0.8, Status.ADMISSIBLE, 0.0, 1),
            candidate("c", 0.7, Status.VIOLATING, 1.0, 2),
        ]
        result = control_topk(window, k=2, q=2)
        self.assertTrue(result.feasible)
        self.assertEqual(result.lambda_star, 0.0)
        self.assertEqual(result.admissible_in_topk, 2)

    def test_minimum_crossing_pressure(self) -> None:
        window = [
            candidate("a1", 0.95, Status.ADMISSIBLE, 0.0, 0),
            candidate("v1", 0.94, Status.VIOLATING, 1.0, 1),
            candidate("v2", 0.93, Status.VIOLATING, 1.0, 2),
            candidate("u1", 0.92, Status.UNKNOWN, 0.5, 3),
            candidate("v3", 0.91, Status.VIOLATING, 1.0, 4),
            candidate("a2", 0.90, Status.ADMISSIBLE, 0.0, 5),
            candidate("v4", 0.89, Status.VIOLATING, 1.0, 6),
            candidate("a3", 0.88, Status.ADMISSIBLE, 0.0, 7),
        ]
        result = control_topk(window, k=5, q=3)
        self.assertTrue(result.feasible)
        self.assertAlmostEqual(result.lambda_star, 0.05)
        self.assertGreaterEqual(result.admissible_in_topk, 3)

    def test_infeasible_window_returns_frozen_topk(self) -> None:
        window = [
            candidate("v1", 0.95, Status.VIOLATING, 1.0, 0),
            candidate("a1", 0.90, Status.ADMISSIBLE, 0.0, 1),
            candidate("u1", 0.85, Status.UNKNOWN, 0.5, 2),
        ]
        result = control_topk(window, k=2, q=2)
        self.assertFalse(result.feasible)
        self.assertEqual(result.lambda_star, 0.0)
        self.assertEqual(
            [item.candidate.identifier for item in result.returned],
            ["v1", "a1"],
        )

    def test_lower_energy_wins_exact_crossing_tie(self) -> None:
        window = [
            candidate("v", 0.90, Status.VIOLATING, 1.0, 0),
            candidate("a", 0.80, Status.ADMISSIBLE, 0.0, 1),
        ]
        result = control_topk(window, k=1, q=1)
        self.assertAlmostEqual(result.lambda_star, 0.10)
        self.assertEqual(result.returned[0].candidate.identifier, "a")

    def test_original_order_is_final_tie_breaker(self) -> None:
        window = [
            candidate("x", 0.80, Status.VIOLATING, 1.0, 0),
            candidate("y", 0.80, Status.VIOLATING, 1.0, 1),
            candidate("a", 0.90, Status.ADMISSIBLE, 0.0, 2),
        ]
        ranked = rank_candidates(window, lambda_value=0.0)
        self.assertEqual(
            [item.candidate.identifier for item in ranked],
            ["a", "x", "y"],
        )

    def test_energy_contract_is_enforced(self) -> None:
        with self.assertRaises(EnergyContractError):
            control_topk(
                [
                    candidate("a", 0.9, Status.ADMISSIBLE, 0.2, 0),
                    candidate("v", 0.8, Status.VIOLATING, 1.0, 1),
                ],
                k=1,
                q=1,
            )

    def test_zero_quota_needs_no_pressure(self) -> None:
        window = [
            candidate("v", 0.9, Status.VIOLATING, 1.0, 0),
            candidate("a", 0.8, Status.ADMISSIBLE, 0.0, 1),
        ]
        result = control_topk(window, k=1, q=0)
        self.assertTrue(result.feasible)
        self.assertEqual(result.lambda_star, 0.0)

    def test_nonadmissible_zero_energy_is_rejected(self) -> None:
        with self.assertRaises(EnergyContractError):
            control_topk(
                [
                    candidate("a", 0.9, Status.ADMISSIBLE, 0.0, 0),
                    candidate("v", 0.8, Status.VIOLATING, 0.0, 1),
                ],
                k=1,
                q=1,
            )

    def test_k_cannot_exceed_materialized_window(self) -> None:
        with self.assertRaises(ValueError):
            control_topk(
                [candidate("a", 0.9, Status.ADMISSIBLE, 0.0, 0)],
                k=2,
                q=1,
            )

    def test_duplicate_identifier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            control_topk(
                [
                    candidate("x", 0.9, Status.ADMISSIBLE, 0.0, 0),
                    candidate("x", 0.8, Status.VIOLATING, 1.0, 1),
                ],
                k=1,
                q=1,
            )


if __name__ == "__main__":
    unittest.main()
