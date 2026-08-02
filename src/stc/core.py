"""Reference implementation of the finite-window OptQ controller.

The implementation follows the deployment contract used in the paper:

* candidates and frozen scores are already materialized;
* admissible candidates have zero semantic energy;
* every non-admissible candidate has strictly positive energy;
* the controller may reorder only the supplied finite window;
* ties are resolved by controlled score, lower energy, frozen score,
  and original candidate order.

Under this energy contract, the smallest scalar pressure in the family

    s'_x(c) = s_x(c) - lambda * e_x(c)

is obtained from the positive threshold crossings relative to the q-th
highest-scoring admissible candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable, Sequence


class Status(str, Enum):
    """Operational semantic status."""

    ADMISSIBLE = "admissible"
    VIOLATING = "violating"
    UNKNOWN = "unknown"


class EnergyContractError(ValueError):
    """Raised when candidate energies do not satisfy the OptQ contract."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """One candidate from a frozen Top-M window."""

    identifier: str
    score: float
    status: Status | str
    energy: float
    original_index: int

    def normalized_status(self) -> Status:
        return self.status if isinstance(self.status, Status) else Status(self.status)


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    """Candidate plus its controlled score and returned rank."""

    candidate: Candidate
    controlled_score: float
    rank: int


@dataclass(frozen=True, slots=True)
class ControlResult:
    """Controller output and finite-window certificate."""

    feasible: bool
    quota: int
    top_k: int
    lambda_star: float
    tau_q: float | None
    positive_crossings: tuple[float, ...]
    admissible_in_window: int
    admissible_in_topk: int
    returned: tuple[RankedCandidate, ...]
    reason: str


def _validate_candidates(candidates: Sequence[Candidate]) -> None:
    if not candidates:
        raise ValueError("The materialized candidate window is empty.")

    seen_indices: set[int] = set()
    seen_ids: set[str] = set()

    for candidate in candidates:
        status = candidate.normalized_status()

        if candidate.identifier in seen_ids:
            raise ValueError(f"Duplicate candidate identifier: {candidate.identifier}")
        seen_ids.add(candidate.identifier)

        if candidate.original_index in seen_indices:
            raise ValueError(
                f"Duplicate original_index: {candidate.original_index}"
            )
        seen_indices.add(candidate.original_index)

        if not isfinite(candidate.score):
            raise ValueError(
                f"Non-finite score for candidate {candidate.identifier}"
            )
        if not isfinite(candidate.energy) or candidate.energy < 0:
            raise ValueError(
                f"Energy must be finite and non-negative for "
                f"{candidate.identifier}"
            )

        if status is Status.ADMISSIBLE and candidate.energy != 0.0:
            raise EnergyContractError(
                "Admissible candidates must have zero energy under the "
                f"reference OptQ contract: {candidate.identifier}"
            )
        if status is not Status.ADMISSIBLE and candidate.energy <= 0.0:
            raise EnergyContractError(
                "Violating and unknown candidates must have strictly positive "
                f"energy under the reference OptQ contract: "
                f"{candidate.identifier}"
            )


def _frozen_key(candidate: Candidate) -> tuple[float, int]:
    return (-candidate.score, candidate.original_index)


def _controlled_key(
    candidate: Candidate,
    lambda_star: float,
) -> tuple[float, float, float, int]:
    controlled_score = candidate.score - lambda_star * candidate.energy
    return (
        -controlled_score,
        candidate.energy,
        -candidate.score,
        candidate.original_index,
    )


def rank_candidates(
    candidates: Iterable[Candidate],
    *,
    lambda_value: float = 0.0,
) -> tuple[RankedCandidate, ...]:
    """Rank a finite window with deterministic STC tie-breaking."""

    if not isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("lambda_value must be finite and non-negative.")

    window = tuple(candidates)
    _validate_candidates(window)

    ordered = sorted(
        window,
        key=lambda candidate: _controlled_key(candidate, lambda_value),
    )
    return tuple(
        RankedCandidate(
            candidate=candidate,
            controlled_score=candidate.score - lambda_value * candidate.energy,
            rank=index,
        )
        for index, candidate in enumerate(ordered, start=1)
    )


def _base_topk(
    candidates: Sequence[Candidate],
    k: int,
) -> tuple[RankedCandidate, ...]:
    ordered = sorted(candidates, key=_frozen_key)
    return tuple(
        RankedCandidate(
            candidate=candidate,
            controlled_score=candidate.score,
            rank=index,
        )
        for index, candidate in enumerate(ordered[:k], start=1)
    )


def control_topk(
    candidates: Iterable[Candidate],
    *,
    k: int,
    q: int,
) -> ControlResult:
    """Return the OptQ-controlled Top-k list.

    The controller computes the q-th admissible score ``tau_q`` and the
    positive crossing thresholds

        gamma(c) = (s(c) - tau_q) / e(c)

    for non-admissible candidates with positive crossings. Let ``b = k - q``.
    If at most ``b`` candidates cross the threshold, no pressure is required.
    Otherwise the minimum pressure is the ``(b + 1)``-th largest crossing.

    When the finite window contains fewer than ``q`` admissible candidates,
    the result is certified infeasible and the frozen base Top-k is returned.
    """

    window = tuple(candidates)
    _validate_candidates(window)

    if k <= 0:
        raise ValueError("k must be positive.")
    if k > len(window):
        raise ValueError(
            f"k={k} exceeds materialized window size M={len(window)}."
        )
    if q < 0 or q > k:
        raise ValueError("q must satisfy 0 <= q <= k.")

    admissible = sorted(
        (
            candidate
            for candidate in window
            if candidate.normalized_status() is Status.ADMISSIBLE
        ),
        key=_frozen_key,
    )

    if len(admissible) < q:
        returned = _base_topk(window, k)
        count = sum(
            ranked.candidate.normalized_status() is Status.ADMISSIBLE
            for ranked in returned
        )
        return ControlResult(
            feasible=False,
            quota=q,
            top_k=k,
            lambda_star=0.0,
            tau_q=None,
            positive_crossings=(),
            admissible_in_window=len(admissible),
            admissible_in_topk=count,
            returned=returned,
            reason=(
                "Finite-window infeasibility: the materialized window contains "
                f"{len(admissible)} admissible candidates, fewer than q={q}. "
                "The frozen base Top-k is returned."
            ),
        )

    if q == 0:
        ranked = rank_candidates(window, lambda_value=0.0)[:k]
        return ControlResult(
            feasible=True,
            quota=0,
            top_k=k,
            lambda_star=0.0,
            tau_q=None,
            positive_crossings=(),
            admissible_in_window=len(admissible),
            admissible_in_topk=sum(
                item.candidate.normalized_status() is Status.ADMISSIBLE
                for item in ranked
            ),
            returned=ranked,
            reason="The requested quota is zero; no pressure is required.",
        )

    tau_q = admissible[q - 1].score
    crossings = sorted(
        (
            (candidate.score - tau_q) / candidate.energy
            for candidate in window
            if candidate.normalized_status() is not Status.ADMISSIBLE
            and candidate.score > tau_q
        ),
        reverse=True,
    )

    b = k - q
    lambda_star = 0.0 if len(crossings) <= b else crossings[b]

    ranked = rank_candidates(window, lambda_value=lambda_star)[:k]
    admissible_in_topk = sum(
        item.candidate.normalized_status() is Status.ADMISSIBLE
        for item in ranked
    )

    if admissible_in_topk < q:
        raise AssertionError(
            "The OptQ postcondition failed despite a valid energy contract. "
            f"Observed {admissible_in_topk} admissible candidates, expected "
            f"at least {q}."
        )

    return ControlResult(
        feasible=True,
        quota=q,
        top_k=k,
        lambda_star=lambda_star,
        tau_q=tau_q,
        positive_crossings=tuple(crossings),
        admissible_in_window=len(admissible),
        admissible_in_topk=admissible_in_topk,
        returned=ranked,
        reason=(
            "Quota satisfied within the fixed materialized window using the "
            "minimum scalar pressure in the controlled-score family."
        ),
    )
