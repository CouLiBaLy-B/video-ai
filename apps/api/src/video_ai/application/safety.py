"""Content safety policy checks for generation requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SafetyDecision(StrEnum):
    """Possible safety decisions."""

    ALLOWED = "allowed"
    NEEDS_REVIEW = "needs_review"
    BLOCKED = "blocked"


class SafetyPolicyViolationError(ValueError):
    """Raised when a request violates or triggers the content policy."""

    def __init__(self, decision: SafetyDecision, reason: str) -> None:
        super().__init__(reason)
        self.decision = decision
        self.reason = reason


@dataclass(frozen=True)
class SafetyResult:
    """Result of a safety policy evaluation."""

    decision: SafetyDecision
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SafetyPolicy:
    """Configurable keyword policy used as a baseline guardrail."""

    enabled: bool
    blocked_terms: tuple[str, ...]
    review_terms: tuple[str, ...]


class SafetyService:
    """Baseline content safety service.

    This keyword-based service is intentionally conservative and lightweight. It
    is a production hook, not a complete moderation system. Public deployments
    should replace or augment it with a real moderation model/provider.
    """

    def __init__(self, policy: SafetyPolicy) -> None:
        self._policy = policy

    def evaluate_prompt(self, prompt: str) -> SafetyResult:
        """Evaluate a user prompt against the configured content policy."""
        if not self._policy.enabled:
            return SafetyResult(SafetyDecision.ALLOWED)
        normalized = _normalize(prompt)
        blocked = [term for term in self._policy.blocked_terms if term and term in normalized]
        if blocked:
            return SafetyResult(
                SafetyDecision.BLOCKED,
                [f"Prompt contains blocked term: {term}" for term in blocked],
            )
        review = [term for term in self._policy.review_terms if term and term in normalized]
        if review:
            return SafetyResult(
                SafetyDecision.NEEDS_REVIEW,
                [f"Prompt requires manual review because of term: {term}" for term in review],
            )
        return SafetyResult(SafetyDecision.ALLOWED)

    def assert_prompt_allowed(self, prompt: str) -> None:
        """Raise when a prompt is blocked or needs manual review."""
        result = self.evaluate_prompt(prompt)
        if result.decision == SafetyDecision.ALLOWED:
            return
        reason = "; ".join(result.reasons) or "Content policy violation"
        raise SafetyPolicyViolationError(result.decision, reason)


def parse_terms(raw: str) -> tuple[str, ...]:
    """Parse comma-separated terms into normalized terms."""
    return tuple(_normalize(item) for item in raw.split(",") if item.strip())


def _normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())
