"""Optional Sentry initialization."""

from __future__ import annotations


def configure_sentry(*, dsn: str | None, traces_sample_rate: float) -> bool:
    """Configure Sentry when DSN and optional dependency are available."""
    if not dsn:
        return False
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the 'observability' extra to enable Sentry") from exc

    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=traces_sample_rate,
        integrations=[FastApiIntegration(), LoggingIntegration(event_level=None)],
    )
    return True
