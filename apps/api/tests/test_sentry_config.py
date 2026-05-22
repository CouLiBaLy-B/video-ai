from video_ai.infrastructure.sentry import configure_sentry


def test_configure_sentry_returns_false_without_dsn() -> None:
    assert configure_sentry(dsn=None, traces_sample_rate=0.1) is False
