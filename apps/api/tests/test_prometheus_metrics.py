from httpx import ASGITransport, AsyncClient

from video_ai.application.metrics import JobMetrics
from video_ai.domain.enums import JobStatus
from video_ai.infrastructure.prometheus import InMemoryPrometheusMetrics, render_prometheus
from video_ai.main import create_app


def test_render_prometheus_includes_http_and_job_metrics() -> None:
    metrics = InMemoryPrometheusMetrics()
    metrics.record_http_request(
        method="GET", path="/api/generations/abc", status_code=200, duration_ms=12.5
    )
    job_metrics = JobMetrics(total=1, by_status={status: 0 for status in JobStatus}, completed=1)
    job_metrics = job_metrics.model_copy(
        update={"by_status": {**job_metrics.by_status, JobStatus.COMPLETED: 1}}
    )

    text = render_prometheus(metrics, job_metrics)

    assert "video_ai_http_requests_total" in text
    assert 'path="/api/generations/{job_id}"' in text
    assert 'status="completed"} 1' in text


async def test_metrics_endpoint_exposes_prometheus_text() -> None:
    app = create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.get("/health")
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert "video_ai_http_requests_total" in response.text
    assert "video_ai_generation_jobs_total" in response.text
