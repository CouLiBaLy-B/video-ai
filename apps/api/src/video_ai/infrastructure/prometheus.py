"""Minimal Prometheus exposition helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from video_ai.application.metrics import JobMetrics


@dataclass
class InMemoryPrometheusMetrics:
    """Small in-process metrics registry.

    This avoids a hard dependency for the MVP while exposing a standard
    Prometheus text endpoint. Multi-process deployments should replace this with
    `prometheus_client` multiprocess mode or an OpenTelemetry collector.
    """

    http_requests: Counter[tuple[str, str, int]] = field(default_factory=Counter)
    http_duration_ms: dict[tuple[str, str], float] = field(
        default_factory=lambda: defaultdict(float)
    )

    def record_http_request(
        self, *, method: str, path: str, status_code: int, duration_ms: float
    ) -> None:
        """Record request count and cumulative duration."""
        normalized = _normalize_path(path)
        self.http_requests[(method, normalized, status_code)] += 1
        self.http_duration_ms[(method, normalized)] += duration_ms


GLOBAL_PROMETHEUS_METRICS = InMemoryPrometheusMetrics()


def render_prometheus(metrics: InMemoryPrometheusMetrics, job_metrics: JobMetrics) -> str:
    """Render metrics in Prometheus text exposition format."""
    lines = [
        "# HELP video_ai_http_requests_total Total HTTP requests.",
        "# TYPE video_ai_http_requests_total counter",
    ]
    for (method, path, status_code), count in sorted(metrics.http_requests.items()):
        lines.append(
            'video_ai_http_requests_total{'
            f'method="{method}",path="{path}",status_code="{status_code}"'
            f"}} {count}"
        )

    lines.extend(
        [
            "# HELP video_ai_http_request_duration_ms_sum Cumulative HTTP duration in ms.",
            "# TYPE video_ai_http_request_duration_ms_sum counter",
        ]
    )
    for (method, path), duration in sorted(metrics.http_duration_ms.items()):
        lines.append(
            'video_ai_http_request_duration_ms_sum{'
            f'method="{method}",path="{path}"'
            f"}} {duration:.3f}"
        )

    lines.extend(
        [
            "# HELP video_ai_generation_jobs_total Generation jobs by status.",
            "# TYPE video_ai_generation_jobs_total gauge",
            f"video_ai_generation_jobs_total {job_metrics.total}",
        ]
    )
    for status, count in sorted(job_metrics.by_status.items(), key=lambda item: item[0].value):
        lines.append(f'video_ai_generation_jobs_by_status{{status="{status.value}"}} {count}')
    return "\n".join(lines) + "\n"


def _normalize_path(path: str) -> str:
    if path.startswith("/api/generations/"):
        parts = path.split("/")
        if len(parts) >= 4:
            suffix = "/" + "/".join(parts[4:]) if len(parts) > 4 else ""
            return "/api/generations/{job_id}" + suffix
    return path
