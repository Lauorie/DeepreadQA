"""Minimal Prometheus-text metrics registry (no external dependency)."""
from __future__ import annotations

import threading
from typing import Callable, Optional

from .jobs import Job

_LATENCY_BUCKETS = (1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0)


class Metrics:
    """Thread-safe counters/histogram rendered in Prometheus text format."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._http: dict[tuple[str, str, int], int] = {}
        self._answers: dict[str, int] = {}
        self._lat_counts = [0] * (len(_LATENCY_BUCKETS) + 1)
        self._lat_sum = 0.0
        self._lat_n = 0
        self._gauges: list[tuple[str, str, Callable[[], float]]] = []

    def observe_http(self, method: str, path: str, status: int) -> None:
        with self._lock:
            key = (method, path, status)
            self._http[key] = self._http.get(key, 0) + 1

    def observe_answer(self, job: Job) -> None:
        with self._lock:
            self._answers[job.status] = self._answers.get(job.status, 0) + 1
            latency = self._latency_s(job)
            if latency is not None:
                for i, edge in enumerate(_LATENCY_BUCKETS):
                    if latency <= edge:
                        self._lat_counts[i] += 1
                        break
                else:
                    self._lat_counts[-1] += 1
                self._lat_sum += latency
                self._lat_n += 1

    @staticmethod
    def _latency_s(job: Job) -> Optional[float]:
        if job.started_at is None or job.finished_at is None:
            return None
        return job.finished_at - job.started_at

    def register_gauge(self, name: str, help_text: str,
                       fn: Callable[[], float]) -> None:
        self._gauges.append((name, help_text, fn))

    def render(self) -> str:
        with self._lock:
            lines = [
                "# HELP deepreadqa_http_requests_total HTTP requests by "
                "method/route/status",
                "# TYPE deepreadqa_http_requests_total counter",
            ]
            for (method, path, status), n in sorted(self._http.items()):
                lines.append(
                    f'deepreadqa_http_requests_total{{method="{method}",'
                    f'path="{path}",status="{status}"}} {n}')
            lines += [
                "# HELP deepreadqa_answers_finished_total answers by final "
                "status",
                "# TYPE deepreadqa_answers_finished_total counter",
            ]
            for status, n in sorted(self._answers.items()):
                lines.append(
                    f'deepreadqa_answers_finished_total{{status="{status}"}} {n}')
            lines += [
                "# HELP deepreadqa_answer_latency_seconds answer wall-clock "
                "latency",
                "# TYPE deepreadqa_answer_latency_seconds histogram",
            ]
            cumulative = 0
            for i, edge in enumerate(_LATENCY_BUCKETS):
                cumulative += self._lat_counts[i]
                lines.append(
                    f'deepreadqa_answer_latency_seconds_bucket{{le="{edge:g}"}}'
                    f" {cumulative}")
            cumulative += self._lat_counts[-1]
            lines.append(
                f'deepreadqa_answer_latency_seconds_bucket{{le="+Inf"}}'
                f" {cumulative}")
            lines.append(f"deepreadqa_answer_latency_seconds_sum "
                         f"{self._lat_sum:.6f}")
            lines.append(f"deepreadqa_answer_latency_seconds_count "
                         f"{self._lat_n}")
        for name, help_text, fn in self._gauges:
            lines += [f"# HELP {name} {help_text}", f"# TYPE {name} gauge",
                      f"{name} {fn():g}"]
        return "\n".join(lines) + "\n"
