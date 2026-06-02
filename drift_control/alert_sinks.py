"""Alert sinks for drift events."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Protocol
from urllib import request


class AlertSink(Protocol):
    def send(self, result: dict[str, dict[str, Any]]) -> Any: ...


class CompositeAlertSink:
    """Fan out drift events to multiple sinks."""

    def __init__(self, sinks: list[AlertSink]) -> None:
        self.sinks = list(sinks)

    def send(self, result: dict[str, dict[str, Any]]) -> None:
        for sink in self.sinks:
            sink.send(result)


class ColumnFilterAlertSink:
    """Forward events to an inner sink only for selected drifting columns."""

    def __init__(self, sink: AlertSink, columns: list[str]) -> None:
        self.sink = sink
        self.columns = set(columns)

    def send(self, result: dict[str, dict[str, Any]]) -> None:
        filtered: dict[str, dict[str, Any]] = {}
        for col, payload in result.items():
            if col in self.columns and bool(payload.get("drift")):
                filtered[col] = payload
        if filtered:
            self.sink.send(filtered)


class LogAlertSink:
    """Log drift payloads to a named logger."""

    def __init__(self, logger_name: str = "drift_control.alerts", level: int = logging.WARNING) -> None:
        self.logger = logging.getLogger(logger_name)
        self.level = level

    def send(self, result: dict[str, dict[str, Any]]) -> None:
        self.logger.log(self.level, "drift_detected payload=%s", json.dumps(result, sort_keys=True))


class WebhookAlertSink:
    """Send drift payloads to an HTTP webhook endpoint."""

    def __init__(self, url: str, timeout_seconds: float = 5.0) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds

    def send(self, result: dict[str, dict[str, Any]]) -> None:
        payload = json.dumps(result).encode("utf-8")
        self.send_payload(payload)

    def send_payload(self, payload: bytes) -> None:
        req = request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds):
            return


class SlackWebhookAlertSink(WebhookAlertSink):
    """Send drift alerts to a Slack incoming webhook."""

    def send(self, result: dict[str, dict[str, Any]]) -> None:
        drifting = [k for k, v in result.items() if bool(v.get("drift"))]
        text = (
            f"Drift detected in {len(drifting)} column(s): {', '.join(drifting)}"
            if drifting
            else "Drift check executed with no drifting columns."
        )
        payload = json.dumps({"text": text, "drift_result": result}).encode("utf-8")
        self.send_payload(payload)


class RetryingWebhookAlertSink(WebhookAlertSink):
    """Webhook sink with bounded retries and fixed backoff."""

    def __init__(
        self,
        url: str,
        timeout_seconds: float = 5.0,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
    ) -> None:
        super().__init__(url=url, timeout_seconds=timeout_seconds)
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be >= 0")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def send_payload(self, payload: bytes) -> None:
        attempts = self.max_retries + 1
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                super().send_payload(payload)
                return
            except Exception as exc:  # pragma: no cover - exercised in tests via monkeypatch
                last_exc = exc
                if i == attempts - 1:
                    break
                if self.backoff_seconds > 0:
                    time.sleep(self.backoff_seconds)
        if last_exc is not None:
            raise last_exc

