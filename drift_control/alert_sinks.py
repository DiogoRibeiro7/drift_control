"""Alert sinks for drift events."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from typing import Any, Protocol
from urllib import request

from dataexcept import WebhookError

AlertResult = dict[str, dict[str, Any]]


class AlertSink(Protocol):
    def send(self, result: AlertResult) -> Any: ...


def _drifting_columns(result: AlertResult) -> list[str]:
    return [column for column, payload in result.items() if bool(payload.get("drift"))]


def _filter_selected_drifting_columns(result: AlertResult, columns: set[str]) -> AlertResult:
    return {
        column: payload
        for column, payload in result.items()
        if column in columns and bool(payload.get("drift"))
    }


def _build_json_request(url: str, payload: bytes) -> request.Request:
    return request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _retry_attempts(max_retries: int) -> Iterable[int]:
    return range(max_retries + 1)


def _validate_non_negative(name: str, value: float | int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


class CompositeAlertSink:
    """Fan out drift events to multiple sinks."""

    def __init__(self, sinks: list[AlertSink]) -> None:
        self.sinks = list(sinks)

    def send(self, result: AlertResult) -> None:
        for sink in self.sinks:
            sink.send(result)


class ColumnFilterAlertSink:
    """Forward events to an inner sink only for selected drifting columns."""

    def __init__(self, sink: AlertSink, columns: list[str]) -> None:
        self.sink = sink
        self.columns = set(columns)

    def send(self, result: AlertResult) -> None:
        filtered = _filter_selected_drifting_columns(result, self.columns)
        if filtered:
            self.sink.send(filtered)


class LogAlertSink:
    """Log drift payloads to a named logger."""

    def __init__(
        self, logger_name: str = "drift_control.alerts", level: int = logging.WARNING
    ) -> None:
        self.logger = logging.getLogger(logger_name)
        self.level = level

    def send(self, result: AlertResult) -> None:
        self.logger.log(self.level, "drift_detected payload=%s", json.dumps(result, sort_keys=True))


class WebhookAlertSink:
    """Send drift payloads to an HTTP webhook endpoint."""

    def __init__(self, url: str, timeout_seconds: float = 5.0) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds

    def send(self, result: AlertResult) -> None:
        payload = json.dumps(result).encode("utf-8")
        self.send_payload(payload)

    def send_payload(self, payload: bytes) -> None:
        req = _build_json_request(self.url, payload)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds):
                return
        except OSError as exc:
            # URL errors include HTTP failures and timeouts. WebhookError
            # removes credential-bearing URL paths from error messages.
            raise WebhookError(self.url, original_exception=exc) from exc


class SlackWebhookAlertSink(WebhookAlertSink):
    """Send drift alerts to a Slack incoming webhook."""

    def send(self, result: AlertResult) -> None:
        drifting = _drifting_columns(result)
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
        _validate_non_negative("max_retries", max_retries)
        _validate_non_negative("backoff_seconds", backoff_seconds)
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def send_payload(self, payload: bytes) -> None:
        last_exc: Exception | None = None
        for attempt in _retry_attempts(self.max_retries):
            try:
                super().send_payload(payload)
                return
            except Exception as exc:  # pragma: no cover - exercised in tests via monkeypatch
                last_exc = exc
                if attempt == self.max_retries:
                    break
                if self.backoff_seconds > 0:
                    time.sleep(self.backoff_seconds)
        if last_exc is not None:
            raise last_exc
