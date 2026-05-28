"""Alert sinks for drift events."""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol
from urllib import request


class AlertSink(Protocol):
    def send(self, result: dict[str, dict[str, Any]]) -> Any: ...


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
        req = request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds):
            return
