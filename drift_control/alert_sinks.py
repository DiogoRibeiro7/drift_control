"""Alert sinks for drift events."""

from __future__ import annotations

import json
import logging
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


class PagerDutyAlertSink(WebhookAlertSink):
    """Send drift alerts to PagerDuty Events API v2."""

    def __init__(
        self,
        routing_key: str,
        source: str = "drift_control",
        component: str = "drift_monitor",
        severity: str = "warning",
        dedup_key: str | None = None,
        events_url: str = "https://events.pagerduty.com/v2/enqueue",
        timeout_seconds: float = 5.0,
    ) -> None:
        super().__init__(url=events_url, timeout_seconds=timeout_seconds)
        self.routing_key = routing_key
        self.source = source
        self.component = component
        self.severity = severity
        self.dedup_key = dedup_key

    def send(self, result: dict[str, dict[str, Any]]) -> None:
        drifting = [k for k, v in result.items() if bool(v.get("drift"))]
        summary = (
            f"Drift detected in {len(drifting)} column(s): {', '.join(drifting)}"
            if drifting
            else "Drift monitor callback without drifting columns."
        )
        payload: dict[str, Any] = {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": summary,
                "source": self.source,
                "severity": self.severity,
                "component": self.component,
                "custom_details": {"drift_result": result},
            },
        }
        if self.dedup_key is not None:
            payload["dedup_key"] = self.dedup_key
        self.send_payload(json.dumps(payload).encode("utf-8"))
