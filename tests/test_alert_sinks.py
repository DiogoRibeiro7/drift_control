import json

from drift_control.alert_sinks import PagerDutyAlertSink, SlackWebhookAlertSink, WebhookAlertSink


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_webhook_alert_sink_posts_json(monkeypatch):
    captured = {}

    def _fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["method"] = req.get_method()
        captured["body"] = req.data
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("drift_control.alert_sinks.request.urlopen", _fake_urlopen)
    sink = WebhookAlertSink(url="https://example.test/webhook", timeout_seconds=3.0)
    sink.send({"x": {"drift": True, "score": 0.9}})

    assert captured["url"] == "https://example.test/webhook"
    assert captured["method"] == "POST"
    assert captured["timeout"] == 3.0
    assert json.loads(captured["body"].decode("utf-8"))["x"]["drift"] is True


def test_slack_webhook_alert_sink_formats_text(monkeypatch):
    captured = {}

    def _fake_urlopen(req, timeout):
        captured["body"] = req.data
        return _FakeResponse()

    monkeypatch.setattr("drift_control.alert_sinks.request.urlopen", _fake_urlopen)
    sink = SlackWebhookAlertSink(url="https://hooks.slack.test/abc")
    sink.send({"x": {"drift": True, "score": 0.9}, "y": {"drift": False, "score": 0.1}})

    payload = json.loads(captured["body"].decode("utf-8"))
    assert "Drift detected in 1 column(s): x" == payload["text"]
    assert payload["drift_result"]["x"]["drift"] is True


def test_pagerduty_alert_sink_formats_event_v2_payload(monkeypatch):
    captured = {}

    def _fake_urlopen(req, timeout):
        captured["body"] = req.data
        return _FakeResponse()

    monkeypatch.setattr("drift_control.alert_sinks.request.urlopen", _fake_urlopen)
    sink = PagerDutyAlertSink(
        routing_key="rk",
        source="unit-tests",
        component="stream",
        severity="error",
        dedup_key="drift-key",
    )
    sink.send({"x": {"drift": True, "score": 0.9}})

    payload = json.loads(captured["body"].decode("utf-8"))
    assert payload["routing_key"] == "rk"
    assert payload["event_action"] == "trigger"
    assert payload["dedup_key"] == "drift-key"
    assert payload["payload"]["source"] == "unit-tests"
    assert payload["payload"]["severity"] == "error"
    assert payload["payload"]["component"] == "stream"
    assert "Drift detected in 1 column(s): x" == payload["payload"]["summary"]
