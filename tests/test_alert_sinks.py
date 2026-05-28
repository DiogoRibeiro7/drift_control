import json

from drift_control.alert_sinks import (
    ColumnFilterAlertSink,
    CompositeAlertSink,
    PagerDutyAlertSink,
    PrometheusAlertSink,
    RetryingWebhookAlertSink,
    SlackWebhookAlertSink,
    WebhookAlertSink,
)


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


def test_composite_alert_sink_fans_out_to_all_sinks():
    calls = {"a": 0, "b": 0}

    class _SinkA:
        def send(self, _result):
            calls["a"] += 1

    class _SinkB:
        def send(self, _result):
            calls["b"] += 1

    sink = CompositeAlertSink([_SinkA(), _SinkB()])
    sink.send({"x": {"drift": True, "score": 0.7}})
    assert calls == {"a": 1, "b": 1}


def test_column_filter_alert_sink_forwards_only_selected_drifting_columns():
    captured = {}

    class _Sink:
        def send(self, result):
            captured["result"] = result

    sink = ColumnFilterAlertSink(_Sink(), columns=["x", "z"])
    sink.send(
        {
            "x": {"drift": True, "score": 0.7},
            "y": {"drift": True, "score": 0.9},
            "z": {"drift": False, "score": 0.1},
        }
    )
    assert captured["result"] == {"x": {"drift": True, "score": 0.7}}


def test_column_filter_alert_sink_noop_when_nothing_matches():
    calls = {"n": 0}

    class _Sink:
        def send(self, _result):
            calls["n"] += 1

    sink = ColumnFilterAlertSink(_Sink(), columns=["x"])
    sink.send({"y": {"drift": True, "score": 0.9}})
    assert calls["n"] == 0


def test_retrying_webhook_alert_sink_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def _fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("temporary failure")
        return _FakeResponse()

    monkeypatch.setattr("drift_control.alert_sinks.request.urlopen", _fake_urlopen)
    monkeypatch.setattr("drift_control.alert_sinks.time.sleep", lambda _: None)
    sink = RetryingWebhookAlertSink(
        url="https://example.test/webhook",
        timeout_seconds=1.0,
        max_retries=3,
        backoff_seconds=0.01,
    )
    sink.send({"x": {"drift": True, "score": 0.9}})
    assert calls["n"] == 3


def test_retrying_webhook_alert_sink_raises_after_exhausted_retries(monkeypatch):
    calls = {"n": 0}

    def _fake_urlopen(req, timeout):
        calls["n"] += 1
        raise RuntimeError("always fails")

    monkeypatch.setattr("drift_control.alert_sinks.request.urlopen", _fake_urlopen)
    monkeypatch.setattr("drift_control.alert_sinks.time.sleep", lambda _: None)
    sink = RetryingWebhookAlertSink(
        url="https://example.test/webhook",
        timeout_seconds=1.0,
        max_retries=2,
        backoff_seconds=0.01,
    )
    try:
        sink.send({"x": {"drift": True, "score": 0.9}})
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert str(exc) == "always fails"
    assert calls["n"] == 3


def test_retrying_webhook_alert_sink_validates_parameters():
    try:
        RetryingWebhookAlertSink(url="https://example.test/webhook", max_retries=-1)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "max_retries" in str(exc)
    try:
        RetryingWebhookAlertSink(url="https://example.test/webhook", backoff_seconds=-0.1)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "backoff_seconds" in str(exc)


def test_prometheus_alert_sink_updates_metrics_with_injected_collectors():
    class _MetricChild:
        def __init__(self):
            self.value = None
            self.increments = 0

        def set(self, value):
            self.value = value

        def inc(self):
            self.increments += 1

    class _Metric:
        def __init__(self):
            self.children = {}

        def labels(self, column):
            if column not in self.children:
                self.children[column] = _MetricChild()
            return self.children[column]

    events = _Metric()
    score = _Metric()
    flag = _Metric()
    sink = PrometheusAlertSink(
        drift_events_total=events,
        drift_score=score,
        drift_flag=flag,
    )
    sink.send(
        {
            "x": {"drift": True, "score": 0.7},
            "y": {"drift": False, "score": 0.1},
        }
    )
    assert score.children["x"].value == 0.7
    assert score.children["y"].value == 0.1
    assert flag.children["x"].value == 1.0
    assert flag.children["y"].value == 0.0
    assert events.children["x"].increments == 1
    assert "y" not in events.children
