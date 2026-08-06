from federated_ts.utils.tracking import Tracker


class _FakeRun:
    def __init__(self):
        self.calls = []

    def log(self, data, step=None):
        self.calls.append((data, step))


def test_tracker_log_records_explicit_step_field():
    tracker = Tracker.__new__(Tracker)
    tracker.run = _FakeRun()

    tracker.log({"round/loss": 0.1}, step=7)

    payload, step = tracker.run.calls[0]
    assert step == 7
    assert payload["tracking/step"] == 7
    assert payload["round/loss"] == 0.1


def test_tracker_log_preserves_existing_step_field():
    tracker = Tracker.__new__(Tracker)
    tracker.run = _FakeRun()

    tracker.log({"tracking/step": 9, "attack/mse": 1.2}, step=4)

    payload, step = tracker.run.calls[0]
    assert step == 4
    assert payload["tracking/step"] == 9
    assert payload["attack/mse"] == 1.2
