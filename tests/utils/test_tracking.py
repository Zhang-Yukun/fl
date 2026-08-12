from federated_ts.utils.tracking import Tracker


class _FakeRun:
    def __init__(self):
        self.calls = []

    def log(self, data, step=None):
        self.calls.append((data, step))


class _FakeImage:
    def __init__(self, image, caption=None):
        self.image = image
        self.caption = caption


class _FakeWandb:
    Image = _FakeImage


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



def test_tracker_log_image_records_step_and_caption():
    tracker = Tracker.__new__(Tracker)
    tracker.run = _FakeRun()
    tracker.wandb = _FakeWandb()

    tracker.log_image("plot/test", image="fake", step=3, caption="demo")

    payload, step = tracker.run.calls[0]
    assert step == 3
    assert payload["tracking/step"] == 3
    assert payload["plot/test"].caption == "demo"


def test_tracker_log_prediction_plot_creates_wandb_image():
    tracker = Tracker.__new__(Tracker)
    tracker.run = _FakeRun()
    tracker.wandb = _FakeWandb()

    tracker.log_prediction_plot(
        "prediction/test",
        prediction=__import__("torch").tensor([[[1.0], [2.0], [3.0]]]),
        target=__import__("torch").tensor([[[1.5], [2.5], [3.5]]]),
        step=5,
        title="prediction demo",
    )

    payload, step = tracker.run.calls[0]
    assert step == 5
    assert payload["prediction/test"].caption == "prediction demo"


def test_tracker_log_attack_reconstruction_creates_wandb_image():
    torch = __import__("torch")
    tracker = Tracker.__new__(Tracker)
    tracker.run = _FakeRun()
    tracker.wandb = _FakeWandb()
    result = type("AttackResultStub", (), {
        "name": "DLG",
        "round_index": 2,
        "real_x": torch.tensor([[[1.0], [2.0], [3.0]]]),
        "reconstructed_x": torch.tensor([[[1.1], [1.9], [3.2]]]),
        "real_y": torch.tensor([[[4.0], [5.0]]]),
        "reconstructed_y": torch.tensor([[[4.1], [4.8]]]),
    })()

    tracker.log_attack_reconstruction("attack/DLG/reconstruction", result, step=2)

    payload, step = tracker.run.calls[0]
    assert step == 2
    assert "DLG" in payload["attack/DLG/reconstruction"].caption
