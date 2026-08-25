import numpy as np

from fedlab.datasets.rare_earth import Standardizer
from fedlab.utils.tracking import Tracker, _attack_reconstruction_figure, _prediction_figure, _to_series


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
        input_series=__import__("torch").tensor([[[0.0], [0.5], [1.0], [1.5]]]),
        prediction=__import__("torch").tensor([[[1.0], [2.0], [3.0]]]),
        target=__import__("torch").tensor([[[1.5], [2.5], [3.5]]]),
        step=5,
        title="prediction demo",
        scaler=Standardizer(mean=np.array([10.0], dtype="float32"), std=np.array([2.0], dtype="float32")),
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
        "reference_label": "nearest_client_train",
        "reference_x": torch.tensor([[[1.0], [2.0], [3.0]]]),
        "reconstructed_x": torch.tensor([[[1.1], [1.9], [3.2]]]),
        "reference_y": torch.tensor([[[4.0], [5.0]]]),
        "reconstructed_y": torch.tensor([[[4.1], [4.8]]]),
    })()

    tracker.log_attack_reconstruction("attack/DLG/reconstruction", result, step=2)

    payload, step = tracker.run.calls[0]
    assert step == 2
    assert "DLG" in payload["attack/DLG/reconstruction"].caption


def test_to_series_preserves_time_axis_for_common_forecasting_shapes():
    torch = __import__("torch")

    assert _to_series(torch.tensor([[[1.0], [2.0], [3.0]]])) == [1.0, 2.0, 3.0]
    assert _to_series(torch.tensor([[1.0], [2.0], [3.0]])) == [1.0, 2.0, 3.0]
    assert _to_series(torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])) == [1.0, 2.0, 3.0]


def test_prediction_figure_draws_single_axis_with_history_and_forecast():
    torch = __import__("torch")

    figure = _prediction_figure(
        torch.tensor([[[0.0], [0.5], [1.0], [1.5]]]),
        torch.tensor([[[1.1], [1.9], [3.2]]]),
        torch.tensor([[[1.0], [2.0], [3.0]]]),
        title="demo",
    )

    assert figure is not None
    assert len(figure.axes) == 1
    axis = figure.axes[0]
    labels = [line.get_label() for line in axis.lines]
    assert "input_x" in labels
    assert "target_y" in labels
    assert "prediction_y" in labels


def test_prediction_figure_can_restore_original_scale_with_scaler():
    torch = __import__("torch")

    scaler = Standardizer(mean=np.array([10.0], dtype="float32"), std=np.array([2.0], dtype="float32"))
    figure = _prediction_figure(
        torch.tensor([[[0.0], [0.5]]]),
        torch.tensor([[[1.0], [2.0]]]),
        torch.tensor([[[1.5], [2.5]]]),
        title="scaled",
        scaler=scaler,
    )

    axis = figure.axes[0]
    input_line = next(line for line in axis.lines if line.get_label() == "input_x")
    target_line = next(line for line in axis.lines if line.get_label() == "target_y")
    prediction_line = next(line for line in axis.lines if line.get_label() == "prediction_y")
    assert list(input_line.get_ydata()) == [10.0, 11.0]
    assert list(target_line.get_ydata()) == [13.0, 15.0]
    assert list(prediction_line.get_ydata()) == [12.0, 14.0]


def test_attack_reconstruction_figure_supports_image_classification_payloads():
    torch = __import__("torch")
    result = type("AttackResultStub", (), {
        "name": "DLG",
        "client_id": "client1",
        "round_index": 0,
        "sample_index": 0,
        "reference_label": "nearest_client_train",
        "reference_x": torch.rand(1, 1, 4, 4),
        "reconstructed_x": torch.rand(1, 1, 4, 4),
        "reference_y": torch.tensor([2]),
        "reconstructed_y": torch.randn(1, 3),
    })()

    figure = _attack_reconstruction_figure(result)

    assert figure is not None
    assert len(figure.axes) == 4
    assert figure.axes[0].images
    assert figure.axes[1].images
