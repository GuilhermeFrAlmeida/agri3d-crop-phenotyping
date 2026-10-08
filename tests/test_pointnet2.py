"""Geometry correctness and CPU network/loss integration on synthetic inputs."""

import pytest
import torch
from torch.utils.data import DataLoader

from agri3d.models import PointNet2Config, PointNet2Segmentation
from agri3d.models import pointnet2_ops as ops
from agri3d.models.pointnet2 import FeaturePropagation, SetAbstraction
from agri3d.training.checkpoints import seed_everything
from agri3d.training.engine import run_epoch
from agri3d.training.losses import masked_cross_entropy


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config():
    return PointNet2Config(npoints=(16, 8, 4, 2), nsamples=(8, 8, 8, 8), chunk_size=8)


def test_distances_and_indexing():
    a = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]]])
    b = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0]]])
    torch.testing.assert_close(
        ops.square_distance(a, b), torch.tensor([[[0.0, 1.0, 9.0], [9.0, 8.0, 6.0]]])
    )
    points = torch.arange(24).reshape(2, 4, 3).float()
    indices = torch.tensor([[[0, 3], [2, 1]], [[3, 0], [1, 2]]])
    grouped = ops.index_points(points, indices)
    assert grouped.shape == (2, 2, 2, 3)
    for batch in range(2):
        torch.testing.assert_close(grouped[batch], points[batch][indices[batch]])
    with pytest.raises(ValueError):
        ops.square_distance(a, b.expand(2, -1, -1))
    with pytest.raises(ValueError):
        ops.index_points(points, torch.tensor([[4], [0]]))
    with pytest.raises(ValueError):
        ops.index_points(points, torch.tensor([[-1], [0]]))
    with pytest.raises(ValueError):
        ops.index_points(points, indices.float())


def test_fps_unique_deterministic_coincident():
    xyz = torch.zeros(2, 7, 3)
    indices = ops.farthest_point_sample(xyz, 10)
    torch.testing.assert_close(indices, torch.arange(7).repeat(2, 1))
    line = torch.zeros(1, 5, 3)
    line[0, :, 0] = torch.arange(5)
    assert ops.farthest_point_sample(line, 3).tolist() == [[0, 4, 2]]
    torch.testing.assert_close(
        ops.farthest_point_sample(line, 3), ops.farthest_point_sample(line, 3)
    )
    with pytest.raises(ValueError):
        ops.farthest_point_sample(line, 0)


def test_radius_query_short_and_empty_balls():
    xyz = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    centers = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    result = ops.query_ball_point(0.1, 5, xyz, centers, chunk_size=1)
    assert result.tolist() == [[[0, 0, 0, 0, 0], [2, 2, 2, 2, 2]]]
    torch.testing.assert_close(result, ops.query_ball_point(0.1, 5, xyz, centers, 128))
    with pytest.raises(ValueError):
        ops.query_ball_point(0, 3, xyz, centers)


def test_grouping_relative_coordinates():
    xyz = torch.randn(2, 11, 3)
    features = torch.randn(2, 11, 5)
    centers, groups = ops.sample_and_group(4, 0.01, 6, xyz, features)
    assert centers.shape == (2, 4, 3) and groups.shape == (2, 4, 6, 8)
    torch.testing.assert_close(groups[..., :3], torch.zeros_like(groups[..., :3]))
    with pytest.raises(ValueError):
        ops.sample_and_group(4, 1.0, 6, xyz, features[:, :-1])


def test_interpolation_values_gradient_and_degenerate_sources():
    sparse = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    dense = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    features = torch.tensor([[[2.0], [6.0]]], requires_grad=True)
    result = ops.interpolate_features(dense, sparse, features, chunk_size=1)
    torch.testing.assert_close(result, torch.tensor([[[2.0], [4.0], [6.0]]]))
    result.sum().backward()
    torch.testing.assert_close(features.grad, torch.tensor([[[1.5], [1.5]]]))
    torch.testing.assert_close(
        ops.interpolate_features(dense, sparse[:, :1], features[:, :1]),
        torch.full((1, 3, 1), 2.0),
    )
    sparse[:] = 0
    torch.testing.assert_close(
        ops.interpolate_features(dense[:, :1], sparse, features),
        torch.tensor([[[4.0]]]),
    )


def test_layer_shapes():
    xyz = torch.randn(2, 24, 3)
    features = torch.randn(2, 24, 5)
    centers, local = SetAbstraction(8, 0.5, 12, 5, (8, 16), chunk_size=3)(xyz, features)
    assert centers.shape == (2, 8, 3) and local.shape == (2, 8, 16)
    dense = FeaturePropagation(21, (16, 8), chunk_size=3)(xyz, centers, features, local)
    assert dense.shape == (2, 24, 8)
    assert FeaturePropagation(16, (8,))(
        xyz, centers[:, :1], None, local[:, :1]
    ).shape == (2, 24, 8)


@pytest.mark.parametrize("batch,n", [(1, 2), (1, 32), (2, 49)])
def test_model_forward_backward(batch, n):
    seed_everything(5)
    model = PointNet2Segmentation(small_config())
    xyz = torch.randn(batch, n, 3)
    logits = model(xyz)
    assert logits.shape == (batch, n, 2) and torch.isfinite(logits).all()
    labels = (xyz[..., 2] > 0).long()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.classifier[-1].weight.detach().clone()
    loss = masked_cross_entropy(
        logits, labels, torch.ones_like(labels, dtype=torch.bool)
    )
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, model.classifier[-1].weight)
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(xyz), model(xyz), rtol=0, atol=0)
        if batch == 2:
            torch.testing.assert_close(
                model(xyz),
                torch.cat([model(xyz[:1]), model(xyz[1:])]),
                rtol=1e-4,
                atol=1e-5,
            )


def test_model_invalid_input():
    model = PointNet2Segmentation(small_config())
    for xyz in (
        torch.zeros(1, 2, 6),
        torch.zeros(1, 1, 3),
        torch.full((1, 2, 3), float("nan")),
    ):
        with pytest.raises(ValueError):
            model(xyz)
    with pytest.raises(TypeError):
        model(torch.zeros(1, 3, 3, dtype=torch.float64))


def test_chunked_search_bounds(monkeypatch):
    original = ops.square_distance
    shapes = []

    def recording(a, b):
        shapes.append((a.shape[1], b.shape[1]))
        return original(a, b)

    monkeypatch.setattr(ops, "square_distance", recording)
    model = PointNet2Segmentation(small_config()).eval()
    with torch.no_grad():
        model(torch.randn(1, 257, 3))
    assert shapes and all(s <= 8 for s, _ in shapes)
    assert (257, 257) not in shapes


def test_masked_loss_and_padding_gradient():
    logits = torch.tensor(
        [[[2.0, -1.0], [-1.0, 2.0], [100.0, -100.0]]], requires_grad=True
    )
    labels = torch.tensor([[0, 1, -999]])
    mask = torch.tensor([[True, True, False]])
    expected = torch.nn.functional.cross_entropy(
        logits[:, :2].reshape(-1, 2), labels[:, :2].reshape(-1)
    )
    loss = masked_cross_entropy(logits, labels, mask)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.equal(logits.grad[:, 2], torch.zeros_like(logits.grad[:, 2]))
    with pytest.raises(ValueError):
        masked_cross_entropy(logits, labels, torch.zeros_like(mask))
    with pytest.raises(ValueError):
        masked_cross_entropy(logits, labels[:, :1], mask)


def test_epoch_metrics_use_valid_points_only():
    class Fixed(torch.nn.Module):
        def forward(self, points):
            return points[..., :2]

    items = [
        {
            "points": torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 5.0, 0.0]]),
            "labels": torch.tensor([0, 1, 0]),
            "valid_mask": torch.tensor([True, True, False]),
        }
    ]
    result = run_epoch(Fixed(), DataLoader(items), torch.device("cpu"))
    assert result["evaluated_points"] == 2 and result["mIoU"] == 1
    assert result["class_1_precision"] == 1 and result["class_0_recall"] == 1
    assert result["loss"] < 0.01


def test_dense_synthetic_overfit_converges_in_eval_mode():
    # Dense, plant-like support keeps the default radii meaningful on a small CPU fixture.
    # A train-mode-only check would miss delayed BatchNorm running-stat convergence.
    seed_everything(42)
    xyz = torch.randn(2, 128, 3) * 0.025
    xyz[:, :, 2] = torch.linspace(-0.8, 0.8, 128)
    labels = (xyz[:, :, 2] > 0).long()
    mask = torch.ones_like(labels, dtype=torch.bool)
    model = PointNet2Segmentation(
        PointNet2Config(npoints=(32, 16, 8, 4), nsamples=(8, 8, 8, 8), chunk_size=16)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.eval()
    with torch.no_grad():
        initial = masked_cross_entropy(model(xyz), labels, mask).item()
    for _ in range(160):
        model.train()
        optimizer.zero_grad()
        loss = masked_cross_entropy(model(xyz), labels, mask)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(xyz)
        final = masked_cross_entropy(logits, labels, mask).item()
    from agri3d.training.evaluate import SegmentationEvaluator

    metrics = SegmentationEvaluator().calculate_metrics(
        logits.argmax(-1).numpy(), labels.numpy()
    )
    assert final < initial * 0.2
    assert metrics["mIoU"] >= 0.95 and metrics["accuracy"] >= 0.98
