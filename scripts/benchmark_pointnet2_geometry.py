"""Synthetic first-SA geometry microbenchmarks; no dataset or experiment writes."""

import argparse
import json
import time
from pathlib import Path

import torch

from agri3d.models.pointnet2_ops import (
    farthest_point_sample,
    index_points,
    interpolate_features,
    query_ball_point,
    square_distance,
)
from agri3d.performance.timing import statistics, telemetry
from agri3d.training.checkpoints import seed_everything
from agri3d.training.pointnet2_runner import environment, select_device


def geometry_benchmark(
    points=10000, centroids=1024, batch_size=2, repeats=3, warmup=1, device_name="cpu"
):
    if min(points, centroids, batch_size, repeats) < 1 or warmup < 0:
        raise ValueError("Invalid geometry benchmark counts")
    seed_everything(42)
    device = select_device(device_name)
    xyz = torch.randn(batch_size, points, 3, device=device)
    # Synthetic clouds are not representative plant neighborhoods or validation metrics.
    indices = farthest_point_sample(xyz, centroids)
    centers = index_points(xyz, indices)
    features = torch.randn(batch_size, centers.shape[1], 64, device=device)
    operations = {
        "fps": lambda: farthest_point_sample(xyz, centroids),
        "square_distance_chunk128": lambda: square_distance(centers[:, :128], xyz),
        "topk_chunk128": lambda: distances.topk(
            min(32, points), dim=-1, largest=False, sorted=True
        ),
        "index_points_centroids": lambda: index_points(xyz, indices),
        "radius_query_all_centroids": lambda: query_ball_point(
            0.1, 32, xyz, centers, 128
        ),
        "interpolation_to_input": lambda: interpolate_features(
            xyz, centers, features, 128
        ),
    }
    distances = square_distance(centers[:, :128], xyz)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    results = {}
    with torch.no_grad():
        for name, operation in operations.items():
            for _ in range(warmup):
                operation()
            sync()
            values = []
            for _ in range(repeats):
                sync()
                start = time.perf_counter()
                operation()
                sync()
                values.append((time.perf_counter() - start) * 1000)
            results[name] = statistics(values)
    return {
        "scope": "Synthetic isolated geometry wall timings; not Colab speedup or end-to-end model performance",
        "batch_size": batch_size,
        "input_points": points,
        "centroids": centers.shape[1],
        "warmup": warmup,
        "repeats": repeats,
        "units": "ms",
        "torch_threads": torch.get_num_threads(),
        "environment": environment(device),
        "telemetry": telemetry(device),
        "results": results,
        "cost": "FPS O(B*N*S), O(B*N) work memory; radius distance O(B*N*S), chunk128 O(B*128*N); 3NN interpolation O(B*N*S) chunked",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=10000)
    parser.add_argument("--centroids", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = geometry_benchmark(
        args.points,
        args.centroids,
        args.batch_size,
        args.repeats,
        args.warmup,
        args.device,
    )
    path = Path(args.output)
    if path.exists():
        parser.error("Output exists; choose a new report path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
