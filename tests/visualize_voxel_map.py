"""Visualize incremental voxel map across captured poses.

Usage:
  python tests/visualize_voxel_map.py --data-dir data/captures_20260123_140948 --voxel-size 0.05

Requirements:
  pip install open3d numpy

The script loads the first cloud_*.npy from each pose_* folder, applies the
corresponding transform from camera_poses_in_base.json, registers into the
voxel map, and visualizes after each registration.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - viewer dependency
    raise SystemExit(
        "open3d is required for visualization. Install with `pip install open3d`."
    ) from exc

# Make sure we can import the consumer and pybind module.
ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
BUILD_DIR = ROOT / "build"
for p in (str(PYTHON_DIR), str(BUILD_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from voxel_mapping_consumer import VoxelMappingConsumer  # noqa: E402


def load_transforms(poses_json: Path) -> dict[int, np.ndarray]:
    with poses_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    transforms = {}
    for entry in data:
        idx = int(entry["pose_index"])
        mat = np.array(entry["T_base_cam"], dtype=np.float32)
        if mat.shape != (4, 4):
            raise ValueError(f"Pose {idx} transform is not 4x4")
        transforms[idx] = mat
    return transforms


def first_cloud_in_pose(pose_dir: Path) -> Path | None:
    clouds = sorted(pose_dir.glob("cloud_*.npy"))
    return clouds[0] if clouds else None


def visualize(points: np.ndarray, title: str) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.visualization.draw_geometries([pcd], window_name=title)


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental voxel map visualization")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "captures_20260123_140948",
        help="Path to captures directory containing pose_* folders",
    )
    parser.add_argument("--voxel-size", type=float, default=0.005, help="Leaf size")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {args.data_dir}")

    poses_json = args.data_dir / "camera_poses_in_base.json"
    if not poses_json.is_file():
        raise SystemExit(f"Missing camera poses file: {poses_json}")

    transforms = load_transforms(poses_json)

    pose_dirs = sorted(p for p in args.data_dir.glob("pose_*"))
    if not pose_dirs:
        raise SystemExit("No pose_* folders found.")

    voxel = VoxelMappingConsumer(voxel_size=args.voxel_size)

    for pose_dir in pose_dirs:
        # parse pose index from folder name suffix
        try:
            pose_idx = int(pose_dir.name.split("_")[-1])
        except ValueError:
            print(f"Skipping non-standard folder name: {pose_dir}")
            continue

        transform = transforms.get(pose_idx)
        if transform is None:
            print(f"No transform for pose index {pose_idx}; skipping")
            continue

        cloud_path = first_cloud_in_pose(pose_dir)
        if cloud_path is None:
            print(f"No cloud_*.npy in {pose_dir}; skipping")
            continue

        points = np.load(cloud_path).astype(np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            print(f"Invalid point cloud shape in {cloud_path}: {points.shape}; skipping")
            continue

        points = points[abs(points).sum(axis=1) > 0]  # remove NaN/inf points
        points = points[abs(points).sum(axis=1) < 10]  # remove NaN/inf points

        print(f"Registering {cloud_path.name} (pose {pose_idx}) with {points.shape[0]} points")
        voxel.register_cloud(points, transform)

        map_points = voxel.retrieve_map()
        if map_points.size == 0:
            print("Map is empty after registration; skipping visualization")
            continue

        print(f"Voxel map now has {map_points.shape[0]} points")
        visualize(map_points, title=f"Voxel map after pose {pose_idx}")


if __name__ == "__main__":
    main()
