#!/usr/bin/env python3
"""
Interactive Multi-Pose Capture (Mech-Eye Python API)

At each pose, capture a 2D image, depth map, and point cloud.

Usage:
    python3 interactive_capture_mecheye.py --out captures --frames-per-pose 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# -----------------------------------------------------------------------------
# Paths / Imports (Voxel Mapping + Diana API)
# -----------------------------------------------------------------------------

VOXEL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(VOXEL_ROOT / "python"))
sys.path.insert(0, str(VOXEL_ROOT / "build"))

try:
    from voxel_mapping_consumer import VoxelMappingConsumer

    VOXEL_AVAILABLE = True
except ImportError:
    VOXEL_AVAILABLE = False
    print("[WARN] VoxelMappingConsumer not available. Build with: cd build && cmake .. && make")

from mecheye.shared import *  # noqa: F403
from mecheye.area_scan_3d_camera import *  # noqa: F403
from mecheye.area_scan_3d_camera_utils import confirm_capture_3d

# Diana API (robot control / pose read)
# Docker container: /diana_api contains DianaApi.py, /diana_lib contains .so files
# Host: Conan package path
DIANA_DOCKER_API = "/diana_api"
DIANA_HOST_PATH = (
    "/home/ahmad.hoteit/.conan/data/diana-api/2.18.1/ar/stable/package/"
    "aeddd718e2f218413aa0b9078e615c0fca8986f5/lib/python3/site-packages/diana_api"
)

if os.path.exists(DIANA_DOCKER_API):
    sys.path.insert(0, DIANA_DOCKER_API)
elif "DIANA_LOCAL_SDK" in globals() and os.path.exists(DIANA_LOCAL_SDK):  # type: ignore[name-defined]
    sys.path.insert(0, DIANA_LOCAL_SDK)  # type: ignore[name-defined]
elif os.path.exists(DIANA_HOST_PATH):
    sys.path.insert(0, DIANA_HOST_PATH)

try:
    import DianaApi  # type: ignore

    DIANA_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] DianaApi import failed: {e}")
    DIANA_AVAILABLE = False

# -----------------------------------------------------------------------------
# Calibration / Extrinsics
# -----------------------------------------------------------------------------
# HAND EYE CALIBRATION EXTRINSICS
# NOTE: interpretation chosen: this matrix is stored and applied as T_ee_cam
# (end-effector FROM camera). Use direct composition:
#   T_base_cam = T_base_ee @ T_ee_cam
EXTRINSICS_CAM_EE = np.array(
    [
        [-0.71365504342923614, -0.70020575462676071, -0.020208418433506396, 0.07363354839981448],
        [0.70042557727115762, -0.71369780379112213, -0.0062813667377777374, 0.13484894916013071],
        [-0.01002445471737245, -0.018637222199069551, 0.99977605705293926, 0.062427763940316391],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

_robot_actually_connected = False


def _timestamp_string() -> str:
    ts = time.time()
    return f"{int(ts)}_{int((ts % 1) * 1e9):09d}"


def assert_se3(T: np.ndarray, name: str) -> None:
    """
    Validate that T is a proper SE(3) homogeneous transformation matrix.
    Bottom row must be [0, 0, 0, 1], and values must be finite.
    """
    T = np.asarray(T)
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-3):
        print(f"[FATAL] {name} bottom row wrong: {T[3, :]}")
        raise ValueError(f"{name} is not a valid SE(3) matrix")
    if not np.isfinite(T).all():
        raise ValueError(f"{name} has NaN/Inf")


def pose_to_matrix(pose: List[float], robot_ip: Optional[str] = None) -> np.ndarray:
    """
    Convert robot pose [x, y, z, rx, ry, rz] (meters, radians) to 4x4 transform.
    Uses DianaApi.pose2Homogeneous for correct conversion.
    Falls back to XYZ extrinsic Euler if DianaApi not available.
    """
    if DIANA_AVAILABLE:
        try:
            # Use Diana's native conversion (flat 16-element output)
            T_flat = [0.0] * 16
            DianaApi.pose2Homogeneous(pose, T_flat)
            # Diana API returns row-major flat array where the caller
            # expects column-major SE(3). Transpose to put translation
            # into the last column and bottom row to [0,0,0,1].
            return np.array(T_flat, dtype=np.float32).reshape(4, 4).T
        except Exception as e:
            print(f"[WARN] DianaApi.pose2Homogeneous failed: {e}, using fallback")

    # Fallback: XYZ extrinsic Euler (equivalent to xyz lowercase in scipy)
    x, y, z, rx, ry, rz = pose
    rot = Rotation.from_euler("xyz", [rx, ry, rz], degrees=False)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot.as_matrix().astype(np.float32)
    T[:3, 3] = [x, y, z]
    return T


def compute_camera_pose_in_base(T_base_ee: np.ndarray, T_cam_ee: np.ndarray) -> np.ndarray:
    """
    Compute T_base_cam from robot end-effector pose and hand-eye extrinsics.

    We interpret the stored extrinsics as T_ee_cam (end-effector FROM camera).
    Therefore use direct composition:
        T_base_cam = T_base_ee @ T_ee_cam
    """
    assert_se3(T_base_ee, "T_base_ee")
    assert_se3(T_cam_ee, "T_cam_ee (extrinsics)")

    # Direct composition (T_cam_ee is treated as T_ee_cam)
    T_base_cam = T_base_ee @ T_cam_ee

    assert_se3(T_base_cam, "T_base_cam")
    return T_base_cam.astype(np.float32)


def _transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply SE(3) transform to Nx3 points."""
    R = T[:3, :3]
    t = T[:3, 3]
    return (points @ R.T) + t


def _voxelize_xyzrgb(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Voxelize colored points by averaging XYZ and RGB per voxel.
    Returns (voxel_xyz, voxel_rgb_uint8).
    """
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    indices = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(indices, axis=0, return_inverse=True)
    n_vox = int(inverse.max()) + 1

    xyz_sum = np.zeros((n_vox, 3), dtype=np.float64)
    rgb_sum = np.zeros((n_vox, 3), dtype=np.float64)
    counts = np.zeros(n_vox, dtype=np.int64)

    np.add.at(xyz_sum, inverse, points.astype(np.float64))
    np.add.at(rgb_sum, inverse, colors.astype(np.float64))
    np.add.at(counts, inverse, 1)

    voxel_xyz = (xyz_sum / counts[:, None]).astype(np.float32)
    voxel_rgb = np.clip(np.round(rgb_sum / counts[:, None]), 0, 255).astype(np.uint8)
    return voxel_xyz, voxel_rgb


def _write_ply_xyzrgb(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write ASCII PLY with x y z red green blue."""
    n = points.shape[0]
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


# -----------------------------------------------------------------------------
# Camera Discovery / Connection
# -----------------------------------------------------------------------------

def print_camera_info(camera_info: Any) -> None:
    """Print camera information."""
    if hasattr(camera_info, "model"):
        print("  Model:", camera_info.model)
    if hasattr(camera_info, "serial_number"):
        print("  Serial Number:", camera_info.serial_number)
    if hasattr(camera_info, "ip_address"):
        print("  IP Address:", camera_info.ip_address)
    if hasattr(camera_info, "subnet_mask"):
        print("  IP Subnet Mask:", camera_info.subnet_mask)
    if hasattr(camera_info, "ip_gateway"):
        print("  IP Gateway:", camera_info.ip_gateway)
    if hasattr(camera_info, "firmware_version"):
        print("  Firmware Version:", camera_info.firmware_version)
    print()


def discover_and_connect(camera: "Camera", index: Optional[int] = None) -> bool:
    print("Discovering all available cameras...")
    camera_infos = Camera.discover_cameras()
    if len(camera_infos) == 0:
        print("No cameras found.")
        return False

    for i in range(len(camera_infos)):
        print("Camera index:", i)
        print_camera_info(camera_infos[i])

    if index is None:
        print("Please enter the index of the camera that you want to connect: ")
        while True:
            user_input = input().strip()
            if user_input.isdigit() and 0 <= int(user_input) < len(camera_infos):
                index = int(user_input)
                break
            print("Input invalid! Please enter a valid camera index: ")

    if index is None or not (0 <= index < len(camera_infos)):
        print("Invalid camera index.")
        return False

    status = camera.connect(camera_infos[index])
    show_error(status, "Connected to the camera successfully.")
    if not status.is_ok():
        return False

    print("Connected to the camera successfully.")
    return True


# -----------------------------------------------------------------------------
# Capture Helpers
# -----------------------------------------------------------------------------

def _capture_single_frame(
    camera: "Camera",
    pose_dir: Path,
    save_textured: bool,
    return_colors: bool = False,
) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Capture frame and return (timestamp, points[, colors]) or Nones on failure.
    points are in meters.
    """
    frame_2d_3d = Frame2DAnd3D()
    status = camera.capture_2d_and_3d(frame_2d_3d)
    show_error(status, "Captured 2D + 3D frame successfully.")
    if not status.is_ok():
        return (None, None, None) if return_colors else (None, None, None)

    ts_str = _timestamp_string()

    # 2D image
    try:
        frame_2d = frame_2d_3d.frame_2d()
    except Exception:
        frame_2d = Frame2D()
        show_error(camera.capture_2d(frame_2d), "Captured 2D frame successfully.")

    if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
        image_2d = frame_2d.get_gray_scale_image()
    else:
        image_2d = frame_2d.get_color_image()

    image_np = image_2d.data()
    cv2.imwrite(str(pose_dir / f"rgb_{ts_str}.png"), image_np)

    # Depth map + point cloud
    try:
        frame_3d = frame_2d_3d.frame_3d()
    except Exception:
        frame_3d = Frame3D()
        show_error(camera.capture_3d(frame_3d), "Captured 3D frame successfully.")

    depth_map = frame_3d.get_depth_map()
    cv2.imwrite(str(pose_dir / f"depth_{ts_str}.tiff"), depth_map.data())

    # Get point cloud as numpy array for voxel mapping
    point_cloud = frame_3d.get_untextured_point_cloud()
    points_data = np.array(point_cloud.data(), dtype=np.float32).reshape(-1, 3)
    # Convert from mm to meters
    points_data = points_data / 1000.0
    # Save as .npy for voxel mapping
    np.save(str(pose_dir / f"cloud_{ts_str}.npy"), points_data)

    cloud_path = pose_dir / f"cloud_{ts_str}.ply"
    show_error(
        frame_3d.save_untextured_point_cloud(FileFormat_PLY, str(cloud_path)),
        f"Saved untextured point cloud: {cloud_path}",
    )

    if save_textured:
        textured_path = pose_dir / f"textured_cloud_{ts_str}.ply"
        show_error(
            frame_2d_3d.save_textured_point_cloud(FileFormat_PLY, str(textured_path)),
            f"Saved textured point cloud: {textured_path}",
        )

    if return_colors:
        if image_np.ndim == 2:
            color_image = np.repeat(image_np[:, :, None], 3, axis=2)
        else:
            color_image = image_np[:, :, :3]
        colors_data = color_image.reshape(-1, 3).astype(np.uint8)
        if colors_data.shape[0] != points_data.shape[0]:
            print(
                f"[WARN] RGB/point count mismatch ({colors_data.shape[0]} vs {points_data.shape[0]}), "
                "skipping textured merge for this frame."
            )
            colors_data = None
        return ts_str, points_data, colors_data

    return ts_str, points_data, None


# -----------------------------------------------------------------------------
# Robot Connection / Pose Read
# -----------------------------------------------------------------------------

def connect_robot(robot_ip: str) -> bool:
    """Connect to Diana robot."""
    global _robot_actually_connected
    _robot_actually_connected = False

    if not DIANA_AVAILABLE:
        print("[WARN] DianaApi not available. Robot pose will be manual input.")
        return False

    try:
        # Clear any stale connection first
        try:
            DianaApi.destroySrv(robot_ip)
            time.sleep(0.5)
        except Exception:
            pass

        # Clear error state before connecting
        try:
            DianaApi.setLastError(0, robot_ip)
        except Exception:
            pass

        # Initialize connection
        # srv_net_st = (ip_address, heartbeat_port, robot_state_port, srv_port, realtime_port, passthrough_port)
        init_ret = DianaApi.initSrv((robot_ip, 0, 0, 0, 0, 0))
        if not init_ret:
            print(f"[ERROR] Robot connection failed at {robot_ip}")
            return False

        # Read TCP pose to verify connection
        test_pose = [0.0] * 6
        ret = DianaApi.getTcpPos(test_pose, ipAddress=robot_ip)

        if not ret or sum(abs(p) for p in test_pose) < 1e-9:
            print(f"[WARN] Connected but received zero pose - robot may need power cycle")
            return False

        _robot_actually_connected = True
        print(f"[INFO] Robot connected: {robot_ip}")
        print(f"[INFO] Current TCP: {[round(p, 4) for p in test_pose]}")
        return True

    except Exception as e:
        print(f"[ERROR] Robot connection failed: {e}")
        return False


def disconnect_robot(robot_ip: str) -> None:
    global _robot_actually_connected
    if DIANA_AVAILABLE and _robot_actually_connected:
        try:
            DianaApi.destroySrv(robot_ip)
            print("[INFO] Robot disconnected")
            _robot_actually_connected = False
        except Exception as e:
            print(f"[WARN] Error during robot disconnect: {e}")


def get_robot_pose(robot_ip: str) -> Optional[List[float]]:
    """Get current robot TCP pose as [x, y, z, rx, ry, rz]."""
    try:
        pose = [0.0] * 6
        DianaApi.getTcpPos(pose, ipAddress=robot_ip)
        return pose
    except Exception as e:
        print(f"[ERROR] Failed to get robot pose: {e}")
        return None


def get_robot_pose_and_matrix(robot_ip: str) -> Tuple[Optional[List[float]], Optional[np.ndarray]]:
    """
    Read current robot TCP pose AND convert to 4x4 matrix atomically.
    Returns (pose, T_base_ee) or (None, None) on failure.
    """
    try:
        pose = [0.0] * 6
        DianaApi.getTcpPos(pose, ipAddress=robot_ip)

        # Convert to homogeneous matrix using Diana's native function (flat 16-element output)
        T_flat = [0.0] * 16
        DianaApi.pose2Homogeneous(pose, T_flat)
        # Diana API returns a row-major flat matrix; transpose to standard SE(3)
        T_base_ee = np.array(T_flat, dtype=np.float32).reshape(4, 4).T

        return pose, T_base_ee
    except Exception as e:
        print(f"[ERROR] Failed to get robot pose: {e}")
        return None, None


# -----------------------------------------------------------------------------
# Predefined Poses
# -----------------------------------------------------------------------------

def _parse_pose_entry(entry: Any, idx: int) -> List[float]:
    """Parse one pose entry into [x, y, z, rx, ry, rz]."""
    if isinstance(entry, dict):
        if "pose" in entry:
            values = entry["pose"]
        else:
            keys = ["x", "y", "z", "rx", "ry", "rz"]
            if not all(k in entry for k in keys):
                raise ValueError(f"Pose #{idx} dict must contain either 'pose' or keys {keys}")
            values = [entry[k] for k in keys]
    else:
        values = entry

    if len(values) != 6:
        raise ValueError(f"Pose #{idx} must have 6 values, got {len(values)}")
    return [float(v) for v in values]


def load_predefined_poses(poses_file: str) -> List[List[float]]:
    """
    Load predefined TCP poses from JSON/TXT/CSV.

    JSON supports either:
      - [ [x,y,z,rx,ry,rz], ... ]
      - { "poses": [ ... ] }
      - list entries as {x,y,z,rx,ry,rz} or {"pose":[...]}

    TXT/CSV supports one pose per line (comma or whitespace separated, 6 values).
    """
    path = Path(poses_file)
    if not path.exists():
        raise FileNotFoundError(f"Poses file not found: {path}")

    poses: List[List[float]] = []
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "poses" not in data:
                raise ValueError("JSON dict poses file must contain key 'poses'")
            data = data["poses"]
        if not isinstance(data, list):
            raise ValueError("JSON poses file must contain a list of poses")
        for i, entry in enumerate(data):
            poses.append(_parse_pose_entry(entry, i))
    else:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.replace(",", " ").split()
                if len(fields) != 6:
                    raise ValueError(f"Line {i + 1} in {path} must contain 6 values")
                poses.append([float(v) for v in fields])

    if len(poses) == 0:
        raise ValueError(f"No poses found in {path}")
    return poses


# -----------------------------------------------------------------------------
# Robot Motion Helpers
# -----------------------------------------------------------------------------

def _wait_until_reached_pose(
    robot_ip: str,
    target_pose: List[float],
    timeout_s: float,
    pos_tol_m: float,
    rot_tol_rad: float,
) -> Tuple[bool, List[float], float, float]:
    """Poll TCP pose until it reaches target tolerances or timeout."""
    deadline = time.time() + timeout_s
    last_pose: Optional[List[float]] = None

    while time.time() < deadline:
        cur_pose = [0.0] * 6
        ok = DianaApi.getTcpPos(cur_pose, ipAddress=robot_ip)
        if ok:
            last_pose = cur_pose
            pos_err = float(np.linalg.norm(np.array(cur_pose[:3]) - np.array(target_pose[:3])))
            rot_err = float(np.linalg.norm(np.array(cur_pose[3:]) - np.array(target_pose[3:])))
            if pos_err <= pos_tol_m and rot_err <= rot_tol_rad:
                return True, cur_pose, pos_err, rot_err
        time.sleep(0.05)

    if last_pose is None:
        last_pose = [0.0] * 6

    pos_err = float(np.linalg.norm(np.array(last_pose[:3]) - np.array(target_pose[:3])))
    rot_err = float(np.linalg.norm(np.array(last_pose[3:]) - np.array(target_pose[3:])))
    return False, last_pose, pos_err, rot_err


def move_robot_to_pose(
    robot_ip: str,
    target_pose: List[float],
    motion: str,
    vel: float,
    acc: float,
    timeout_s: float,
    pos_tol_m: float,
    rot_tol_rad: float,
    settle_time_s: float,
) -> Tuple[bool, Optional[List[float]], Optional[float], Optional[float]]:
    """Move robot to target TCP pose and wait until within tolerance."""
    if motion == "movej":
        ok = DianaApi.moveJToPose(target_pose, vel, acc, ipAddress=robot_ip)
    elif motion == "movel":
        ok = DianaApi.moveLToPose(target_pose, vel, acc, ipAddress=robot_ip)
    else:
        raise ValueError(f"Unsupported motion type: {motion}")

    if not ok:
        return False, None, None, None

    reached, reached_pose, pos_err, rot_err = _wait_until_reached_pose(
        robot_ip, target_pose, timeout_s, pos_tol_m, rot_tol_rad
    )
    if reached and settle_time_s > 0:
        time.sleep(settle_time_s)
    return reached, reached_pose, pos_err, rot_err


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive multi-pose capture with Mech-Eye (Python SDK)")
    parser.add_argument("--out", type=str, default="captures", help="Output directory")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index from discovery")
    parser.add_argument("--frames-per-pose", type=int, default=1, help="Frames to capture per pose")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between frames at a pose")
    parser.add_argument("--textured", action="store_true", help="Save textured point clouds")
    parser.add_argument("--ip", type=str, default="192.168.10.75", help="Diana robot IP address")
    parser.add_argument("--voxel-size", type=float, default=0.002, help="Voxel size in meters (default: 5mm)")
    parser.add_argument("--no-voxel", action="store_true", help="Disable real-time voxel mapping")
    parser.add_argument(
        "--verify-frames",
        type=int,
        default=5,
        help="Number of frames for calibration verification (default: 5)",
    )
    parser.add_argument(
        "--poses-file",
        type=str,
        default=None,
        help="Path to predefined TCP poses for fully automatic capture",
    )
    parser.add_argument(
        "--motion",
        choices=["movej", "movel"],
        default="movej",
        help="Robot motion primitive for automatic mode",
    )
    parser.add_argument(
        "--motion-vel",
        type=float,
        default=0.1,
        help="Motion velocity parameter for Diana moveJToPose/moveLToPose",
    )
    parser.add_argument(
        "--motion-acc",
        type=float,
        default=0.1,
        help="Motion acceleration parameter for Diana moveJToPose/moveLToPose",
    )
    parser.add_argument(
        "--motion-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for robot to reach each target pose",
    )
    parser.add_argument(
        "--motion-pos-tol",
        type=float,
        default=0.003,
        help="Position tolerance in meters for automatic mode",
    )
    parser.add_argument(
        "--motion-rot-tol",
        type=float,
        default=0.03,
        help="Rotation tolerance in radians for automatic mode",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.3,
        help="Extra settle time in seconds after robot reaches target",
    )
    args = parser.parse_args()

    print("\n=== SAFETY NOTICE ===")
    print("Ensure the workspace is clear. Eyes on robot, hand on E-stop!")
    print("Press ENTER to continue...")
    input()

    output_root = Path(args.out)
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(output_root), 0o777)
        except Exception:
            pass
    except Exception as e:
        print(f"[WARN] Could not create output directory '{output_root}': {e}")
        fallback = Path("/tmp") / f"voxelmapping_captures_{int(time.time())}"
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            os.chmod(str(fallback), 0o777)
            print(f"[INFO] Falling back to: {fallback}")
            output_root = fallback
        except Exception as e2:
            print(f"[ERROR] Failed to create fallback output directory: {e2}")
            return

    session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_root / session_name
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            os.chmod(str(output_dir), 0o777)
        except Exception:
            pass
    except FileExistsError:
        output_dir = output_root / f"{session_name}_{int(time.time() * 1000) % 1000:03d}"
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            os.chmod(str(output_dir), 0o777)
        except Exception:
            pass
    except Exception as e:
        print(f"[ERROR] Failed to create session directory '{output_dir}': {e}")
        return

    print(f"[INFO] Capture session directory: {output_dir}")

    camera = Camera()
    if not discover_and_connect(camera, args.camera_index):
        return

    if not confirm_capture_3d():
        camera.disconnect()
        return

    robot_connected = connect_robot(args.ip)

    auto_poses: Optional[List[List[float]]] = None
    if args.poses_file is not None:
        if not robot_connected:
            print("[ERROR] Automatic mode requires robot connection.")
            camera.disconnect()
            return
        try:
            auto_poses = load_predefined_poses(args.poses_file)
            print(f"[AUTO] Loaded {len(auto_poses)} target poses from {args.poses_file}")
            print(
                f"[AUTO] Motion={args.motion} vel={args.motion_vel} acc={args.motion_acc} "
                f"timeout={args.motion_timeout}s"
            )
        except Exception as e:
            print(f"[ERROR] Failed to load poses file: {e}")
            camera.disconnect()
            if robot_connected:
                disconnect_robot(args.ip)
            return

    voxel_map = None
    if VOXEL_AVAILABLE and not args.no_voxel:
        voxel_map = VoxelMappingConsumer(voxel_size=args.voxel_size)
        print(f"[VOXEL] Initialized voxel map with {args.voxel_size * 1000:.1f}mm voxel size")
    elif not args.no_voxel:
        print("[WARN] Voxel mapping disabled (module not available)")

    poses: List[List[float]] = []
    pose_data: List[Dict[str, Any]] = []
    camera_poses_in_base: List[Dict[str, Any]] = []

    textured_world_points: List[np.ndarray] = []
    textured_world_colors: List[np.ndarray] = []

    pose_idx = 0
    last_captured_pose: Optional[List[float]] = None

    while True:
        if auto_poses is not None and pose_idx >= len(auto_poses):
            print("\n[AUTO] Completed all predefined poses.")
            break

        print("\n" + "=" * 50)
        print(f"POSE {pose_idx}")
        print("=" * 50)

        if auto_poses is None:
            print("Press ENTER to capture, 'done' to finish, 'skip' to skip, or 'pose x y z rx ry rz'")
        else:
            print("Automatic mode: moving robot to predefined target and capturing.")

        # Preview pose in manual mode if robot is connected
        if auto_poses is None and robot_connected:
            preview_pose = get_robot_pose(args.ip)
            if preview_pose is not None:
                print(f"Current robot pose (preview): {[round(p, 6) for p in preview_pose]}")

                if last_captured_pose is not None:
                    trans_diff = float(np.linalg.norm(np.array(preview_pose[:3]) - np.array(last_captured_pose[:3])))
                    rot_diff = float(np.linalg.norm(np.array(preview_pose[3:]) - np.array(last_captured_pose[3:])))
                    if trans_diff < 0.005 and rot_diff < 0.01:
                        print(f"[WARN] Robot has NOT moved since last capture! (diff: {trans_diff * 1000:.1f}mm)")
                        print("       Move robot to a new pose, or type 'skip' to skip.")

        # Manual mode input
        manual_pose: Optional[List[float]] = None
        if auto_poses is None:
            user_input = input("> ").strip().lower()
            if user_input == "done":
                break
            if user_input == "skip":
                continue

            if user_input.startswith("pose "):
                try:
                    parts = user_input.replace("pose ", "").split()
                    manual_pose = [float(p) for p in parts]
                    if len(manual_pose) != 6:
                        print("[ERROR] Pose must have 6 values: x y z rx ry rz")
                        continue
                except ValueError:
                    print("[ERROR] Invalid pose format. Use: pose x y z rx ry rz")
                    continue
            elif user_input != "":
                print(f"[WARN] Unknown command: {user_input}")
                continue

        # === CRITICAL: Read pose RIGHT BEFORE capture ===
        current_pose: Optional[List[float]] = None
        T_base_ee: Optional[np.ndarray] = None

        if auto_poses is not None:
            target_pose = auto_poses[pose_idx]
            print(f"[AUTO] Target TCP pose: {[round(p, 6) for p in target_pose]}")

            moved, reached_pose, pos_err, rot_err = move_robot_to_pose(
                args.ip,
                target_pose,
                args.motion,
                args.motion_vel,
                args.motion_acc,
                args.motion_timeout,
                args.motion_pos_tol,
                args.motion_rot_tol,
                args.settle_time,
            )
            if not moved:
                print("[ERROR] Failed to reach target pose. Stopping automatic capture.")
                if reached_pose is not None:
                    print(
                        f"[ERROR] Last pose error: pos={pos_err:.4f}m rot={rot_err:.4f}rad, "
                        f"last_pose={[round(p, 6) for p in reached_pose]}"
                    )
                break

            print(f"[AUTO] Reached pose. Errors: pos={pos_err:.4f}m rot={rot_err:.4f}rad")

            current_pose, T_base_ee = get_robot_pose_and_matrix(args.ip)
            if current_pose is None or T_base_ee is None:
                print("[ERROR] Failed to read robot pose at capture time.")
                break
            print(f"[AUTO] Captured robot pose: {[round(p, 6) for p in current_pose]}")

        elif manual_pose is not None:
            current_pose = manual_pose
            T_base_ee = pose_to_matrix(current_pose, args.ip)

        elif robot_connected:
            current_pose, T_base_ee = get_robot_pose_and_matrix(args.ip)
            if current_pose is None or T_base_ee is None:
                print("[ERROR] Failed to read robot pose at capture time.")
                continue
            print(f"Captured robot pose: {[round(p, 6) for p in current_pose]}")

        else:
            print("[ERROR] No pose available. Enter manually: pose x y z rx ry rz")
            continue

        pose_dir = output_dir / f"pose_{pose_idx:02d}"
        try:
            pose_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(str(pose_dir), 0o777)
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] Could not create pose dir '{pose_dir}': {e}")
            try:
                pose_dir = output_dir / "pose_fallback"
                pose_dir.mkdir(parents=True, exist_ok=True)
                os.chmod(str(pose_dir), 0o777)
                print(f"[INFO] Using fallback pose dir: {pose_dir}")
            except Exception as e2:
                print(f"[ERROR] Failed to create any pose directory: {e2}")
                continue

        # Compute camera pose in base frame
        T_base_cam = compute_camera_pose_in_base(T_base_ee, EXTRINSICS_CAM_EE)

        timestamps: List[str] = []
        all_points: List[np.ndarray] = []
        all_colors: List[Optional[np.ndarray]] = []

        for i in range(args.frames_per_pose):
            if args.textured:
                ts, points, colors = _capture_single_frame(
                    camera,
                    pose_dir,
                    args.textured,
                    return_colors=True,
                )
            else:
                ts, points, colors = _capture_single_frame(camera, pose_dir, args.textured, return_colors=False)

            if ts is not None:
                timestamps.append(ts)
                if points is not None:
                    all_points.append(points)
                    all_colors.append(colors)

            if i + 1 < args.frames_per_pose:
                time.sleep(max(0.0, args.interval))

        if len(timestamps) == 0:
            print("[WARN] No frames captured at this pose.")
            continue

        # Register to voxel map and accumulate colored points for textured merge.
        if len(all_points) > 0:
            for frame_idx, points in enumerate(all_points):
                valid = (np.abs(points).sum(axis=1) > 0) & (np.abs(points).sum(axis=1) < 10)
                valid_points = points[valid]

                if voxel_map is not None and valid_points.shape[0] > 0:
                    voxel_map.register_cloud(valid_points, T_base_cam)

                frame_colors = all_colors[frame_idx] if frame_idx < len(all_colors) else None
                if args.textured and frame_colors is not None and frame_colors.shape[0] == points.shape[0]:
                    valid_colors = frame_colors[valid]
                    if valid_points.shape[0] > 0:
                        world_points = _transform_points(T_base_cam, valid_points)
                        textured_world_points.append(world_points)
                        textured_world_colors.append(valid_colors)

            if voxel_map is not None:
                map_points = voxel_map.retrieve_map()
                print(f"[VOXEL] Registered {len(all_points)} cloud(s). Map now has {map_points.shape[0]} voxels")

        pose_info: Dict[str, Any] = {
            "pose_index": pose_idx,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "pose": current_pose,
            "folder": f"pose_{pose_idx:02d}",
            "frames": timestamps,
            "frames_per_pose": args.frames_per_pose,
        }
        if auto_poses is not None:
            pose_info["target_pose"] = auto_poses[pose_idx]

        pose_data.append(pose_info)
        poses.append(current_pose)
        last_captured_pose = current_pose

        camera_poses_in_base.append(
            {
                "pose_index": pose_idx,
                "T_base_cam": T_base_cam.tolist(),
                "T_base_ee": T_base_ee.tolist(),
            }
        )

        pose_idx += 1
        print(f"[SAVE] Pose {pose_idx - 1} saved to {pose_dir}")

    index_path = output_dir / "capture_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(pose_data, f, indent=2)
    print(f"\n[SAVE] Capture index saved to {index_path}")

    poses_path = output_dir / "camera_poses_in_base.json"
    with open(poses_path, "w", encoding="utf-8") as f:
        json.dump(camera_poses_in_base, f, indent=2)
    print(f"[SAVE] Camera poses saved to {poses_path}")

    extrinsics_path = output_dir / "extrinsics_used.json"
    # Save using explicit key `T_ee_cam` to indicate the stored matrix is T_ee_cam
    with open(extrinsics_path, "w", encoding="utf-8") as f:
        json.dump({"T_ee_cam": EXTRINSICS_CAM_EE.tolist()}, f, indent=2)
    print(f"[SAVE] Extrinsics saved to {extrinsics_path} (key: T_ee_cam)")

    # Save final voxel map
    if voxel_map is not None:
        map_points = voxel_map.retrieve_map()
        voxel_map_path = output_dir / "voxel_map_merged.npy"
        try:
            np.save(str(voxel_map_path), map_points)
            print(f"[SAVE] Final voxel map saved to {voxel_map_path} ({map_points.shape[0]} points)")
        except Exception as e:
            print(f"[WARN] Failed to save voxel map to {voxel_map_path}: {e}")
            fallback_map = Path("/tmp") / f"voxel_map_merged_{int(time.time())}.npy"
            try:
                np.save(str(fallback_map), map_points)
                print(f"[SAVE] Final voxel map saved to fallback {fallback_map} ({map_points.shape[0]} points)")
            except Exception as e2:
                print(f"[ERROR] Failed to save fallback voxel map: {e2}")

    if args.textured:
        if len(textured_world_points) == 0:
            print("[WARN] --textured was enabled but no valid RGB points were accumulated for merged textured voxel map.")
        else:
            merged_xyz = np.concatenate(textured_world_points, axis=0)
            merged_rgb = np.concatenate(textured_world_colors, axis=0)
            voxel_xyz, voxel_rgb = _voxelize_xyzrgb(merged_xyz, merged_rgb, args.voxel_size)
            textured_voxel_map_path = output_dir / "voxel_map_merged_textured.ply"
            try:
                _write_ply_xyzrgb(str(textured_voxel_map_path), voxel_xyz, voxel_rgb)
                print(
                    f"[SAVE] Final textured voxel map saved to {textured_voxel_map_path} "
                    f"({voxel_xyz.shape[0]} points)"
                )
            except Exception as e:
                print(f"[WARN] Failed to save textured voxel map to {textured_voxel_map_path}: {e}")

    camera.disconnect()
    if robot_connected:
        disconnect_robot(args.ip)
    print("Disconnected from the camera successfully.")


if __name__ == "__main__":
    main()