#!/usr/bin/env python3
"""
Interactive Multi-Pose Capture (Mech-Eye Python API)

At each pose, capture a 2D image, depth map, and point cloud.

Usage:
    python3 interactive_capture_mecheye.py --out captures --frames-per-pose 1
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Add voxelmapping paths
VOXEL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(VOXEL_ROOT / "python"))
sys.path.insert(0, str(VOXEL_ROOT / "build"))

try:
    from voxel_mapping_consumer import VoxelMappingConsumer
    VOXEL_AVAILABLE = True
except ImportError:
    VOXEL_AVAILABLE = False
    print("[WARN] VoxelMappingConsumer not available. Build with: cd build && cmake .. && make")

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import confirm_capture_3d

# Diana API (robot control / pose read)
# Docker container: /diana_api contains DianaApi.py, /diana_lib contains .so files
# Host: Conan package path
DIANA_DOCKER_API = "/diana_api"
DIANA_HOST_PATH = "/home/ahmad.hoteit/.conan/data/diana-api/2.18.1/ar/stable/package/aeddd718e2f218413aa0b9078e615c0fca8986f5/lib/python3/site-packages/diana_api"

if os.path.exists(DIANA_DOCKER_API):
    sys.path.insert(0, DIANA_DOCKER_API)
elif os.path.exists(DIANA_HOST_PATH):
    sys.path.insert(0, DIANA_HOST_PATH)

try:
    import DianaApi
    DIANA_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] DianaApi import failed: {e}")
    DIANA_AVAILABLE = False


# HAND EYE CALIBRATION EXTRINSICS
EXTRINSICS_CAM_EE = np.array([
    [-0.71365504342923614, -0.70020575462676071, -0.020208418433506396, 0.07363354839981448],
    [ 0.70042557727115762, -0.71369780379112213, -0.0062813667377777374, 0.13484894916013071],
    [-0.01002445471737245, -0.018637222199069551,  0.99977605705293926, 0.062427763940316391],
    [ 0.0,                  0.0,                   0.0,                  1.0]
], dtype=np.float32)


def pose_to_matrix(pose, robot_ip=None):
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
    rot = Rotation.from_euler('xyz', [rx, ry, rz], degrees=False)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot.as_matrix().astype(np.float32)
    T[:3, 3] = [x, y, z]
    return T


def assert_se3(T, name):
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


def compute_camera_pose_in_base(T_base_ee, T_cam_ee):
    """
    Compute T_base_cam from robot end-effector pose and hand-eye extrinsics.
    
    For eye-in-hand: T_base_cam = T_base_ee @ inv(T_cam_ee)
    For eye-to-hand: T_base_cam = T_cam_ee (fixed)
    """
    # Validate inputs
    assert_se3(T_base_ee, "T_base_ee")
    assert_se3(T_cam_ee, "T_cam_ee (extrinsics)")
    
    # Eye-in-hand configuration
    T_ee_cam = np.linalg.inv(T_cam_ee)
    T_base_cam = T_base_ee @ T_ee_cam
    
    # Validate output
    assert_se3(T_base_cam, "T_base_cam")
    return T_base_cam.astype(np.float32)


def print_camera_info(camera_info):
    """Print camera information."""
    if hasattr(camera_info, 'model'):
        print("  Model:", camera_info.model)
    if hasattr(camera_info, 'serial_number'):
        print("  Serial Number:", camera_info.serial_number)
    if hasattr(camera_info, 'ip_address'):
        print("  IP Address:", camera_info.ip_address)
    if hasattr(camera_info, 'subnet_mask'):
        print("  IP Subnet Mask:", camera_info.subnet_mask)
    if hasattr(camera_info, 'ip_gateway'):
        print("  IP Gateway:", camera_info.ip_gateway)
    if hasattr(camera_info, 'firmware_version'):
        print("  Firmware Version:", camera_info.firmware_version)
    print()


def discover_and_connect(camera, index=None):
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

    if not (0 <= index < len(camera_infos)):
        print("Invalid camera index.")
        return False

    status = camera.connect(camera_infos[index])
    show_error(status, "Connected to the camera successfully.")
    if not status.is_ok():
        return False

    print("Connected to the camera successfully.")
    return True


def _timestamp_string():
    ts = time.time()
    return f"{int(ts)}_{int((ts % 1) * 1e9):09d}"


def _capture_single_frame(camera, pose_dir, save_textured):
    """Capture frame and return (timestamp, points_array) or (None, None)."""
    frame_2d_3d = Frame2DAnd3D()
    status = camera.capture_2d_and_3d(frame_2d_3d)
    show_error(status, "Captured 2D + 3D frame successfully.")
    if not status.is_ok():
        return None, None

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

    cv2.imwrite(str(pose_dir / f"rgb_{ts_str}.png"), image_2d.data())

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
        f"Saved untextured point cloud: {cloud_path}"
    )

    if save_textured:
        textured_path = pose_dir / f"textured_cloud_{ts_str}.ply"
        show_error(
            frame_2d_3d.save_textured_point_cloud(FileFormat_PLY, str(textured_path)),
            f"Saved textured point cloud: {textured_path}"
        )

    return ts_str, points_data

_robot_actually_connected = False

def connect_robot(robot_ip):
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


def get_robot_pose(robot_ip):
    """Get current robot TCP pose as [x, y, z, rx, ry, rz]."""
    try:
        pose = [0.0] * 6
        DianaApi.getTcpPos(pose, ipAddress=robot_ip)
        return pose
    except Exception as e:
        print(f"[ERROR] Failed to get robot pose: {e}")
        return None


def get_robot_pose_and_matrix(robot_ip):
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


def disconnect_robot(robot_ip):
    global _robot_actually_connected
    if DIANA_AVAILABLE and _robot_actually_connected:
        try:
            DianaApi.destroySrv(robot_ip)
            print("[INFO] Robot disconnected")
            _robot_actually_connected = False
        except Exception as e:
            print(f"[WARN] Error during robot disconnect: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive multi-pose capture with Mech-Eye (Python SDK)"
    )
    parser.add_argument("--out", type=str, default="interactive_captures", help="Output directory")
    parser.add_argument("--camera-index", type=int, default=None, help="Camera index from discovery")
    parser.add_argument("--frames-per-pose", type=int, default=1, help="Frames to capture per pose")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between frames at a pose")
    parser.add_argument("--textured", action="store_true", help="Save textured point clouds")
    parser.add_argument("--ip", type=str, default="192.168.10.75", help="Diana robot IP address")
    parser.add_argument("--voxel-size", type=float, default=0.005, help="Voxel size in meters (default: 5mm)")
    parser.add_argument("--no-voxel", action="store_true", help="Disable real-time voxel mapping")
    parser.add_argument("--verify-frames", type=int, default=5, 
                        help="Number of frames for calibration verification (default: 5)")
    args = parser.parse_args()

    print("\n=== SAFETY NOTICE ===")
    print("Ensure the workspace is clear. Eyes on robot, hand on E-stop!")
    print("Press ENTER to continue...")
    input()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = Camera()
    if not discover_and_connect(camera, args.camera_index):
        return

    if not confirm_capture_3d():
        camera.disconnect()
        return

    robot_connected = connect_robot(args.ip)

    # Initialize voxel mapping
    voxel_map = None
    if VOXEL_AVAILABLE and not args.no_voxel:
        voxel_map = VoxelMappingConsumer(voxel_size=args.voxel_size)
        print(f"[VOXEL] Initialized voxel map with {args.voxel_size*1000:.1f}mm voxel size")
    elif not args.no_voxel:
        print("[WARN] Voxel mapping disabled (module not available)")

    poses = []
    pose_data = []
    camera_poses_in_base = []
    pose_idx = 0
    last_captured_pose = None  # Track last captured pose for duplicate warning

    while True:
        print("\n" + "=" * 50)
        print(f"POSE {pose_idx}")
        print("=" * 50)
        print("Press ENTER to capture, 'done' to finish, 'skip' to skip, or 'pose x y z rx ry rz'")
        
        # Show current pose for user reference (but don't use this value for capture)
        if robot_connected:
            preview_pose = get_robot_pose(args.ip)
            if preview_pose is not None:
                print(f"Current robot pose (preview): {[round(p, 6) for p in preview_pose]}")
                
                # Warn if robot hasn't moved since last capture
                if last_captured_pose is not None:
                    trans_diff = np.linalg.norm(np.array(preview_pose[:3]) - np.array(last_captured_pose[:3]))
                    rot_diff = np.linalg.norm(np.array(preview_pose[3:]) - np.array(last_captured_pose[3:]))
                    if trans_diff < 0.005 and rot_diff < 0.01:
                        print(f"[WARN] Robot has NOT moved since last capture! (diff: {trans_diff*1000:.1f}mm)")
                        print("       Move robot to a new pose, or type 'skip' to skip.")

        user_input = input("> ").strip().lower()
        if user_input == "done":
            break
        if user_input == "skip":
            continue

        # Manual pose entry
        manual_pose = None
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
        current_pose = None
        T_base_ee = None
        
        if manual_pose is not None:
            # provide manual pose
            current_pose = manual_pose
            T_base_ee = pose_to_matrix(current_pose, args.ip)
        elif robot_connected:
            # Read pose atomically with matrix conversion
            current_pose, T_base_ee = get_robot_pose_and_matrix(args.ip)
            if current_pose is None:
                print("[ERROR] Failed to read robot pose at capture time.")
                continue
            print(f"Captured robot pose: {[round(p, 6) for p in current_pose]}")
        else:
            print("[ERROR] No pose available. Enter manually: pose x y z rx ry rz")
            continue

        pose_dir = output_dir / f"pose_{pose_idx:02d}"
        pose_dir.mkdir(parents=True, exist_ok=True)

        # Compute camera pose in base frame
        T_base_cam = compute_camera_pose_in_base(T_base_ee, EXTRINSICS_CAM_EE)

        timestamps = []
        all_points = []
        for i in range(args.frames_per_pose):
            ts, points = _capture_single_frame(camera, pose_dir, args.textured)
            if ts is not None:
                timestamps.append(ts)
                if points is not None:
                    all_points.append(points)
            if i + 1 < args.frames_per_pose:
                time.sleep(max(0.0, args.interval))

        if len(timestamps) == 0:
            print("[WARN] No frames captured at this pose.")
            continue

        # Register to voxel map
        if voxel_map is not None and len(all_points) > 0:
            for points in all_points:
                # Filter invalid points (zeros, NaN, too far)
                valid = (np.abs(points).sum(axis=1) > 0) & (np.abs(points).sum(axis=1) < 10)
                valid_points = points[valid]
                if valid_points.shape[0] > 0:
                    voxel_map.register_cloud(valid_points, T_base_cam)
            map_points = voxel_map.retrieve_map()
            print(f"[VOXEL] Registered {len(all_points)} cloud(s). Map now has {map_points.shape[0]} voxels")

        pose_info = {
            "pose_index": pose_idx,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "pose": current_pose,
            "folder": f"pose_{pose_idx:02d}",
            "frames": timestamps,
            "frames_per_pose": args.frames_per_pose,
        }
        pose_data.append(pose_info)
        poses.append(current_pose)
        last_captured_pose = current_pose  # Track for duplicate pose warning
        
        # Save camera pose in base frame
        camera_poses_in_base.append({
            "pose_index": pose_idx,
            "T_base_cam": T_base_cam.tolist(),
            "T_base_ee": T_base_ee.tolist(),
        })
        
        pose_idx += 1

        print(f"[SAVE] Pose {pose_idx - 1} saved to {pose_dir}")

    index_path = output_dir / "capture_index.json"
    with open(index_path, "w") as f:
        json.dump(pose_data, f, indent=2)
    print(f"\n[SAVE] Capture index saved to {index_path}")

    # Save camera poses in base frame
    poses_path = output_dir / "camera_poses_in_base.json"
    with open(poses_path, "w") as f:
        json.dump(camera_poses_in_base, f, indent=2)
    print(f"[SAVE] Camera poses saved to {poses_path}")

    # Save extrinsics used
    extrinsics_path = output_dir / "extrinsics_used.json"
    with open(extrinsics_path, "w") as f:
        json.dump({"T_cam_ee": EXTRINSICS_CAM_EE.tolist()}, f, indent=2)
    print(f"[SAVE] Extrinsics saved to {extrinsics_path}")

    # Save final voxel map
    if voxel_map is not None:
        map_points = voxel_map.retrieve_map()
        voxel_map_path = output_dir / "voxel_map_merged.npy"
        np.save(str(voxel_map_path), map_points)
        print(f"[SAVE] Final voxel map saved to {voxel_map_path} ({map_points.shape[0]} points)")

    camera.disconnect()
    if robot_connected:
        disconnect_robot(args.ip)
    print("Disconnected from the camera successfully.")


if __name__ == "__main__":
    main()
