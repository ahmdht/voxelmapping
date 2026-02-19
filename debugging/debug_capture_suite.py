#!/usr/bin/env python3
"""
Automated 2-pose capture suite for verifying hand-eye extrinsics.

Performs:
 - Connect camera and robot
 - Capture frames at Pose A
 - Move robot by a small displacement (auto or manual)
 - Capture frames at Pose B
 - Transform captured clouds using both extrinsics conventions
 - Fuse into voxel maps and save results per-run

Usage example:
  python3 debug_capture_suite.py --axis x --displacement 0.10 --frames-per-pose 1 --out captures/debug_move_x

Notes:
 - If `--auto-move` is provided the script will attempt to call DianaApi move functions
   (best-effort). If the call fails the script falls back to prompting the user to move
   the robot manually.
 - This script uses the existing interactive capture routines to perform live Mech-Eye captures.
"""

import argparse
import os
import time
from pathlib import Path
import numpy as np

# Import existing capture helpers
import interactive_capture_mecheye as ic
from python.voxel_mapping_consumer import VoxelMappingConsumer
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def save_ply(points, path):
    # Simple ASCII PLY writer for point clouds
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def attempt_auto_move(DianaApi, target_pose, ip, v=0.02, a=0.2):
    # Try a list of possible move function names (best-effort). Wrap in try/except.
    candidates = [
        "moveAbsJ",
        "moveJ",
        "moveL",
        "moveAbsTcp",
        "moveTcp",
        "moveTo",
        "move",
    ]
    for name in candidates:
        if hasattr(DianaApi, name):
            fn = getattr(DianaApi, name)
            try:
                print(f"[INFO] Attempting auto-move with DianaApi.{name}...")
                # Call with common kwarg ipAddress when available
                # Prefer calls that accept speed/accel
                try:
                    ret = fn(target_pose, v=v, a=a, ipAddress=ip)
                except TypeError:
                    try:
                        ret = fn(target_pose, v, a, ip)
                    except TypeError:
                        try:
                            ret = fn(target_pose, ipAddress=ip)
                        except TypeError:
                            try:
                                ret = fn(target_pose, ip)
                            except TypeError:
                                ret = fn(target_pose)
                print(f"[INFO] DianaApi.{name} returned: {ret}")
                return True
            except Exception as e:
                print(f"[WARN] DianaApi.{name} failed: {e}")
    return False


def collect_and_save_frames(camera, pose_dir, frames_per_pose, interval):
    pose_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(str(pose_dir), 0o777)
    except Exception:
        pass
    timestamps = []
    saved_points = []
    for i in range(frames_per_pose):
        ts, points = ic._capture_single_frame(camera, pose_dir, save_textured=False)
        if ts is not None:
            timestamps.append(ts)
        if points is not None:
            saved_points.append(points)
        if i + 1 < frames_per_pose:
            time.sleep(max(0.0, interval))
    # Ensure files in the pose_dir are writable by host users (permissive)
    try:
        for root, dirs, files in os.walk(pose_dir):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root, d), 0o777)
                except Exception:
                    pass
            for fn in files:
                try:
                    os.chmod(os.path.join(root, fn), 0o666)
                except Exception:
                    pass
    except Exception:
        pass
    return timestamps, saved_points


def run_single_test(args):
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    camera = ic.Camera()
    if not ic.discover_and_connect(camera, None):
        raise RuntimeError("Failed to connect to camera")
    if not ic.confirm_capture_3d():
        camera.disconnect()
        raise RuntimeError("Camera capture not confirmed")

    robot_connected = ic.connect_robot(args.ip)
    if not robot_connected:
        print("[WARN] Robot not connected; script will still attempt to read poses if possible.")

    # Pose A
    print("\n=== Capturing Pose A ===")
    pose_a, T_base_ee_a = ic.get_robot_pose_and_matrix(args.ip)
    if pose_a is None or T_base_ee_a is None:
        raise RuntimeError("Failed to read robot pose A")
    run_name = f"axis_{args.axis}_d{int(args.displacement*1000)}mm"
    run_dir = out_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pose_dir_a = run_dir / "pose_00"
    ts_a, points_a_list = collect_and_save_frames(camera, pose_dir_a, args.frames_per_pose, args.interval)

    # Compute target pose
    idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    target_pose = list(pose_a)
    target_pose[idx] += args.displacement

    # Move robot (auto or manual)
    print(f"\nTarget pose for Pose B: {target_pose}")
    if args.auto_move and ic.DIANA_AVAILABLE:
        confirm = input("Proceed with automatic move? Type YES to continue: ")
        if confirm.strip() == "YES":
            moved = False
            try:
                # Preferred API: moveLToPose + wait_move
                if hasattr(ic, "DianaApi") and hasattr(ic.DianaApi, "moveLToPose"):
                    try:
                        ic.DianaApi.moveLToPose(target_pose, v=args.move_speed, a=args.move_accel, ipAddress=args.ip)
                        if hasattr(ic.DianaApi, "wait_move"):
                            ic.DianaApi.wait_move()
                        elif hasattr(ic.DianaApi, "waitMove"):
                            ic.DianaApi.waitMove()
                        else:
                            time.sleep(1.0)
                        moved = True
                        print("Auto-move (moveLToPose) completed.")
                    except Exception as e:
                        print(f"[WARN] moveLToPose failed: {e}")
                # Fallback to best-effort mover
                if not moved:
                    ok = attempt_auto_move(ic.DianaApi, target_pose, args.ip, v=args.move_speed, a=args.move_accel)
                    if not ok:
                        input("Auto-move failed. Move robot to target pose manually, then press ENTER...")
                    else:
                        print("Auto-move invoked (fallback). Give robot time to move, then press ENTER to continue.")
                        input()
            except Exception as e:
                print(f"[WARN] Automatic move attempt raised: {e}")
                input("Automatic move raised an exception. Move robot manually, then press ENTER...")
        else:
            input("Automatic move cancelled. Move robot to target pose manually, then press ENTER...")
    else:
        # Manual mode
        input("Please move robot to the target pose, then press ENTER to continue...")

    # Pose B
    print("\n=== Capturing Pose B ===")
    pose_b, T_base_ee_b = ic.get_robot_pose_and_matrix(args.ip)
    if pose_b is None or T_base_ee_b is None:
        raise RuntimeError("Failed to read robot pose B")
    pose_dir_b = run_dir / "pose_01"
    ts_b, points_b_list = collect_and_save_frames(camera, pose_dir_b, args.frames_per_pose, args.interval)

    # Save pose metadata
    meta = {
        "axis": args.axis,
        "displacement_m": args.displacement,
        "pose_a": pose_a,
        "pose_b": pose_b,
        "timestamps_a": ts_a,
        "timestamps_b": ts_b,
    }
    import json
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    try:
        os.chmod(str(run_dir / "meta.json"), 0o666)
    except Exception:
        pass

    # Prepare point-cloud lists (concatenate frames per pose)
    def concat(points_list):
        if not points_list:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack(points_list)

    cloud_a = concat(points_a_list)
    cloud_b = concat(points_b_list)

    # Two extrinsics options
    T_cam_ee = ic.EXTRINSICS_CAM_EE.astype(np.float64)

    options = {
        "inv": lambda T_base_ee: T_base_ee @ np.linalg.inv(T_cam_ee),
        "direct": lambda T_base_ee: T_base_ee @ T_cam_ee,
    }

    modes = ["inv", "direct"] if args.extrinsics_mode == "both" else [args.extrinsics_mode]

    for mode in modes:
        print(f"\n=== Building fused map for mode: {mode} ===")
        T_base_cam_a = options[mode](T_base_ee_a)
        T_base_cam_b = options[mode](T_base_ee_b)

        # Transform clouds into base frame
        def transform_cloud(points, T):
            if points.shape[0] == 0:
                return points
            ones = np.ones((points.shape[0], 1), dtype=np.float64)
            ph = np.hstack([points.astype(np.float64), ones])
            pts_b = (T @ ph.T).T[:, :3]
            return pts_b.astype(np.float32)

        pts_a_base = transform_cloud(cloud_a, T_base_cam_a)
        pts_b_base = transform_cloud(cloud_b, T_base_cam_b)

        # Fuse via VoxelMappingConsumer if available
        vm = VoxelMappingConsumer(voxel_size=args.voxel_size)
        # Filter invalid points (zeros, NaN, too far) using same heuristic as interactive_capture_mecheye
        def _filter_valid(points):
            if points is None or points.shape[0] == 0:
                return points
            mask = (np.abs(points).sum(axis=1) > 0) & (np.abs(points).sum(axis=1) < 10)
            return points[mask]

        pts_a_base = _filter_valid(pts_a_base)
        pts_b_base = _filter_valid(pts_b_base)

        if pts_a_base.shape[0] > 0:
            vm.register_cloud(pts_a_base, np.eye(4, dtype=np.float32))
        if pts_b_base.shape[0] > 0:
            vm.register_cloud(pts_b_base, np.eye(4, dtype=np.float32))
        fused = vm.retrieve_map()

        out_npy = run_dir / f"voxel_map_{mode}.npy"
        np.save(str(out_npy), fused)
        out_ply = run_dir / f"voxel_map_{mode}.ply"
        save_ply(fused, out_ply)
        try:
            os.chmod(str(out_npy), 0o666)
        except Exception:
            pass
        try:
            os.chmod(str(out_ply), 0o666)
        except Exception:
            pass
        print(f"[SAVE] Saved fused map ({fused.shape[0]} points): {out_npy} and {out_ply}")

    # Compare inv vs direct maps if both present
    def _load_map(p: Path):
        if not p.exists():
            return None
        return np.load(str(p))

    inv_map = _load_map(run_dir / "voxel_map_inv.npy")
    direct_map = _load_map(run_dir / "voxel_map_direct.npy")

    def _map_stats(points):
        if points is None or points.size == 0:
            return None
        return {
            "count": int(points.shape[0]),
            "centroid": points.mean(axis=0).tolist(),
            "bbox_min": points.min(axis=0).tolist(),
            "bbox_max": points.max(axis=0).tolist(),
        }

    compare_report = {
        "run_dir": str(run_dir),
        "inv_present": inv_map is not None,
        "direct_present": direct_map is not None,
        "inv_stats": _map_stats(inv_map),
        "direct_stats": _map_stats(direct_map),
    }

    def _symmetric_nn(a, b):
        if a is None or b is None or a.size == 0 or b.size == 0:
            return None
        if cKDTree is not None:
            ta = cKDTree(a)
            tb = cKDTree(b)
            da, _ = tb.query(a, k=1)
            db, _ = ta.query(b, k=1)
            return {
                "a_to_b_mean": float(da.mean()),
                "a_to_b_median": float(np.median(da)),
                "b_to_a_mean": float(db.mean()),
                "b_to_a_median": float(np.median(db)),
                "symmetric_mean": float((da.mean() + db.mean()) / 2.0),
            }
        # fallback brute force if sizes small enough
        na = a.shape[0]
        nb = b.shape[0]
        if na * nb <= 5_000_000:
            dists = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
            da = dists.min(axis=1)
            db = dists.min(axis=0)
            return {
                "a_to_b_mean": float(da.mean()),
                "a_to_b_median": float(np.median(da)),
                "b_to_a_mean": float(db.mean()),
                "b_to_a_median": float(np.median(db)),
                "symmetric_mean": float((da.mean() + db.mean()) / 2.0),
            }
        # chunked computation for larger sets
        def nn_mean(src, dst):
            chunk = 10000
            parts = []
            for i in range(0, src.shape[0], chunk):
                s = src[i : i + chunk]
                d = np.sqrt(((s[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2))
                parts.append(d.min(axis=1))
            return np.hstack(parts)

        da = nn_mean(a, b)
        db = nn_mean(b, a)
        return {
            "a_to_b_mean": float(da.mean()),
            "a_to_b_median": float(np.median(da)),
            "b_to_a_mean": float(db.mean()),
            "b_to_a_median": float(np.median(db)),
            "symmetric_mean": float((da.mean() + db.mean()) / 2.0),
        }

    try:
        compare_report["nn_metrics"] = _symmetric_nn(inv_map, direct_map)
    except Exception as e:
        compare_report["nn_error"] = str(e)

    # write compare report
    try:
        import json

        out_report = run_dir / "compare_report.json"
        with out_report.open("w") as f:
            json.dump(compare_report, f, indent=2)
        try:
            os.chmod(str(out_report), 0o666)
        except Exception:
            pass
        print(f"[COMPARE] Saved comparison report: {out_report}")
    except Exception as e:
        print(f"[WARN] Failed to write compare report: {e}")

    camera.disconnect()
    if robot_connected:
        ic.disconnect_robot(args.ip)


def main():
    parser = argparse.ArgumentParser(description="Automated 2-pose capture + extrinsics test")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="x")
    parser.add_argument("--displacement", type=float, default=0.10, help="Meters to move between poses (default 0.10)")
    parser.add_argument("--frames-per-pose", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--ip", type=str, default="192.168.10.75", help="Robot IP")
    parser.add_argument("--out", type=str, default="captures/debug_suite", help="Output base folder")
    parser.add_argument("--auto-move", action="store_true", help="Attempt to command robot automatically")
    parser.add_argument("--move-speed", type=float, default=0.02, help="Move speed for DianaApi move calls (default: 0.02)")
    parser.add_argument("--move-accel", type=float, default=0.2, help="Move accel for DianaApi move calls (default: 0.2)")
    parser.add_argument("--extrinsics-mode", choices=["inv", "direct", "both"], default="both")
    parser.add_argument("--voxel-size", type=float, default=0.005)
    args = parser.parse_args()

    run_single_test(args)


if __name__ == "__main__":
    main()
