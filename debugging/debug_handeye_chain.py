#!/usr/bin/env python3
"""
Quick hand-eye debugging tool for pose-dependent ghosting.

Implements fast checks:
1) Extrinsics direction flip test
2) Camera origin print at two poses
3) TCP vs flange mismatch check (optional tool offset compensation)

Usage examples:
  python3 debug_handeye_chain.py --ip 192.168.10.75
  python3 debug_handeye_chain.py --ip 192.168.10.75 --extrinsics captures/extrinsics_used.json
  python3 debug_handeye_chain.py --ip 192.168.10.75 --calib-frame flange \
      --flange-to-tcp 0.0 0.0 0.12 0.0 0.0 0.0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Diana API paths (same pattern as interactive_capture_mecheye.py)
DIANA_DOCKER_API = "/diana_api"
DIANA_HOST_PATH = "/home/ahmad.hoteit/.conan/data/diana-api/2.18.1/ar/stable/package/aeddd718e2f218413aa0b9078e615c0fca8986f5/lib/python3/site-packages/diana_api"

if os.path.exists(DIANA_DOCKER_API):
    sys.path.insert(0, DIANA_DOCKER_API)
elif os.path.exists(DIANA_HOST_PATH):
    sys.path.insert(0, DIANA_HOST_PATH)

try:
    import DianaApi
    DIANA_AVAILABLE = True
except ImportError as exc:
    print(f"[ERROR] DianaApi import failed: {exc}")
    DIANA_AVAILABLE = False


DEFAULT_EXTRINSICS_CAM_EE = np.array([
    [-0.71365504342923614, -0.70020575462676071, -0.020208418433506396, 0.07363354839981448],
    [0.70042557727115762, -0.71369780379112213, -0.0062813667377777374, 0.13484894916013071],
    [-0.01002445471737245, -0.018637222199069551, 0.99977605705293926, 0.062427763940316391],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def assert_se3(T, name):
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"{name} shape is {T.shape}, expected (4,4)")
    if not np.isfinite(T).all():
        raise ValueError(f"{name} has NaN/Inf")
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} bottom row invalid: {T[3, :]}")
    R = T[:3, :3]
    det = np.linalg.det(R)
    if abs(det - 1.0) > 1e-2:
        raise ValueError(f"{name} rotation det={det:.6f}, expected ~1")


def pose_to_matrix_xyz(pose_xyzrpy):
    x, y, z, rx, ry, rz = pose_xyzrpy
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_euler("xyz", [rx, ry, rz], degrees=False).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def load_extrinsics(path):
    if path is None:
        return DEFAULT_EXTRINSICS_CAM_EE.copy()
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "T_cam_ee" in data:
            T = np.array(data["T_cam_ee"], dtype=np.float64)
        elif "T_ee_cam" in data:
            # Keep as-is; direction tests will evaluate both formulas.
            T = np.array(data["T_ee_cam"], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported extrinsics JSON keys in {path}")
    else:
        T = np.array(data, dtype=np.float64)
    assert_se3(T, "Extrinsics")
    return T


def connect_robot(ip):
    if not DIANA_AVAILABLE:
        return False
    try:
        try:
            DianaApi.destroySrv(ip)
            time.sleep(0.3)
        except Exception:
            pass
        DianaApi.setLastError(0, ip)
    except Exception:
        pass
    ok = DianaApi.initSrv((ip, 0, 0, 0, 0, 0))
    if not ok:
        print(f"[ERROR] Robot connection failed at {ip}")
        return False
    print(f"[OK] Connected to robot at {ip}")
    return True


def disconnect_robot(ip):
    if not DIANA_AVAILABLE:
        return
    try:
        DianaApi.destroySrv(ip)
        print("[INFO] Robot disconnected")
    except Exception as exc:
        print(f"[WARN] Robot disconnect issue: {exc}")


def get_pose_and_matrix(ip):
    pose = [0.0] * 6
    ok = DianaApi.getTcpPos(pose, ipAddress=ip)
    if not ok:
        raise RuntimeError("getTcpPos failed")

    # Keep same matrix convention as interactive_capture_mecheye.py.
    T_flat = [0.0] * 16
    DianaApi.pose2Homogeneous(pose, T_flat)
    T_base_tcp = np.array(T_flat, dtype=np.float64).reshape(4, 4).T
    assert_se3(T_base_tcp, "T_base_tcp")
    return pose, T_base_tcp


def maybe_compensate_tcp_to_flange(T_base_tcp, calib_frame, T_flange_tcp):
    if calib_frame == "tcp":
        return T_base_tcp
    if T_flange_tcp is None:
        print("[WARN] Calibration frame set to flange, but no --flange-to-tcp given.")
        print("       Using TCP directly; this can hide a TCP/flange mismatch.")
        return T_base_tcp
    assert_se3(T_flange_tcp, "T_flange_tcp")
    T_base_flange = T_base_tcp @ np.linalg.inv(T_flange_tcp)
    assert_se3(T_base_flange, "T_base_flange")
    return T_base_flange


def compute_options(T_base_ee_like, T_cam_ee):
    # Option A: assume matrix is T_cam_ee.
    T_base_cam_inv = T_base_ee_like @ np.linalg.inv(T_cam_ee)
    # Option B: assume matrix is already T_ee_cam.
    T_base_cam_direct = T_base_ee_like @ T_cam_ee
    assert_se3(T_base_cam_inv, "T_base_cam_inv")
    assert_se3(T_base_cam_direct, "T_base_cam_direct")
    return T_base_cam_inv, T_base_cam_direct


def fmt_xyz(v):
    return f"[{v[0]: .6f}, {v[1]: .6f}, {v[2]: .6f}]"


def print_pose_block(tag, pose, T_base_ee_like, T_cam_inv, T_cam_direct):
    print("\n" + "=" * 72)
    print(tag)
    print("=" * 72)
    print(f"Robot pose [x,y,z,rx,ry,rz]: {[round(x, 6) for x in pose]}")
    print(f"EE-like origin (base):        {fmt_xyz(T_base_ee_like[:3, 3])}")
    print(f"Camera origin (INV option):   {fmt_xyz(T_cam_inv[:3, 3])}")
    print(f"Camera origin (DIRECT option):{fmt_xyz(T_cam_direct[:3, 3])}")


def print_delta_summary(name, p0, p1):
    d = p1 - p0
    print(f"{name} delta: {fmt_xyz(d)}  |norm|={np.linalg.norm(d):.6f} m")


def delta_transform(T_a, T_b):
    return np.linalg.inv(T_a) @ T_b


def se3_error(T_meas, T_pred):
    T_err = np.linalg.inv(T_pred) @ T_meas
    R_err = T_err[:3, :3]
    t_err = T_err[:3, 3]
    rot_deg = np.degrees(Rotation.from_matrix(R_err).magnitude())
    trans_mm = np.linalg.norm(t_err) * 1000.0
    return rot_deg, trans_mm


def print_relative_motion_metrics(T_base_ee_a, T_base_ee_b, T_cam_a_inv, T_cam_b_inv, T_cam_a_direct, T_cam_b_direct, T_cam_ee):
    d_ee = delta_transform(T_base_ee_a, T_base_ee_b)
    d_cam_inv = delta_transform(T_cam_a_inv, T_cam_b_inv)
    d_cam_direct = delta_transform(T_cam_a_direct, T_cam_b_direct)

    # Assumption A: given matrix is T_cam_ee
    pred_from_cam_ee = T_cam_ee @ d_ee @ np.linalg.inv(T_cam_ee)
    # Assumption B: given matrix is T_ee_cam
    pred_from_ee_cam = np.linalg.inv(T_cam_ee) @ d_ee @ T_cam_ee

    print("\n" + "=" * 72)
    print("RELATIVE-MOTION CONJUGATION CHECK")
    print("=" * 72)
    print("Reported as: rotation_error_deg, translation_error_mm")

    r1, t1 = se3_error(d_cam_inv, pred_from_cam_ee)
    r2, t2 = se3_error(d_cam_direct, pred_from_cam_ee)
    print(f"Assume given T is T_cam_ee  -> INV: ({r1:.4f} deg, {t1:.3f} mm), DIRECT: ({r2:.4f} deg, {t2:.3f} mm)")

    r3, t3 = se3_error(d_cam_inv, pred_from_ee_cam)
    r4, t4 = se3_error(d_cam_direct, pred_from_ee_cam)
    print(f"Assume given T is T_ee_cam  -> INV: ({r3:.4f} deg, {t3:.3f} mm), DIRECT: ({r4:.4f} deg, {t4:.3f} mm)")

    print("\nNote:")
    print("- This check is algebraically consistent with the chain, so it is not alone a fusion-ground-truth test.")
    print("- Use it to catch implementation mistakes, then decide with 2-pose cloud overlap (INV run vs DIRECT run).")


def read_flange_to_tcp(vals):
    if vals is None:
        return None
    if len(vals) != 6:
        raise ValueError("--flange-to-tcp needs 6 values: x y z rx ry rz")
    return pose_to_matrix_xyz(vals)


def main():
    parser = argparse.ArgumentParser(description="Debug hand-eye chain (2-pose fast test).")
    parser.add_argument("--ip", default="192.168.10.75", help="Robot IP")
    parser.add_argument(
        "--extrinsics",
        default="captures/extrinsics_used.json",
        help="Path to JSON containing T_cam_ee (or raw 4x4).",
    )
    parser.add_argument(
        "--calib-frame",
        choices=["tcp", "flange"],
        default="tcp",
        help="Frame used during hand-eye calibration.",
    )
    parser.add_argument(
        "--flange-to-tcp",
        type=float,
        nargs=6,
        metavar=("x", "y", "z", "rx", "ry", "rz"),
        help="Known T_flange_tcp (meters/radians) for TCP->flange compensation.",
    )
    args = parser.parse_args()

    print("\nHand-Eye Debug Sequence")
    print("1) Capture pose A")
    print("2) Move robot")
    print("3) Capture pose B")
    print("4) Compare camera-origin deltas for INV vs DIRECT extrinsics")
    print("5) Optional TCP/flange compensation applied before both tests")

    extrinsics_path = Path(args.extrinsics)
    if args.extrinsics and not extrinsics_path.exists():
        print(f"[WARN] Extrinsics file not found: {args.extrinsics}")
        print("[INFO] Falling back to hard-coded extrinsics from interactive_capture_mecheye.py")
        T_cam_ee = DEFAULT_EXTRINSICS_CAM_EE.copy()
    else:
        T_cam_ee = load_extrinsics(args.extrinsics if extrinsics_path.exists() else None)

    T_flange_tcp = read_flange_to_tcp(args.flange_to_tcp)

    if not connect_robot(args.ip):
        sys.exit(1)

    try:
        input("\nPress ENTER at Pose A...")
        pose_a, T_base_tcp_a = get_pose_and_matrix(args.ip)
        T_base_ee_a = maybe_compensate_tcp_to_flange(T_base_tcp_a, args.calib_frame, T_flange_tcp)
        T_cam_a_inv, T_cam_a_direct = compute_options(T_base_ee_a, T_cam_ee)
        print_pose_block("POSE A", pose_a, T_base_ee_a, T_cam_a_inv, T_cam_a_direct)

        input("\nMove robot to Pose B, then press ENTER...")
        pose_b, T_base_tcp_b = get_pose_and_matrix(args.ip)
        T_base_ee_b = maybe_compensate_tcp_to_flange(T_base_tcp_b, args.calib_frame, T_flange_tcp)
        T_cam_b_inv, T_cam_b_direct = compute_options(T_base_ee_b, T_cam_ee)
        print_pose_block("POSE B", pose_b, T_base_ee_b, T_cam_b_inv, T_cam_b_direct)

        print("\n" + "=" * 72)
        print("DELTA COMPARISON (A -> B)")
        print("=" * 72)
        print_delta_summary("EE-like origin", T_base_ee_a[:3, 3], T_base_ee_b[:3, 3])
        print_delta_summary("Camera INV", T_cam_a_inv[:3, 3], T_cam_b_inv[:3, 3])
        print_delta_summary("Camera DIRECT", T_cam_a_direct[:3, 3], T_cam_b_direct[:3, 3])
        print_relative_motion_metrics(
            T_base_ee_a,
            T_base_ee_b,
            T_cam_a_inv,
            T_cam_b_inv,
            T_cam_a_direct,
            T_cam_b_direct,
            T_cam_ee,
        )

        print("\nInterpretation:")
        print("- This script narrows implementation/frame issues, but final direction choice should use fusion quality.")
        print("- Run 2-pose capture twice (cam_to_ee vs ee_to_cam) and keep whichever removes the double plane.")
        print("- If both are poor, check TCP vs flange mismatch and capture-time synchronization.")

    finally:
        disconnect_robot(args.ip)


if __name__ == "__main__":
    main()
