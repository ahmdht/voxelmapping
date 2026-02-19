#!/usr/bin/env python3
"""
Simple DianaApi move demo and output-folder helper.

Usage:
  python3 diana_move_and_dirs.py

This script:
 - Connects to Diana robot
 - Reads current TCP pose
 - Moves through a small sequence of relative TCP targets (safe, slow speeds)
 - Demonstrates creating the captures folder layout used by the project
"""

import time
from pathlib import Path
import numpy as np
import os
import sys

# Discover DianaApi in same way as other scripts
DIANA_DOCKER_API = "/diana_api"
DIANA_HOST_PATH = "/home/ahmad.hoteit/.conan/data/diana-api/2.18.1/ar/stable/package/aeddd718e2f218413aa0b9078e615c0fca8986f5/lib/python3/site-packages/diana_api"

if Path(DIANA_DOCKER_API).exists():
    sys.path.insert(0, DIANA_DOCKER_API)
elif Path(DIANA_HOST_PATH).exists():
    sys.path.insert(0, DIANA_HOST_PATH)

try:
    import DianaApi
    DIANA_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] DianaApi import failed: {e}")
    DIANA_AVAILABLE = False


def init_robot(ip: str) -> bool:
    if not DIANA_AVAILABLE:
        print("[WARN] DianaApi not available")
        return False
    try:
        try:
            DianaApi.destroySrv(ip)
            time.sleep(0.2)
        except Exception:
            pass
        ok = DianaApi.initSrv((ip, 0, 0, 0, 0, 0))
        print(f"initSrv returned: {ok}")
        time.sleep(0.5)
        return bool(ok)
    except Exception as e:
        print(f"[ERROR] init failed: {e}")
        return False


def read_tcp(ip: str):
    if not DIANA_AVAILABLE:
        return None
    tcp = np.zeros(6, dtype=float)
    try:
        DianaApi.getTcpPos(tcp, ipAddress=ip)
        return tcp
    except Exception as e:
        print(f"[ERROR] getTcpPos failed: {e}")
        return None


def move_sequence(ip: str, base_tcp, rel_moves, v=0.02, a=0.2):
    """Move through a sequence of relative tcp offsets.

    - base_tcp: numpy array shape (6,) containing starting tcp
    - rel_moves: list of (dx,dy,dz,drx,dry,drz)
    """
    if not DIANA_AVAILABLE:
        print("[WARN] DianaApi not available; skipping moves")
        return

    for i, rel in enumerate(rel_moves):
        target = base_tcp + np.array(rel, dtype=float)
        print(f"Moving to target {i+1}: {target}")
        try:
            # Many Diana APIs use names like moveLToPose or moveL; best-effort call
            if hasattr(DianaApi, "moveLToPose"):
                DianaApi.moveLToPose(target, v=v, a=a, ipAddress=ip)
            elif hasattr(DianaApi, "moveL"):
                DianaApi.moveL(target, v=v, a=a, ipAddress=ip)
            else:
                # Try generic move functions
                if hasattr(DianaApi, "move"):
                    DianaApi.move(target, ipAddress=ip)
                else:
                    print("[WARN] No known move function on DianaApi; aborting moves")
                    return
            # Wait for completion if wait_move exists
            if hasattr(DianaApi, "wait_move"):
                DianaApi.wait_move()
            elif hasattr(DianaApi, "waitMove"):
                DianaApi.waitMove()
            else:
                # conservative sleep
                time.sleep(1.0)
        except Exception as e:
            print(f"[WARN] Move to target {i+1} failed: {e}")
            return


def make_capture_dirs(base: str):
    """Create the captures folder structure used by the repository.

    Creates:
      base/
        capture_index.json (placeholder)
        camera_poses_in_base.json (placeholder)
        extrinsics_used.json (placeholder)
        pose_00/
        pose_01/
    Returns the Path to the base folder.
    """
    basep = Path(base)
    basep.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(str(basep), 0o777)
    except Exception:
        pass
    # Create two pose subfolders for the 2-pose test
    for i in range(2):
        p = basep / f"pose_{i:02d}"
        p.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(p), 0o777)
        except Exception:
            pass

    # Write minimal placeholder JSONs so tools see the files
    import json
    (basep / "capture_index.json").write_text(json.dumps([], indent=2))
    (basep / "camera_poses_in_base.json").write_text(json.dumps([], indent=2))
    (basep / "extrinsics_used.json").write_text(json.dumps({"T_cam_ee": []}, indent=2))

    print(f"Created capture folder skeleton at: {basep}")
    return basep


def main():
    ip = "192.168.10.75"
    print("Eyes on robot, hand on stop! Starting in 2s...")
    time.sleep(2)

    if not init_robot(ip):
        print("Failed to init robot; exiting")
        return

    tcp = read_tcp(ip)
    if tcp is None:
        print("Failed to read TCP; exiting")
        return

    print(f"Current TCP: {tcp}")

    # Relative moves (meters)
    rels = [
        (0.0, 0.0, 0.05, 0.0, 0.0, 0.0),
        (0.0, 0.0, -0.05, 0.0, 0.0, 0.0),
        (0.05, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-0.05, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.05, 0.0, 0.0, 0.0, 0.0),
        (0.0, -0.05, 0.0, 0.0, 0.0, 0.0),
    ]

    move_sequence(ip, tcp, rels, v=0.02, a=0.2)

    # Create capture folder skeleton
    make_capture_dirs("captures/debug_demo")

    try:
        DianaApi.destroySrv(ip)
    except Exception:
        pass


if __name__ == "__main__":
    main()
