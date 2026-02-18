#!/usr/bin/env python3
"""
Transform Chain Diagnostic Tool

Checks 3 common issues:
1. Extrinsic matrix direction (T_cam_ee vs T_ee_cam)
2. TCP vs Flange frame mismatch
3. Units mismatch (meters vs mm)

Usage:
    python3 diagnose_transforms.py --ip 192.168.10.75
    python3 diagnose_transforms.py --ip 192.168.10.75 --capture  # Capture and visualize
"""

import argparse
import sys
import time
import os
from pathlib import Path

import numpy as np

# Diana API paths
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
    print(f"[ERROR] DianaApi not available: {e}")
    DIANA_AVAILABLE = False

# Your hand-eye calibration extrinsics
EXTRINSICS_CAM_EE = np.array([
    [-0.71365504342923614, -0.70020575462676071, -0.020208418433506396, 0.07363354839981448],
    [ 0.70042557727115762, -0.71369780379112213, -0.0062813667377777374, 0.13484894916013071],
    [-0.01002445471737245, -0.018637222199069551,  0.99977605705293926, 0.062427763940316391],
    [ 0.0,                  0.0,                   0.0,                  1.0]
], dtype=np.float32)


def connect_robot(robot_ip):
    """Connect to Diana robot."""
    if not DIANA_AVAILABLE:
        return False
    
    try:
        try:
            DianaApi.destroySrv(robot_ip)
            time.sleep(0.3)
        except:
            pass
        
        init_ret = DianaApi.initSrv((robot_ip, 0, 0, 0, 0, 0))
        if not init_ret:
            print(f"[ERROR] Robot connection failed")
            return False
        
        print(f"[OK] Connected to robot at {robot_ip}")
        return True
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return False


def get_tcp_pose(robot_ip):
    """Get TCP pose [x, y, z, rx, ry, rz]."""
    pose = [0.0] * 6
    DianaApi.getTcpPos(pose, ipAddress=robot_ip)
    return pose


def pose_to_matrix_diana(pose, robot_ip):
    """Convert pose to 4x4 matrix using Diana's native function."""
    T_flat = [0.0] * 16
    DianaApi.pose2Homogeneous(pose, T_flat)
    return np.array(T_flat, dtype=np.float32).reshape(4, 4)


def analyze_extrinsics():
    """Analyze the extrinsic matrix."""
    print("\n" + "=" * 60)
    print("1. EXTRINSIC MATRIX ANALYSIS")
    print("=" * 60)
    
    T = EXTRINSICS_CAM_EE
    
    # Extract rotation and translation
    R = T[:3, :3]
    t = T[:3, 3]
    
    print(f"\nEXTRINSICS_CAM_EE:")
    print(T)
    
    print(f"\nTranslation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
    print(f"  = [{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}] mm")
    
    # Check if rotation is valid (det should be +1)
    det = np.linalg.det(R)
    print(f"\nRotation matrix determinant: {det:.6f} (should be +1.0)")
    
    # Check orthogonality
    RRT = R @ R.T
    ortho_error = np.linalg.norm(RRT - np.eye(3))
    print(f"Orthogonality error: {ortho_error:.6f} (should be ~0)")
    
    # Extract Euler angles for intuition
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(R)
    euler = r.as_euler('xyz', degrees=True)
    print(f"\nEuler angles (XYZ, degrees): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]")
    
    # Compute inverse for comparison
    T_inv = np.linalg.inv(T)
    t_inv = T_inv[:3, 3]
    print(f"\nInverse translation: [{t_inv[0]:.4f}, {t_inv[1]:.4f}, {t_inv[2]:.4f}] m")
    print(f"  = [{t_inv[0]*1000:.1f}, {t_inv[1]*1000:.1f}, {t_inv[2]*1000:.1f}] mm")
    
    print("\n" + "-" * 40)
    print("INTERPRETATION:")
    print("-" * 40)
    print(f"If this is T_cam_ee (camera FROM end-effector):")
    print(f"  Camera is located at ~{np.linalg.norm(t)*1000:.0f}mm from EE origin")
    print(f"  Use: T_base_cam = T_base_ee @ inv(T_cam_ee)  ← Current code")
    print(f"\nIf this is T_ee_cam (end-effector FROM camera):")
    print(f"  Use: T_base_cam = T_base_ee @ T_ee_cam  ← Alternative")


def check_units(robot_ip):
    """Check if robot returns meters or millimeters."""
    print("\n" + "=" * 60)
    print("2. UNITS CHECK (meters vs mm)")
    print("=" * 60)
    
    pose = get_tcp_pose(robot_ip)
    
    print(f"\nCurrent TCP pose from getTcpPos():")
    print(f"  Position: x={pose[0]:.6f}, y={pose[1]:.6f}, z={pose[2]:.6f}")
    print(f"  Rotation: rx={pose[3]:.6f}, ry={pose[4]:.6f}, rz={pose[5]:.6f}")
    
    # Heuristic check
    pos_magnitude = np.linalg.norm(pose[:3])
    
    print(f"\nPosition magnitude: {pos_magnitude:.4f}")
    
    if pos_magnitude < 5:
        print("  → Likely METERS (typical robot reach is 0.5-1.5m)")
        units = "meters"
    elif pos_magnitude > 100:
        print("  → Likely MILLIMETERS (values > 100)")
        units = "mm"
        print("  ⚠ WARNING: Your code assumes meters! Divide by 1000!")
    else:
        print("  → AMBIGUOUS - could be either. Do the movement test below.")
        units = "unknown"
    
    print("\n" + "-" * 40)
    print("MANUAL VERIFICATION:")
    print("-" * 40)
    print("1. Note current X position")
    print("2. Jog robot +100mm in X direction")
    print("3. Read pose again")
    print("4. If delta is ~0.1 → meters")
    print("   If delta is ~100 → millimeters")
    
    return units


def check_tcp_vs_flange(robot_ip):
    """Check TCP vs flange frame issue."""
    print("\n" + "=" * 60)
    print("3. TCP vs FLANGE FRAME CHECK")
    print("=" * 60)
    
    # Try to get tool info if available
    try:
        tool_data = [0.0] * 6
        # Some Diana APIs have getToolData or similar
        print("\nAttempting to read tool/TCP offset...")
        
        # Try various Diana API calls
        has_tool = False
        try:
            DianaApi.getToolData(tool_data, ipAddress=robot_ip)
            has_tool = True
            print(f"Tool offset: {tool_data}")
        except:
            pass
        
        if not has_tool:
            print("  Could not read tool offset via API")
    except Exception as e:
        print(f"  Tool check failed: {e}")
    
    print("\n" + "-" * 40)
    print("IMPORTANT QUESTIONS:")
    print("-" * 40)
    print("1. During hand-eye calibration, did you use:")
    print("   [ ] Flange frame (no tool offset)")
    print("   [ ] TCP frame (with tool offset)")
    print()
    print("2. Is there a tool defined on the robot?")
    print("   - Check robot pendant → Tool settings")
    print("   - If tool offset exists, you may need to compensate")
    print()
    print("3. If calibration used flange but runtime uses TCP:")
    print("   T_base_flange = T_base_tcp @ inv(T_flange_tcp)")


def test_both_transform_options(robot_ip, points_file=None):
    """Test both transform interpretations."""
    print("\n" + "=" * 60)
    print("4. TRANSFORM OPTIONS TEST")
    print("=" * 60)
    
    pose = get_tcp_pose(robot_ip)
    T_base_ee = pose_to_matrix_diana(pose, robot_ip)
    
    print(f"\nT_base_ee (robot pose as matrix):")
    print(T_base_ee)
    
    # Option 1: Current code (T_cam_ee interpretation)
    T_ee_cam_opt1 = np.linalg.inv(EXTRINSICS_CAM_EE)
    T_base_cam_opt1 = T_base_ee @ T_ee_cam_opt1
    
    # Option 2: Alternative (T_ee_cam interpretation)
    T_base_cam_opt2 = T_base_ee @ EXTRINSICS_CAM_EE
    
    print(f"\nOption 1: T_base_cam = T_base_ee @ inv(EXTRINSICS)")
    print(f"  Camera position in base: [{T_base_cam_opt1[0,3]:.4f}, {T_base_cam_opt1[1,3]:.4f}, {T_base_cam_opt1[2,3]:.4f}]")
    
    print(f"\nOption 2: T_base_cam = T_base_ee @ EXTRINSICS (no inverse)")
    print(f"  Camera position in base: [{T_base_cam_opt2[0,3]:.4f}, {T_base_cam_opt2[1,3]:.4f}, {T_base_cam_opt2[2,3]:.4f}]")
    
    # If points file provided, transform and save both versions
    if points_file is not None:
        print(f"\nTransforming point cloud: {points_file}")
        points = np.load(points_file)
        
        # Filter valid points
        valid = (np.abs(points).sum(axis=1) > 0) & (np.abs(points).sum(axis=1) < 10)
        points = points[valid]
        
        # Make homogeneous
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.hstack([points, ones])
        
        # Transform with both options
        points_opt1 = (T_base_cam_opt1 @ points_h.T).T[:, :3]
        points_opt2 = (T_base_cam_opt2 @ points_h.T).T[:, :3]
        
        # Save both for visualization
        out_dir = Path(points_file).parent
        np.save(out_dir / "cloud_option1_inv.npy", points_opt1)
        np.save(out_dir / "cloud_option2_direct.npy", points_opt2)
        
        print(f"\nSaved transformed clouds:")
        print(f"  {out_dir}/cloud_option1_inv.npy    (current code: inv)")
        print(f"  {out_dir}/cloud_option2_direct.npy (alternative: no inv)")
        print(f"\nVisualize both in CloudCompare/MeshLab to see which looks correct!")
        
        # Print bounding boxes for quick comparison
        print(f"\nOption 1 bounding box:")
        print(f"  X: [{points_opt1[:,0].min():.3f}, {points_opt1[:,0].max():.3f}]")
        print(f"  Y: [{points_opt1[:,1].min():.3f}, {points_opt1[:,1].max():.3f}]")
        print(f"  Z: [{points_opt1[:,2].min():.3f}, {points_opt1[:,2].max():.3f}]")
        
        print(f"\nOption 2 bounding box:")
        print(f"  X: [{points_opt2[:,0].min():.3f}, {points_opt2[:,0].max():.3f}]")
        print(f"  Y: [{points_opt2[:,1].min():.3f}, {points_opt2[:,1].max():.3f}]")
        print(f"  Z: [{points_opt2[:,2].min():.3f}, {points_opt2[:,2].max():.3f}]")
    
    return T_base_cam_opt1, T_base_cam_opt2


def quick_movement_test(robot_ip):
    """Interactive test to verify units by movement."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MOVEMENT TEST")
    print("=" * 60)
    
    pose1 = get_tcp_pose(robot_ip)
    print(f"\nCurrent position:")
    print(f"  X: {pose1[0]:.6f}")
    print(f"  Y: {pose1[1]:.6f}")
    print(f"  Z: {pose1[2]:.6f}")
    
    print("\n>>> Jog the robot ~100mm in any direction <<<")
    print(">>> Then press ENTER <<<")
    input()
    
    pose2 = get_tcp_pose(robot_ip)
    print(f"\nNew position:")
    print(f"  X: {pose2[0]:.6f}")
    print(f"  Y: {pose2[1]:.6f}")
    print(f"  Z: {pose2[2]:.6f}")
    
    delta = np.array(pose2[:3]) - np.array(pose1[:3])
    delta_mag = np.linalg.norm(delta)
    
    print(f"\nDelta: [{delta[0]:.6f}, {delta[1]:.6f}, {delta[2]:.6f}]")
    print(f"Magnitude: {delta_mag:.6f}")
    
    if delta_mag < 0.01:
        print("\n⚠ Robot didn't move much. Try again with larger movement.")
    elif 0.05 < delta_mag < 0.5:
        print(f"\n✓ Units are METERS (moved ~{delta_mag*1000:.0f}mm)")
    elif 50 < delta_mag < 500:
        print(f"\n⚠ Units are MILLIMETERS (moved ~{delta_mag:.0f}mm)")
        print("  Your code needs to divide positions by 1000!")
    else:
        print(f"\n? Unusual magnitude. Check movement was ~100mm")


def main():
    parser = argparse.ArgumentParser(description="Diagnose transform chain issues")
    parser.add_argument("--ip", type=str, default="192.168.10.75", help="Robot IP")
    parser.add_argument("--capture", action="store_true", help="Capture a frame and test transforms")
    parser.add_argument("--cloud", type=str, default=None, help="Path to existing .npy cloud to test")
    parser.add_argument("--movement-test", action="store_true", help="Interactive movement test for units")
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRANSFORM CHAIN DIAGNOSTIC")
    print("=" * 60)
    
    # Always analyze extrinsics (no robot needed)
    analyze_extrinsics()
    
    # Connect to robot for other checks
    if not connect_robot(args.ip):
        print("\n[WARN] Cannot connect to robot. Only extrinsic analysis available.")
        return
    
    # Check units
    check_units(args.ip)
    
    # Check TCP vs flange
    check_tcp_vs_flange(args.ip)
    
    # Test both transform options
    cloud_file = args.cloud
    if cloud_file is None:
        # Find a cloud file in captures
        captures = Path("captures")
        if captures.exists():
            clouds = list(captures.rglob("cloud_*.npy"))
            if clouds:
                cloud_file = str(clouds[0])
                print(f"\nUsing existing cloud: {cloud_file}")
    
    if cloud_file:
        test_both_transform_options(args.ip, cloud_file)
    else:
        test_both_transform_options(args.ip, None)
    
    # Interactive movement test
    if args.movement_test:
        quick_movement_test(args.ip)
    
    print("\n" + "=" * 60)
    print("SUMMARY - WHAT TO CHECK")
    print("=" * 60)
    print("""
1. EXTRINSIC DIRECTION:
   - Visualize cloud_option1_inv.npy and cloud_option2_direct.npy
   - The correct one will align with your robot base frame
   - Wrong one will be flipped/mirrored

2. UNITS:
   - If pose values are > 100, robot returns mm → divide by 1000
   - Use --movement-test to verify

3. TCP vs FLANGE:
   - Check if tool offset is defined on robot
   - If calibration used flange, runtime must use flange
   - Or compensate with: T_base_flange = T_base_tcp @ inv(T_flange_tcp)
""")
    
    try:
        DianaApi.destroySrv(args.ip)
    except:
        pass


if __name__ == "__main__":
    main()
