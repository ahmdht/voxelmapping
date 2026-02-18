#!/usr/bin/env python3
"""
Simple Voxel Map Visualization Tool

Opens voxel maps in MeshLab, CloudCompare, or Open3D.
Automatically converts .npy to .ply if needed.

Usage:
    python3 visualize_voxelmap_simple.py captures/
    python3 visualize_voxelmap_simple.py captures/voxel_map_merged.npy --viewer meshlab
    python3 visualize_voxelmap_simple.py captures/voxel_map_merged.ply --viewer cloudcompare
"""

import argparse
import subprocess
import shutil
import sys
from pathlib import Path

import numpy as np

# Check for Open3D (needed for conversion)
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


def find_voxel_map(path: Path) -> Path:
    """Find voxel map file in path."""
    path = Path(path)
    
    if path.is_file():
        return path
    
    if path.is_dir():
        # Priority: PLY > NPY
        ply_path = path / "voxel_map_merged.ply"
        npy_path = path / "voxel_map_merged.npy"
        
        if ply_path.exists():
            return ply_path
        elif npy_path.exists():
            return npy_path
        else:
            # Search for any ply/npy
            plys = list(path.glob("*.ply"))
            npys = list(path.glob("*.npy"))
            if plys:
                return plys[0]
            elif npys:
                return npys[0]
    
    raise FileNotFoundError(f"No voxel map found in {path}")


def convert_npy_to_ply(npy_path: Path) -> Path:
    """Convert .npy to .ply for external viewers."""
    if not OPEN3D_AVAILABLE:
        print("Error: open3d required for .npy conversion")
        print("Install with: pip install open3d")
        sys.exit(1)
    
    ply_path = npy_path.with_suffix(".ply")
    
    if ply_path.exists():
        print(f"Using existing: {ply_path}")
        return ply_path
    
    print(f"Converting {npy_path.name} to PLY...")
    points = np.load(npy_path)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"Saved: {ply_path}")
    
    return ply_path


def print_stats(path: Path):
    """Print basic statistics."""
    if path.suffix == ".npy":
        points = np.load(path)
    elif path.suffix == ".ply" and OPEN3D_AVAILABLE:
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)
    else:
        print(f"[Cannot read stats from {path.suffix}]")
        return
    
    if len(points) == 0:
        print("[ERROR] No points in file!")
        return
    
    print(f"\n{'='*40}")
    print("VOXEL MAP STATS")
    print(f"{'='*40}")
    print(f"Points:     {len(points):,}")
    
    dims = points.max(axis=0) - points.min(axis=0)
    print(f"Size:       {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f} m")
    
    print(f"X range:    [{points[:,0].min():.3f}, {points[:,0].max():.3f}] m")
    print(f"Y range:    [{points[:,1].min():.3f}, {points[:,1].max():.3f}] m")
    print(f"Z range:    [{points[:,2].min():.3f}, {points[:,2].max():.3f}] m")
    print(f"{'='*40}\n")


def find_viewer(name: str) -> str:
    """Find viewer executable."""
    viewers = {
        "meshlab": ["meshlab", "MeshLab"],
        "cloudcompare": ["cloudcompare", "CloudCompare", "cloudcompare.CloudCompare"],
        "pcl": ["pcl_viewer"],
        "open3d": ["open3d"]  # Special case - use Python
    }
    
    if name not in viewers:
        return None
    
    for cmd in viewers[name]:
        if shutil.which(cmd):
            return cmd
    
    return None


def open_meshlab(ply_path: Path):
    """Open in MeshLab."""
    cmd = find_viewer("meshlab")
    if not cmd:
        print("MeshLab not found. Install with: sudo apt install meshlab")
        return False
    
    print(f"Opening in MeshLab: {ply_path}")
    subprocess.Popen([cmd, str(ply_path)], 
                     stdout=subprocess.DEVNULL, 
                     stderr=subprocess.DEVNULL)
    return True


def open_cloudcompare(ply_path: Path):
    """Open in CloudCompare."""
    cmd = find_viewer("cloudcompare")
    if not cmd:
        print("CloudCompare not found. Install with: sudo snap install cloudcompare")
        return False
    
    print(f"Opening in CloudCompare: {ply_path}")
    subprocess.Popen([cmd, str(ply_path)],
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    return True


def open_pcl_viewer(ply_path: Path):
    """Open in PCL Viewer."""
    cmd = find_viewer("pcl")
    if not cmd:
        print("PCL Viewer not found. Install with: sudo apt install pcl-tools")
        return False
    
    print(f"Opening in PCL Viewer: {ply_path}")
    subprocess.Popen([cmd, str(ply_path)],
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    return True


def open_open3d(path: Path):
    """Open in Open3D viewer."""
    if not OPEN3D_AVAILABLE:
        print("Open3D not found. Install with: pip install open3d")
        return False
    
    print(f"Opening in Open3D: {path}")
    
    if path.suffix == ".npy":
        points = np.load(path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    else:
        pcd = o3d.io.read_point_cloud(str(path))
    
    # Color by height
    points = np.asarray(pcd.points)
    if len(points) > 0:
        z = points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = z_norm        # Red = high
        colors[:, 2] = 1 - z_norm    # Blue = low
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Add coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    o3d.visualization.draw_geometries(
        [pcd, frame],
        window_name="Voxel Map",
        width=1200, height=800
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Simple voxel map visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Viewers:
  meshlab      - MeshLab (install: sudo apt install meshlab)
  cloudcompare - CloudCompare (install: sudo snap install cloudcompare)
  pcl          - PCL Viewer (install: sudo apt install pcl-tools)
  open3d       - Open3D Python viewer (pip install open3d)
  auto         - Try viewers in order: meshlab > cloudcompare > pcl > open3d

Examples:
  python3 visualize_voxelmap_simple.py captures/
  python3 visualize_voxelmap_simple.py captures/voxel_map_merged.npy --viewer meshlab
  python3 visualize_voxelmap_simple.py --stats-only captures/
"""
    )
    parser.add_argument("path", type=Path, help="Path to .npy, .ply file or captures directory")
    parser.add_argument("--viewer", "-v", type=str, default="auto",
                        choices=["meshlab", "cloudcompare", "pcl", "open3d", "auto"],
                        help="Viewer to use (default: auto)")
    parser.add_argument("--stats-only", "-s", action="store_true",
                        help="Print statistics only, don't open viewer")
    parser.add_argument("--no-stats", action="store_true",
                        help="Skip statistics printout")
    
    args = parser.parse_args()
    
    # Find voxel map file
    try:
        file_path = find_voxel_map(args.path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Found: {file_path}")
    
    # Print statistics
    if not args.no_stats:
        print_stats(file_path)
    
    if args.stats_only:
        return
    
    # Convert .npy to .ply if needed for external viewers
    ply_path = file_path
    if file_path.suffix == ".npy" and args.viewer != "open3d":
        ply_path = convert_npy_to_ply(file_path)
    
    # Open viewer
    viewer = args.viewer
    success = False
    
    if viewer == "auto":
        # Try viewers in order
        for v in ["meshlab", "cloudcompare", "pcl", "open3d"]:
            if v == "meshlab":
                success = open_meshlab(ply_path)
            elif v == "cloudcompare":
                success = open_cloudcompare(ply_path)
            elif v == "pcl":
                success = open_pcl_viewer(ply_path)
            elif v == "open3d":
                success = open_open3d(file_path)
            
            if success:
                break
        
        if not success:
            print("\nNo viewer found! Install one of:")
            print("  sudo apt install meshlab")
            print("  sudo snap install cloudcompare")
            print("  sudo apt install pcl-tools")
            print("  pip install open3d")
    else:
        if viewer == "meshlab":
            success = open_meshlab(ply_path)
        elif viewer == "cloudcompare":
            success = open_cloudcompare(ply_path)
        elif viewer == "pcl":
            success = open_pcl_viewer(ply_path)
        elif viewer == "open3d":
            success = open_open3d(file_path)


if __name__ == "__main__":
    main()
