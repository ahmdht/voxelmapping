#!/usr/bin/env python3
"""
Voxel Map Visualization Tool

Multiple visualization methods for voxel maps:
1. Basic point cloud view
2. Voxel grid (cubes)
3. Color by height (Z-axis)
4. Color by density
5. Cross-section slices
6. Statistical analysis
7. Export to various formats

Usage:
    python3 visualize_voxelmap.py /path/to/voxel_map_merged.npy
    python3 visualize_voxelmap.py /path/to/voxel_map_merged.ply --method voxels
    python3 visualize_voxelmap.py /path/to/captures --method all
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d required. Install with: pip install open3d")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_voxel_map(path: Path) -> np.ndarray:
    """Load voxel map from .npy or .ply file."""
    path = Path(path)
    
    if path.is_dir():
        # Look for voxel_map_merged in directory
        npy_path = path / "voxel_map_merged.npy"
        ply_path = path / "voxel_map_merged.ply"
        if npy_path.exists():
            path = npy_path
        elif ply_path.exists():
            path = ply_path
        else:
            raise FileNotFoundError(f"No voxel map found in {path}")
    
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".ply":
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def create_point_cloud(points: np.ndarray, colors: np.ndarray = None) -> o3d.geometry.PointCloud:
    """Create Open3D point cloud from numpy array."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def visualize_basic(points: np.ndarray):
    """Method 1: Basic point cloud visualization."""
    print(f"\n=== Basic Point Cloud ===")
    print(f"Points: {len(points)}")
    
    pcd = create_point_cloud(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.8])  # Light blue
    
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Voxel Map - Basic View",
        width=1200, height=800
    )


def visualize_height_colored(points: np.ndarray):
    """Method 2: Color points by height (Z-axis)."""
    print(f"\n=== Height-Colored View ===")
    
    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_normalized = (z_vals - z_min) / (z_max - z_min + 1e-8)
    
    # Use turbo colormap (blue=low, red=high)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = z_normalized  # Red increases with height
    colors[:, 1] = 0.3
    colors[:, 2] = 1 - z_normalized  # Blue decreases with height
    
    print(f"Height range: {z_min:.3f}m to {z_max:.3f}m")
    
    pcd = create_point_cloud(points, colors)
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Voxel Map - Height Colored (Blue=Low, Red=High)",
        width=1200, height=800
    )


def visualize_voxel_grid(points: np.ndarray, voxel_size: float = 0.01):
    """Method 3: Show as actual voxel cubes."""
    print(f"\n=== Voxel Grid View ===")
    print(f"Voxel size: {voxel_size*1000:.1f}mm")
    
    pcd = create_point_cloud(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    
    o3d.visualization.draw_geometries(
        [voxel_grid],
        window_name=f"Voxel Grid ({voxel_size*1000:.1f}mm)",
        width=1200, height=800
    )


def visualize_with_axes(points: np.ndarray):
    """Method 4: Point cloud with coordinate axes."""
    print(f"\n=== View with Coordinate Frame ===")
    
    pcd = create_point_cloud(points)
    pcd.paint_uniform_color([0.6, 0.6, 0.9])
    
    # Create coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.2, origin=[0, 0, 0]
    )
    
    # Create bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0)
    
    print(f"Bounding box: {bbox.get_min_bound()} to {bbox.get_max_bound()}")
    
    o3d.visualization.draw_geometries(
        [pcd, coord_frame, bbox],
        window_name="Voxel Map with Axes",
        width=1200, height=800
    )


def visualize_cross_sections(points: np.ndarray, n_slices: int = 5):
    """Method 5: Show horizontal cross-sections."""
    print(f"\n=== Cross-Section Slices ===")
    
    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    slice_heights = np.linspace(z_min, z_max, n_slices + 2)[1:-1]
    
    thickness = (z_max - z_min) / (n_slices * 2)
    
    geometries = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
    
    for i, z_height in enumerate(slice_heights):
        mask = np.abs(points[:, 2] - z_height) < thickness
        slice_points = points[mask]
        
        if len(slice_points) > 0:
            pcd = create_point_cloud(slice_points)
            pcd.paint_uniform_color(colors[i % len(colors)])
            geometries.append(pcd)
            print(f"  Slice at z={z_height:.3f}m: {len(slice_points)} points")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Cross Sections ({n_slices} slices)",
        width=1200, height=800
    )


def visualize_density(points: np.ndarray, radius: float = 0.02):
    """Method 6: Color by local point density."""
    print(f"\n=== Density-Colored View ===")
    print(f"Computing density with radius={radius*1000:.1f}mm...")
    
    pcd = create_point_cloud(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    densities = np.zeros(len(points))
    for i in range(len(points)):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
        densities[i] = k
    
    # Normalize densities
    d_min, d_max = densities.min(), densities.max()
    d_normalized = (densities - d_min) / (d_max - d_min + 1e-8)
    
    # Color: blue=sparse, green=medium, red=dense
    colors = np.zeros((len(points), 3))
    colors[:, 0] = d_normalized  # Red
    colors[:, 1] = 1 - np.abs(d_normalized - 0.5) * 2  # Green peaks in middle
    colors[:, 2] = 1 - d_normalized  # Blue
    
    print(f"Density range: {d_min:.0f} to {d_max:.0f} neighbors")
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Voxel Map - Density (Blue=Sparse, Red=Dense)",
        width=1200, height=800
    )


def visualize_normals(points: np.ndarray):
    """Method 7: Estimate and show surface normals."""
    print(f"\n=== Surface Normals View ===")
    
    pcd = create_point_cloud(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    
    # Color by normal direction (useful for seeing surfaces)
    normals = np.asarray(pcd.normals)
    colors = np.abs(normals)  # RGB = abs(nx, ny, nz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Surface Normals (RGB = |nx|, |ny|, |nz|)",
        width=1200, height=800,
        point_show_normal=True
    )


def print_statistics(points: np.ndarray):
    """Print statistical analysis of the voxel map."""
    print("\n" + "=" * 50)
    print("VOXEL MAP STATISTICS")
    print("=" * 50)
    
    print(f"\nPoint Count: {len(points):,}")
    
    print(f"\nBounding Box:")
    print(f"  X: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}] m")
    print(f"  Y: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}] m")
    print(f"  Z: [{points[:, 2].min():.4f}, {points[:, 2].max():.4f}] m")
    
    dims = points.max(axis=0) - points.min(axis=0)
    print(f"\nDimensions: {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f} m")
    
    centroid = points.mean(axis=0)
    print(f"\nCentroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")
    
    # Estimate voxel size from point spacing
    pcd = create_point_cloud(points)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f"\nEstimated voxel size: {avg_dist*1000:.2f} mm")
    
    print("=" * 50)


def plot_histograms(points: np.ndarray):
    """Plot 2D histograms using matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        print("[WARN] matplotlib not available for histograms")
        return
    
    print("\n=== 2D Histograms ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # XY projection (top view)
    axes[0].hist2d(points[:, 0], points[:, 1], bins=100, cmap='viridis')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Top View (XY)')
    axes[0].set_aspect('equal')
    
    # XZ projection (front view)
    axes[1].hist2d(points[:, 0], points[:, 2], bins=100, cmap='viridis')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Z (m)')
    axes[1].set_title('Front View (XZ)')
    axes[1].set_aspect('equal')
    
    # YZ projection (side view)
    axes[2].hist2d(points[:, 1], points[:, 2], bins=100, cmap='viridis')
    axes[2].set_xlabel('Y (m)')
    axes[2].set_ylabel('Z (m)')
    axes[2].set_title('Side View (YZ)')
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('voxel_map_projections.png', dpi=150)
    print("Saved to: voxel_map_projections.png")
    plt.show()


def export_formats(points: np.ndarray, output_dir: Path):
    """Export voxel map to multiple formats."""
    print("\n=== Exporting to multiple formats ===")
    
    output_dir = Path(output_dir)
    pcd = create_point_cloud(points)
    
    # PLY
    ply_path = output_dir / "voxel_map.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"  PLY: {ply_path}")
    
    # PCD
    pcd_path = output_dir / "voxel_map.pcd"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    print(f"  PCD: {pcd_path}")
    
    # XYZ (ASCII)
    xyz_path = output_dir / "voxel_map.xyz"
    np.savetxt(xyz_path, points, fmt='%.6f', delimiter=' ')
    print(f"  XYZ: {xyz_path}")
    
    # CSV
    csv_path = output_dir / "voxel_map.csv"
    np.savetxt(csv_path, points, fmt='%.6f', delimiter=',', header='x,y,z', comments='')
    print(f"  CSV: {csv_path}")


def interactive_viewer(points: np.ndarray):
    """Interactive viewer with keyboard controls."""
    print("\n=== Interactive Viewer ===")
    print("Controls:")
    print("  Mouse: Rotate/Pan/Zoom")
    print("  R: Reset view")
    print("  P: Capture screenshot")
    print("  Q/Esc: Quit")
    
    pcd = create_point_cloud(points)
    
    # Color by height
    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_normalized = (z_vals - z_min) / (z_max - z_min + 1e-8)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = z_normalized
    colors[:, 1] = 0.5 - np.abs(z_normalized - 0.5)
    colors[:, 2] = 1 - z_normalized
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Voxel Map Viewer", width=1400, height=900)
    vis.add_geometry(pcd)
    vis.add_geometry(coord_frame)
    
    # Set render options
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.15])
    
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize voxel maps with multiple methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  basic      - Simple point cloud view
  height     - Color by Z-height
  voxels     - Show as voxel cubes
  axes       - With coordinate frame and bounding box
  slices     - Horizontal cross-sections
  density    - Color by local point density
  normals    - Show estimated surface normals
  stats      - Print statistics only
  histogram  - 2D projections (requires matplotlib)
  export     - Export to PLY, PCD, XYZ, CSV
  interactive - Interactive viewer with controls
  all        - Run all visualization methods

Examples:
  python3 visualize_voxelmap.py captures/voxel_map_merged.npy
  python3 visualize_voxelmap.py captures/ --method height
  python3 visualize_voxelmap.py captures/voxel_map_merged.ply --method all
"""
    )
    parser.add_argument("path", type=Path, help="Path to .npy, .ply file or captures directory")
    parser.add_argument("--method", "-m", type=str, default="interactive",
                        choices=["basic", "height", "voxels", "axes", "slices", 
                                "density", "normals", "stats", "histogram", 
                                "export", "interactive", "all"],
                        help="Visualization method (default: interactive)")
    parser.add_argument("--voxel-size", type=float, default=0.01,
                        help="Voxel size for grid view (default: 0.01m)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for exports")
    
    args = parser.parse_args()
    
    # Load voxel map
    print(f"Loading: {args.path}")
    points = load_voxel_map(args.path)
    print(f"Loaded {len(points):,} points")
    
    # Always print basic stats
    print_statistics(points)
    
    method = args.method
    
    if method == "basic" or method == "all":
        visualize_basic(points)
    
    if method == "height" or method == "all":
        visualize_height_colored(points)
    
    if method == "voxels" or method == "all":
        visualize_voxel_grid(points, args.voxel_size)
    
    if method == "axes" or method == "all":
        visualize_with_axes(points)
    
    if method == "slices" or method == "all":
        visualize_cross_sections(points)
    
    if method == "density":
        visualize_density(points)  # Skip in 'all' - too slow
    
    if method == "normals" or method == "all":
        visualize_normals(points)
    
    if method == "stats":
        pass  # Already printed
    
    if method == "histogram" or method == "all":
        plot_histograms(points)
    
    if method == "export":
        output_dir = args.output_dir or args.path.parent
        export_formats(points, output_dir)
    
    if method == "interactive":
        interactive_viewer(points)


if __name__ == "__main__":
    main()
