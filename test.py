import sys
import json
import numpy as np
from pathlib import Path

# Add voxelmapping paths
sys.path.insert(0, "/home/ahmad.hoteit/voxelmapping/python")
sys.path.insert(0, "/home/ahmad.hoteit/voxelmapping/build")

from voxel_mapping_consumer import VoxelMappingConsumer

# Initialize voxel map
voxel = VoxelMappingConsumer(voxel_size=0.005)  # 5mm voxels

# Load transforms
data_dir = Path("/home/ahmad.hoteit/3d_reconstruction_pipeline/captures_20260123_140948")
with open(data_dir / "camera_poses_in_base.json") as f:
    poses = json.load(f)

# Register each cloud
for entry in poses:
    pose_idx = entry["pose_index"]
    T = np.array(entry["T_base_cam"], dtype=np.float32)  # 4x4 transform
    
    pose_dir = data_dir / f"pose_{pose_idx:02d}"
    cloud_files = sorted(pose_dir.glob("cloud_*.npy"))
    
    for cloud_file in cloud_files:
        points = np.load(cloud_file).astype(np.float32)
        # Filter invalid points
        valid = (np.abs(points).sum(axis=1) > 0) & (np.abs(points).sum(axis=1) < 10)
        points = points[valid]
        
        voxel.register_cloud(points, T)

# Retrieve merged map
map_points = voxel.retrieve_map()
print(f"Final voxel map: {map_points.shape[0]} points")

# Save or visualize
np.save("voxel_map_merged.npy", map_points)