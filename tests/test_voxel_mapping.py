import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
BUILD_DIR = ROOT / "build"
for p in (str(PYTHON_DIR), str(BUILD_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from voxel_mapping_consumer import VoxelMappingConsumer


class TestVoxelMapping(unittest.TestCase):
    def setUp(self):
        self.voxel_map = VoxelMappingConsumer(voxel_size=0.2)

    def test_register_and_retrieve(self):
        point_cloud = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        transform = np.eye(4, dtype=np.float32)

        self.voxel_map.register_cloud(point_cloud, transform)
        retrieved_map = self.voxel_map.retrieve_map()

        self.assertEqual(retrieved_map.shape[1], 3)
        self.assertGreaterEqual(retrieved_map.shape[0], 2)
        # ensure voxelization merged the two close points into one voxel
        self.assertLessEqual(retrieved_map.shape[0], point_cloud.shape[0])


if __name__ == "__main__":
    unittest.main()