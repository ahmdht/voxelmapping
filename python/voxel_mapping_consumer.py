"""Thin Python convenience wrapper around the pybind11 module."""

import numpy as np

import voxel_mapping_py


class VoxelMappingConsumer:
    def __init__(self, voxel_size: float = 0.1):
        self._voxel_map = voxel_mapping_py.VoxelMapping(voxel_size)

    def register_cloud(self, point_cloud: np.ndarray, transform: np.ndarray) -> None:
        points = np.asarray(point_cloud, dtype=np.float32)
        tf = np.asarray(transform, dtype=np.float32)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("point_cloud must be shaped (N, 3)")
        if tf.shape != (4, 4):
            raise ValueError("transform must be shaped (4, 4)")

        self._voxel_map.register_cloud(points, tf)

    def retrieve_map(self) -> np.ndarray:
        return np.asarray(self._voxel_map.retrieve_map_points(), dtype=np.float32)


__all__ = ["VoxelMappingConsumer"]