# VoxelMapping

## Overview
This repository contains a C++ class `VoxelMapping` that utilizes the Point Cloud Library (PCL) for voxelization. A pybind11 module (`voxel_mapping_py`) exposes the class to Python, and a thin consumer wrapper demonstrates usage from NumPy arrays.

## Project Structure
- `src/`: Contains the C++ source files.
- `include/`: Contains the header files.
- `python/`: Contains the Python consumer code.
- `CMakeLists.txt`: CMake configuration file for building the project.
- `README.md`: Project documentation.

## Building the Project
1. Install dependencies:
   - CMake, PCL, pybind11 (CMake config), Eigen3, Python 3 with `pytest`.
2. Build the project (library + pybind11 module):
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage
### C++ Class
- **Initialization**: `VoxelMapping(float voxel_size)`
- **Register Point Cloud**: `void registerCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const Eigen::Matrix4f& transform)`
- **Retrieve Map**: `pcl::PointCloud<pcl::PointXYZ>::Ptr retrieveMap()`

### Python Consumer
```python
import numpy as np
from voxel_mapping_consumer import VoxelMappingConsumer

voxel_map = VoxelMappingConsumer(voxel_size=0.1)
points = np.array([[0, 0, 0], [0.05, 0, 0], [1, 1, 1]], dtype=np.float32)
transform = np.eye(4, dtype=np.float32)

voxel_map.register_cloud(points, transform)
map_points = voxel_map.retrieve_map()
```

## Testing

From the `build` directory after running CMake:
```bash
ctest
```

## License
This project is licensed under the MIT License.