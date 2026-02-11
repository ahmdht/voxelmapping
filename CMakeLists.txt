# CMakeLists.txt for VoxelMapping

cmake_minimum_required(VERSION 3.16)
project(VoxelMapping LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(MPI REQUIRED)
find_package(PCL REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

add_definitions(${PCL_DEFINITIONS})

add_library(VoxelMapping src/VoxelMapping.cpp)
target_include_directories(VoxelMapping PUBLIC include ${PCL_INCLUDE_DIRS})
target_link_libraries(VoxelMapping PUBLIC ${PCL_LIBRARIES} Eigen3::Eigen)

pybind11_add_module(voxel_mapping_py src/bindings.cpp)
target_link_libraries(voxel_mapping_py PRIVATE VoxelMapping ${PCL_LIBRARIES} Eigen3::Eigen)
target_include_directories(voxel_mapping_py PRIVATE include ${PCL_INCLUDE_DIRS})

enable_testing()
add_subdirectory(tests)