#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

#include "VoxelMapping.h"

namespace py = pybind11;

namespace {

pcl::PointCloud<pcl::PointXYZ>::Ptr numpy_to_cloud(const py::array_t<float>& points) {
    const auto buf = points.request();
    if (buf.ndim != 2 || buf.shape[1] != 3) {
        throw std::runtime_error("points must have shape (N, 3)");
    }

    const size_t n_points = static_cast<size_t>(buf.shape[0]);
    const auto* data = static_cast<const float*>(buf.ptr);

    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->resize(n_points);
    for (size_t i = 0; i < n_points; ++i) {
        (*cloud)[i].x = data[3 * i + 0];
        (*cloud)[i].y = data[3 * i + 1];
        (*cloud)[i].z = data[3 * i + 2];
    }

    return cloud;
}

}  // namespace

PYBIND11_MODULE(voxel_mapping_py, m) {
    m.doc() = "Pybind11 bindings for the VoxelMapping class";

    py::class_<VoxelMapping>(m, "VoxelMapping")
        .def(py::init<float>(), py::arg("voxel_size"))
        .def("register_cloud",
             [](VoxelMapping& self,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& points,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& transform) {
                 auto cloud = numpy_to_cloud(points);

                 // Validate transform shape and map to Eigen::Matrix4f (row-major)
                 const auto tbuf = transform.request();
                 if (tbuf.ndim != 2 || tbuf.shape[0] != 4 || tbuf.shape[1] != 4) {
                     throw std::runtime_error("transform must have shape (4,4)");
                 }
                 const float* tdata = static_cast<const float*>(tbuf.ptr);
                 Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> tf_map(tdata);

                 self.registerCloud(cloud, tf_map);
             },
             py::arg("points"), py::arg("transform"),
             "Register a cloud (Nx3 float32 array) with a 4x4 transform")
        .def("retrieve_map_points", &VoxelMapping::retrieveMapPoints,
             "Retrieve the voxelized map as a list of 3D points")
        .def_property_readonly("voxel_size", &VoxelMapping::voxelSize,
                               "Leaf size used for voxelization");
}
