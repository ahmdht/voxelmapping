// VoxelMapping Implementation

#include "VoxelMapping.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <memory>

VoxelMapping::VoxelMapping(float voxel_size) : voxel_size_(voxel_size) {
    voxel_map_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
}

void VoxelMapping::registerCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
                                 const Eigen::Matrix4f& transform) {
    auto transformed_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform);

    *voxel_map_ += *transformed_cloud;

    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    voxel_filter.setInputCloud(voxel_map_);

    pcl::PointCloud<pcl::PointXYZ> filtered;
    voxel_filter.filter(filtered);
    voxel_map_ = filtered.makeShared();
}

pcl::PointCloud<pcl::PointXYZ>::ConstPtr VoxelMapping::retrieveMap() const {
    return voxel_map_;
}

std::vector<Eigen::Vector3f> VoxelMapping::retrieveMapPoints() const {
    std::vector<Eigen::Vector3f> points;
    points.reserve(voxel_map_->size());
    for (const auto& p : voxel_map_->points) {
        points.emplace_back(p.x, p.y, p.z);
    }
    return points;
}