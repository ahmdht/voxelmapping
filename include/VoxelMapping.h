// VoxelMapping Class

#ifndef VOXELMAPPING_H
#define VOXELMAPPING_H

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

class VoxelMapping {
public:
    explicit VoxelMapping(float voxel_size);

    void registerCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud,
                       const Eigen::Matrix4f& transform);

    pcl::PointCloud<pcl::PointXYZ>::ConstPtr retrieveMap() const;

    std::vector<Eigen::Vector3f> retrieveMapPoints() const;

    float voxelSize() const { return voxel_size_; }

private:
    float voxel_size_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_map_;
};

#endif // VOXELMAPPING_H