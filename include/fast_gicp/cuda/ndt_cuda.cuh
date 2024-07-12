#ifndef FAST_GICP_NDT_CUDA_CUH
#define FAST_GICP_NDT_CUDA_CUH

#include <memory>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <fast_gicp/ndt/ndt_settings.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {
namespace cuda {

class GaussianVoxelMap;

class NDTCudaCore {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NDTCudaCore();
  ~NDTCudaCore();

  void set_distance_mode(fast_gicp::NDTDistanceMode mode);
  void set_resolution(double resolution);
  void set_regularization_method(fast_gicp::RegularizationMethod method);
  void set_kernel_widht(double kernel_width);
  void set_kernel_method(fast_gicp::KernelMethod method);
  void set_neighbor_search_method(fast_gicp::NeighborSearchMethod method, double radius);

  void swap_source_and_target();
  void set_source_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud);
  void set_target_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud);

  void create_voxelmaps();
  void create_target_voxelmap();
  void create_source_voxelmap();

  void update_correspondences(const Eigen::Isometry3d& trans);
  double compute_error(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) const;

private:
  struct Impl;
  std::unique_ptr<Impl> pimpl;
};

}  // namespace cuda
}  // namespace fast_gicp

#endif