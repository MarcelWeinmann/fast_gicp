#include <fast_gicp/cuda/ndt_cuda.cuh>

#include <thrust/device_vector.h>

#include <fast_gicp/cuda/gaussian_voxelmap.cuh>
#include <fast_gicp/cuda/covariance_regularization.cuh>
#include <fast_gicp/cuda/find_voxel_correspondences.cuh>
#include <fast_gicp/cuda/ndt_compute_derivatives.cuh>

namespace fast_gicp {
namespace cuda {

using Points = thrust::device_vector<Eigen::Vector3f, thrust::device_allocator<Eigen::Vector3f>>;
using Indices = thrust::device_vector<int, thrust::device_allocator<int>>;
using Matrices = thrust::device_vector<Eigen::Matrix3f, thrust::device_allocator<Eigen::Matrix3f>>;
using Correspondences = thrust::device_vector<thrust::pair<int, int>, thrust::device_allocator<thrust::pair<int, int>>>;
using VoxelCoordinates = thrust::device_vector<Eigen::Vector3i, thrust::device_allocator<Eigen::Vector3i>>;

struct NDTCudaCore::Impl {
  fast_gicp::NDTDistanceMode distance_mode;
  fast_gicp::KernelMethod kernel_method;
  fast_gicp::RegularizationMethod regularization_method;
  double resolution;
  double kernel_width;
  std::unique_ptr<thrust::device_vector<Eigen::Vector3i>> offsets;

  std::unique_ptr<thrust::device_vector<Eigen::Vector3f>> source_points;
  std::unique_ptr<thrust::device_vector<Eigen::Vector3f>> target_points;

  std::unique_ptr<GaussianVoxelMap> source_voxelmap;
  std::unique_ptr<GaussianVoxelMap> target_voxelmap;

  Eigen::Isometry3f linearized_x;
  std::unique_ptr<thrust::device_vector<thrust::pair<int, int>>> correspondences;

  Impl() {
    cudaDeviceSynchronize();
    resolution = 1.0;
    kernel_width = 0.5;
    linearized_x.setIdentity();

    offsets.reset(new thrust::device_vector<Eigen::Vector3i>(1));
    (*offsets)[0] = Eigen::Vector3i::Zero().eval();

    kernel_method = fast_gicp::KernelMethod::None;
    regularization_method = fast_gicp::RegularizationMethod::MIN_EIG;
    distance_mode = fast_gicp::NDTDistanceMode::D2D;
  }
};

NDTCudaCore::NDTCudaCore() : pimpl(std::make_unique<Impl>()) {}

NDTCudaCore::~NDTCudaCore() {}

void NDTCudaCore::set_distance_mode(fast_gicp::NDTDistanceMode mode) {
  pimpl->distance_mode = mode;
}

void NDTCudaCore::set_resolution(double resolution) {
  pimpl->resolution = resolution;
}

void NDTCudaCore::set_regularization_method(fast_gicp::RegularizationMethod method) {
  pimpl->regularization_method = method;
}

void NDTCudaCore::set_kernel_widht(double kernel_width) {
  pimpl->kernel_width = kernel_width;
}

void NDTCudaCore::set_kernel_method(fast_gicp::KernelMethod method) {
  pimpl->kernel_method = method;
}

void NDTCudaCore::set_neighbor_search_method(fast_gicp::NeighborSearchMethod method, double radius) {
  thrust::host_vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> h_offsets;

  switch (method) {
    default:
      std::cerr << "here must not be reached" << std::endl;
      abort();

    case fast_gicp::NeighborSearchMethod::DIRECT1:
      h_offsets.resize(1);
      h_offsets[0] = Eigen::Vector3i::Zero();
      break;

    case fast_gicp::NeighborSearchMethod::DIRECT7:
      h_offsets.resize(7);
      h_offsets[0] = Eigen::Vector3i(0, 0, 0);
      h_offsets[1] = Eigen::Vector3i(1, 0, 0);
      h_offsets[2] = Eigen::Vector3i(-1, 0, 0);
      h_offsets[3] = Eigen::Vector3i(0, 1, 0);
      h_offsets[4] = Eigen::Vector3i(0, -1, 0);
      h_offsets[5] = Eigen::Vector3i(0, 0, 1);
      h_offsets[6] = Eigen::Vector3i(0, 0, -1);
      break;

    case fast_gicp::NeighborSearchMethod::DIRECT27:
      h_offsets.reserve(27);
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < 3; k++) {
            h_offsets.push_back(Eigen::Vector3i(i - 1, j - 1, k - 1));
          }
        }
      }
      break;

    case fast_gicp::NeighborSearchMethod::DIRECT_RADIUS:
      h_offsets.reserve(50);
      int range = std::ceil(radius);
      for (int i = -range; i <= range; i++) {
        for (int j = -range; j <= range; j++) {
          for (int k = -range; k <= range; k++) {
            Eigen::Vector3i offset(i, j, k);
            if (offset.cast<double>().norm() <= radius + 1e-3) {
              h_offsets.push_back(offset);
            }
          }
        }
      }

      break;
  }

  *pimpl->offsets = h_offsets;
}

void NDTCudaCore::swap_source_and_target() {
  pimpl->source_points.swap(pimpl->target_points);
  pimpl->source_voxelmap.swap(pimpl->target_voxelmap);
}

void NDTCudaCore::set_source_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud) {
  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud.begin(), cloud.end());
  if (!pimpl->source_points) {
    pimpl->source_points.reset(new Points());
  }

  *pimpl->source_points = points;
  pimpl->source_voxelmap.reset();
}

void NDTCudaCore::set_target_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud) {
  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud.begin(), cloud.end());
  if (!pimpl->target_points) {
    pimpl->target_points.reset(new Points());
  }

  *pimpl->target_points = points;
  pimpl->target_voxelmap.reset();
}

void NDTCudaCore::create_voxelmaps() {
  create_source_voxelmap();
  create_target_voxelmap();
}

void NDTCudaCore::create_source_voxelmap() {
  assert(pimpl->source_points);
  if (pimpl->source_voxelmap || pimpl->distance_mode == fast_gicp::NDTDistanceMode::P2D) {
    return;
  }

  pimpl->source_voxelmap.reset(new GaussianVoxelMap(pimpl->resolution));
  pimpl->source_voxelmap->create_voxelmap(*pimpl->source_points);
  covariance_regularization(pimpl->source_voxelmap->voxel_means, pimpl->source_voxelmap->voxel_covs, pimpl->regularization_method);
}

void NDTCudaCore::create_target_voxelmap() {
  assert(pimpl->target_points);
  if (pimpl->target_voxelmap) {
    return;
  }

  pimpl->target_voxelmap.reset(new GaussianVoxelMap(pimpl->resolution));
  pimpl->target_voxelmap->create_voxelmap(*pimpl->target_points);
  covariance_regularization(pimpl->target_voxelmap->voxel_means, pimpl->target_voxelmap->voxel_covs, pimpl->regularization_method);
}

void NDTCudaCore::update_correspondences(const Eigen::Isometry3d& trans) {
  thrust::device_vector<Eigen::Isometry3f> trans_ptr(1);
  trans_ptr[0] = trans.cast<float>();

  if (pimpl->correspondences == nullptr) {
    pimpl->correspondences.reset(new Correspondences());
  }
  pimpl->linearized_x = trans.cast<float>();

  switch (pimpl->distance_mode) {
    case fast_gicp::NDTDistanceMode::P2D:
      find_voxel_correspondences(*pimpl->source_points, *pimpl->target_voxelmap, trans_ptr.data(), *pimpl->offsets, *pimpl->correspondences);
      break;

    case fast_gicp::NDTDistanceMode::D2D:
      find_voxel_correspondences(pimpl->source_voxelmap->voxel_means, *pimpl->target_voxelmap, trans_ptr.data(), *pimpl->offsets, *pimpl->correspondences);
      break;
  }
}

double NDTCudaCore::compute_error(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) const {
  thrust::host_vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> trans_(2);
  trans_[0] = pimpl->linearized_x;
  trans_[1] = trans.cast<float>();

  thrust::device_vector<Eigen::Isometry3f> trans_ptr = trans_;

  thrust::device_vector<fast_gicp::KernelMethod> kernel_method_ptr(1);
  kernel_method_ptr[0] = pimpl->kernel_method;
  thrust::device_vector<float> kernel_width_ptr(1);
  kernel_width_ptr[0] = pimpl->kernel_width;

  switch (pimpl->distance_mode) {
    default:
    case fast_gicp::NDTDistanceMode::P2D:
      return p2d_ndt_compute_derivatives(*pimpl->target_voxelmap, *pimpl->source_points, *pimpl->correspondences, trans_ptr.data(),
                                         trans_ptr.data() + 1, kernel_method_ptr.data(), kernel_width_ptr.data(), H, b);

    case fast_gicp::NDTDistanceMode::D2D:
      return d2d_ndt_compute_derivatives(*pimpl->target_voxelmap, *pimpl->source_voxelmap, *pimpl->correspondences, trans_ptr.data(),
                                         trans_ptr.data() + 1, kernel_method_ptr.data(), kernel_width_ptr.data(), H, b);
  }
}

}  // namespace cuda

}  // namespace fast_gicp