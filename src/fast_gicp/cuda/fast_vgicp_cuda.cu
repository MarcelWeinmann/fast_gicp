#include <fast_gicp/cuda/fast_vgicp_cuda.cuh>

#include <thrust/device_new.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <fast_gicp/cuda/brute_force_knn.cuh>
#include <fast_gicp/cuda/covariance_estimation.cuh>
#include <fast_gicp/cuda/covariance_regularization.cuh>
#include <fast_gicp/cuda/gaussian_voxelmap.cuh>
#include <fast_gicp/cuda/compute_mahalanobis.cuh>
#include <fast_gicp/cuda/compute_derivatives.cuh>
#include <fast_gicp/cuda/find_voxel_correspondences.cuh>

namespace fast_gicp {
namespace cuda {

using Points = thrust::device_vector<Eigen::Vector3f, thrust::device_allocator<Eigen::Vector3f>>;
using Indices = thrust::device_vector<int, thrust::device_allocator<int>>;
using Matrices = thrust::device_vector<Eigen::Matrix3f, thrust::device_allocator<Eigen::Matrix3f>>;
using Correspondences = thrust::device_vector<thrust::pair<int, int>, thrust::device_allocator<thrust::pair<int, int>>>;
using VoxelCoordinates = thrust::device_vector<Eigen::Vector3i, thrust::device_allocator<Eigen::Vector3i>>;

struct FastVGICPCudaCore::Impl {
  double resolution;
  double kernel_width;
  double kernel_max_dist;
  std::unique_ptr<thrust::device_vector<Eigen::Vector3i>> offsets;

  std::unique_ptr<thrust::device_vector<Eigen::Vector3f>> source_points;
  std::unique_ptr<thrust::device_vector<Eigen::Vector3f>> target_points;

  std::unique_ptr<thrust::device_vector<int>> source_neighbors;
  std::unique_ptr<thrust::device_vector<int>> target_neighbors;

  std::unique_ptr<thrust::device_vector<Eigen::Matrix3f>> source_covariances;
  std::unique_ptr<thrust::device_vector<Eigen::Matrix3f>> target_covariances;

  std::unique_ptr<GaussianVoxelMap> voxelmap;

  Eigen::Isometry3f linearized_x;
  std::unique_ptr<thrust::device_vector<thrust::pair<int, int>>> voxel_correspondences;

  Impl() {
    // Warming up GPU
    cudaDeviceSynchronize();

    resolution = 1.0;
    linearized_x.setIdentity();

    kernel_width = 0.25;
    kernel_max_dist = 3.0;

    offsets.reset(new thrust::device_vector<Eigen::Vector3i>(1));
    (*offsets)[0] = Eigen::Vector3i::Zero().eval();
  }
};

FastVGICPCudaCore::FastVGICPCudaCore() : pimpl(new Impl()) {}

FastVGICPCudaCore ::~FastVGICPCudaCore() {}

void FastVGICPCudaCore::set_resolution(double resolution) {
  pimpl->resolution = resolution;
}

void FastVGICPCudaCore::set_kernel_params(double kernel_width, double kernel_max_dist) {
  pimpl->kernel_width = kernel_width;
  pimpl->kernel_max_dist = kernel_max_dist;
}

void FastVGICPCudaCore::set_neighbor_search_method(fast_gicp::NeighborSearchMethod method, double radius) {
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
            if(offset.cast<double>().norm() <= radius + 1e-3) {
              h_offsets.push_back(offset);
            }
          }
        }
      }

      break;
  }

  *pimpl->offsets = h_offsets;
}

void FastVGICPCudaCore::swap_source_and_target() {
  pimpl->source_points.swap(pimpl->target_points);
  pimpl->source_neighbors.swap(pimpl->target_neighbors);
  pimpl->source_covariances.swap(pimpl->target_covariances);

  if(!pimpl->target_points || !pimpl->target_covariances) {
    return;
  }

  create_target_voxelmap();
}

void FastVGICPCudaCore::set_source_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud) {
  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud.begin(), cloud.end());
  if(!pimpl->source_points) {
    pimpl->source_points.reset(new Points());
  }

  *pimpl->source_points = points;
}

void FastVGICPCudaCore::set_target_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud) {
  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud.begin(), cloud.end());
  if(!pimpl->target_points) {
    pimpl->target_points.reset(new Points());
  }

  *pimpl->target_points = points;
}

void FastVGICPCudaCore::set_source_neighbors(int k, const std::vector<int>& neighbors) {
  assert(k * pimpl->source_points->size() == neighbors.size());
  thrust::host_vector<int> k_neighbors(neighbors.begin(), neighbors.end());

  if(!pimpl->source_neighbors) {
    pimpl->source_neighbors.reset(new thrust::device_vector<int>());
  }

  *pimpl->source_neighbors = k_neighbors;
}

void FastVGICPCudaCore::set_target_neighbors(int k, const std::vector<int>& neighbors) {
  assert(k * pimpl->target_points->size() == neighbors.size());
  thrust::host_vector<int> k_neighbors(neighbors.begin(), neighbors.end());

  if(!pimpl->target_neighbors) {
    pimpl->target_neighbors.reset(new thrust::device_vector<int>());
  }

  *pimpl->target_neighbors = k_neighbors;
}

struct untie_pair_second {
  __device__ int operator()(thrust::pair<float, int>& p) const {
    return p.second;
  }
};

void FastVGICPCudaCore::find_source_neighbors(int k) {
  assert(pimpl->source_points);

  thrust::device_vector<thrust::pair<float, int>> k_neighbors;
  brute_force_knn_search(*pimpl->source_points, *pimpl->source_points, k, k_neighbors);

  if(!pimpl->source_neighbors) {
    pimpl->source_neighbors.reset(new thrust::device_vector<int>(k_neighbors.size()));
  } else {
    pimpl->source_neighbors->resize(k_neighbors.size());
  }
  thrust::transform(k_neighbors.begin(), k_neighbors.end(), pimpl->source_neighbors->begin(), untie_pair_second());
}

void FastVGICPCudaCore::find_target_neighbors(int k) {
  assert(pimpl->target_points);

  thrust::device_vector<thrust::pair<float, int>> k_neighbors;
  brute_force_knn_search(*pimpl->target_points, *pimpl->target_points, k, k_neighbors);

  if(!pimpl->target_neighbors) {
    pimpl->target_neighbors.reset(new thrust::device_vector<int>(k_neighbors.size()));
  } else {
    pimpl->target_neighbors->resize(k_neighbors.size());
  }
  thrust::transform(k_neighbors.begin(), k_neighbors.end(), pimpl->target_neighbors->begin(), untie_pair_second());
}

void FastVGICPCudaCore::calculate_source_covariances(RegularizationMethod method) {
  assert(pimpl->source_points && pimpl->source_neighbors);
  int k = pimpl->source_neighbors->size() / pimpl->source_points->size();

  if(!pimpl->source_covariances) {
    pimpl->source_covariances.reset(new thrust::device_vector<Eigen::Matrix3f>(pimpl->source_points->size()));
  }
  covariance_estimation(*pimpl->source_points, k, *pimpl->source_neighbors, *pimpl->source_covariances);
  covariance_regularization(*pimpl->source_points, *pimpl->source_covariances, method);
}

void FastVGICPCudaCore::calculate_target_covariances(RegularizationMethod method) {
  assert(pimpl->target_points && pimpl->target_neighbors);
  int k = pimpl->target_neighbors->size() / pimpl->target_points->size();

  if(!pimpl->target_covariances) {
    pimpl->target_covariances.reset(new thrust::device_vector<Eigen::Matrix3f>(pimpl->target_points->size()));
  }
  covariance_estimation(*pimpl->target_points, k, *pimpl->target_neighbors, *pimpl->target_covariances);
  covariance_regularization(*pimpl->target_points, *pimpl->target_covariances, method);
}

void FastVGICPCudaCore::calculate_source_covariances_kernel(RegularizationMethod method, KernelMethod kernel) {
  if(!pimpl->source_covariances) {
    pimpl->source_covariances.reset(new thrust::device_vector<Eigen::Matrix3f>(pimpl->source_points->size()));
  }
  covariance_estimation_kernelized(*pimpl->source_points, pimpl->kernel_width, pimpl->kernel_max_dist, kernel, *pimpl->source_covariances);
  covariance_regularization(*pimpl->source_points, *pimpl->source_covariances, method);
}

void FastVGICPCudaCore::calculate_target_covariances_kernel(RegularizationMethod method, KernelMethod kernel) {
  if(!pimpl->target_covariances) {
    pimpl->target_covariances.reset(new thrust::device_vector<Eigen::Matrix3f>(pimpl->target_points->size()));
  }
  covariance_estimation_kernelized(*pimpl->target_points, pimpl->kernel_width, pimpl->kernel_max_dist, kernel, *pimpl->target_covariances);
  covariance_regularization(*pimpl->target_points, *pimpl->target_covariances, method);
}

void FastVGICPCudaCore::get_voxel_correspondences(std::vector<std::pair<int, int>>& correspondences) const {
  thrust::host_vector<thrust::pair<int, int>> corrs = *pimpl->voxel_correspondences;
  correspondences.resize(corrs.size());
  std::transform(corrs.begin(), corrs.end(), correspondences.begin(), [](const auto& x) { return std::make_pair(x.first, x.second); });
}

void FastVGICPCudaCore::get_voxel_num_points(std::vector<int>& num_points) const {
  thrust::host_vector<int> voxel_num_points = pimpl->voxelmap->num_points;
  num_points.resize(voxel_num_points.size());
  std::copy(voxel_num_points.begin(), voxel_num_points.end(), num_points.begin());
}

void FastVGICPCudaCore::get_voxel_means(std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& means) const {
  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> voxel_means = pimpl->voxelmap->voxel_means;
  means.resize(voxel_means.size());
  std::copy(voxel_means.begin(), voxel_means.end(), means.begin());
}

void FastVGICPCudaCore::get_voxel_covs(std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>>& covs) const {
  thrust::host_vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> voxel_covs = pimpl->voxelmap->voxel_covs;
  covs.resize(voxel_covs.size());
  std::copy(voxel_covs.begin(), voxel_covs.end(), covs.begin());
}

void FastVGICPCudaCore::get_source_covariances(std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>>& covs) const {
  thrust::host_vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> c = *pimpl->source_covariances;
  covs.resize(c.size());
  std::copy(c.begin(), c.end(), covs.begin());
}

void FastVGICPCudaCore::get_target_covariances(std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>>& covs) const {
  thrust::host_vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> c = *pimpl->target_covariances;
  covs.resize(c.size());
  std::copy(c.begin(), c.end(), covs.begin());
}

void FastVGICPCudaCore::create_target_voxelmap() {
  assert(pimpl->target_points && pimpl->target_covariances);
  if(!pimpl->voxelmap) {
    pimpl->voxelmap.reset(new GaussianVoxelMap(pimpl->resolution));
  }
  pimpl->voxelmap->create_voxelmap(*pimpl->target_points, *pimpl->target_covariances);
}

void FastVGICPCudaCore::update_correspondences(const Eigen::Isometry3d& trans) {
  thrust::device_vector<Eigen::Isometry3f> trans_ptr(1);
  trans_ptr[0] = trans.cast<float>();

  if(pimpl->voxel_correspondences == nullptr) {
    pimpl->voxel_correspondences.reset(new Correspondences());
  }
  pimpl->linearized_x = trans.cast<float>();
  find_voxel_correspondences(*pimpl->source_points, *pimpl->voxelmap, trans_ptr.data(), *pimpl->offsets, *pimpl->voxel_correspondences);
}

double FastVGICPCudaCore::compute_error(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) const {
  thrust::host_vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> trans_(2);
  trans_[0] = pimpl->linearized_x;
  trans_[1] = trans.cast<float>();

  thrust::device_vector<Eigen::Isometry3f> trans_ptr = trans_;

  return compute_derivatives(*pimpl->source_points, *pimpl->source_covariances, *pimpl->voxelmap, *pimpl->voxel_correspondences, trans_ptr.data(), trans_ptr.data() + 1, H, b);
}

}  // namespace cuda
}  // namespace fast_gicp
