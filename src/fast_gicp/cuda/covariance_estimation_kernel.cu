#include <fast_gicp/cuda/covariance_estimation.cuh>

#include <thrust/device_vector.h>

#include <thrust/async/for_each.h>
#include <thrust/async/transform.h>

namespace fast_gicp {
namespace cuda {

struct NormalDistribution {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  __host__ __device__ NormalDistribution() {}

  static __host__ __device__ NormalDistribution zero() {
    NormalDistribution dist;
    dist.sum_weights = 0.0f;
    dist.mean.setZero();
    dist.cov.setZero();
    return dist;
  }

  __host__ __device__ NormalDistribution operator+(const NormalDistribution& rhs) const {
    NormalDistribution sum;
    sum.sum_weights = sum_weights + rhs.sum_weights;
    sum.mean = mean + rhs.mean;
    sum.cov = cov + rhs.cov;
    return sum;
  }

  __host__ __device__ NormalDistribution& operator+=(const NormalDistribution& rhs) {
    sum_weights += rhs.sum_weights;
    mean += rhs.mean;
    cov += rhs.cov;
    return *this;
  }

  __host__ __device__ void accumulate(const float w, const Eigen::Vector3f& x) {
    sum_weights += w;
    mean += w * x;
    cov += w * x * x.transpose();
  }

  __host__ __device__ NormalDistribution& finalize() {
    Eigen::Vector3f sum_pt = mean;
    mean /= sum_weights;
    cov = (cov - mean * sum_pt.transpose()) / sum_weights;

    return *this;
  }

  float sum_weights;
  Eigen::Vector3f mean;
  Eigen::Matrix3f cov;
};

struct covariance_estimation_kernel {
  static const int BLOCK_SIZE = 512;

  covariance_estimation_kernel(thrust::device_ptr<const float> kernel_width_ptr, thrust::device_ptr<const float> max_dist_ptr, thrust::device_ptr<const KernelMethod> kernel_ptr, thrust::device_ptr<const Eigen::Vector3f> points_ptr)
  : kernel_width_ptr(kernel_width_ptr),
    max_dist_ptr(max_dist_ptr),
    kernel_ptr(kernel_ptr),
    points_ptr(points_ptr) {}

  __host__ __device__ NormalDistribution operator()(const Eigen::Vector3f& x) const {
    float kernel_width = *thrust::raw_pointer_cast(kernel_width_ptr);
    const float max_dist = *thrust::raw_pointer_cast(max_dist_ptr);
    const float max_dist_sq = max_dist * max_dist;
    const KernelMethod kernel = *thrust::raw_pointer_cast(kernel_ptr);
    const Eigen::Vector3f* points = thrust::raw_pointer_cast(points_ptr);

    NormalDistribution dist = NormalDistribution::zero();
    for (int i = 0; i < BLOCK_SIZE; i++) {
      float sq_d = (x - points[i]).squaredNorm();
      if (sq_d > max_dist_sq) {
        continue;
      }

      float w = calculate_kernel(kernel_width, x - points[i], kernel);
      dist.accumulate(w, points[i]);
    }

    return dist;
  }

  __host__ __device__ inline float square(float x) const { return x * x; }

  __host__ __device__ float calculate_kernel(float kernel_width, const Eigen::Vector3f& error, KernelMethod method) const {
      switch(method) {
          case KernelMethod::RBF:
              return exp(- kernel_width * error.squaredNorm());
              break;
          case KernelMethod::L1:
              return 1.0 / error.template lpNorm<1>();
              break;
          case KernelMethod::Geman_McClure:
              return square(kernel_width) / square((kernel_width + error.squaredNorm()));
              break;
          case KernelMethod::Welsch:
              return exp(- (error / kernel_width).squaredNorm());
              break;
          case KernelMethod::Switchable_Constraint: {
              float squared_error = error.squaredNorm();
              if (squared_error <= kernel_width) {
                  return 1.0;
              } else {
                  return 4 * square(kernel_width) / square(kernel_width + squared_error);
              }}
              break;
          default:
              return 1.0;
      }
  }

  thrust::device_ptr<const float> kernel_width_ptr;
  thrust::device_ptr<const float> max_dist_ptr;
  thrust::device_ptr<const KernelMethod> kernel_ptr;
  thrust::device_ptr<const Eigen::Vector3f> points_ptr;
};

struct finalization_kernel {
  finalization_kernel(const int stride, const thrust::device_vector<NormalDistribution>& accumulated_dists)
  : stride(stride),
    accumulated_dists_first(accumulated_dists.data()),
    accumulated_dists_last(accumulated_dists.data() + accumulated_dists.size()) {}

  __host__ __device__ Eigen::Matrix3f operator()(int index) const {
    const NormalDistribution* dists = thrust::raw_pointer_cast(accumulated_dists_first);
    const NormalDistribution* dists_last = thrust::raw_pointer_cast(accumulated_dists_last);
    const int num_dists = dists_last - dists;

    NormalDistribution sum = dists[index];
    for (int dist_index = index + stride; dist_index < num_dists; dist_index += stride) {
      sum += dists[dist_index];
    }

    return sum.finalize().cov;
  }

  const int stride;
  thrust::device_ptr<const NormalDistribution> accumulated_dists_first;
  thrust::device_ptr<const NormalDistribution> accumulated_dists_last;
};

void covariance_estimation_kernelized(const thrust::device_vector<Eigen::Vector3f>& points, double kernel_width, double max_dist, KernelMethod kernel, thrust::device_vector<Eigen::Matrix3f>& covariances) {
  covariances.resize(points.size());

  thrust::device_vector<float> constants(2);
  constants[0] = kernel_width;
  constants[1] = max_dist;
  thrust::device_ptr<const float> kernel_width_ptr = constants.data();
  thrust::device_ptr<const float> max_dist_ptr = constants.data() + 1;

  thrust::device_vector<KernelMethod> kernel_type(1);
  kernel_type[0] = kernel;
  thrust::device_ptr<const KernelMethod> kernel_ptr = kernel_type.data();

  int num_blocks = (points.size() + (covariance_estimation_kernel::BLOCK_SIZE - 1)) / covariance_estimation_kernel::BLOCK_SIZE;
  // padding
  thrust::device_vector<Eigen::Vector3f> ext_points(num_blocks * covariance_estimation_kernel::BLOCK_SIZE);
  thrust::copy(points.begin(), points.end(), ext_points.begin());
  thrust::fill(ext_points.begin() + points.size(), ext_points.end(), Eigen::Vector3f(0.0f, 0.0f, 0.0f));

  thrust::device_vector<NormalDistribution> accumulated_dists(points.size() * num_blocks);

  thrust::system::cuda::detail::unique_stream stream;
  std::vector<thrust::system::cuda::unique_eager_event> events(num_blocks);

  // accumulate kerneled point distributions
  for (int i = 0; i < num_blocks; i++) {
    covariance_estimation_kernel kernel(kernel_width_ptr, max_dist_ptr, kernel_ptr, ext_points.data() + covariance_estimation_kernel::BLOCK_SIZE * i);
    auto event = thrust::async::transform(points.begin(), points.end(), accumulated_dists.begin() + points.size() * i, kernel);
    events[i] = std::move(event);
    thrust::system::cuda::detail::create_dependency(stream, events[i]);
  }

  // finalize distributions
  thrust::transform(
    thrust::cuda::par.on(stream.native_handle()),
    thrust::counting_iterator<int>(0),
    thrust::counting_iterator<int>(points.size()),
    covariances.begin(),
    finalization_kernel(points.size(), accumulated_dists));
}

}  // namespace cuda
}  // namespace fast_gicp