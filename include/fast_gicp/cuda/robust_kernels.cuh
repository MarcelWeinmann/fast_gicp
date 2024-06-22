#ifndef FAST_GICP_CUDA_ROBUST_KERNELS_CUH
#define FAST_GICP_CUDA_ROBUST_KERNELS_CUH

#include <Eigen/Core>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {
  namespace cuda {

  __host__ __device__ inline float square(float x) { return x * x; }

  __host__ __device__ inline float calculate_kernel(float kernel_width, const Eigen::Vector3f& error, KernelMethod method) {
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
  }
}  // namespace fast_gicp

#endif