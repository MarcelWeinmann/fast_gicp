#ifndef ROBUST_KERNELS_HPP
#define ROBUST_KERNELS_HPP

#include <Eigen/Core>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {

template <typename T>
inline T square(T x) { return x * x; }

template <typename R, typename T>
R calculate_kernel(R kernel_width, T error, KernelMethod method) {
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
        case KernelMethod::Cauchy:
            return 1 / ( 1 + (error / kernel_width).squaredNorm());
            break;
        case KernelMethod::Switchable_Constraint: {
            R squared_error = error.squaredNorm();
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
}  // namespace fast_gicp

#endif
