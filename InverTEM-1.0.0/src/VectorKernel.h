#pragma once

#include <complex>
#include <cstddef>

namespace pytem {

// Inputs of one step-response evaluation, laid out for VectorKernel.cpp.
struct KernelView {
    const double *lambdaSquared, *hankelFactors;
    const double *lambdas;
    std::size_t lambdaCount;
    const std::complex<double> *s, *coefficients;
    std::size_t frequencies;
    const double *thicknesses, *resistivities;
    const char *sameRho, *sameLayer;
    std::size_t layers;
};

// True when this build contains the AVX2 kernel and the CPU can run it.
bool vectorKernelAvailable();

// Response at step time t (and, when jacobian is non-null, its n layer
// derivatives) with the wavenumber loop innermost so it vectorises.
// Only call when vectorKernelAvailable() is true.
void vectorKernelTime(const KernelView &view, std::size_t t, double &response, double *jacobian);

} // namespace pytem
