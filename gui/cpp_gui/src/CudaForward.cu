#include "CudaForward.h"
#include "TemSolver.h"
#include "TransformWeights.h"

#include <cuda_runtime.h>
#include <thrust/complex.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

namespace pytem {
namespace {

constexpr int hankelCount = 101;
constexpr int fourierCount = 81;
constexpr int maximumLayers = 128;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double mu0 = 4.0e-7 * pi;

__constant__ double cFourierBase[fourierCount];
__constant__ double cFourierSin[fourierCount];

std::once_flag weightsOnce;
bool weightsReady = false;
std::string weightsError;

std::string cudaMessage(cudaError_t error)
{
    return error == cudaSuccess ? std::string() : cudaGetErrorString(error);
}

void initializeWeights()
{
    cudaError_t status = cudaMemcpyToSymbol(
        cFourierBase, weights::fourier_base_81.data(),
        sizeof(double) * fourierCount);
    if (status == cudaSuccess)
        status = cudaMemcpyToSymbol(cFourierSin, weights::fourier_sin_81.data(),
                                    sizeof(double) * fourierCount);
    weightsReady = status == cudaSuccess;
    if (!weightsReady)
        weightsError = cudaMessage(status);
}

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count)
    {
        if (count > 0)
            m_status = cudaMalloc(reinterpret_cast<void **>(&m_pointer),
                                  count * sizeof(T));
    }
    ~DeviceBuffer() { if (m_pointer) cudaFree(m_pointer); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    T *get() const { return m_pointer; }
    bool valid() const { return m_status == cudaSuccess; }
    cudaError_t status() const { return m_status; }
private:
    T *m_pointer = nullptr;
    cudaError_t m_status = cudaSuccess;
};

__device__ thrust::complex<double> filterStage(
    thrust::complex<double> omega, double frequency, int order, bool highPass)
{
    const thrust::complex<double> imaginary(0.0, 1.0);
    const thrust::complex<double> s = imaginary * omega;
    const double cutoff = 2.0 * pi * frequency;
    if (order == 2) {
        const auto denominator = s * s + std::sqrt(2.0) * cutoff * s
                               + cutoff * cutoff;
        return highPass ? s * s / denominator
                        : thrust::complex<double>(cutoff * cutoff, 0.0) / denominator;
    }
    return highPass ? s / (s + cutoff)
                    : thrust::complex<double>(cutoff, 0.0) / (s + cutoff);
}

__global__ void frequencyContributions(
    const double *times, int timeCount,
    const double *thicknesses, const double *conductivity, int layerCount,
    const double *lambdas, const double *lambdaSquared,
    const double *hankelFactors,
    const double *lowPass, const int *lowPassOrders, int lowPassCount,
    const double *highPass, const int *highPassOrders, int highPassCount,
    double *contributions)
{
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = timeCount * fourierCount;
    if (flat >= count)
        return;
    const int timeIndex = flat / fourierCount;
    const int frequencyIndex = flat % fourierCount;
    const double time = times[timeIndex];
    const double omegaValue = cFourierBase[frequencyIndex] / time;
    const thrust::complex<double> omega(omegaValue, 0.0);
    const thrust::complex<double> imaginary(0.0, 1.0);
    thrust::complex<double> field(0.0, 0.0);

    for (int hankel = 0; hankel < hankelCount; ++hankel) {
        const double lambda = lambdas[hankel];
        const double lambda2 = lambdaSquared[hankel];
        thrust::complex<double> reflected(0.0, 0.0);
        thrust::complex<double> current = thrust::sqrt(
            thrust::complex<double>(lambda2, 0.0)
            + imaginary * omega * conductivity[layerCount - 1]);
        for (int layer = layerCount - 1; layer >= 0; --layer) {
            const auto above = layer == 0
                ? thrust::complex<double>(lambda, 0.0)
                : thrust::sqrt(thrust::complex<double>(lambda2, 0.0)
                    + imaginary * omega * conductivity[layer - 1]);
            const auto psi = (above - current) / (above + current);
            const auto decay = layer < layerCount - 1
                ? thrust::exp(-2.0 * current * thicknesses[layer])
                : thrust::complex<double>(0.0, 0.0);
            reflected = (psi + reflected * decay)
                      / (thrust::complex<double>(1.0, 0.0)
                         + psi * reflected * decay);
            current = above;
        }
        field += reflected * hankelFactors[hankel];
    }
    field *= 0.5;

    thrust::complex<double> transfer(1.0, 0.0);
    for (int stage = 0; stage < lowPassCount; ++stage)
        transfer *= filterStage(omega, lowPass[stage], lowPassOrders[stage], false);
    for (int stage = 0; stage < highPassCount; ++stage)
        transfer *= filterStage(omega, highPass[stage], highPassOrders[stage], true);
    contributions[flat] = mu0 * (field * transfer).imag()
                        * cFourierSin[frequencyIndex];
}

__global__ void sumFrequencies(const double *times, int timeCount,
                               const double *contributions, double *response)
{
    const int timeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (timeIndex >= timeCount)
        return;
    double sum = 0.0;
    for (int frequency = 0; frequency < fourierCount; ++frequency)
        sum += contributions[timeIndex * fourierCount + frequency];
    response[timeIndex] = -sum / times[timeIndex] * 2.0 / pi;
}

template<typename T>
bool copyToDevice(DeviceBuffer<T> &buffer, const std::vector<T> &values,
                  std::string *error)
{
    if (!buffer.valid()) {
        if (error) *error = cudaMessage(buffer.status());
        return false;
    }
    if (values.empty())
        return true;
    const cudaError_t status = cudaMemcpy(buffer.get(), values.data(),
        values.size() * sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        if (error) *error = cudaMessage(status);
        return false;
    }
    return true;
}

} // namespace

bool cudaForwardAvailable(std::string *reason)
{
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count < 1) {
        if (reason)
            *reason = status == cudaSuccess ? "No CUDA-capable GPU detected"
                                            : cudaMessage(status);
        return false;
    }
    std::call_once(weightsOnce, initializeWeights);
    if (!weightsReady && reason)
        *reason = weightsError;
    return weightsReady;
}

bool cudaStepResponse(const ForwardModel &model,
                      const std::vector<double> &evaluationTimes,
                      std::vector<double> &response, std::string *error)
{
    if (!cudaForwardAvailable(error))
        return false;
    if (model.transform != TransformMethod::DigitalLinearFilter) {
        if (error) *error = "CUDA currently accelerates the DLF transform only";
        return false;
    }
    if (model.resistivities.empty()
        || model.resistivities.size() > maximumLayers
        || model.thicknesses.size() + 1 != model.resistivities.size()) {
        if (error) *error = "CUDA forward model supports 1 to 128 valid layers";
        return false;
    }

    std::vector<int> lowOrders(model.lowPassFrequencies.size(), 1);
    std::vector<int> highOrders(model.highPassFrequencies.size(), 1);
    for (std::size_t i = 0; i < model.lowPassOrders.size()
         && i < lowOrders.size(); ++i)
        lowOrders[i] = model.lowPassOrders[i];
    for (std::size_t i = 0; i < model.highPassOrders.size()
         && i < highOrders.size(); ++i)
        highOrders[i] = model.highPassOrders[i];

    std::vector<double> conductivity(model.resistivities.size());
    for (std::size_t layer = 0; layer < model.resistivities.size(); ++layer)
        conductivity[layer] = mu0 / model.resistivities[layer];
    std::vector<double> lambdas(hankelCount);
    std::vector<double> lambdaSquared(hankelCount);
    std::vector<double> hankelFactors(hankelCount);
    for (int hankel = 0; hankel < hankelCount; ++hankel) {
        const double lambda = weights::hankel_base_101[static_cast<std::size_t>(hankel)]
                            / model.txSize;
        const double geometryFactor = model.geometry == Geometry::CircleOffset
            ? std::cyl_bessel_j(0.0, lambda * model.rxX) : 1.0;
        lambdas[static_cast<std::size_t>(hankel)] = lambda;
        lambdaSquared[static_cast<std::size_t>(hankel)] = lambda * lambda;
        hankelFactors[static_cast<std::size_t>(hankel)] = lambda
            * weights::hankel_j1_101[static_cast<std::size_t>(hankel)]
            * geometryFactor;
    }

    DeviceBuffer<double> dTimes(evaluationTimes.size());
    DeviceBuffer<double> dThicknesses(model.thicknesses.size());
    DeviceBuffer<double> dConductivity(conductivity.size());
    DeviceBuffer<double> dLambdas(lambdas.size());
    DeviceBuffer<double> dLambdaSquared(lambdaSquared.size());
    DeviceBuffer<double> dHankelFactors(hankelFactors.size());
    DeviceBuffer<double> dLowPass(model.lowPassFrequencies.size());
    DeviceBuffer<int> dLowOrders(lowOrders.size());
    DeviceBuffer<double> dHighPass(model.highPassFrequencies.size());
    DeviceBuffer<int> dHighOrders(highOrders.size());
    DeviceBuffer<double> dContributions(evaluationTimes.size() * fourierCount);
    DeviceBuffer<double> dResponse(evaluationTimes.size());
    if (!copyToDevice(dTimes, evaluationTimes, error)
        || !copyToDevice(dThicknesses, model.thicknesses, error)
        || !copyToDevice(dConductivity, conductivity, error)
        || !copyToDevice(dLambdas, lambdas, error)
        || !copyToDevice(dLambdaSquared, lambdaSquared, error)
        || !copyToDevice(dHankelFactors, hankelFactors, error)
        || !copyToDevice(dLowPass, model.lowPassFrequencies, error)
        || !copyToDevice(dLowOrders, lowOrders, error)
        || !copyToDevice(dHighPass, model.highPassFrequencies, error)
        || !copyToDevice(dHighOrders, highOrders, error)
        || !dContributions.valid() || !dResponse.valid()) {
        if (error && error->empty())
            *error = "CUDA device-memory allocation failed";
        return false;
    }

    constexpr int threads = 128;
    const int contributionCount = static_cast<int>(evaluationTimes.size())
                                * fourierCount;
    frequencyContributions<<<(contributionCount + threads - 1) / threads, threads>>>(
        dTimes.get(), static_cast<int>(evaluationTimes.size()),
        dThicknesses.get(), dConductivity.get(),
        static_cast<int>(model.resistivities.size()), dLambdas.get(),
        dLambdaSquared.get(), dHankelFactors.get(),
        dLowPass.get(), dLowOrders.get(), static_cast<int>(lowOrders.size()),
        dHighPass.get(), dHighOrders.get(), static_cast<int>(highOrders.size()),
        dContributions.get());
    cudaError_t status = cudaGetLastError();
    if (status == cudaSuccess) {
        sumFrequencies<<<(static_cast<int>(evaluationTimes.size()) + threads - 1)
                         / threads, threads>>>(
            dTimes.get(), static_cast<int>(evaluationTimes.size()),
            dContributions.get(), dResponse.get());
        status = cudaGetLastError();
    }
    if (status == cudaSuccess)
        status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        if (error) *error = cudaMessage(status);
        return false;
    }

    response.resize(evaluationTimes.size());
    status = cudaMemcpy(response.data(), dResponse.get(),
                        response.size() * sizeof(double),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        if (error) *error = cudaMessage(status);
        response.clear();
        return false;
    }
    return true;
}

} // namespace pytem
