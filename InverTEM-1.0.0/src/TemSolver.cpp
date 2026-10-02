#include "TemSolver.h"
#include "CudaForward.h"
#include "JacobianCache.h"
#include "TransformWeights.h"
#include "VectorKernel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_map>

#ifdef INVERTEM_USE_OPENMP
#include <omp.h>
#endif
#if defined(INVERTEM_VECTOR_KERNEL) && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace pytem {
namespace {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double mu0 = 4.0e-7 * pi;
// Values that display as 1.00 at two decimal places count as converged.
constexpr double rmsDisplayConvergenceThreshold = 1.005;
constexpr double minimumAlpha = 1.0e-30;
constexpr double maximumAlpha = 1.0e30;
using Complex = std::complex<double>;
using Matrix = std::vector<std::vector<double>>;

double boundedAlpha(double alpha)
{
    if (!std::isfinite(alpha))
        return maximumAlpha;
    return std::clamp(alpha, minimumAlpha, maximumAlpha);
}

double strongerAlphaFactor(double alphaLogStep)
{
    // A half-decade jump recovers from an over-aggressive update within the
    // small trial budget without making the ordinary improving sweep coarse.
    return std::pow(10.0, std::max(0.5, alphaLogStep));
}

double weakerAlphaFactor(double alphaLogStep)
{
    return std::pow(10.0, std::max(0.01, alphaLogStep));
}

bool applyBroydenRankOneUpdate(
    Matrix &jacobian, const std::vector<double> &oldModel,
    const std::vector<double> &newModel,
    const std::vector<double> &oldPrediction,
    const std::vector<double> &newPrediction)
{
    if (oldModel.size() != newModel.size()
        || oldPrediction.size() != newPrediction.size()
        || jacobian.size() != oldPrediction.size()
        || oldModel.empty() || oldPrediction.empty())
        return false;
    for (const auto &row : jacobian)
        if (row.size() != oldModel.size())
            return false;

    std::vector<double> step(oldModel.size());
    double denominator = 0.0;
    for (std::size_t parameter = 0; parameter < step.size(); ++parameter) {
        step[parameter] = newModel[parameter] - oldModel[parameter];
        denominator += step[parameter] * step[parameter];
    }
    if (!(std::isfinite(denominator) && denominator > 1.0e-16))
        return false;

    std::vector<double> secantResidual(oldPrediction.size());
    double residualNormSquared = 0.0;
    double jacobianNormSquared = 0.0;
    for (std::size_t row = 0; row < jacobian.size(); ++row) {
        if (!(oldPrediction[row] > 0.0 && newPrediction[row] > 0.0
              && std::isfinite(oldPrediction[row])
              && std::isfinite(newPrediction[row])))
            return false;
        double projected = 0.0;
        for (std::size_t parameter = 0;
             parameter < step.size(); ++parameter) {
            projected += jacobian[row][parameter] * step[parameter];
            jacobianNormSquared += jacobian[row][parameter]
                * jacobian[row][parameter];
        }
        const double observedChange = std::log(newPrediction[row])
            - std::log(oldPrediction[row]);
        secantResidual[row] = observedChange - projected;
        if (!std::isfinite(secantResidual[row]))
            return false;
        residualNormSquared += secantResidual[row] * secantResidual[row];
    }

    const double correctionNorm = std::sqrt(residualNormSquared / denominator);
    const double jacobianNorm = std::sqrt(jacobianNormSquared);
    if (!std::isfinite(correctionNorm)
        || correctionNorm > 10.0 * std::max(jacobianNorm, 1.0e-12))
        return false;

    for (std::size_t row = 0; row < jacobian.size(); ++row) {
        const double factor = secantResidual[row] / denominator;
        for (std::size_t parameter = 0;
             parameter < step.size(); ++parameter) {
            jacobian[row][parameter] += factor * step[parameter];
            if (!std::isfinite(jacobian[row][parameter]))
                return false;
        }
    }
    return true;
}

bool wasCancelled(const CancelCallback &callback)
{
    return callback && callback();
}

std::string fixedTwoDecimals(double value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

Complex reflectionCoefficient(double lambda, double lambdaSquared,
                              const std::vector<Complex> &layerTerms,
                              const std::vector<double> &thicknesses,
                              std::vector<Complex> &gammaLayer)
{
    const std::size_t count = layerTerms.size();
    for (std::size_t layer = 0; layer < count; ++layer)
        gammaLayer[layer] = std::sqrt(lambdaSquared + layerTerms[layer]);

    Complex reflected = 0.0;
    for (std::size_t reverse = count; reverse-- > 0;) {
        const Complex above = reverse == 0 ? Complex(lambda, 0.0) : gammaLayer[reverse - 1];
        const Complex current = gammaLayer[reverse];
        const Complex psi = (above - current) / (above + current);
        const Complex decay = reverse < count - 1
            ? std::exp(-2.0 * current * thicknesses[reverse]) : Complex(0.0, 0.0);
        reflected = (psi + reflected * decay) / (1.0 + psi * reflected * decay);
    }
    return reflected;
}

Complex circleFrequencyKernel(const std::vector<Complex> &layerTerms,
                              const ForwardModel &model,
                              const std::vector<double> &lambdas,
                              const std::vector<double> &lambdaSquared,
                              const std::vector<double> &hankelFactors,
                              std::vector<Complex> &gammaLayer)
{
    Complex field = 0.0;
    for (std::size_t index = 0; index < lambdas.size(); ++index)
        field += reflectionCoefficient(lambdas[index], lambdaSquared[index],
                                       layerTerms, model.thicknesses, gammaLayer)
               * hankelFactors[index];
    return 0.5 * field;
}

Complex filterStage(Complex omega, double frequency, int order, bool highPass)
{
    const Complex s = Complex(0.0, 1.0) * omega;
    const double cutoff = 2.0 * pi * frequency;
    if (order == 2) {
        const Complex denominator = s * s + std::sqrt(2.0) * cutoff * s + cutoff * cutoff;
        return highPass ? s * s / denominator : cutoff * cutoff / denominator;
    }
    return highPass ? s / (s + cutoff) : cutoff / (s + cutoff);
}

Complex systemTransfer(Complex omega, const ForwardModel &model)
{
    Complex result = 1.0;
    for (std::size_t i = 0; i < model.lowPassFrequencies.size(); ++i) {
        const int order = i < model.lowPassOrders.size() ? model.lowPassOrders[i] : 1;
        result *= filterStage(omega, model.lowPassFrequencies[i], order, false);
    }
    for (std::size_t i = 0; i < model.highPassFrequencies.size(); ++i) {
        const int order = i < model.highPassOrders.size() ? model.highPassOrders[i] : 1;
        result *= filterStage(omega, model.highPassFrequencies[i], order, true);
    }
    return result;
}

std::vector<double> eulerWeights(int order)
{
    if (order < 1 || order > 30)
        throw std::invalid_argument("Euler order must be between 1 and 30");
    std::vector<double> eta(static_cast<std::size_t>(2 * order + 1), 1.0);
    eta.front() = 0.5;
    double cumulative = 0.0;
    double binomial = 1.0;
    const double denominator = std::ldexp(1.0, order);
    for (int j = 0; j < order; ++j) {
        if (j > 0)
            binomial *= static_cast<double>(order - j + 1) / static_cast<double>(j);
        cumulative += binomial / denominator;
        eta[static_cast<std::size_t>(order + 1 + j)] = 1.0 - cumulative;
    }
    return eta;
}

template<typename Function>
void parallelFor(std::size_t count, unsigned preferredThreads,
                 const Function &function, bool forceParallel = false)
{
    const unsigned available = preferredThreads > 0
        ? preferredThreads : std::max(1u, std::thread::hardware_concurrency());
    const unsigned threadCount = std::min<unsigned>(available, static_cast<unsigned>(count));
    if (threadCount <= 1 || (count < 24 && !forceParallel)) {
        for (std::size_t index = 0; index < count; ++index)
            function(index);
        return;
    }
#ifdef INVERTEM_USE_OPENMP
    std::exception_ptr failure;
    std::mutex failureMutex;
    std::atomic_bool failed{false};
#pragma omp parallel for schedule(dynamic) num_threads(threadCount)
    for (std::ptrdiff_t rawIndex = 0;
         rawIndex < static_cast<std::ptrdiff_t>(count); ++rawIndex) {
        if (failed.load(std::memory_order_relaxed))
            continue;
        try {
            function(static_cast<std::size_t>(rawIndex));
        } catch (...) {
            std::lock_guard<std::mutex> lock(failureMutex);
            if (!failure)
                failure = std::current_exception();
            failed.store(true, std::memory_order_relaxed);
        }
    }
    if (failure)
        std::rethrow_exception(failure);
#else
    std::atomic_size_t next{0};
    std::exception_ptr failure;
    std::mutex failureMutex;
    std::vector<std::thread> workers;
    workers.reserve(threadCount);
    for (unsigned worker = 0; worker < threadCount; ++worker) {
        workers.emplace_back([&]() {
            try {
                while (true) {
                    const std::size_t index = next.fetch_add(1);
                    if (index >= count)
                        break;
                    function(index);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(failureMutex);
                if (!failure)
                    failure = std::current_exception();
                next.store(count);
            }
        });
    }
    for (auto &worker : workers)
        worker.join();
    if (failure)
        std::rethrow_exception(failure);
#endif
}

std::vector<double> stepResponse(const ForwardModel &model,
                                 const std::vector<double> &evaluationTimes,
                                 const CancelCallback &cancelled)
{
    if (model.useGpu && model.transform == TransformMethod::DigitalLinearFilter && model.altitude == 0.0) {
        std::vector<double> gpuResponse;
        if (cudaStepResponse(model, evaluationTimes, gpuResponse))
            return gpuResponse;
    }
    const std::size_t hankelCount = weights::hankel_base_101.size();
    std::vector<double> lambdas(hankelCount);
    std::vector<double> lambdaSquared(hankelCount);
    std::vector<double> hankelFactors(hankelCount);
    for (std::size_t index = 0; index < hankelCount; ++index) {
        const double lambda = weights::hankel_base_101[index] / model.txSize;
        const double geometryFactor = (model.geometry == Geometry::CircleOffset
            ? std::cyl_bessel_j(0.0, lambda * model.rxX) : 1.0) * std::exp(-lambda * model.altitude);
        lambdas[index] = lambda;
        lambdaSquared[index] = lambda * lambda;
        hankelFactors[index] = lambda * weights::hankel_j1_101[index]
                             * geometryFactor;
    }

    std::vector<double> conductivity(model.resistivities.size());
    for (std::size_t layer = 0; layer < model.resistivities.size(); ++layer)
        conductivity[layer] = mu0 / model.resistivities[layer];

    const std::vector<double> eta = model.transform == TransformMethod::Euler
        ? eulerWeights(model.eulerOrder) : std::vector<double>{};
    const std::size_t frequencyCount = model.transform == TransformMethod::DigitalLinearFilter
        ? weights::fourier_base_81.size() : eta.size();
    std::vector<Complex> angularFrequencies(evaluationTimes.size() * frequencyCount);
    std::vector<Complex> transfers(evaluationTimes.size() * frequencyCount);
    std::vector<double> responseScales(evaluationTimes.size());
    for (std::size_t gate = 0; gate < evaluationTimes.size(); ++gate) {
        const double time = evaluationTimes[gate];
        if (time <= 0.0)
            throw std::invalid_argument("Gate times must be positive");
        const std::size_t offset = gate * frequencyCount;
        if (model.transform == TransformMethod::DigitalLinearFilter) {
            for (std::size_t frequency = 0; frequency < frequencyCount; ++frequency) {
                const Complex omega(weights::fourier_base_81[frequency] / time, 0.0);
                angularFrequencies[offset + frequency] = omega;
                transfers[offset + frequency] = systemTransfer(omega, model);
            }
            responseScales[gate] = -2.0 / (pi * time);
        } else {
            constexpr double eulerA = 18.4;
            const double realShift = eulerA / (2.0 * time);
            const double spacing = pi / time;
            for (std::size_t frequency = 0; frequency < frequencyCount; ++frequency) {
                const Complex omega(static_cast<double>(frequency) * spacing,
                                    -realShift);
                angularFrequencies[offset + frequency] = omega;
                transfers[offset + frequency] = systemTransfer(omega, model);
            }
            responseScales[gate] = std::exp(eulerA / 2.0) / time;
        }
    }

    std::vector<double> response(evaluationTimes.size());
    auto evaluate = [&](std::size_t gate) {
        if (wasCancelled(cancelled))
            throw std::runtime_error("Inversion cancelled");
        std::vector<Complex> gammaLayer(model.resistivities.size());
        std::vector<Complex> layerTerms(model.resistivities.size());
        const std::size_t offset = gate * frequencyCount;
        if (model.transform == TransformMethod::DigitalLinearFilter) {
            double transformSum = 0.0;
            for (std::size_t frequency = 0; frequency < frequencyCount; ++frequency) {
                const Complex omega = angularFrequencies[offset + frequency];
                const Complex s = Complex(0.0, 1.0) * omega;
                for (std::size_t layer = 0; layer < conductivity.size(); ++layer)
                    layerTerms[layer] = s * conductivity[layer];
                const Complex field = circleFrequencyKernel(
                    layerTerms, model, lambdas, lambdaSquared, hankelFactors,
                    gammaLayer) * transfers[offset + frequency];
                transformSum += mu0 * field.imag() * weights::fourier_sin_81[frequency];
            }
            response[gate] = responseScales[gate] * transformSum;
        } else {
            double transformSum = 0.0;
            for (std::size_t frequency = 0; frequency < frequencyCount; ++frequency) {
                const Complex omega = angularFrequencies[offset + frequency];
                const Complex s = Complex(0.0, 1.0) * omega;
                for (std::size_t layer = 0; layer < conductivity.size(); ++layer)
                    layerTerms[layer] = s * conductivity[layer];
                const Complex field = circleFrequencyKernel(
                    layerTerms, model, lambdas, lambdaSquared, hankelFactors,
                    gammaLayer) * transfers[offset + frequency];
                const double sign = (frequency & 1U) ? -1.0 : 1.0;
                transformSum += eta[frequency] * sign * (mu0 * field).real();
            }
            response[gate] = responseScales[gate] * transformSum;
        }
    };
    parallelFor(evaluationTimes.size(), model.maxThreads, evaluate);
    return response;
}

struct ReflectionSensitivity {
    Complex reflection;
    std::vector<Complex> derivative;
};

ReflectionSensitivity reflectionSensitivity(double lambda, Complex omega,
                                             const ForwardModel &model)
{
    const std::size_t count = model.resistivities.size();
    std::vector<Complex> gamma(count);
    std::vector<Complex> gammaDerivative(count);
    std::vector<Complex> psi(count);
    std::vector<Complex> decay(count);
    std::vector<Complex> below(count);
    const Complex s = Complex(0.0, 1.0) * omega;
    for (std::size_t layer = 0; layer < count; ++layer) {
        const Complex conductivityTerm = s * mu0 / model.resistivities[layer];
        gamma[layer] = std::sqrt(lambda * lambda + conductivityTerm);
        gammaDerivative[layer] = -conductivityTerm / (2.0 * gamma[layer]);
    }

    Complex reflected = 0.0;
    for (std::size_t reverse = count; reverse-- > 0;) {
        const Complex above = reverse == 0 ? Complex(lambda, 0.0) : gamma[reverse - 1];
        psi[reverse] = (above - gamma[reverse]) / (above + gamma[reverse]);
        decay[reverse] = reverse < count - 1
            ? std::exp(-2.0 * gamma[reverse] * model.thicknesses[reverse])
            : Complex(0.0, 0.0);
        below[reverse] = reflected;
        reflected = (psi[reverse] + reflected * decay[reverse])
                  / (1.0 + psi[reverse] * reflected * decay[reverse]);
    }

    std::vector<Complex> derivative(count, 0.0);
    Complex adjoint = 1.0;
    for (std::size_t layer = 0; layer < count; ++layer) {
        const Complex above = layer == 0 ? Complex(lambda, 0.0) : gamma[layer - 1];
        const Complex denominator = 1.0 + psi[layer] * below[layer] * decay[layer];
        const Complex denominatorSquared = denominator * denominator;
        const Complex belowDecay = below[layer] * decay[layer];
        const Complex dReflectionDPsi = (1.0 - belowDecay * belowDecay) / denominatorSquared;
        const Complex dReflectionDDecay = below[layer] * (1.0 - psi[layer] * psi[layer])
                                        / denominatorSquared;
        const Complex dReflectionDBelow = decay[layer] * (1.0 - psi[layer] * psi[layer])
                                        / denominatorSquared;
        const Complex gammaSum = above + gamma[layer];
        const Complex dPsiDCurrent = -2.0 * above / (gammaSum * gammaSum);
        const Complex dDecayDCurrent = layer < count - 1
            ? -2.0 * model.thicknesses[layer] * decay[layer] : Complex(0.0, 0.0);
        derivative[layer] += adjoint
            * (dReflectionDPsi * dPsiDCurrent + dReflectionDDecay * dDecayDCurrent)
            * gammaDerivative[layer];
        if (layer > 0) {
            const Complex dPsiDAbove = 2.0 * gamma[layer] / (gammaSum * gammaSum);
            derivative[layer - 1] += adjoint * dReflectionDPsi * dPsiDAbove
                                   * gammaDerivative[layer - 1];
        }
        adjoint *= dReflectionDBelow;
    }
    return {reflected, std::move(derivative)};
}

struct FrequencySensitivity {
    Complex field;
    std::vector<Complex> derivative;
};

FrequencySensitivity circleFrequencySensitivity(Complex omega, const ForwardModel &model)
{
    FrequencySensitivity result{0.0, std::vector<Complex>(model.resistivities.size(), 0.0)};
    for (std::size_t index = 0; index < weights::hankel_base_101.size(); ++index) {
        const double lambda = weights::hankel_base_101[index] / model.txSize;
        double extra = 1.0;
        if (model.geometry == Geometry::CircleOffset)
            extra = std::cyl_bessel_j(0.0, lambda * model.rxX);
        extra *= std::exp(-lambda * model.altitude);
        const double factor = lambda * weights::hankel_j1_101[index] * extra;
        const auto sensitivity = reflectionSensitivity(lambda, omega, model);
        result.field += sensitivity.reflection * factor;
        for (std::size_t layer = 0; layer < result.derivative.size(); ++layer)
            result.derivative[layer] += sensitivity.derivative[layer] * factor;
    }
    result.field *= 0.5;
    for (Complex &value : result.derivative)
        value *= 0.5;
    return result;
}

struct ModelSensitivity {
    std::vector<double> response;
    Matrix absoluteJacobian;
};

ModelSensitivity stepSensitivity(const ForwardModel &model,
                                 const std::vector<double> &evaluationTimes,
                                 const CancelCallback &cancelled)
{
    ModelSensitivity result{
        std::vector<double>(evaluationTimes.size(), 0.0),
        Matrix(evaluationTimes.size(), std::vector<double>(model.resistivities.size(), 0.0))};
    auto evaluate = [&](std::size_t gate) {
        if (wasCancelled(cancelled))
            throw std::runtime_error("Inversion cancelled");
        const double time = evaluationTimes[gate];
        if (time <= 0.0)
            throw std::invalid_argument("Gate times must be positive");
        if (model.transform == TransformMethod::DigitalLinearFilter) {
            double fieldSum = 0.0;
            std::vector<double> derivativeSum(model.resistivities.size(), 0.0);
            for (std::size_t frequency = 0; frequency < weights::fourier_base_81.size(); ++frequency) {
                const double omega = weights::fourier_base_81[frequency] / time;
                auto sensitivity = circleFrequencySensitivity(omega, model);
                const Complex transfer = systemTransfer(omega, model);
                const double weight = weights::fourier_sin_81[frequency];
                fieldSum += (mu0 * sensitivity.field * transfer).imag() * weight;
                for (std::size_t layer = 0; layer < derivativeSum.size(); ++layer)
                    derivativeSum[layer] += (mu0 * sensitivity.derivative[layer] * transfer).imag() * weight;
            }
            const double factor = -2.0 / (pi * time);
            result.response[gate] = factor * fieldSum;
            for (std::size_t layer = 0; layer < derivativeSum.size(); ++layer)
                result.absoluteJacobian[gate][layer] = factor * derivativeSum[layer];
        } else {
            constexpr double eulerA = 18.4;
            const auto eta = eulerWeights(model.eulerOrder);
            const double realShift = eulerA / (2.0 * time);
            const double spacing = pi / time;
            double fieldSum = 0.0;
            std::vector<double> derivativeSum(model.resistivities.size(), 0.0);
            for (std::size_t k = 0; k < eta.size(); ++k) {
                const Complex omega(static_cast<double>(k) * spacing, -realShift);
                auto sensitivity = circleFrequencySensitivity(omega, model);
                const Complex transfer = systemTransfer(omega, model);
                const double weight = eta[k] * ((k & 1U) ? -1.0 : 1.0);
                fieldSum += (mu0 * sensitivity.field * transfer).real() * weight;
                for (std::size_t layer = 0; layer < derivativeSum.size(); ++layer)
                    derivativeSum[layer] += (mu0 * sensitivity.derivative[layer] * transfer).real() * weight;
            }
            const double factor = std::exp(eulerA / 2.0) / time;
            result.response[gate] = factor * fieldSum;
            for (std::size_t layer = 0; layer < derivativeSum.size(); ++layer)
                result.absoluteJacobian[gate][layer] = factor * derivativeSum[layer];
        }
    };
    parallelFor(evaluationTimes.size(), model.maxThreads, evaluate);
    return result;
}

ModelSensitivity modelSensitivity(const ForwardModel &model,
                                  const CancelCallback &cancelled)
{
    if (model.responseMatrix.empty())
        return stepSensitivity(model, model.times, cancelled);
    if (model.responseMatrix.size() != model.times.size() || model.stepTimes.empty())
        throw std::invalid_argument("Invalid waveform/gate response matrix dimensions");
    const auto step = stepSensitivity(model, model.stepTimes, cancelled);
    ModelSensitivity gated{
        std::vector<double>(model.responseMatrix.size(), 0.0),
        Matrix(model.responseMatrix.size(), std::vector<double>(model.resistivities.size(), 0.0))};
    for (std::size_t row = 0; row < model.responseMatrix.size(); ++row) {
        if (model.responseMatrix[row].size() != step.response.size())
            throw std::invalid_argument("Response matrix column count does not match step-time grid");
        for (std::size_t column = 0; column < step.response.size(); ++column) {
            const double matrixValue = model.responseMatrix[row][column];
            gated.response[row] += matrixValue * step.response[column];
            for (std::size_t layer = 0; layer < model.resistivities.size(); ++layer)
                gated.absoluteJacobian[row][layer] += matrixValue * step.absoluteJacobian[column][layer];
        }
    }
    return gated;
}

double rms(const std::vector<double> &observed, const std::vector<double> &predicted,
           const std::vector<double> &weights)
{
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        if (observed[i] > 0.0 && predicted[i] > 0.0 && std::isfinite(predicted[i])) {
            const double residual = weights[i] * (std::log(observed[i]) - std::log(predicted[i]));
            sum += residual * residual;
            ++count;
        }
    }
    return count == 0 ? std::numeric_limits<double>::infinity()
                      : std::sqrt(sum / static_cast<double>(count));
}

Matrix roughness(const std::vector<double> &model, RegularizationNorm norm)
{
    const std::size_t count = model.size();
    Matrix result(count, std::vector<double>(count, 0.0));
    for (std::size_t i = 0; i < count; ++i)
        result[i][i] = 1.0e-4;
    for (std::size_t i = 0; i + 1 < count; ++i) {
        double weight = 1.0;
        if (norm == RegularizationNorm::L1Blocky) {
            constexpr double irlsEpsilon = 0.05;
            const double difference = model[i + 1] - model[i];
            weight = 1.0 / std::sqrt(difference * difference
                                   + irlsEpsilon * irlsEpsilon);
        }
        result[i][i] += weight;
        result[i + 1][i + 1] += weight;
        result[i][i + 1] -= weight;
        result[i + 1][i] -= weight;
    }
    return result;
}

std::vector<double> solveLinear(Matrix matrix, std::vector<double> rhs)
{
    const std::size_t count = rhs.size();
    // Partial-pivot LU/Gaussian elimination. The earlier prototype used
    // Gauss-Jordan elimination, which unnecessarily eliminated above every
    // pivot before the factorization was complete.
    for (std::size_t pivot = 0; pivot < count; ++pivot) {
        std::size_t best = pivot;
        for (std::size_t row = pivot + 1; row < count; ++row)
            if (std::abs(matrix[row][pivot]) > std::abs(matrix[best][pivot]))
                best = row;
        std::swap(matrix[pivot], matrix[best]);
        std::swap(rhs[pivot], rhs[best]);
        if (std::abs(matrix[pivot][pivot]) < 1.0e-14)
            matrix[pivot][pivot] += matrix[pivot][pivot] < 0.0
                ? -1.0e-10 : 1.0e-10;
        const double divisor = matrix[pivot][pivot];
        for (std::size_t row = pivot + 1; row < count; ++row) {
            const double factor = matrix[row][pivot] / divisor;
            matrix[row][pivot] = 0.0;
            for (std::size_t column = pivot + 1;
                 column < count; ++column)
                matrix[row][column] -= factor * matrix[pivot][column];
            rhs[row] -= factor * rhs[pivot];
        }
    }
    std::vector<double> solution(count, 0.0);
    for (std::size_t reverse = count; reverse-- > 0;) {
        double value = rhs[reverse];
        for (std::size_t column = reverse + 1; column < count; ++column)
            value -= matrix[reverse][column] * solution[column];
        solution[reverse] = value / matrix[reverse][reverse];
    }
    return solution;
}

double fittedTargetAlpha(const std::vector<double> &alphas,
                         const std::vector<double> &rmsValues)
{
    if (alphas.size() != rmsValues.size() || alphas.size() < 2
        || *std::min_element(rmsValues.begin(), rmsValues.end()) >= 1.0)
        return std::numeric_limits<double>::quiet_NaN();

    std::vector<double> x(alphas.size());
    std::transform(alphas.begin(), alphas.end(), x.begin(),
                   [](double alpha) { return std::log10(alpha); });
    std::size_t bracket = x.size();
    for (std::size_t index = 1; index < x.size(); ++index) {
        if ((rmsValues[index - 1] - 1.0)
                * (rmsValues[index] - 1.0) <= 0.0) {
            bracket = index;
            break;
        }
    }
    if (bracket == x.size())
        return std::numeric_limits<double>::quiet_NaN();
    const double xLow = std::min(x[bracket - 1], x[bracket]);
    const double xHigh = std::max(x[bracket - 1], x[bracket]);
    std::vector<double> roots;

    if (x.size() == 2) {
        const double slope = (rmsValues[1] - rmsValues[0]) / (x[1] - x[0]);
        if (std::abs(slope) > 1.0e-14)
            roots.push_back(x[0] + (1.0 - rmsValues[0]) / slope);
    } else {
        double sx = 0.0, sx2 = 0.0, sx3 = 0.0, sx4 = 0.0;
        double sy = 0.0, sxy = 0.0, sx2y = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double x2 = x[i] * x[i];
            sx += x[i];
            sx2 += x2;
            sx3 += x2 * x[i];
            sx4 += x2 * x2;
            sy += rmsValues[i];
            sxy += x[i] * rmsValues[i];
            sx2y += x2 * rmsValues[i];
        }
        const auto coefficients = solveLinear(
            {{sx4, sx3, sx2},
             {sx3, sx2, sx},
             {sx2, sx, static_cast<double>(x.size())}},
            {sx2y, sxy, sy});
        const double a = coefficients[0];
        const double b = coefficients[1];
        const double c = coefficients[2] - 1.0;
        if (std::abs(a) < 1.0e-14) {
            if (std::abs(b) > 1.0e-14)
                roots.push_back(-c / b);
        } else {
            const double discriminant = b * b - 4.0 * a * c;
            if (discriminant >= 0.0) {
                const double root = std::sqrt(discriminant);
                roots.push_back((-b - root) / (2.0 * a));
                roots.push_back((-b + root) / (2.0 * a));
            }
        }
    }

    double selected = -std::numeric_limits<double>::infinity();
    for (double root : roots) {
        if (std::isfinite(root) && root >= xLow && root <= xHigh)
            selected = std::max(selected, root);
    }
    if (std::isfinite(selected))
        return std::pow(10.0, selected);

    // A local quadratic can occasionally have no real root inside the
    // measured bracket. Fall back to interpolation in log(alpha), still
    // requiring no additional bracketing forward models.
    const double y0 = rmsValues[bracket - 1];
    const double y1 = rmsValues[bracket];
    if (std::abs(y1 - y0) <= 1.0e-14)
        return std::numeric_limits<double>::quiet_NaN();
    const double fraction = std::clamp((1.0 - y0) / (y1 - y0), 0.0, 1.0);
    return std::pow(10.0,
                    x[bracket - 1] + fraction * (x[bracket] - x[bracket - 1]));
}

std::vector<double> alphaFactors(double alpha, const std::vector<double> &thicknesses,
                                 std::size_t parameterCount)
{
    if (parameterCount == 1)
        return {alpha};
    std::vector<double> midpoints;
    midpoints.reserve(parameterCount);
    double top = 0.0;
    for (double thickness : thicknesses) {
        midpoints.push_back(top + 0.5 * thickness);
        top += thickness;
    }
    const double lastThickness = thicknesses.empty() ? 1.0 : thicknesses.back();
    midpoints.push_back(top + 0.5 * lastThickness);
    std::vector<double> spacing(parameterCount - 1);
    for (std::size_t i = 0; i + 1 < parameterCount; ++i)
        spacing[i] = std::max(1.0e-12, midpoints[i + 1] - midpoints[i]);
    std::vector<double> factors(parameterCount);
    factors.front() = alpha / spacing.front();
    factors.back() = alpha / spacing.back();
    for (std::size_t i = 1; i + 1 < parameterCount; ++i)
        factors[i] = alpha * (1.0 / spacing[i - 1] + 1.0 / spacing[i]);
    return factors;
}

std::vector<double> gaussNewtonStep(const Matrix &jacobian,
                                    const std::vector<double> &weightedResidual,
                                    const std::vector<double> &weights,
                                    const Matrix &regularization,
                                    const std::vector<double> &model,
                                    double alpha,
                                    const std::vector<double> &thicknesses)
{
    const std::size_t dataCount = jacobian.size();
    const std::size_t parameterCount = model.size();
    Matrix lhs(parameterCount, std::vector<double>(parameterCount, 0.0));
    std::vector<double> rhs(parameterCount, 0.0);
    const auto alphaVector = alphaFactors(alpha, thicknesses, parameterCount);

    for (std::size_t row = 0; row < dataCount; ++row) {
        for (std::size_t i = 0; i < parameterCount; ++i) {
            const double ji = jacobian[row][i] * weights[row];
            rhs[i] += ji * weightedResidual[row];
            for (std::size_t j = 0; j < parameterCount; ++j)
                lhs[i][j] += ji * jacobian[row][j] * weights[row];
        }
    }

    for (std::size_t i = 0; i < parameterCount; ++i) {
        for (std::size_t j = 0; j < parameterCount; ++j) {
            const double ar = alphaVector[i] * regularization[i][j];
            lhs[i][j] += ar;
            rhs[i] -= ar * model[j];
        }
    }
    return solveLinear(std::move(lhs), std::move(rhs));
}

std::vector<double> forwardLogModel(const InversionOptions &options,
                                    const std::vector<double> &logModel,
                                    const CancelCallback &cancelled)
{
    ForwardModel model = options.model;
    model.resistivities.resize(logModel.size());
    std::transform(logModel.begin(), logModel.end(), model.resistivities.begin(),
                   [](double value) { return std::exp(value); });
    std::vector<double> result = TemSolver::forward(model, cancelled);
    for (const auto &dataSet : options.additionalDataSets) {
        ForwardModel additional = dataSet.model;
        additional.resistivities = model.resistivities;
        const auto response = TemSolver::forward(additional, cancelled);
        result.insert(result.end(), response.begin(), response.end());
    }
    return result;
}

Matrix finiteDifferenceJacobian(const InversionOptions &options,
                                const std::vector<double> &logModel,
                                const std::vector<double> &base,
                                const CancelCallback &cancelled)
{
    Matrix jacobian(base.size(), std::vector<double>(logModel.size(), 0.0));
    for (std::size_t parameter = 0; parameter < logModel.size(); ++parameter) {
        if (wasCancelled(cancelled))
            throw std::runtime_error("Inversion cancelled");
        std::vector<double> perturbed = logModel;
        const double step = options.jacobianStep * std::max(1.0, std::abs(logModel[parameter]));
        perturbed[parameter] += step;
        const auto response = forwardLogModel(options, perturbed, cancelled);
        for (std::size_t row = 0; row < base.size(); ++row) {
            if (base[row] > 0.0 && response[row] > 0.0)
                jacobian[row][parameter] = (std::log(response[row]) - std::log(base[row])) / step;
        }
    }
    return jacobian;
}

Matrix analyticalJacobian(const InversionOptions &options,
                          const std::vector<double> &logModel,
                          const std::vector<double> &base,
                          const CancelCallback &cancelled)
{
    ForwardModel model = options.model;
    model.resistivities.resize(logModel.size());
    std::transform(logModel.begin(), logModel.end(), model.resistivities.begin(),
                   [](double value) { return std::exp(value); });
    std::vector<ModelSensitivity> sensitivities;
    sensitivities.push_back(modelSensitivity(model, cancelled));
    for (const auto &dataSet : options.additionalDataSets) {
        ForwardModel additional = dataSet.model;
        additional.resistivities = model.resistivities;
        sensitivities.push_back(modelSensitivity(additional, cancelled));
    }
    Matrix jacobian(base.size(), std::vector<double>(logModel.size(), 0.0));
    std::size_t outputRow = 0;
    for (const auto &sensitivity : sensitivities) {
        for (std::size_t row = 0; row < sensitivity.response.size(); ++row, ++outputRow) {
            if (base[outputRow] <= 0.0)
                continue;
            for (std::size_t parameter = 0; parameter < logModel.size(); ++parameter)
                jacobian[outputRow][parameter] = sensitivity.absoluteJacobian[row][parameter] / base[outputRow];
        }
    }
    return jacobian;
}

template<typename Value>
void appendSignatureValue(std::string &signature, const Value &value)
{
    static_assert(std::is_trivially_copyable<Value>::value,
                  "Cache signature values must be trivially copyable");
    signature.append(reinterpret_cast<const char *>(&value), sizeof(Value));
}

template<typename Value>
void appendSignatureVector(std::string &signature,
                           const std::vector<Value> &values)
{
    const std::uint64_t size = values.size();
    appendSignatureValue(signature, size);
    if (!values.empty())
        signature.append(reinterpret_cast<const char *>(values.data()),
                         values.size() * sizeof(Value));
}

void appendForwardSignature(std::string &signature, const ForwardModel &model,
                            const std::vector<double> &resistivities)
{
    const std::uint32_t geometry = static_cast<std::uint32_t>(model.geometry);
    const std::uint32_t transform = static_cast<std::uint32_t>(model.transform);
    appendSignatureValue(signature, geometry);
    appendSignatureValue(signature, transform);
    appendSignatureValue(signature, model.eulerOrder);
    appendSignatureValue(signature, model.txSize);
    appendSignatureValue(signature, model.rxX);
    if (model.altitude != 0.0) // keeps existing on-ground cache keys valid
        appendSignatureValue(signature, model.altitude);
    appendSignatureVector(signature, model.thicknesses);
    appendSignatureVector(signature, resistivities);
    appendSignatureVector(signature, model.times);
    appendSignatureVector(signature, model.stepTimes);
    const std::uint64_t matrixRows = model.responseMatrix.size();
    appendSignatureValue(signature, matrixRows);
    for (const auto &row : model.responseMatrix)
        appendSignatureVector(signature, row);
    appendSignatureVector(signature, model.lowPassFrequencies);
    appendSignatureVector(signature, model.lowPassOrders);
    appendSignatureVector(signature, model.highPassFrequencies);
    appendSignatureVector(signature, model.highPassOrders);
}

// Signature for the expensive part of a forward/sensitivity calculation.
// Gate times and response-matrix rows are deliberately excluded: compatible
// soundings may select different gates from the same physical system.
void appendSharedSystemSignature(std::string &signature,
                                 const ForwardModel &model,
                                 const std::vector<double> &resistivities)
{
    const std::uint32_t geometry = static_cast<std::uint32_t>(model.geometry);
    const std::uint32_t transform = static_cast<std::uint32_t>(model.transform);
    const bool waveformGated = !model.responseMatrix.empty();
    appendSignatureValue(signature, geometry);
    appendSignatureValue(signature, transform);
    appendSignatureValue(signature, model.eulerOrder);
    appendSignatureValue(signature, model.txSize);
    appendSignatureValue(signature, model.rxX);
    appendSignatureValue(signature, waveformGated);
    appendSignatureVector(signature, model.thicknesses);
    appendSignatureVector(signature, resistivities);
    appendSignatureVector(signature, model.stepTimes);
    appendSignatureVector(signature, model.lowPassFrequencies);
    appendSignatureVector(signature, model.lowPassOrders);
    appendSignatureVector(signature, model.highPassFrequencies);
    appendSignatureVector(signature, model.highPassOrders);
}

std::string selectedGateSignature(const ForwardModel &model,
                                  std::size_t row)
{
    std::string signature = "invertem-selected-gate-v1";
    appendSignatureValue(signature, model.times[row]);
    if (!model.responseMatrix.empty())
        appendSignatureVector(signature, model.responseMatrix[row]);
    return signature;
}

struct SharedInitialBatch {
    std::vector<std::vector<double>> responses;
    std::vector<Matrix> jacobians;
    std::vector<bool> available;
    std::vector<bool> jacobianAvailable;
    std::vector<bool> sharedAcrossSoundings;
    std::size_t systemGroups = 0;
};

// Compute the starting response and analytical Jacobian on a union of all
// selected gates for each compatible physical system. This keeps every 1-D
// inversion independent, but avoids repeating the same full waveform grid for
// hundreds of soundings that differ only by gate selection and observations.
SharedInitialBatch sharedInitialAnalyticalBatch(
    const std::vector<InversionOptions> &options,
    const std::vector<std::vector<double>> &logModels,
    const std::vector<bool> &needJacobians,
    unsigned maximumThreads, const CancelCallback &cancelled)
{
    if (options.size() != logModels.size()
        || options.size() != needJacobians.size())
        throw std::invalid_argument(
            "Shared initial option/model counts differ");

    struct Request {
        std::size_t sounding = 0;
        std::size_t outputOffset = 0;
        std::vector<std::size_t> unionRows;
    };
    struct Group {
        ForwardModel model;
        std::unordered_map<std::string, std::size_t> gateRows;
        std::vector<Request> requests;
        bool needsJacobian = false;
    };

    SharedInitialBatch result;
    result.responses.resize(options.size());
    result.jacobians.resize(options.size());
    result.available.assign(options.size(), false);
    result.jacobianAvailable.assign(options.size(), false);
    result.sharedAcrossSoundings.assign(options.size(), false);
    std::vector<Group> groups;
    std::unordered_map<std::string, std::size_t> signatureToGroup;

    for (std::size_t sounding = 0; sounding < options.size(); ++sounding) {
        if (options[sounding].jacobianMethod != JacobianMethod::Analytical)
            continue;
        std::vector<double> resistivities(logModels[sounding].size());
        std::transform(logModels[sounding].begin(), logModels[sounding].end(),
                       resistivities.begin(),
                       [](double value) { return std::exp(value); });
        std::size_t rowCount = options[sounding].model.times.size();
        for (const auto &dataSet : options[sounding].additionalDataSets)
            rowCount += dataSet.model.times.size();
        result.responses[sounding].resize(rowCount);
        if (needJacobians[sounding]) {
            result.jacobians[sounding] = Matrix(
                rowCount,
                std::vector<double>(logModels[sounding].size(), 0.0));
            result.jacobianAvailable[sounding] = true;
        }
        result.available[sounding] = true;

        std::size_t outputOffset = 0;
        auto addDataSet = [&](const ForwardModel &source) {
            std::string signature = "invertem-shared-system-v1";
            appendSharedSystemSignature(signature, source, resistivities);
            const auto inserted = signatureToGroup.emplace(
                signature, groups.size());
            if (inserted.second) {
                Group group;
                group.model = source;
                group.model.resistivities = resistivities;
                group.model.times.clear();
                group.model.responseMatrix.clear();
                groups.push_back(std::move(group));
            }
            Group &group = groups[inserted.first->second];
            group.needsJacobian = group.needsJacobian
                || needJacobians[sounding];
            Request request;
            request.sounding = sounding;
            request.outputOffset = outputOffset;
            request.unionRows.reserve(source.times.size());
            for (std::size_t row = 0; row < source.times.size(); ++row) {
                const std::string gate = selectedGateSignature(source, row);
                const auto gateInserted = group.gateRows.emplace(
                    gate, group.model.times.size());
                if (gateInserted.second) {
                    group.model.times.push_back(source.times[row]);
                    if (!source.responseMatrix.empty())
                        group.model.responseMatrix.push_back(
                            source.responseMatrix[row]);
                }
                request.unionRows.push_back(gateInserted.first->second);
            }
            group.requests.push_back(std::move(request));
            outputOffset += source.times.size();
        };
        addDataSet(options[sounding].model);
        for (const auto &dataSet : options[sounding].additionalDataSets)
            addDataSet(dataSet.model);
    }

    result.systemGroups = groups.size();
    // Usually there are only one or two groups (LM/HM). Run each union grid
    // with all requested cores; parallelising the tiny group count would leave
    // most cores idle during the expensive per-time sensitivity calculation.
    for (Group &group : groups) {
        if (wasCancelled(cancelled))
            throw std::runtime_error("Inversion cancelled");
        group.model.maxThreads = maximumThreads;
        ModelSensitivity sensitivity;
        if (group.needsJacobian) {
            sensitivity = modelSensitivity(group.model, cancelled);
        } else {
            sensitivity.response = TemSolver::forward(
                group.model, cancelled);
        }
        std::vector<std::size_t> soundings;
        soundings.reserve(group.requests.size());
        for (const Request &request : group.requests) {
            if (std::find(soundings.begin(), soundings.end(), request.sounding)
                    == soundings.end())
                soundings.push_back(request.sounding);
            for (std::size_t row = 0; row < request.unionRows.size(); ++row) {
                const std::size_t output = request.outputOffset + row;
                const std::size_t source = request.unionRows[row];
                const double response = sensitivity.response[source];
                result.responses[request.sounding][output] = response;
                if (!needJacobians[request.sounding]
                    || response <= 0.0)
                    continue;
                for (std::size_t parameter = 0;
                     parameter < logModels[request.sounding].size();
                     ++parameter) {
                    result.jacobians[request.sounding][output][parameter]
                        = sensitivity.absoluteJacobian[source][parameter]
                        / response;
                }
            }
        }
        if (soundings.size() > 1)
            for (const std::size_t sounding : soundings)
                result.sharedAcrossSoundings[sounding] = true;
    }
    return result;
}

std::string forwardBatchSignature(const InversionOptions &options,
                                  const std::vector<double> &logModel)
{
    std::string signature = "invertem-forward-batch-v1";
    std::vector<double> resistivities(logModel.size());
    std::transform(logModel.begin(), logModel.end(), resistivities.begin(),
                   [](double value) { return std::exp(value); });
    appendForwardSignature(signature, options.model, resistivities);
    const std::uint64_t additionalCount = options.additionalDataSets.size();
    appendSignatureValue(signature, additionalCount);
    for (const auto &dataSet : options.additionalDataSets)
        appendForwardSignature(signature, dataSet.model, resistivities);
    return signature;
}

std::vector<std::vector<double>> forwardLogModelsBatch(
    const std::vector<InversionOptions> &options,
    const std::vector<std::vector<double>> &logModels,
    unsigned maximumThreads, const CancelCallback &cancelled)
{
    if (options.size() != logModels.size())
        throw std::invalid_argument("Batch forward option/model counts differ");
    std::vector<std::vector<double>> responses(options.size());
    if (options.empty())
        return responses;

    struct UniqueRequest {
        std::size_t representative = 0;
        std::vector<std::size_t> destinations;
    };
    std::vector<UniqueRequest> unique;
    std::unordered_map<std::string, std::size_t> signatureToRequest;
    for (std::size_t index = 0; index < options.size(); ++index) {
        const std::string signature = forwardBatchSignature(
            options[index], logModels[index]);
        const auto inserted = signatureToRequest.emplace(
            signature, unique.size());
        if (inserted.second)
            unique.push_back({index, {index}});
        else
            unique[inserted.first->second].destinations.push_back(index);
    }

    std::vector<std::vector<double>> uniqueResponses(unique.size());
    parallelFor(unique.size(), maximumThreads, [&](std::size_t requestIndex) {
        if (wasCancelled(cancelled))
            throw std::runtime_error("Inversion cancelled");
        const std::size_t source = unique[requestIndex].representative;
        InversionOptions serial = options[source];
        const unsigned threadsPerRequest = unique.size() == 1
            ? maximumThreads : 1u;
        serial.model.maxThreads = threadsPerRequest;
        for (auto &dataSet : serial.additionalDataSets)
            dataSet.model.maxThreads = threadsPerRequest;
        uniqueResponses[requestIndex] = forwardLogModel(
            serial, logModels[source], cancelled);
    }, true);
    for (std::size_t requestIndex = 0; requestIndex < unique.size(); ++requestIndex)
        for (const std::size_t destination : unique[requestIndex].destinations)
            responses[destination] = uniqueResponses[requestIndex];
    return responses;
}

std::string firstJacobianSignature(const InversionOptions &options,
                                   const std::vector<double> &logModel)
{
    std::string signature = "pytem-first-jacobian-v2";
    const std::uint32_t method = static_cast<std::uint32_t>(options.jacobianMethod);
    appendSignatureValue(signature, method);
    appendSignatureValue(signature, options.jacobianStep);
    std::vector<double> resistivities(logModel.size());
    std::transform(logModel.begin(), logModel.end(), resistivities.begin(),
                   [](double value) { return std::exp(value); });
    appendForwardSignature(signature, options.model, resistivities);
    const std::uint64_t additionalCount = options.additionalDataSets.size();
    appendSignatureValue(signature, additionalCount);
    for (const auto &dataSet : options.additionalDataSets)
        appendForwardSignature(signature, dataSet.model, resistivities);
    return signature;
}

Matrix buildJacobian(const InversionOptions &options,
                     const std::vector<double> &logModel,
                     const std::vector<double> &base,
                     const CancelCallback &cancelled)
{
    return options.jacobianMethod == JacobianMethod::Analytical
        ? analyticalJacobian(options, logModel, base, cancelled)
        : finiteDifferenceJacobian(options, logModel, base, cancelled);
}


// Plain-formula complex division/exp/sqrt for the kernels below. The library
// versions handle inf/NaN edge cases that cannot occur here (Re(gamma^2) > 0)
// and dominate the runtime of the TE recursion.
inline Complex fastDiv(Complex a, Complex b)
{
    const double d = 1.0 / (b.real() * b.real() + b.imag() * b.imag());
    return {(a.real() * b.real() + a.imag() * b.imag()) * d,
            (a.imag() * b.real() - a.real() * b.imag()) * d};
}

inline Complex fastExp(Complex z)
{
    const double e = std::exp(z.real());
    return {e * std::cos(z.imag()), e * std::sin(z.imag())};
}

inline Complex fastSqrt(Complex z)
{
    // Valid for Re(z) > 0, which holds for lambda^2 + s*mu0/rho here.
    const double modulus = std::hypot(z.real(), z.imag());
    const double re = std::sqrt(0.5 * (modulus + z.real()));
    return {re, 0.5 * z.imag() / re};
}

// Model-independent part of the circular-loop step response on a fixed time
// grid. Each (time, frequency) sample stores s = i*omega and a coefficient
// folding the system filter, transform weight and scale, so that
// response(t) = sum_k Re(coefficient[t,k] * mu0 * field(s[t,k])).
struct StepKernel {
    std::vector<double> lambdas, lambdaSquared, hankelFactors;
    std::vector<Complex> s, coefficients;
    std::size_t times = 0, frequencies = 0;
    bool vectorized = false;
};

StepKernel stepKernel(const ForwardModel &model, const std::vector<double> &times)
{
    StepKernel k;
    for (std::size_t i = 0; i < weights::hankel_base_101.size(); ++i) {
        const double lambda = weights::hankel_base_101[i] / model.txSize;
        const double offset = (model.geometry == Geometry::CircleOffset
            ? std::cyl_bessel_j(0.0, lambda * model.rxX) : 1.0) * std::exp(-lambda * model.altitude);
        k.lambdas.push_back(lambda);
        k.lambdaSquared.push_back(lambda * lambda);
        k.hankelFactors.push_back(0.5 * mu0 * lambda * weights::hankel_j1_101[i] * offset);
    }
    const bool euler = model.transform == TransformMethod::Euler;
    const auto eta = euler ? eulerWeights(model.eulerOrder) : std::vector<double>{};
    k.times = times.size();
    k.frequencies = euler ? eta.size() : weights::fourier_base_81.size();
    constexpr double eulerA = 18.4;
    for (const double t : times) {
        if (t <= 0.0)
            throw std::invalid_argument("Gate times must be positive");
        for (std::size_t f = 0; f < k.frequencies; ++f) {
            const Complex omega = euler
                ? Complex(f * pi / t, -eulerA / (2.0 * t))
                : Complex(weights::fourier_base_81[f] / t, 0.0);
            const Complex weight = euler
                ? std::exp(eulerA / 2.0) / t * eta[f] * ((f & 1U) ? -1.0 : 1.0)
                : Complex(0.0, 2.0 / (pi * t) * weights::fourier_sin_81[f]); // -2/(pi t) Im(z) = Re(i 2/(pi t) z)
            k.s.push_back(Complex(0.0, 1.0) * omega);
            k.coefficients.push_back(weight * systemTransfer(omega, model));
        }
    }
    return k;
}

// Step response and, when requested, d(response)/d(ln rho) as a
// times x layers row-major matrix. The derivative uses the adjoint of the
// upward TE recursion, so one pass gives all layers.
void evaluateStep(const StepKernel &k, const std::vector<double> &thicknesses,
                  const std::vector<double> &resistivities, unsigned threads,
                  const CancelCallback &cancelled, std::vector<double> &response,
                  std::vector<double> *jacobian)
{
    const std::size_t n = resistivities.size();
    response.assign(k.times, 0.0);
    if (jacobian)
        jacobian->assign(k.times * n, 0.0);
    // Layers repeating the one above (e.g. the refined DOI model) reuse its
    // gamma and, with equal thickness, its decay factor.
    std::vector<char> sameRho(n, 0), sameLayer(n, 0);
    for (std::size_t j = 1; j < n; ++j) {
        sameRho[j] = resistivities[j] == resistivities[j - 1];
        sameLayer[j] = sameRho[j] && j + 1 < n && thicknesses[j] == thicknesses[j - 1];
    }
    if (k.vectorized) {
        const KernelView view{k.lambdaSquared.data(), k.hankelFactors.data(), k.lambdas.data(),
                              k.lambdas.size(), k.s.data(), k.coefficients.data(), k.frequencies,
                              thicknesses.data(), resistivities.data(), sameRho.data(),
                              sameLayer.data(), n};
        parallelFor(k.times, threads, [&](std::size_t t) {
            if (wasCancelled(cancelled))
                throw std::runtime_error("Inversion cancelled");
            vectorKernelTime(view, t, response[t], jacobian ? jacobian->data() + t * n : nullptr);
        }, true);
        return;
    }
    parallelFor(k.times, threads, [&](std::size_t t) {
        if (wasCancelled(cancelled))
            throw std::runtime_error("Inversion cancelled");
        std::vector<Complex> term(n), gamma(n), dGamma(n), psi(n), decay(n), below(n), field(n);
        for (std::size_t f = 0; f < k.frequencies; ++f) {
            const Complex s = k.s[t * k.frequencies + f];
            for (std::size_t j = 0; j < n; ++j)
                term[j] = s * mu0 / resistivities[j];
            Complex total = 0.0;
            std::fill(field.begin(), field.end(), Complex(0.0));
            for (std::size_t l = 0; l < k.lambdas.size(); ++l) {
                const double lambda = k.lambdas[l];
                Complex reflected = 0.0;
                // Stop at the layer below which the two-way attenuation
                // exceeds e^-50: deeper layers cannot change R in double precision.
                std::size_t depth = n;
                double attenuation = 0.0;
                for (std::size_t j = 0; j < depth; ++j) {
                    gamma[j] = sameRho[j] ? gamma[j - 1] : fastSqrt(k.lambdaSquared[l] + term[j]);
                    decay[j] = sameLayer[j] ? decay[j - 1]
                        : j + 1 < n ? fastExp(-2.0 * thicknesses[j] * gamma[j]) : Complex(0.0);
                    if (j + 1 < n && (attenuation += 2.0 * thicknesses[j] * gamma[j].real()) > 50.0) {
                        decay[j] = 0.0;
                        depth = j + 1;
                    }
                }
                for (std::size_t j = depth; j-- > 0;) {
                    below[j] = reflected;
                    if (sameRho[j]) { // no interface: psi = 0
                        psi[j] = 0.0;
                        reflected *= decay[j];
                        continue;
                    }
                    const Complex above = j == 0 ? Complex(lambda) : gamma[j - 1];
                    psi[j] = fastDiv(above - gamma[j], above + gamma[j]);
                    const Complex rd = reflected * decay[j];
                    reflected = fastDiv(psi[j] + rd, 1.0 + psi[j] * rd);
                }
                total += reflected * k.hankelFactors[l];
                if (!jacobian)
                    continue;
                for (std::size_t j = 0; j < depth; ++j)
                    dGamma[j] = sameRho[j] ? dGamma[j - 1] : fastDiv(-0.5 * term[j], gamma[j]);
                Complex adjoint = k.hankelFactors[l];
                for (std::size_t j = 0; j < depth; ++j) {
                    const Complex dDecay = -2.0 * (j + 1 < n ? thicknesses[j] : 0.0) * decay[j];
                    if (sameRho[j]) { // psi = 0 and above = gamma: 1/(above+gamma)^2 terms reduce to 1/(2 gamma)
                        const Complex bd = below[j] * decay[j];
                        const Complex q = fastDiv(0.5, gamma[j]) * adjoint * (1.0 - bd * bd) * dGamma[j];
                        field[j] += adjoint * below[j] * dDecay * dGamma[j] - q;
                        field[j - 1] += q;
                        adjoint *= decay[j];
                        continue;
                    }
                    const Complex above = j == 0 ? Complex(lambda) : gamma[j - 1];
                    const Complex bd = below[j] * decay[j];
                    const Complex inv2 = fastDiv(1.0, (1.0 + psi[j] * bd) * (1.0 + psi[j] * bd));
                    const Complex dPsi = (1.0 - bd * bd) * inv2;
                    const Complex invSum2 = fastDiv(1.0, (above + gamma[j]) * (above + gamma[j]));
                    const Complex onePsi2 = (1.0 - psi[j] * psi[j]) * inv2;
                    field[j] += adjoint * (-2.0 * above * invSum2 * dPsi + below[j] * onePsi2 * dDecay)
                        * dGamma[j];
                    if (j > 0)
                        field[j - 1] += adjoint * dPsi * 2.0 * gamma[j] * invSum2 * dGamma[j - 1];
                    adjoint *= decay[j] * onePsi2;
                }
            }
            const Complex c = k.coefficients[t * k.frequencies + f];
            response[t] += (c * total).real();
            if (jacobian)
                for (std::size_t j = 0; j < n; ++j)
                    (*jacobian)[t * n + j] += (c * field[j]).real();
        }
    }, true);
}

// Step response and Jacobian over a whole step grid at one model, computed once
// per process for each system and shared by every sounding (and worker) asking
// for the same inputs. Used for the common starting model of a batch.
struct SharedStep {
    std::vector<double> y, dy;
    bool computedHere = false;
};

SharedStep sharedStep(const ForwardModel &m, const std::vector<double> &grid,
                      const std::vector<double> &thicknesses, const std::vector<double> &rho,
                      bool vectorized, unsigned threads, const CancelCallback &cancelled)
{
    std::vector<double> key{double(m.geometry), double(m.transform), double(m.eulerOrder), m.txSize, m.rxX,
                            m.altitude, double(vectorized)};
    for (const auto *v : {&m.lowPassFrequencies, &m.highPassFrequencies, &grid, &thicknesses, &rho}) {
        key.push_back(double(v->size()));
        key.insert(key.end(), v->begin(), v->end());
    }
    for (const auto *v : {&m.lowPassOrders, &m.highPassOrders}) {
        key.push_back(double(v->size()));
        key.insert(key.end(), v->begin(), v->end());
    }
    using Entry = std::shared_future<std::shared_ptr<const SharedStep>>;
    static std::mutex mutex;
    static std::map<std::vector<double>, Entry> cache;
    std::promise<std::shared_ptr<const SharedStep>> promise;
    Entry entry;
    bool owner = false;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = cache.find(key);
        owner = found == cache.end();
        entry = owner ? cache.emplace(key, promise.get_future().share()).first->second : found->second;
    }
    if (owner) {
        try {
            StepKernel k = stepKernel(m, grid);
            k.vectorized = vectorized;
            auto step = std::make_shared<SharedStep>();
            evaluateStep(k, thicknesses, rho, threads, cancelled, step->y, &step->dy);
            promise.set_value(step);
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                cache.erase(key); // a cancelled or failed evaluation is not cached
            }
            promise.set_exception(std::current_exception());
        }
    }
    SharedStep result = *entry.get();
    result.computedHere = owner;
    return result;
}

// One sounding's pyTEM-joint problem: step kernels shared by datasets of one
// system, gated predictions, their log-Jacobian and the sensitivity DOI.
struct JointProblem {
    using Clock = std::chrono::steady_clock;
    static double seconds(Clock::time_point since)
    {
        return std::chrono::duration<double>(Clock::now() - since).count();
    }
    static const std::vector<double> &grid(const ForwardModel *m)
    {
        return m->responseMatrix.empty() ? m->times : m->stepTimes;
    }
    std::vector<const ForwardModel *> models;
    std::vector<double> observed, weights;
    std::vector<StepKernel> kernels;
    std::vector<std::size_t> kernelOf, kernelModel;
    std::vector<std::vector<std::size_t>> columns;
    unsigned threads = 0;
    CancelCallback cancelled;
    InversionResult *result = nullptr; // receives timings, first-Jacobian source and DOI
    bool linearJacobian = false;       // d prediction / d ln(rho) instead of d ln(prediction) / d ln(rho)

    JointProblem(const InversionOptions &options, const CancelCallback &cancel, InversionResult &sink)
        : threads(options.model.maxThreads), cancelled(cancel), result(&sink)
    {
        // Datasets sharing a step grid and system share one step-response evaluation.
        models = {&options.model};
        observed = options.observed;
        std::vector<double> noise = options.noiseStd;
        for (const auto &dataSet : options.additionalDataSets) {
            models.push_back(&dataSet.model);
            observed.insert(observed.end(), dataSet.observed.begin(), dataSet.observed.end());
            noise.insert(noise.end(), dataSet.noiseStd.begin(), dataSet.noiseStd.end());
        }
        for (std::size_t i = 0; i < observed.size(); ++i)
            weights.push_back(observed[i] / noise[i]);
        for (std::size_t d = 0; d < models.size(); ++d) {
            const ForwardModel *m = models[d];
            std::size_t match = d;
            for (std::size_t e = 0; e < d && match == d; ++e) {
                const ForwardModel *o = models[e];
                if (grid(o) == grid(m) && o->geometry == m->geometry && o->transform == m->transform
                    && o->eulerOrder == m->eulerOrder && o->txSize == m->txSize && o->rxX == m->rxX
                    && o->altitude == m->altitude
                    && o->lowPassFrequencies == m->lowPassFrequencies && o->lowPassOrders == m->lowPassOrders
                    && o->highPassFrequencies == m->highPassFrequencies && o->highPassOrders == m->highPassOrders)
                    match = e;
            }
            kernelOf.push_back(match == d ? kernelModel.size() : kernelOf[match]);
            if (match == d)
                kernelModel.push_back(d);
        }
        // Only step times with a nonzero weight in some gate are evaluated; the
        // padded ends of the shared grid typically fall outside every gate.
        columns.assign(kernelModel.size(), {});
        for (std::size_t u = 0; u < kernelModel.size(); ++u) {
            const auto &times = grid(models[kernelModel[u]]);
            std::vector<double> active;
            for (std::size_t c = 0; c < times.size(); ++c) {
                bool used = false;
                for (std::size_t d = 0; d < models.size() && !used; ++d)
                    if (kernelOf[d] == u)
                        for (const auto &row : models[d]->responseMatrix)
                            used = used || row[c] != 0.0;
                if (used || models[kernelModel[u]]->responseMatrix.empty()) {
                    columns[u].push_back(c);
                    active.push_back(times[c]);
                }
            }
            kernels.push_back(stepKernel(*models[kernelModel[u]], active));
            kernels.back().vectorized = options.vectorizedKernel && vectorKernelAvailable();
        }
    }

    // Gated prediction and, optionally, d ln(prediction) / d ln(rho).
    void evaluate(const std::vector<double> &thicknesses, const std::vector<double> &logModel,
                  std::vector<double> &predicted, Matrix *jacobian, bool shared = false)
    {
        const auto t0 = Clock::now();
        std::vector<double> rho(logModel.size());
        std::transform(logModel.begin(), logModel.end(), rho.begin(), [](double v) { return std::exp(v); });
        const std::size_t n = rho.size();
        std::vector<std::vector<double>> step(kernels.size()), stepJac(kernels.size());
        for (std::size_t u = 0; u < kernels.size(); ++u) {
            if (shared) { // whole grid, so soundings with different enabled gates share it too
                auto common = sharedStep(*models[kernelModel[u]], grid(models[kernelModel[u]]), thicknesses, rho,
                                         kernels[u].vectorized, threads, cancelled);
                result->firstJacobianSource = common.computedHere ? "computed once, shared" : "shared";
                step[u] = std::move(common.y);
                stepJac[u] = std::move(common.dy);
                continue;
            }
            std::vector<double> y, dy;
            evaluateStep(kernels[u], thicknesses, rho, threads, cancelled, y, jacobian ? &dy : nullptr);
            const std::size_t full = grid(models[kernelModel[u]]).size();
            step[u].assign(full, 0.0);
            stepJac[u].assign(jacobian ? full * n : 0, 0.0);
            for (std::size_t i = 0; i < columns[u].size(); ++i) {
                step[u][columns[u][i]] = y[i];
                if (jacobian)
                    std::copy_n(dy.begin() + i * n, n, stepJac[u].begin() + columns[u][i] * n);
            }
        }
        predicted.clear();
        if (jacobian)
            jacobian->clear();
        for (std::size_t d = 0; d < models.size(); ++d) {
            const auto &matrix = models[d]->responseMatrix;
            const auto &y = step[kernelOf[d]];
            const auto &dy = stepJac[kernelOf[d]];
            const std::size_t rows = matrix.empty() ? y.size() : matrix.size();
            for (std::size_t r = 0; r < rows; ++r) {
                double value = 0.0;
                std::vector<double> row(jacobian ? n : 0, 0.0);
                for (std::size_t c = 0; c < y.size(); ++c) {
                    const double w = matrix.empty() ? (c == r) : matrix[r][c];
                    if (w == 0.0)
                        continue;
                    value += w * y[c];
                    for (std::size_t j = 0; j < row.size(); ++j)
                        row[j] += w * dy[c * n + j];
                }
                if (!linearJacobian)
                    for (double &v : row)
                        v = value > 0.0 ? v / value : 0.0;
                predicted.push_back(value);
                if (jacobian)
                    jacobian->push_back(std::move(row));
            }
        }
        if (jacobian) {
            result->timing.jacobianSeconds += seconds(t0);
            ++result->timing.jacobianEvaluations;
        } else {
            result->timing.alphaForwardSeconds += seconds(t0);
            ++result->timing.alphaTrialForwards;
        }
    }

    // Cumulative-sensitivity DOI on a refined copy of the final model.
    void doi(const InversionOptions &options, const std::vector<double> &model, int refinement = 0)
    {
        const std::vector<double> &thicknesses = options.model.thicknesses;
        const auto t0 = Clock::now();
        const int refine = refinement > 0 ? refinement : options.doiRefinement;
        std::vector<double> fineThick, fineModel;
        for (std::size_t j = 0; j < thicknesses.size(); ++j)
            for (int k = 0; k < refine; ++k) {
                fineThick.push_back(thicknesses[j] / refine);
                fineModel.push_back(model[j]);
            }
        fineModel.push_back(model.back());
        std::vector<double> finePrediction;
        Matrix fineJacobian;
        evaluate(fineThick, fineModel, finePrediction, &fineJacobian);
        std::vector<double> sensitivity(fineModel.size(), 0.0), tops{0.0};
        for (std::size_t i = 0; i < fineJacobian.size(); ++i)
            for (std::size_t j = 0; j < sensitivity.size(); ++j)
                sensitivity[j] += std::abs(fineJacobian[i][j] * weights[i]);
        for (double h : fineThick)
            tops.push_back(tops.back() + h);
        std::vector<double> cumulative(sensitivity.size());
        std::partial_sum(sensitivity.rbegin(), sensitivity.rend(), cumulative.rbegin());
        auto crossing = [&](double level, bool &capped) {
            capped = false;
            if (cumulative.front() <= level)
                return 0.0;
            if (cumulative.back() >= level) {
                capped = true;
                return tops.back();
            }
            std::size_t i = 0;
            while (cumulative[i + 1] >= level)
                ++i;
            return tops[i] + (level - cumulative[i]) / (cumulative[i + 1] - cumulative[i])
                * (tops[i + 1] - tops[i]);
        };
        result->doiStandard = crossing(options.doiThreshold, result->doiStandardCapped);
        result->doiConservative = crossing(options.doiConservativeThreshold, result->doiConservativeCapped);
        result->sensitivity.assign(model.size(), 0.0);
        for (std::size_t j = 0; j < sensitivity.size(); ++j)
            result->sensitivity[std::min(j / refine, model.size() - 1)] += sensitivity[j];
        result->timing.sensitivitySeconds = seconds(t0);
    }
};

double trialLogRms(const std::vector<double> &observed, const std::vector<double> &predicted,
                   const std::vector<double> &weights)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        if (!(std::isfinite(predicted[i]) && predicted[i] > 0.0))
            return std::numeric_limits<double>::infinity();
        const double r = weights[i] * (std::log(observed[i]) - std::log(predicted[i]));
        sum += r * r;
    }
    return observed.empty() ? std::numeric_limits<double>::infinity()
                            : std::sqrt(sum / static_cast<double>(observed.size()));
}


// Edges of the Delaunay triangulation of planar points (Bowyer-Watson). Points
// without an edge (e.g. an exactly collinear set) are joined to their nearest
// neighbour so every sounding is constrained.
std::vector<std::pair<std::size_t, std::size_t>> delaunayEdges(std::vector<double> x, std::vector<double> y)
{
    const std::size_t n = x.size();
    std::set<std::pair<std::size_t, std::size_t>> edges;
    if (n < 2)
        return {};
    const double mx = std::accumulate(x.begin(), x.end(), 0.0) / n, my = std::accumulate(y.begin(), y.end(), 0.0) / n;
    double extent = 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        x[i] -= mx - 1e-7 * static_cast<double>(i); // tiny jitter separates coincident points
        y[i] -= my + 1.3e-7 * static_cast<double>(i % 7);
        extent = std::max({extent, std::abs(x[i]), std::abs(y[i])});
    }
    x.insert(x.end(), {-40 * extent, 40 * extent, 0.0});
    y.insert(y.end(), {-40 * extent, -40 * extent, 40 * extent});
    struct Triangle { std::size_t v[3]; double cx, cy, r2; };
    auto triangle = [&](std::size_t a, std::size_t b, std::size_t c) {
        const double ax = x[a], ay = y[a], bx = x[b] - ax, by = y[b] - ay, cx = x[c] - ax, cy = y[c] - ay;
        const double d = 2.0 * (bx * cy - by * cx);
        const double ux = (cy * (bx * bx + by * by) - by * (cx * cx + cy * cy)) / d;
        const double uy = (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by)) / d;
        return Triangle{{a, b, c}, ax + ux, ay + uy, ux * ux + uy * uy};
    };
    std::vector<Triangle> triangles{triangle(n, n + 1, n + 2)};
    for (std::size_t p = 0; p < n; ++p) {
        std::map<std::pair<std::size_t, std::size_t>, int> boundary;
        std::vector<Triangle> kept;
        for (const auto &t : triangles) {
            const double dx = x[p] - t.cx, dy = y[p] - t.cy;
            if (dx * dx + dy * dy < t.r2)
                for (int k = 0; k < 3; ++k)
                    ++boundary[std::minmax(t.v[k], t.v[(k + 1) % 3])];
            else
                kept.push_back(t);
        }
        for (const auto &[edge, count] : boundary)
            if (count == 1)
                kept.push_back(triangle(edge.first, edge.second, p));
        triangles = std::move(kept);
    }
    std::vector<int> degree(n, 0);
    for (const auto &t : triangles)
        for (int k = 0; k < 3; ++k) {
            const auto edge = std::minmax(t.v[k], t.v[(k + 1) % 3]);
            if (edge.second < n && edges.insert(edge).second)
                ++degree[edge.first], ++degree[edge.second];
        }
    for (std::size_t i = 0; i < n; ++i)
        if (degree[i] == 0) {
            std::size_t nearest = i == 0 ? 1 : 0;
            for (std::size_t j = 0; j < n; ++j)
                if (j != i && std::hypot(x[j] - x[i], y[j] - y[i]) < std::hypot(x[nearest] - x[i], y[nearest] - y[i]))
                    nearest = j;
            edges.insert(std::minmax(i, nearest));
        }
    return {edges.begin(), edges.end()};
}

} // namespace

bool vectorKernelAvailable()
{
#if !defined(INVERTEM_VECTOR_KERNEL)
    return false;
#elif defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7)
        return false;
    __cpuid(info, 1);
    const bool osxsave = info[2] & (1 << 27), avx = info[2] & (1 << 28), fma = info[2] & (1 << 12);
    if (!(osxsave && avx && fma) || (_xgetbv(0) & 6) != 6)
        return false;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}

std::vector<double> TemSolver::forward(const ForwardModel &model,
                                       const CancelCallback &cancelled)
{
    if (model.resistivities.empty() || model.thicknesses.size() + 1 != model.resistivities.size())
        throw std::invalid_argument("Forward model layer arrays are inconsistent");
    if (model.txSize <= 0.0)
        throw std::invalid_argument("Transmitter size must be positive");

    if (model.responseMatrix.empty())
        return stepResponse(model, model.times, cancelled);
    if (model.responseMatrix.size() != model.times.size() || model.stepTimes.empty())
        throw std::invalid_argument("Invalid waveform/gate response matrix dimensions");
    const auto step = stepResponse(model, model.stepTimes, cancelled);
    std::vector<double> gated(model.responseMatrix.size(), 0.0);
    for (std::size_t row = 0; row < model.responseMatrix.size(); ++row) {
        if (model.responseMatrix[row].size() != step.size())
            throw std::invalid_argument("Response matrix column count does not match step-time grid");
        for (std::size_t column = 0; column < step.size(); ++column)
            gated[row] += model.responseMatrix[row][column] * step[column];
    }
    return gated;
}

std::vector<double> TemSolver::logSpacedThicknesses(int layerCount, double firstDepth,
                                                    double maxDepth)
{
    if (layerCount < 2 || !(firstDepth > 0.0) || !(maxDepth > firstDepth))
        throw std::invalid_argument("Require at least two layers and 0 < first depth < maximum depth");
    std::vector<double> thicknesses;
    double top = 0.0;
    for (int k = 0; k + 1 < layerCount; ++k) {
        const double base = layerCount == 2 ? maxDepth
            : firstDepth * std::pow(maxDepth / firstDepth, k / static_cast<double>(layerCount - 2));
        thicknesses.push_back(base - top);
        top = base;
    }
    return thicknesses;
}

std::vector<double> TemSolver::geometricThicknesses(int layerCount,
                                                    double maxDepth)
{
    if (layerCount < 3)
        throw std::invalid_argument("At least three layers are required");
    const int finiteLayers = layerCount - 1;
    if (!std::isfinite(maxDepth) || maxDepth < finiteLayers)
        throw std::invalid_argument(
            "Maximum finite-layer depth must be at least N-1 metres");
    auto seriesSum = [finiteLayers](double ratio) {
        double sum = 0.0;
        double thickness = 1.0;
        for (int layer = 0; layer < finiteLayers; ++layer) {
            sum += thickness;
            thickness *= ratio;
        }
        return sum;
    };
    double lower = 1.0;
    double upper = 2.0;
    while (seriesSum(upper) < maxDepth)
        upper *= 2.0;
    for (int iteration = 0; iteration < 100; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if (seriesSum(middle) < maxDepth)
            lower = middle;
        else
            upper = middle;
    }
    const double ratio = 0.5 * (lower + upper);
    std::vector<double> thicknesses(static_cast<std::size_t>(finiteLayers), 1.0);
    for (int layer = 1; layer < finiteLayers; ++layer)
        thicknesses[static_cast<std::size_t>(layer)]
            = thicknesses[static_cast<std::size_t>(layer - 1)] * ratio;
    const double total = std::accumulate(thicknesses.begin(), thicknesses.end(), 0.0);
    thicknesses.back() += maxDepth - total;
    return thicknesses;
}

double TemSolver::depthOfInvestigation(
    const std::vector<double> &thicknesses,
    const std::vector<double> &sensitivity,
    double cumulativeFraction)
{
    if (thicknesses.empty()
        || sensitivity.size() != thicknesses.size() + 1
        || !(cumulativeFraction > 0.0 && cumulativeFraction < 1.0))
        return -1.0;

    // Squared column norms are the diagonal information contributions of the
    // noise-weighted Jacobian. Only finite layers can be assigned a depth.
    std::vector<double> information(thicknesses.size(), 0.0);
    double total = 0.0;
    for (std::size_t layer = 0; layer < thicknesses.size(); ++layer) {
        const double value = sensitivity[layer];
        if (std::isfinite(value) && value > 0.0
            && std::isfinite(thicknesses[layer]) && thicknesses[layer] > 0.0) {
            information[layer] = value * value;
            total += information[layer];
        }
    }
    if (!(total > 0.0) || !std::isfinite(total))
        return -1.0;

    const double target = cumulativeFraction * total;
    double cumulative = 0.0;
    double top = 0.0;
    for (std::size_t layer = 0; layer < thicknesses.size(); ++layer) {
        const double next = cumulative + information[layer];
        if (next >= target && information[layer] > 0.0) {
            const double fraction = std::clamp(
                (target - cumulative) / information[layer], 0.0, 1.0);
            return top + fraction * thicknesses[layer];
        }
        cumulative = next;
        top += thicknesses[layer];
    }
    return -1.0;
}

std::vector<std::vector<double>> TemSolver::logJacobian(
    const ForwardModel &model, JacobianMethod method,
    double finiteDifferenceStep, const CancelCallback &cancelled)
{
    InversionOptions options;
    options.model = model;
    options.jacobianMethod = method;
    options.jacobianStep = finiteDifferenceStep;
    std::vector<double> logModel(model.resistivities.size());
    std::transform(model.resistivities.begin(), model.resistivities.end(), logModel.begin(),
                   [](double value) {
                       if (value <= 0.0)
                           throw std::invalid_argument("Resistivities must be positive");
                       return std::log(value);
                   });
    const auto base = forward(model, cancelled);
    return buildJacobian(options, logModel, base, cancelled);
}

void TemSolver::validate(const InversionOptions &options)
{
    auto validateDataSet = [](const ForwardModel &model,
                              const std::vector<double> &observed,
                              const std::vector<double> &noiseStd) {
        if (model.times.size() < 3 || observed.size() != model.times.size()
            || noiseStd.size() != model.times.size())
            throw std::invalid_argument("Times, observed data, and errors must have the same length (at least 3)");
        if (model.resistivities.empty() || model.thicknesses.size() + 1 != model.resistivities.size())
            throw std::invalid_argument("There must be one fewer thickness than resistivity values");
        if (model.transform == TransformMethod::Euler && (model.eulerOrder < 1 || model.eulerOrder > 30))
            throw std::invalid_argument("Euler order must be between 1 and 30");
        for (std::size_t i = 0; i < model.times.size(); ++i) {
            if (!(model.times[i] > 0.0 && observed[i] > 0.0 && noiseStd[i] > 0.0))
                throw std::invalid_argument("Times, observed magnitudes, and errors must be positive");
            if (i > 0 && model.times[i] <= model.times[i - 1])
                throw std::invalid_argument("Gate times must be strictly increasing within each moment");
        }
        if (std::any_of(model.thicknesses.begin(), model.thicknesses.end(), [](double value) { return value <= 0.0; })
            || std::any_of(model.resistivities.begin(), model.resistivities.end(), [](double value) { return value <= 0.0; }))
            throw std::invalid_argument("Thicknesses and resistivities must be positive");
    };
    validateDataSet(options.model, options.observed, options.noiseStd);
    for (const auto &dataSet : options.additionalDataSets) {
        validateDataSet(dataSet.model, dataSet.observed, dataSet.noiseStd);
        if (dataSet.model.thicknesses != options.model.thicknesses
            || dataSet.model.resistivities.size() != options.model.resistivities.size())
            throw std::invalid_argument("Joint datasets must use the same layered-earth parameterization");
    }
    if (!(options.rhoMin > 0.0 && options.rhoMin < options.rhoMax))
        throw std::invalid_argument("Invalid resistivity bounds");
    if (options.broydenRefreshInterval < 1)
        throw std::invalid_argument(
            "Broyden full-Jacobian refresh interval must be positive");
}

InversionResult TemSolver::invertJoint(const InversionOptions &options,
                                       const ProgressCallback &progress,
                                       const CancelCallback &cancelled)
{
    using Clock = std::chrono::steady_clock;
    auto seconds = [](Clock::time_point since) {
        return std::chrono::duration<double>(Clock::now() - since).count();
    };
    const auto start = Clock::now();
    validate(options);
    if (options.alphaSteps < 1 || options.doiRefinement < 1)
        throw std::invalid_argument("Alpha steps and DOI refinement must be positive");

    InversionResult result;
    JointProblem problem(options, cancelled, result);
    const auto &observed = problem.observed;
    const auto &weights = problem.weights;
    const std::vector<double> &thicknesses = options.model.thicknesses;
    const double lower = std::log(options.rhoMin), upper = std::log(options.rhoMax);
    std::vector<double> model(options.model.resistivities.size());
    std::transform(options.model.resistivities.begin(), options.model.resistivities.end(),
                   model.begin(), [](double v) { return std::log(v); });
    Matrix roughnessMatrix = roughness(model, RegularizationNorm::L2Smooth);
    // The adaptive search fits alpha to RMS 1 instead of accepting the first undershoot.
    const double target = options.adaptiveAlphaSearch ? rmsDisplayConvergenceThreshold : 1.0;

    std::vector<double> predicted;
    Matrix jacobian;
    // Optionally start from the best of 33 homogeneous half-spaces (1 Ohm m to
    // 10 kOhm m, 8 per decade), whose responses are shared across the batch.
    if (options.halfSpaceStart) {
        double best = std::numeric_limits<double>::infinity();
        for (int g = 0; g <= 32; ++g) {
            const std::vector<double> uniform(model.size(), std::clamp(std::log(10.0) * g / 8.0, lower, upper));
            problem.evaluate(thicknesses, uniform, predicted, nullptr, true);
            if (const double value = trialLogRms(observed, predicted, weights); value < best) {
                best = value;
                model = uniform;
            }
        }
    }
    // Soundings of a system that start from the same model share their first
    // step response and Jacobian, computed once across the batch.
    problem.evaluate(thicknesses, model, predicted, &jacobian, true);
    double rmsValue = trialLogRms(observed, predicted, weights);
    auto weightedResidual = [&]() {
        std::vector<double> r(observed.size());
        for (std::size_t i = 0; i < r.size(); ++i)
            r[i] = weights[i] * (std::log(observed[i]) - std::log(std::max(predicted[i], 1e-300)));
        return r;
    };
    // Initial alpha: infinity norm of the weighted gradient Jw^T (w r).
    double alpha = 0.0;
    {
        const auto r = weightedResidual();
        for (std::size_t j = 0; j < model.size(); ++j) {
            double g = 0.0;
            for (std::size_t i = 0; i < r.size(); ++i)
                g += jacobian[i][j] * weights[i] * r[i];
            alpha = std::max(alpha, std::abs(g));
        }
        alpha += 1e-30;
    }
    bool haveJacobian = true;
    std::string termination = "max_iterations";
    result.rmsHistory.push_back(rmsValue);

    for (int iteration = 0; iteration < options.maxIterations; ++iteration) {
        if (!std::isfinite(rmsValue)) {
            termination = "invalid_response";
            break;
        }
        if (rmsValue <= target)
            break;
        ++result.iterations;
        if (!haveJacobian)
            problem.evaluate(thicknesses, model, predicted, &jacobian);
        haveJacobian = false;
        const auto r = weightedResidual();
        if (options.regularizationNorm == RegularizationNorm::L1Blocky) // IRLS weights from the current model
            roughnessMatrix = roughness(model, RegularizationNorm::L1Blocky);

        // Log-spaced alpha ladder; each trial halves its step until the RMS improves.
        std::vector<double> alphas, rmsValues;
        std::vector<std::vector<double>> trials, predictions;
        auto tryAlpha = [&](double a) {
            const auto delta = gaussNewtonStep(jacobian, r, weights, roughnessMatrix, model, a, thicknesses);
            std::vector<double> trial(model.size()), trialPrediction;
            double trialRms = 0.0;
            double step = 1.0;
            for (int halving = 0; halving <= 6; ++halving, step *= 0.5) {
                double bound = step;
                for (int b = 0; b <= 10; ++b, bound *= 0.5) {
                    bool inside = true;
                    for (std::size_t j = 0; j < model.size(); ++j) {
                        trial[j] = model[j] + bound * delta[j];
                        inside = inside && trial[j] >= lower && trial[j] <= upper;
                    }
                    if (inside)
                        break;
                    if (b == 10)
                        for (double &v : trial)
                            v = std::clamp(v, lower, upper);
                }
                problem.evaluate(thicknesses, trial, trialPrediction, nullptr);
                trialRms = trialLogRms(observed, trialPrediction, weights);
                if (halving == 6 || trialRms < rmsValue)
                    break;
            }
            alphas.push_back(a);
            rmsValues.push_back(trialRms);
            trials.push_back(trial);
            predictions.push_back(std::move(trialPrediction));
        };
        for (int i = 0; i < options.alphaSteps; ++i) {
            tryAlpha(alpha * std::pow(10.0, -options.alphaLogStep * i));
            const double trialRms = rmsValues.back();
            if (trialRms < 1.0) {
                // Adaptive: one extra trial at the alpha interpolated to RMS 1 in the bracket.
                const double fitted = options.adaptiveAlphaSearch && trialRms < 0.995
                    ? fittedTargetAlpha(alphas, rmsValues) : std::numeric_limits<double>::quiet_NaN();
                if (std::isfinite(fitted))
                    tryAlpha(fitted);
                break;
            }
            if (rmsValues.size() > 1 && trialRms > rmsValues[rmsValues.size() - 2]
                && *std::min_element(rmsValues.begin(), rmsValues.end() - 1) < rmsValue)
                break;
        }

        // Strongest regularisation reaching RMS <= 1, else the lowest RMS.
        std::size_t best = std::min_element(rmsValues.begin(), rmsValues.end()) - rmsValues.begin();
        for (std::size_t i = 0; i < rmsValues.size(); ++i)
            if (rmsValues[i] <= target && (rmsValues[best] > target || alphas[i] > alphas[best]))
                best = i;
        if (!std::isfinite(rmsValues[best]) || rmsValues[best] >= rmsValue) {
            termination = "stalled";
            break;
        }
        const double previous = rmsValue;
        model = trials[best];
        predicted = predictions[best];
        rmsValue = rmsValues[best];
        result.alphaFinal = alphas[best];
        alpha = alphas[best] * std::pow(10.0, options.alphaLogStep);
        result.rmsHistory.push_back(rmsValue);
        if (progress)
            progress(result.iterations, rmsValue, "alpha " + fixedTwoDecimals(result.alphaFinal));
        if (rmsValue > target && previous - rmsValue <= 1e-8 * std::max(1.0, previous)) {
            termination = "stalled";
            break;
        }
        // Stagnation: less than 1% RMS improvement over the last 5 iterations.
        const auto &history = result.rmsHistory;
        if (history.size() > 5 && history[history.size() - 6] - rmsValue < 0.01 * history[history.size() - 6]) {
            termination = "stagnated";
            break;
        }
    }

    result.converged = std::isfinite(rmsValue) && rmsValue <= target;
    if (result.converged)
        termination = "converged";
    else if (!std::isfinite(rmsValue))
        termination = "invalid_response";
    result.message = termination;
    result.predicted = predicted;
    result.resistivities.resize(model.size());
    std::transform(model.begin(), model.end(), result.resistivities.begin(),
                   [](double v) { return std::exp(v); });

    if (options.calculateSensitivity && std::isfinite(rmsValue))
        problem.doi(options, model);
    result.timing.totalSeconds = seconds(start);
    return result;
}

std::vector<InversionResult> TemSolver::invertSci(const std::vector<InversionOptions> &soundings,
                                                  const std::vector<double> &eastings,
                                                  const std::vector<double> &northings,
                                                  const std::vector<double> &elevations,
                                                  unsigned maximumThreads,
                                                  const BatchProgressCallback &progress,
                                                  const CancelCallback &cancelled)
{
    const std::size_t S = soundings.size();
    if (S == 0)
        return {};
    if (eastings.size() != S || northings.size() != S || elevations.size() != S)
        throw std::invalid_argument("SCI needs one position per sounding");
    const SciSettings &sci = soundings.front().sci;
    if (sci.adaptive) {
        const auto factor = [](double value) { std::ostringstream text; text << std::setprecision(3) << value; return text.str(); };
        auto options = soundings;
        const std::vector<double> scales{0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0};
        std::vector<InversionResult> results;
        for (std::size_t k = 0; k < scales.size(); ++k) {
            for (auto &o : options) {
                o.sci = sci;
                o.sci.adaptive = false;
                o.sci.stopAtHalfFit = false; // a partly fitted run would leave poorly fitting lines behind
                o.sci.verticalFactor = std::pow(sci.verticalFactor, scales[k]);
                o.sci.lateralFactor = std::pow(sci.lateralFactor, scales[k]);
            }
            const std::string step = "adaptive " + std::to_string(k + 1) + "/" + std::to_string(scales.size())
                + " (vertical " + factor(options.front().sci.verticalFactor) + ", lateral "
                + factor(options.front().sci.lateralFactor) + "): ";
            results = invertSci(options, eastings, northings, elevations, maximumThreads,
                [&](std::size_t s, int iteration, double rms, const std::string &detail) {
                    if (progress)
                        progress(s, iteration, rms, step + detail);
                }, cancelled);
            // Total RMS, so lines that fit poorly count (the median would hide them).
            double sum = 0.0, count = 0.0;
            for (const auto &result : results)
                if (!result.rmsHistory.empty()) {
                    sum += std::pow(result.rmsHistory.back(), 2) * result.predicted.size();
                    count += static_cast<double>(result.predicted.size());
                }
            if (count > 0.0 && std::sqrt(sum / count) <= 1.0)
                break;
            for (std::size_t s = 0; s < S; ++s) { // the next, looser run starts here
                options[s].model.resistivities = results[s].resistivities;
                options[s].halfSpaceStart = false;
            }
        }
        for (auto &result : results) {
            result.sci.adaptive = true;
            result.message = "SCI adaptive (vertical " + factor(result.sci.verticalFactor) + ", lateral "
                + factor(result.sci.lateralFactor) + "): " + result.message;
        }
        return results;
    }
    const auto &thicknesses = soundings.front().model.thicknesses;
    const std::size_t L = soundings.front().model.resistivities.size();
    for (const auto &options : soundings) {
        validate(options);
        if (options.model.thicknesses != thicknesses)
            throw std::invalid_argument("SCI soundings must share one layer grid");
    }
    const unsigned threads = maximumThreads > 0 ? maximumThreads : std::max(1u, std::thread::hardware_concurrency());
    auto checkCancelled = [&] {
        if (wasCancelled(cancelled))
            throw std::runtime_error("Inversion cancelled");
    };

    std::vector<InversionResult> results(S);
    std::vector<std::unique_ptr<JointProblem>> problems;
    for (std::size_t s = 0; s < S; ++s) {
        problems.push_back(std::make_unique<JointProblem>(soundings[s], cancelled, results[s]));
        problems.back()->threads = 1; // parallel over soundings instead
        problems.back()->linearJacobian = true; // d g / d ln(rho); log space rescales it below
    }
    std::vector<double> model(S * L);
    for (std::size_t s = 0; s < S; ++s)
        for (std::size_t j = 0; j < L; ++j)
            model[s * L + j] = std::log(soundings[s].model.resistivities[j]);
    const double lower = std::log(soundings.front().rhoMin), upper = std::log(soundings.front().rhoMax);

    // Constraint rows sum(c_k m_k) with weight 1/sigma^2, applied as C^T C:
    // vertical rows between adjacent layers, and lateral rows between Delaunay
    // neighbours. With elevation constraints each layer is compared with the
    // neighbour's layers it intersects at the same elevation, weighted by the
    // overlap; otherwise layer by layer at the same depth below the surface.
    struct Row {
        std::vector<std::pair<std::size_t, double>> terms;
        double weight;
        std::size_t a, b; // soundings involved
    };
    std::vector<Row> rows;
    const double vertical = 1.0 / std::pow(std::log(sci.verticalFactor), 2);
    for (std::size_t s = 0; s < S; ++s)
        for (std::size_t j = 0; j + 1 < L; ++j)
            rows.push_back({{{s * L + j, -1.0}, {s * L + j + 1, 1.0}}, vertical, s, s});
    // Layer j spans [tops[j], tops[j + 1]] below the surface; the half-space is
    // given the last layer's thickness on the own side and no bottom on the other.
    std::vector<double> tops{0.0};
    for (const double h : thicknesses)
        tops.push_back(tops.back() + h);
    const double halfSpace = thicknesses.empty() ? 1.0 : thicknesses.back();
    auto overlapTerms = [&](std::size_t q, double upper, double lower) {
        std::vector<std::pair<std::size_t, double>> terms;
        double total = 0.0;
        for (std::size_t k = 0; k < L; ++k) {
            const double overlap = std::min(lower, k + 1 < L ? tops[k + 1] : lower) - std::max(upper, tops[k]);
            if (overlap > 0.0) {
                terms.push_back({q * L + k, overlap});
                total += overlap;
            }
        }
        for (auto &term : terms)
            term.second /= total;
        return terms; // empty above the neighbour's ground surface
    };
    for (const auto &[a, b] : delaunayEdges(eastings, northings)) {
        const double distance = std::max(1.0, std::hypot(eastings[a] - eastings[b], northings[a] - northings[b]));
        const double sigma = std::log(sci.lateralFactor) * std::pow(distance / sci.referenceDistance, sci.distancePower);
        const double weight = 1.0 / (sigma * sigma);
        if (!sci.elevationConstraints) {
            for (std::size_t j = 0; j < L; ++j)
                rows.push_back({{{a * L + j, -1.0}, {b * L + j, 1.0}}, weight, a, b});
            continue;
        }
        // From each side, so the pair stays symmetric; on flat ground this equals the depth rows.
        for (const auto &[p, q] : {std::pair{a, b}, std::pair{b, a}})
            for (std::size_t j = 0; j < L; ++j) {
                const double shift = elevations[q] - elevations[p];
                auto terms = overlapTerms(q, tops[j] + shift, (j + 1 < L ? tops[j + 1] : tops[j] + halfSpace) + shift);
                if (terms.empty())
                    continue;
                terms.push_back({p * L + j, -1.0});
                rows.push_back({std::move(terms), 0.5 * weight, p, q});
            }
    }
    // Soundings above RMS 1 that stop improving are abandoned: their model is
    // frozen and they leave the data, constraint and misfit terms.
    std::vector<int> abandonedAt(S, 0);
    auto moves = [&](std::size_t s) { return abandonedAt[s] == 0; };
    const auto &inSystem = moves;
    auto addConstraints = [&](const std::vector<double> &x, std::vector<double> &y) { // y += C^T C x
        for (const auto &row : rows) {
            if (!inSystem(row.a) || !inSystem(row.b))
                continue;
            double value = 0.0;
            for (const auto &[k, c] : row.terms)
                value += c * x[k];
            for (const auto &[k, c] : row.terms)
                y[k] += row.weight * c * value;
        }
    };
    std::vector<Matrix> constraintBlocks(S, Matrix(L, std::vector<double>(L, 0.0))); // per-sounding part of C^T C

    // Predictions (and Jacobians) of every sounding; returns the data misfit sum of squares.
    std::vector<std::vector<double>> predicted(S);
    std::vector<Matrix> jacobians(S);
    std::vector<double> rms(S);
    // Residual of gate i in units of its noise. Linear space (as in Lupus):
    // (d - g) / sigma, defined when a trial model gives a negative response.
    // Log space: ln(d / g) d / sigma, continued linearly below g = d / 1000 so
    // a sign-changing response stays finite and differentiable.
    auto residual = [&](std::size_t s, std::size_t i, double g) {
        const double obs = problems[s]->observed[i], w = problems[s]->weights[i], floor = 1e-3 * obs;
        if (!sci.logDataSpace)
            return w * (obs - g) / obs;
        return g >= floor ? w * std::log(obs / g) : w * (std::log(obs / floor) + (floor - g) / floor);
    };
    // Per-sounding RMS in log space (whatever the fitting space), whose median
    // over the active soundings decides acceptance, so a few badly fitting
    // soundings cannot dominate it.
    auto logRms = [&](std::size_t s, const std::vector<double> &g) {
        double sum = 0.0;
        for (std::size_t i = 0; i < g.size(); ++i) {
            const double obs = problems[s]->observed[i], floor = 1e-3 * obs;
            sum += std::pow(problems[s]->weights[i]
                * (g[i] >= floor ? std::log(obs / g[i]) : std::log(obs / floor) + (floor - g[i]) / floor), 2);
        }
        return std::sqrt(sum / static_cast<double>(g.size()));
    };
    auto medianRms = [&](const std::vector<std::vector<double>> &g) {
        std::vector<double> values;
        for (std::size_t s = 0; s < S; ++s)
            if (moves(s))
                values.push_back(logRms(s, g[s]));
        if (values.empty())
            return 0.0;
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        return values[values.size() / 2];
    };
    auto forwardAll = [&](const std::vector<double> &x, std::vector<std::vector<double>> &out,
                          bool withJacobian, bool shared) {
        parallelFor(S, threads, [&](std::size_t s) {
            if (!moves(s))
                return;
            const std::vector<double> slice(x.begin() + s * L, x.begin() + (s + 1) * L);
            problems[s]->evaluate(thicknesses, slice, out[s], withJacobian ? &jacobians[s] : nullptr, shared);
        }, true);
    };

    // Start each sounding from its best homogeneous half-space: 33 resistivities
    // from 1 Ohm m to 10 kOhm m (8 per decade). A half-space step response
    // depends only on the system, so each is computed once per system and
    // shared by every sounding, which only gates it and scores its log RMS.
    if (soundings.front().halfSpaceStart) {
        std::vector<double> bestRms(S, std::numeric_limits<double>::infinity());
        for (int g = 0; g <= 32; ++g) {
            const std::vector<double> uniform(L, std::clamp(std::log(10.0) * g / 8.0, lower, upper));
            parallelFor(S, threads, [&](std::size_t s) {
                std::vector<double> response;
                problems[s]->evaluate(thicknesses, uniform, response, nullptr, true);
                const double value = logRms(s, response);
                if (value < bestRms[s]) {
                    bestRms[s] = value;
                    std::fill_n(model.begin() + s * L, L, uniform.front());
                }
            }, true);
        }
    }
    forwardAll(model, predicted, true, true);
    double measure = medianRms(predicted);
    for (std::size_t s = 0; s < S; ++s)
        results[s].rmsHistory.push_back(logRms(s, predicted[s]));

    double stepMax = sci.stepMax;
    int previousLevel = 1, iterations = 0;
    std::string termination = "maximum iterations";
    bool haveJacobian = true;

    for (int iteration = 1; iteration <= sci.maxIterations; ++iteration) {
        checkCancelled();
        if (!std::isfinite(measure)) {
            termination = "invalid response";
            break;
        }
        if (std::none_of(abandonedAt.begin(), abandonedAt.end(), [](int a) { return a == 0; })) {
            termination = "every sounding abandoned";
            break;
        }
        if (!haveJacobian)
            forwardAll(model, predicted, true, false);
        haveJacobian = false;
        for (auto &block : constraintBlocks)
            for (auto &line : block)
                std::fill(line.begin(), line.end(), 0.0);
        for (const auto &row : rows)
            if (inSystem(row.a) && inSystem(row.b))
                for (const auto &[i, ci] : row.terms)
                    for (const auto &[k, ck] : row.terms)
                        if (i / L == k / L)
                            constraintBlocks[i / L][i % L][k % L] += row.weight * ci * ck;

        // Block-diagonal Gauss-Newton term and gradient per sounding.
        std::vector<Matrix> blocks(S, Matrix(L, std::vector<double>(L, 0.0)));
        std::vector<double> rhs(S * L, 0.0);
        parallelFor(S, threads, [&](std::size_t s) {
            if (!moves(s))
                return; // abandoned: its step is zero
            const auto &J = jacobians[s];
            const auto &w = problems[s]->weights;
            const auto &obs = problems[s]->observed;
            for (std::size_t i = 0; i < J.size(); ++i) {
                // Row of d residual / d ln(rho) (sign aside) = weight * J.
                const double weight = sci.logDataSpace ? w[i] / std::max(predicted[s][i], 1e-3 * obs[i]) : w[i] / obs[i];
                const double r = residual(s, i, predicted[s][i]);
                for (std::size_t a = 0; a < L; ++a) {
                    rhs[s * L + a] += weight * J[i][a] * r;
                    for (std::size_t b = 0; b < L; ++b)
                        blocks[s][a][b] += weight * weight * J[i][a] * J[i][b];
                }
            }
        }, true);
        {
            std::vector<double> cm(S * L, 0.0);
            addConstraints(model, cm);
            for (std::size_t i = 0; i < rhs.size(); ++i)
                rhs[i] = moves(i / L) ? rhs[i] - cm[i] : 0.0;
        }
        double meanDiagonal = 0.0;
        for (std::size_t s = 0; s < S; ++s)
            for (std::size_t j = 0; j < L; ++j)
                meanDiagonal += blocks[s][j][j] + constraintBlocks[s][j][j];
        meanDiagonal /= static_cast<double>(S * L);

        // (G + C^T C + lambda I) delta = rhs by conjugate gradients with a
        // Cholesky block-Jacobi preconditioner (one block per sounding).
        auto solve = [&](double lambda) {
            std::vector<Matrix> factors(S);
            parallelFor(S, threads, [&](std::size_t s) {
                Matrix m = blocks[s];
                for (std::size_t j = 0; j < L; ++j)
                    for (std::size_t k = 0; k < L; ++k)
                        m[j][k] = moves(s) ? m[j][k] + constraintBlocks[s][j][k] + (j == k ? lambda : 0.0)
                                           : (j == k ? 1.0 : 0.0);
                for (std::size_t j = 0; j < L; ++j) { // in-place Cholesky, lower triangle
                    for (std::size_t k = 0; k < j; ++k)
                        m[j][j] -= m[j][k] * m[j][k];
                    m[j][j] = std::sqrt(std::max(m[j][j], 1e-300));
                    for (std::size_t i = j + 1; i < L; ++i) {
                        for (std::size_t k = 0; k < j; ++k)
                            m[i][j] -= m[i][k] * m[j][k];
                        m[i][j] /= m[j][j];
                    }
                }
                factors[s] = std::move(m);
            }, true);
            // The per-sounding parts of each CG iteration run in parallel; the
            // sparse constraint product (scattered writes) stays serial.
            auto precondition = [&](const std::vector<double> &r, std::vector<double> &z) {
                parallelFor(S, threads, [&](std::size_t s) {
                    const auto &f = factors[s];
                    double *v = &z[s * L];
                    for (std::size_t j = 0; j < L; ++j) {
                        v[j] = r[s * L + j];
                        for (std::size_t k = 0; k < j; ++k)
                            v[j] -= f[j][k] * v[k];
                        v[j] /= f[j][j];
                    }
                    for (std::size_t j = L; j-- > 0;) {
                        for (std::size_t k = j + 1; k < L; ++k)
                            v[j] -= f[k][j] * v[k];
                        v[j] /= f[j][j];
                    }
                });
            };
            auto apply = [&](const std::vector<double> &x, std::vector<double> &y) {
                parallelFor(S, threads, [&](std::size_t s) {
                    for (std::size_t a = 0; a < L; ++a) {
                        double v = lambda * x[s * L + a];
                        for (std::size_t b = 0; b < L; ++b)
                            v += blocks[s][a][b] * x[s * L + b];
                        y[s * L + a] = v;
                    }
                });
                addConstraints(x, y);
                for (std::size_t i = 0; i < y.size(); ++i)
                    if (!moves(i / L))
                        y[i] = x[i]; // abandoned unknowns: identity rows, and x stays 0 there
            };
            const std::size_t N = S * L;
            std::vector<double> x(N, 0.0), r = rhs, z(N), p(N), q(N);
            precondition(r, z);
            p = z;
            double rz = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
            const double target = 1e-10 * std::inner_product(rhs.begin(), rhs.end(), rhs.begin(), 0.0);
            for (int k = 0; k < 500 && std::inner_product(r.begin(), r.end(), r.begin(), 0.0) > target; ++k) {
                apply(p, q);
                const double alpha = rz / std::inner_product(p.begin(), p.end(), q.begin(), 0.0);
                for (std::size_t i = 0; i < N; ++i) {
                    x[i] += alpha * p[i];
                    r[i] -= alpha * q[i];
                }
                precondition(r, z);
                const double rzNext = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
                for (std::size_t i = 0; i < N; ++i)
                    p[i] = z[i] + rzNext / rz * p[i];
                rz = rzNext;
            }
            return x;
        };

        // Marquardt ladder (x3 per level), starting three levels below the
        // last accepted one; the first step within the step limit that lowers
        // the median log-space RMS enough is accepted.
        bool accepted = false, finished = false;
        int level = std::max(1, previousLevel - 3);
        std::vector<double> candidate(S * L);
        std::vector<std::vector<double>> candidatePredicted(S);
        double candidateMeasure = 0.0;
        for (; level <= 40 && !accepted && !finished; ++level) {
            checkCancelled();
            const auto delta = solve(1e-4 * meanDiagonal * std::pow(3.0, level - 1));
            double step = 0.0;
            for (double d : delta)
                step = std::max(step, std::abs(d));
            if (!std::isfinite(step) || step > stepMax)
                continue;
            for (std::size_t i = 0; i < candidate.size(); ++i)
                candidate[i] = std::clamp(model[i] + delta[i], lower, upper);
            candidatePredicted = predicted; // abandoned soundings keep theirs
            forwardAll(candidate, candidatePredicted, false, false);
            // A step that is right for the batch can overshoot a single sounding
            // badly (a gate response near a sign change). Such soundings back
            // off towards their current model, and stay there if still worse.
            parallelFor(S, threads, [&](std::size_t s) {
                if (!moves(s))
                    return;
                const double now = logRms(s, predicted[s]);
                for (int halving = 0; halving < 4 && !(logRms(s, candidatePredicted[s]) <= std::sqrt(2.0) * now); ++halving) {
                    for (std::size_t j = s * L; j < (s + 1) * L; ++j)
                        candidate[j] = 0.5 * (model[j] + candidate[j]);
                    const std::vector<double> slice(candidate.begin() + s * L, candidate.begin() + (s + 1) * L);
                    problems[s]->evaluate(thicknesses, slice, candidatePredicted[s], nullptr);
                }
                if (!(logRms(s, candidatePredicted[s]) <= std::sqrt(2.0) * now)) {
                    std::copy_n(model.begin() + s * L, L, candidate.begin() + s * L);
                    candidatePredicted[s] = predicted[s];
                }
            }, true);
            candidateMeasure = medianRms(candidatePredicted);
            const double change = (measure - candidateMeasure) / measure;
            if (std::isfinite(change) && change >= sci.relativeChangeThreshold) {
                accepted = true;
            } else if (stepMax > sci.stepMin) {
                stepMax = std::max(stepMax / sci.stepShrink, sci.stepMin);
            } else if (std::isfinite(change) && change > 0.0) {
                accepted = finished = true; // improving, but by less than the threshold
                termination = "minimum step reached";
            } // a worse fit at the minimum step: damp harder
        }
        if (!accepted && !finished) {
            termination = "no damping level improved the fit";
            break;
        }
        if (accepted) {
            ++iterations;
            previousLevel = level - 1;
            model = candidate;
            predicted = candidatePredicted;
            stepMax *= sci.stepGrowth;
            const double acceptedMedian = medianRms(predicted);
            for (std::size_t s = 0; s < S; ++s) {
                rms[s] = logRms(s, predicted[s]);
                results[s].rmsHistory.push_back(rms[s]);
                // Abandon a misfitting sounding that improved < 1% over 3 iterations.
                const auto &h = results[s].rmsHistory;
                if (moves(s) && rms[s] > rmsDisplayConvergenceThreshold && h.size() >= 4
                    && h[h.size() - 4] - rms[s] < 0.01 * h[h.size() - 4])
                    abandonedAt[s] = iterations;
            }
            measure = medianRms(predicted); // over the soundings still active
            const auto abandonedCount = std::count_if(abandonedAt.begin(), abandonedAt.end(), [](int a) { return a > 0; });
            // Total log-space data RMS of the soundings still in the system (reported).
            double sum = 0.0, count = 0.0;
            for (std::size_t s = 0; s < S; ++s)
                if (inSystem(s)) {
                    sum += rms[s] * rms[s] * problems[s]->observed.size();
                    count += problems[s]->observed.size();
                }
            const double totalRms = count > 0.0 ? std::sqrt(sum / count) : 0.0;
            if (sci.stopAtHalfFit && acceptedMedian <= 1.0) { // half the soundings fit: stop before the rest over-fit
                finished = true;
                termination = "median RMS below 1";
            }
            if (progress)
                for (std::size_t s = 0; s < S; ++s)
                    progress(s, iterations, rms[s], "SCI iteration, total RMS " + fixedTwoDecimals(totalRms)
                             + ", median RMS " + fixedTwoDecimals(acceptedMedian)
                             + "; " + std::to_string(S - abandonedCount) + " of " + std::to_string(S) + " soundings active"
                             + (abandonedAt[s] == iterations ? "; this sounding abandoned (no improvement)" : ""));
        }
        if (finished)
            break;
    }

    parallelFor(S, threads, [&](std::size_t s) {
        auto &result = results[s];
        const std::vector<double> slice(model.begin() + s * L, model.begin() + (s + 1) * L);
        result.iterations = iterations;
        result.sci = sci;
        result.predicted = predicted[s];
        result.resistivities.resize(L);
        std::transform(slice.begin(), slice.end(), result.resistivities.begin(), [](double v) { return std::exp(v); });
        result.converged = result.rmsHistory.back() <= rmsDisplayConvergenceThreshold;
        result.message = abandonedAt[s] > 0
            ? "SCI: abandoned after iteration " + std::to_string(abandonedAt[s]) + " (no improvement)"
            : "SCI: " + termination;
        problems[s]->linearJacobian = false;
        if (soundings[s].calculateSensitivity && std::isfinite(result.rmsHistory.back()))
            problems[s]->doi(soundings[s], slice, 3); // 3x refinement: the DOI cost of 10x, a third of it
        result.timing.totalSeconds = result.timing.jacobianSeconds + result.timing.alphaForwardSeconds
            + result.timing.sensitivitySeconds;
    }, true);
    return results;
}

InversionResult TemSolver::invert(const InversionOptions &options,
                                  const ProgressCallback &progress,
                                  const CancelCallback &cancelled)
{
    using Clock = std::chrono::steady_clock;
    const auto inversionStart = Clock::now();
    auto elapsedSeconds = [](const Clock::time_point &start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    validate(options);
    InversionResult result;
    std::vector<double> model(options.model.resistivities.size());
    std::transform(options.model.resistivities.begin(), options.model.resistivities.end(), model.begin(),
                   [](double value) { return std::log(value); });
    const double lower = std::log(options.rhoMin);
    const double upper = std::log(options.rhoMax);
    std::vector<double> observed = options.observed;
    std::vector<double> noiseStd = options.noiseStd;
    for (const auto &dataSet : options.additionalDataSets) {
        observed.insert(observed.end(), dataSet.observed.begin(), dataSet.observed.end());
        noiseStd.insert(noiseStd.end(), dataSet.noiseStd.begin(), dataSet.noiseStd.end());
    }
    std::vector<double> weights(observed.size());
    for (std::size_t i = 0; i < weights.size(); ++i)
        weights[i] = observed[i] / noiseStd[i];
    const auto initialForwardStart = Clock::now();
    auto predicted = forwardLogModel(options, model, cancelled);
    result.timing.initialForwardSeconds = elapsedSeconds(initialForwardStart);
    double currentRms = rms(observed, predicted, weights);
    double alphaStart = 1.0;
    bool adaptiveAlphaActive = options.adaptiveAlphaSearch;
    Matrix broydenJacobian;
    bool broydenJacobianValid = false;
    int broydenAge = 0;

    for (int iteration = 0; iteration < options.maxIterations; ++iteration) {
        if (wasCancelled(cancelled))
            throw std::runtime_error("Inversion cancelled");
        result.rmsHistory.push_back(currentRms);
        const bool useBroyden = options.jacobianUpdateMethod
                == JacobianUpdateMethod::BroydenRankOne
            && broydenJacobianValid
            && broydenAge < options.broydenRefreshInterval;
        if (progress)
            progress(iteration + 1, currentRms,
                     useBroyden
                         ? "Using guarded Broyden Jacobian update"
                     : iteration == 0 && options.cacheFirstJacobian
                         ? "Loading or building shared first Jacobian"
                         : (options.jacobianMethod == JacobianMethod::Analytical
                             ? "Building analytical Jacobian" : "Building finite-difference Jacobian"));
        if (currentRms <= rmsDisplayConvergenceThreshold) {
            result.converged = true;
            break;
        }

        Matrix jacobian;
        if (useBroyden) {
            jacobian = broydenJacobian;
        } else {
            const auto jacobianStart = Clock::now();
            if (iteration == 0 && options.cacheFirstJacobian) {
            const auto cached = JacobianCache::loadOrCompute(
                firstJacobianSignature(options, model),
                options.jacobianCacheDirectory,
                [&]() { return buildJacobian(options, model, predicted, cancelled); });
            jacobian = cached.matrix;
            if (cached.source == JacobianCacheSource::Memory)
                result.firstJacobianSource = "memory cache";
            else if (cached.source == JacobianCacheSource::Disk)
                result.firstJacobianSource = "disk cache";
            else
                result.firstJacobianSource = "computed and cached";
            if (progress)
                progress(iteration + 1, currentRms,
                         "First Jacobian: " + result.firstJacobianSource);
            } else {
                jacobian = buildJacobian(options, model, predicted, cancelled);
                if (iteration == 0)
                    result.firstJacobianSource = "computed (cache disabled)";
            }
            result.timing.jacobianSeconds += elapsedSeconds(jacobianStart);
            ++result.timing.jacobianEvaluations;
            if (options.jacobianUpdateMethod
                == JacobianUpdateMethod::BroydenRankOne) {
                broydenJacobian = jacobian;
                broydenJacobianValid = true;
                broydenAge = 0;
            }
        }
        const Matrix regularization = roughness(model, options.regularizationNorm);
        std::vector<double> weightedResidual(observed.size());
        for (std::size_t i = 0; i < observed.size(); ++i)
            weightedResidual[i] = weights[i] * (std::log(observed[i]) - std::log(predicted[i]));

        if (iteration == 0) {
            alphaStart = 1.0e-30;
            for (std::size_t parameter = 0; parameter < model.size(); ++parameter) {
                double gradient = 0.0;
                for (std::size_t row = 0; row < jacobian.size(); ++row)
                    gradient += jacobian[row][parameter] * weights[row] * weightedResidual[row];
                alphaStart = std::max(alphaStart, std::abs(gradient));
            }
        }

        struct AlphaTrial {
            double alpha = 0.0;
            double rms = std::numeric_limits<double>::infinity();
            std::vector<double> model;
            std::vector<double> prediction;
        };
        std::vector<AlphaTrial> trials;
        bool reachedTarget = false;
        bool undershotTarget = false;
        bool targetFitApplied = false;
        bool strengtheningAlpha = false;
        double nextAdaptiveAlpha = boundedAlpha(alphaStart);

        for (int trialIndex = 0; trialIndex < options.alphaSteps; ++trialIndex) {
            if (wasCancelled(cancelled))
                throw std::runtime_error("Inversion cancelled");
            const double alpha = adaptiveAlphaActive
                ? nextAdaptiveAlpha
                : boundedAlpha(alphaStart * std::pow(
                      10.0, -options.alphaLogStep * trialIndex));
            const auto linearSolveStart = Clock::now();
            const auto delta = gaussNewtonStep(
                jacobian, weightedResidual, weights, regularization,
                model, alpha, options.model.thicknesses);
            result.timing.linearSolveSeconds
                += elapsedSeconds(linearSolveStart);
            double scale = 1.0;
            std::vector<double> trial(model.size());
            for (int backtrack = 0; backtrack < 11; ++backtrack) {
                for (std::size_t i = 0; i < model.size(); ++i)
                    trial[i] = model[i] + scale * delta[i];
                if (std::all_of(trial.begin(), trial.end(), [&](double value) { return value >= lower && value <= upper; }))
                    break;
                scale *= 0.5;
            }
            for (double &value : trial)
                value = std::clamp(value, lower, upper);
            const auto alphaForwardStart = Clock::now();
            const auto trialPrediction = forwardLogModel(options, trial, cancelled);
            result.timing.alphaForwardSeconds += elapsedSeconds(alphaForwardStart);
            ++result.timing.alphaTrialForwards;
            const double trialRms = rms(observed, trialPrediction, weights);
            if (!std::isfinite(trialRms)) {
                if (progress)
                    progress(iteration + 1, currentRms,
                             adaptiveAlphaActive
                                 ? "Invalid alpha trial; increasing regularization"
                                 : "Invalid alpha trial rejected");
                if (adaptiveAlphaActive && !strengtheningAlpha) {
                    strengtheningAlpha = true;
                    nextAdaptiveAlpha = boundedAlpha(
                        alpha * strongerAlphaFactor(options.alphaLogStep));
                } else if (adaptiveAlphaActive) {
                    // This sounding alone returns to the original fixed sweep.
                    adaptiveAlphaActive = false;
                }
                continue;
            }
            if (progress)
                progress(iteration + 1, trialRms,
                         "Alpha " + fixedTwoDecimals(alpha)
                         + ", step " + fixedTwoDecimals(scale));
            trials.push_back({alpha, trialRms, std::move(trial),
                              std::move(trialPrediction)});
            if (trialRms <= rmsDisplayConvergenceThreshold) {
                reachedTarget = true;
                undershotTarget = trialRms < 1.0;
                if (progress)
                    progress(iteration + 1, trialRms,
                             undershotTarget
                                 ? "RMS below 1.00; fitting RMS=1.00 target"
                                 : "RMS rounds to 1.00; accepting as converged");
                break;
            }
            if (adaptiveAlphaActive && strengtheningAlpha) {
                if (trialRms < currentRms)
                    break;
                adaptiveAlphaActive = false;
                continue;
            }
            if (trials.size() > 1
                && trials.back().rms > trials[trials.size() - 2].rms) {
                double priorBest = std::numeric_limits<double>::infinity();
                for (std::size_t i = 0; i + 1 < trials.size(); ++i)
                    priorBest = std::min(priorBest, trials[i].rms);
                if (priorBest < currentRms) {
                    if (progress)
                        progress(iteration + 1, trialRms,
                                 "RMS increased; accepting the best trial and advancing iteration");
                    break;
                }
            }
            if (adaptiveAlphaActive) {
                if (trialRms < currentRms) {
                    nextAdaptiveAlpha = boundedAlpha(
                        alpha / weakerAlphaFactor(options.alphaLogStep));
                } else {
                    strengtheningAlpha = true;
                    nextAdaptiveAlpha = boundedAlpha(
                        alpha * strongerAlphaFactor(options.alphaLogStep));
                }
            }
        }

        if (trials.empty()) {
            result.message = "No regularized update improved the data fit";
            break;
        }

        if (undershotTarget) {
            if (trials.size() == 1) {
                // There is no alpha bracket when the first trial undershoots.
                // Interpolate the model update from the current RMS to the
                // undershoot and perform exactly one target forward model.
                const auto &undershoot = trials.front();
                const double denominator = currentRms - undershoot.rms;
                const double fraction = denominator > 1.0e-14
                    ? std::clamp((currentRms - 1.0) / denominator, 0.0, 1.0)
                    : 1.0;
                std::vector<double> fittedModel(model.size());
                for (std::size_t parameter = 0; parameter < model.size(); ++parameter)
                    fittedModel[parameter] = std::clamp(
                        model[parameter]
                            + fraction * (undershoot.model[parameter] - model[parameter]),
                        lower, upper);
                const auto alphaForwardStart = Clock::now();
                auto fittedPrediction = forwardLogModel(
                    options, fittedModel, cancelled);
                result.timing.alphaForwardSeconds += elapsedSeconds(alphaForwardStart);
                ++result.timing.alphaTrialForwards;
                const double fittedRms = rms(observed, fittedPrediction, weights);
                if (std::isfinite(fittedRms)) {
                    trials.push_back({undershoot.alpha, fittedRms,
                                      std::move(fittedModel),
                                      std::move(fittedPrediction)});
                    targetFitApplied = true;
                }
                if (progress)
                    progress(iteration + 1, fittedRms,
                             "Single-step RMS=1 fit from first-alpha undershoot, fraction "
                             + fixedTwoDecimals(fraction)
                             + ", actual RMS " + fixedTwoDecimals(fittedRms));
            } else {
                // Use only the local points around the crossing. Distant alpha
                // trials can bend a global quadratic away from the RMS=1 root.
                const std::size_t first = trials.size() > 3
                    ? trials.size() - 3 : 0;
                std::vector<double> alphaHistory;
                std::vector<double> rmsHistory;
                for (std::size_t index = first; index < trials.size(); ++index) {
                    alphaHistory.push_back(trials[index].alpha);
                    rmsHistory.push_back(trials[index].rms);
                }
                const double fittedAlpha = fittedTargetAlpha(
                    alphaHistory, rmsHistory);
                if (std::isfinite(fittedAlpha) && fittedAlpha > 0.0) {
                    const auto linearSolveStart = Clock::now();
                    const auto delta = gaussNewtonStep(
                        jacobian, weightedResidual, weights, regularization, model,
                        fittedAlpha, options.model.thicknesses);
                    result.timing.linearSolveSeconds += elapsedSeconds(linearSolveStart);
                    double scale = 1.0;
                    std::vector<double> fittedModel(model.size());
                    for (int backtrack = 0; backtrack < 11; ++backtrack) {
                        for (std::size_t i = 0; i < model.size(); ++i)
                            fittedModel[i] = model[i] + scale * delta[i];
                        if (std::all_of(fittedModel.begin(), fittedModel.end(),
                                        [&](double value) {
                                            return value >= lower && value <= upper;
                                        }))
                            break;
                        scale *= 0.5;
                    }
                    for (double &value : fittedModel)
                        value = std::clamp(value, lower, upper);
                    const auto alphaForwardStart = Clock::now();
                    auto fittedPrediction = forwardLogModel(
                        options, fittedModel, cancelled);
                    result.timing.alphaForwardSeconds += elapsedSeconds(alphaForwardStart);
                    ++result.timing.alphaTrialForwards;
                    const double fittedRms = rms(
                        observed, fittedPrediction, weights);
                    if (std::isfinite(fittedRms)) {
                        trials.push_back({fittedAlpha, fittedRms,
                                          std::move(fittedModel),
                                          std::move(fittedPrediction)});
                        targetFitApplied = true;
                    }
                    if (progress)
                        progress(iteration + 1, fittedRms,
                                 "Single parabolic RMS=1 fit, alpha "
                                 + fixedTwoDecimals(fittedAlpha)
                                 + ", actual RMS " + fixedTwoDecimals(fittedRms));
                }
            }
        }

        std::size_t selected = 0;
        if (reachedTarget) {
            selected = trials.size() - 1;
        } else {
            for (std::size_t i = 1; i < trials.size(); ++i)
                if (trials[i].rms < trials[selected].rms)
                    selected = i;
        }
        if (!(trials[selected].rms < currentRms)) {
            result.message = "No regularized update improved the data fit";
            break;
        }
        if (options.jacobianUpdateMethod
            == JacobianUpdateMethod::BroydenRankOne) {
            const auto broydenStart = Clock::now();
            Matrix updated = jacobian;
            broydenJacobianValid = applyBroydenRankOneUpdate(
                updated, model, trials[selected].model,
                predicted, trials[selected].prediction);
            result.timing.broydenUpdateSeconds
                += elapsedSeconds(broydenStart);
            if (broydenJacobianValid) {
                broydenJacobian = std::move(updated);
                ++broydenAge;
                ++result.timing.broydenUpdates;
            } else {
                broydenAge = options.broydenRefreshInterval;
                if (progress)
                    progress(iteration + 1, trials[selected].rms,
                             "Broyden guard requested a full Jacobian");
            }
        }
        model = std::move(trials[selected].model);
        predicted = std::move(trials[selected].prediction);
        currentRms = trials[selected].rms;
        alphaStart = boundedAlpha(trials[selected].alpha
            * weakerAlphaFactor(options.alphaLogStep));
        result.iterations = iteration + 1;
        if (reachedTarget) {
            result.message = targetFitApplied
                ? "Single-step RMS=1 target fit applied"
                : "RMS rounds to 1.00; converged";
            break;
        }
    }

    if (result.rmsHistory.empty() || std::abs(result.rmsHistory.back() - currentRms) > 1.0e-12)
        result.rmsHistory.push_back(currentRms);
    result.converged = result.converged
        || currentRms <= rmsDisplayConvergenceThreshold;
    result.predicted = predicted;
    result.resistivities.resize(model.size());
    std::transform(model.begin(), model.end(), result.resistivities.begin(),
                   [](double value) { return std::exp(value); });

    if (options.calculateSensitivity && !wasCancelled(cancelled)) {
        const auto sensitivityStart = Clock::now();
        const Matrix jacobian = buildJacobian(options, model, predicted, cancelled);
        result.sensitivity.assign(model.size(), 0.0);
        for (std::size_t parameter = 0; parameter < model.size(); ++parameter) {
            for (std::size_t row = 0; row < jacobian.size(); ++row) {
                const double weighted = jacobian[row][parameter] * weights[row];
                result.sensitivity[parameter] += weighted * weighted;
            }
            result.sensitivity[parameter] = std::sqrt(result.sensitivity[parameter]);
        }
        result.timing.sensitivitySeconds = elapsedSeconds(sensitivityStart);
    }
    if (result.message.empty())
        result.message = result.converged ? "Target RMS reached" : "Maximum iterations reached";
    result.timing.totalSeconds = elapsedSeconds(inversionStart);
    return result;
}

std::vector<InversionResult> TemSolver::invertIndependentBatch(
    const std::vector<InversionOptions> &inputOptions,
    unsigned maximumThreads, const BatchProgressCallback &progress,
    const CancelCallback &cancelled,
    const BatchResultCallback &completed)
{
    using Clock = std::chrono::steady_clock;
    const auto batchStart = Clock::now();
    auto elapsedSeconds = [](const Clock::time_point &start) {
        return std::chrono::duration<double>(Clock::now() - start).count();
    };
    if (inputOptions.empty())
        return {};
    if (maximumThreads == 0)
        maximumThreads = std::max(1u, std::thread::hardware_concurrency());

    struct State {
        InversionOptions options;
        InversionResult result;
        std::vector<double> model;
        std::vector<double> observed;
        std::vector<double> noiseStd;
        std::vector<double> weights;
        std::vector<double> predicted;
        Matrix broydenJacobian;
        double currentRms = std::numeric_limits<double>::infinity();
        double alphaStart = 1.0;
        double lower = 0.0;
        double upper = 0.0;
        int iteration = 0;
        int broydenAge = 0;
        bool adaptiveAlphaActive = true;
        bool broydenJacobianValid = false;
        bool done = false;
        bool finalized = false;
    };
    struct AlphaTrial {
        double alpha = 0.0;
        double rms = std::numeric_limits<double>::infinity();
        std::vector<double> model;
        std::vector<double> prediction;
    };
    struct IterationWork {
        Matrix jacobian;
        Matrix regularization;
        std::vector<double> weightedResidual;
        std::vector<AlphaTrial> trials;
        bool reachedTarget = false;
        bool undershotTarget = false;
        bool targetFitApplied = false;
        bool strengtheningAlpha = false;
        double nextAdaptiveAlpha = 1.0;
    };

    std::vector<State> states(inputOptions.size());
    std::vector<InversionOptions> batchOptions;
    std::vector<std::vector<double>> batchModels;
    batchOptions.reserve(inputOptions.size());
    batchModels.reserve(inputOptions.size());
    for (std::size_t index = 0; index < inputOptions.size(); ++index) {
        validate(inputOptions[index]);
        auto &state = states[index];
        state.options = inputOptions[index];
        state.adaptiveAlphaActive = state.options.adaptiveAlphaSearch;
        state.options.model.maxThreads = 1;
        for (auto &dataSet : state.options.additionalDataSets)
            dataSet.model.maxThreads = 1;
        state.model.resize(state.options.model.resistivities.size());
        std::transform(state.options.model.resistivities.begin(),
                       state.options.model.resistivities.end(),
                       state.model.begin(),
                       [](double value) { return std::log(value); });
        state.lower = std::log(state.options.rhoMin);
        state.upper = std::log(state.options.rhoMax);
        state.observed = state.options.observed;
        state.noiseStd = state.options.noiseStd;
        for (const auto &dataSet : state.options.additionalDataSets) {
            state.observed.insert(state.observed.end(),
                                  dataSet.observed.begin(),
                                  dataSet.observed.end());
            state.noiseStd.insert(state.noiseStd.end(),
                                  dataSet.noiseStd.begin(),
                                  dataSet.noiseStd.end());
        }
        state.weights.resize(state.observed.size());
        for (std::size_t row = 0; row < state.weights.size(); ++row)
            state.weights[row] = state.observed[row] / state.noiseStd[row];
        batchOptions.push_back(state.options);
        batchModels.push_back(state.model);
    }

    std::vector<Matrix> loadedFirstJacobians(states.size());
    std::vector<std::string> loadedFirstSources(states.size());
    std::vector<bool> needSharedJacobian(states.size(), false);
    for (std::size_t index = 0; index < states.size(); ++index) {
        if (states[index].options.jacobianMethod
                != JacobianMethod::Analytical)
            continue;
        needSharedJacobian[index] = true;
        if (!states[index].options.cacheFirstJacobian)
            continue;
        JacobianCacheResult cached;
        if (JacobianCache::tryLoad(
                firstJacobianSignature(states[index].options,
                                       states[index].model),
                states[index].options.jacobianCacheDirectory, cached)) {
            loadedFirstJacobians[index] = std::move(cached.matrix);
            loadedFirstSources[index]
                = cached.source == JacobianCacheSource::Memory
                ? "memory cache" : "disk cache";
            needSharedJacobian[index] = false;
        }
    }

    const auto initialForwardStart = Clock::now();
    SharedInitialBatch sharedInitial = sharedInitialAnalyticalBatch(
        batchOptions, batchModels, needSharedJacobian,
        maximumThreads, cancelled);
    std::vector<std::vector<double>> initialPredictions(states.size());
    std::vector<std::size_t> fallbackIndices;
    std::vector<InversionOptions> fallbackOptions;
    std::vector<std::vector<double>> fallbackModels;
    for (std::size_t index = 0; index < states.size(); ++index) {
        if (sharedInitial.available[index]) {
            initialPredictions[index] = sharedInitial.responses[index];
        } else {
            fallbackIndices.push_back(index);
            fallbackOptions.push_back(batchOptions[index]);
            fallbackModels.push_back(batchModels[index]);
        }
    }
    if (!fallbackIndices.empty()) {
        auto fallbackPredictions = forwardLogModelsBatch(
            fallbackOptions, fallbackModels, maximumThreads, cancelled);
        for (std::size_t index = 0; index < fallbackIndices.size(); ++index)
            initialPredictions[fallbackIndices[index]]
                = std::move(fallbackPredictions[index]);
    }
    const double initialForwardSeconds = elapsedSeconds(initialForwardStart);
    for (std::size_t index = 0; index < states.size(); ++index) {
        states[index].predicted = std::move(initialPredictions[index]);
        states[index].currentRms = rms(
            states[index].observed, states[index].predicted,
            states[index].weights);
        states[index].result.timing.initialForwardSeconds
            = initialForwardSeconds;
    }

    auto finalizeReady = [&]() {
        std::vector<std::size_t> ready;
        for (std::size_t index = 0; index < states.size(); ++index)
            if (states[index].done && !states[index].finalized)
                ready.push_back(index);
        parallelFor(ready.size(), maximumThreads,
            [&](std::size_t position) {
                const std::size_t index = ready[position];
                auto &state = states[index];
                if (state.result.rmsHistory.empty()
                    || std::abs(state.result.rmsHistory.back()
                                - state.currentRms) > 1.0e-12)
                    state.result.rmsHistory.push_back(state.currentRms);
                state.result.converged = state.result.converged
                    || state.currentRms <= rmsDisplayConvergenceThreshold;
                state.result.predicted = state.predicted;
                state.result.resistivities.resize(state.model.size());
                std::transform(state.model.begin(), state.model.end(),
                               state.result.resistivities.begin(),
                               [](double value) { return std::exp(value); });
                if (state.options.calculateSensitivity
                    && !wasCancelled(cancelled)) {
                    const auto sensitivityStart = Clock::now();
                    try {
                        const Matrix jacobian = buildJacobian(
                            state.options, state.model, state.predicted,
                            cancelled);
                        state.result.sensitivity.assign(
                            state.model.size(), 0.0);
                        for (std::size_t parameter = 0;
                             parameter < state.model.size(); ++parameter) {
                            for (std::size_t row = 0;
                                 row < jacobian.size(); ++row) {
                                const double weighted
                                    = jacobian[row][parameter]
                                    * state.weights[row];
                                state.result.sensitivity[parameter]
                                    += weighted * weighted;
                            }
                            state.result.sensitivity[parameter] = std::sqrt(
                                state.result.sensitivity[parameter]);
                        }
                        state.result.timing.sensitivitySeconds
                            = elapsedSeconds(sensitivityStart);
                    } catch (...) {
                        if (!wasCancelled(cancelled))
                            throw;
                        // The inversion result is already complete. A kill
                        // request may omit only the optional final DOI pass.
                    }
                }
                if (state.result.message.empty())
                    state.result.message = state.result.converged
                        ? "Target RMS reached"
                        : "Maximum iterations reached";
                state.result.timing.totalSeconds
                    = elapsedSeconds(batchStart);
                state.finalized = true;
                if (completed)
                    completed(index, state.result);
            }, true);
    };

    while (true) {
        if (wasCancelled(cancelled))
            throw std::runtime_error("Inversion cancelled");
        std::vector<std::size_t> active;
        for (std::size_t index = 0; index < states.size(); ++index) {
            auto &state = states[index];
            if (state.done)
                continue;
            if (state.iteration >= state.options.maxIterations) {
                state.done = true;
                continue;
            }
            state.result.rmsHistory.push_back(state.currentRms);
            const bool useBroyden = state.options.jacobianUpdateMethod
                    == JacobianUpdateMethod::BroydenRankOne
                && state.broydenJacobianValid
                && state.broydenAge
                    < state.options.broydenRefreshInterval;
            if (progress)
                progress(index, state.iteration + 1, state.currentRms,
                    useBroyden
                        ? "Using guarded Broyden Jacobian update"
                    : state.iteration == 0 && state.options.cacheFirstJacobian
                        ? "Loading or building shared first Jacobian"
                        : (state.options.jacobianMethod
                                == JacobianMethod::Analytical
                            ? "Building batched analytical Jacobian"
                            : "Building batched finite-difference Jacobian"));
            if (state.currentRms <= rmsDisplayConvergenceThreshold) {
                state.result.converged = true;
                state.done = true;
            } else {
                active.push_back(index);
            }
        }
        finalizeReady();
        if (active.empty())
            break;

        std::vector<IterationWork> work(states.size());
        parallelFor(active.size(), maximumThreads, [&](std::size_t position) {
            const std::size_t index = active[position];
            auto &state = states[index];
            auto &iteration = work[index];
            const bool useBroyden = state.options.jacobianUpdateMethod
                    == JacobianUpdateMethod::BroydenRankOne
                && state.broydenJacobianValid
                && state.broydenAge
                    < state.options.broydenRefreshInterval;
            if (useBroyden) {
                iteration.jacobian = state.broydenJacobian;
                return;
            }
            const auto jacobianStart = Clock::now();
            if (state.iteration == 0
                && !loadedFirstJacobians[index].empty()) {
                iteration.jacobian
                    = std::move(loadedFirstJacobians[index]);
                state.result.firstJacobianSource
                    = loadedFirstSources[index];
                if (progress)
                    progress(index, state.iteration + 1, state.currentRms,
                             "First Jacobian: "
                                 + state.result.firstJacobianSource);
            } else if (state.iteration == 0
                       && sharedInitial.jacobianAvailable[index]) {
                if (state.options.cacheFirstJacobian) {
                    const auto cached = JacobianCache::loadOrCompute(
                        firstJacobianSignature(state.options, state.model),
                        state.options.jacobianCacheDirectory,
                        [&]() { return sharedInitial.jacobians[index]; });
                    iteration.jacobian = cached.matrix;
                    if (cached.source == JacobianCacheSource::Memory)
                        state.result.firstJacobianSource = "memory cache";
                    else if (cached.source == JacobianCacheSource::Disk)
                        state.result.firstJacobianSource = "disk cache";
                    else
                        state.result.firstJacobianSource
                            = sharedInitial.sharedAcrossSoundings[index]
                            ? "shared full-system batch and cached"
                            : "computed and cached";
                } else {
                    iteration.jacobian
                        = std::move(sharedInitial.jacobians[index]);
                    state.result.firstJacobianSource
                        = sharedInitial.sharedAcrossSoundings[index]
                        ? "shared full-system batch"
                        : "computed (cache disabled)";
                }
                if (progress)
                    progress(index, state.iteration + 1, state.currentRms,
                             "First Jacobian: "
                                 + state.result.firstJacobianSource);
            } else if (state.iteration == 0
                       && state.options.cacheFirstJacobian) {
                const auto cached = JacobianCache::loadOrCompute(
                    firstJacobianSignature(state.options, state.model),
                    state.options.jacobianCacheDirectory,
                    [&]() {
                        return buildJacobian(state.options, state.model,
                                             state.predicted, cancelled);
                    });
                iteration.jacobian = cached.matrix;
                if (cached.source == JacobianCacheSource::Memory)
                    state.result.firstJacobianSource = "memory cache";
                else if (cached.source == JacobianCacheSource::Disk)
                    state.result.firstJacobianSource = "disk cache";
                else
                    state.result.firstJacobianSource = "computed and cached";
                if (progress)
                    progress(index, state.iteration + 1, state.currentRms,
                             "First Jacobian: "
                                 + state.result.firstJacobianSource);
            } else {
                iteration.jacobian = buildJacobian(
                    state.options, state.model, state.predicted, cancelled);
                if (state.iteration == 0)
                    state.result.firstJacobianSource
                        = "computed (cache disabled)";
            }
            state.result.timing.jacobianSeconds
                += elapsedSeconds(jacobianStart);
            ++state.result.timing.jacobianEvaluations;
            if (state.options.jacobianUpdateMethod
                == JacobianUpdateMethod::BroydenRankOne) {
                state.broydenJacobian = iteration.jacobian;
                state.broydenJacobianValid = true;
                state.broydenAge = 0;
            }
        }, true);

        for (const std::size_t index : active) {
            auto &state = states[index];
            auto &iteration = work[index];
            iteration.regularization = roughness(
                state.model, state.options.regularizationNorm);
            iteration.weightedResidual.resize(state.observed.size());
            for (std::size_t row = 0; row < state.observed.size(); ++row) {
                iteration.weightedResidual[row] = state.weights[row]
                    * (std::log(state.observed[row])
                       - std::log(state.predicted[row]));
            }
            if (state.iteration == 0) {
                state.alphaStart = 1.0e-30;
                for (std::size_t parameter = 0;
                     parameter < state.model.size(); ++parameter) {
                    double gradient = 0.0;
                    for (std::size_t row = 0;
                         row < iteration.jacobian.size(); ++row) {
                        gradient += iteration.jacobian[row][parameter]
                            * state.weights[row]
                            * iteration.weightedResidual[row];
                    }
                    state.alphaStart = std::max(
                        state.alphaStart, std::abs(gradient));
                }
            }
            iteration.nextAdaptiveAlpha = boundedAlpha(state.alphaStart);
        }

        std::vector<std::size_t> searching = active;
        for (int trialIndex = 0; !searching.empty(); ++trialIndex) {
            std::vector<std::size_t> candidateStates;
            std::vector<InversionOptions> candidateOptions;
            std::vector<std::vector<double>> candidateModels;
            std::vector<double> candidateAlphas;
            std::vector<double> candidateScales;
            for (const std::size_t index : searching) {
                auto &state = states[index];
                if (trialIndex >= state.options.alphaSteps)
                    continue;
                auto &iteration = work[index];
                const double alpha = state.adaptiveAlphaActive
                    ? iteration.nextAdaptiveAlpha
                    : boundedAlpha(state.alphaStart * std::pow(
                          10.0,
                          -state.options.alphaLogStep * trialIndex));
                const auto linearSolveStart = Clock::now();
                const auto delta = gaussNewtonStep(
                    iteration.jacobian, iteration.weightedResidual,
                    state.weights, iteration.regularization,
                    state.model, alpha,
                    state.options.model.thicknesses);
                state.result.timing.linearSolveSeconds
                    += elapsedSeconds(linearSolveStart);
                double scale = 1.0;
                std::vector<double> trial(state.model.size());
                for (int backtrack = 0; backtrack < 11; ++backtrack) {
                    for (std::size_t parameter = 0;
                         parameter < state.model.size(); ++parameter)
                        trial[parameter] = state.model[parameter]
                            + scale * delta[parameter];
                    if (std::all_of(trial.begin(), trial.end(),
                            [&](double value) {
                                return value >= state.lower
                                    && value <= state.upper;
                            }))
                        break;
                    scale *= 0.5;
                }
                for (double &value : trial)
                    value = std::clamp(value, state.lower, state.upper);
                candidateStates.push_back(index);
                candidateOptions.push_back(state.options);
                candidateModels.push_back(std::move(trial));
                candidateAlphas.push_back(alpha);
                candidateScales.push_back(scale);
            }
            if (candidateStates.empty())
                break;
            const auto alphaForwardStart = Clock::now();
            auto predictions = forwardLogModelsBatch(
                candidateOptions, candidateModels, maximumThreads,
                cancelled);
            const double alphaForwardSeconds
                = elapsedSeconds(alphaForwardStart);
            std::vector<std::size_t> nextSearching;
            for (std::size_t candidate = 0;
                 candidate < candidateStates.size(); ++candidate) {
                const std::size_t index = candidateStates[candidate];
                auto &state = states[index];
                auto &iteration = work[index];
                state.result.timing.alphaForwardSeconds
                    += alphaForwardSeconds;
                ++state.result.timing.alphaTrialForwards;
                const double trialRms = rms(
                    state.observed, predictions[candidate], state.weights);
                if (!std::isfinite(trialRms)) {
                    if (progress)
                        progress(index, state.iteration + 1,
                            state.currentRms,
                            state.adaptiveAlphaActive
                                ? "Invalid alpha trial; increasing regularization"
                                : "Invalid alpha trial rejected");
                    if (state.adaptiveAlphaActive
                        && !iteration.strengtheningAlpha) {
                        iteration.strengtheningAlpha = true;
                        iteration.nextAdaptiveAlpha = boundedAlpha(
                            candidateAlphas[candidate]
                            * strongerAlphaFactor(
                                state.options.alphaLogStep));
                    } else if (state.adaptiveAlphaActive) {
                        state.adaptiveAlphaActive = false;
                        if (progress)
                            progress(index, state.iteration + 1,
                                state.currentRms,
                                "Adaptive alpha guard activated; using fixed search");
                    }
                    if (trialIndex + 1 < state.options.alphaSteps)
                        nextSearching.push_back(index);
                    continue;
                }
                if (progress)
                    progress(index, state.iteration + 1, trialRms,
                        "Batched alpha "
                            + fixedTwoDecimals(candidateAlphas[candidate])
                            + ", step "
                            + fixedTwoDecimals(candidateScales[candidate]));
                iteration.trials.push_back({
                    candidateAlphas[candidate], trialRms,
                    std::move(candidateModels[candidate]),
                    std::move(predictions[candidate])});
                if (trialRms <= rmsDisplayConvergenceThreshold) {
                    iteration.reachedTarget = true;
                    iteration.undershotTarget = trialRms < 1.0;
                    if (progress)
                        progress(index, state.iteration + 1, trialRms,
                            iteration.undershotTarget
                                ? "RMS below 1.00; fitting RMS=1.00 target"
                                : "RMS rounds to 1.00; accepting as converged");
                    continue;
                }
                if (state.adaptiveAlphaActive
                    && iteration.strengtheningAlpha) {
                    if (trialRms < state.currentRms)
                        continue;
                    state.adaptiveAlphaActive = false;
                    if (progress)
                        progress(index, state.iteration + 1,
                            state.currentRms,
                            "Adaptive alpha guard activated; using fixed search");
                    if (trialIndex + 1 < state.options.alphaSteps)
                        nextSearching.push_back(index);
                    continue;
                }
                bool stop = false;
                if (iteration.trials.size() > 1
                    && iteration.trials.back().rms
                        > iteration.trials[iteration.trials.size() - 2].rms) {
                    double priorBest = std::numeric_limits<double>::infinity();
                    for (std::size_t trial = 0;
                         trial + 1 < iteration.trials.size(); ++trial)
                        priorBest = std::min(
                            priorBest, iteration.trials[trial].rms);
                    if (priorBest < state.currentRms) {
                        stop = true;
                        if (progress)
                            progress(index, state.iteration + 1, trialRms,
                                "RMS increased; accepting the best trial "
                                "and advancing iteration");
                    }
                }
                if (!stop && state.adaptiveAlphaActive) {
                    if (trialRms < state.currentRms) {
                        iteration.nextAdaptiveAlpha = boundedAlpha(
                            candidateAlphas[candidate]
                            / weakerAlphaFactor(
                                state.options.alphaLogStep));
                    } else {
                        iteration.strengtheningAlpha = true;
                        iteration.nextAdaptiveAlpha = boundedAlpha(
                            candidateAlphas[candidate]
                            * strongerAlphaFactor(
                                state.options.alphaLogStep));
                    }
                }
                if (!stop && trialIndex + 1 < state.options.alphaSteps)
                    nextSearching.push_back(index);
            }
            searching = std::move(nextSearching);
        }

        std::vector<std::size_t> fitStates;
        std::vector<InversionOptions> fitOptions;
        std::vector<std::vector<double>> fitModels;
        std::vector<double> fitAlphas;
        std::vector<double> fitScales;
        std::vector<double> fitFractions;
        for (const std::size_t index : active) {
            auto &state = states[index];
            auto &iteration = work[index];
            if (!iteration.undershotTarget || iteration.trials.empty())
                continue;
            std::vector<double> fittedModel(state.model.size());
            double fittedAlpha = iteration.trials.back().alpha;
            double scale = 1.0;
            double fraction = -1.0;
            if (iteration.trials.size() == 1) {
                const auto &undershoot = iteration.trials.front();
                const double denominator
                    = state.currentRms - undershoot.rms;
                fraction = denominator > 1.0e-14
                    ? std::clamp((state.currentRms - 1.0) / denominator,
                                 0.0, 1.0)
                    : 1.0;
                for (std::size_t parameter = 0;
                     parameter < state.model.size(); ++parameter) {
                    fittedModel[parameter] = std::clamp(
                        state.model[parameter]
                            + fraction * (undershoot.model[parameter]
                                          - state.model[parameter]),
                        state.lower, state.upper);
                }
            } else {
                const std::size_t first = iteration.trials.size() > 3
                    ? iteration.trials.size() - 3 : 0;
                std::vector<double> alphaHistory;
                std::vector<double> rmsHistory;
                for (std::size_t trial = first;
                     trial < iteration.trials.size(); ++trial) {
                    alphaHistory.push_back(iteration.trials[trial].alpha);
                    rmsHistory.push_back(iteration.trials[trial].rms);
                }
                fittedAlpha = fittedTargetAlpha(alphaHistory, rmsHistory);
                if (!(std::isfinite(fittedAlpha) && fittedAlpha > 0.0))
                    continue;
                const auto linearSolveStart = Clock::now();
                const auto delta = gaussNewtonStep(
                    iteration.jacobian, iteration.weightedResidual,
                    state.weights, iteration.regularization, state.model,
                    fittedAlpha, state.options.model.thicknesses);
                state.result.timing.linearSolveSeconds
                    += elapsedSeconds(linearSolveStart);
                for (int backtrack = 0; backtrack < 11; ++backtrack) {
                    for (std::size_t parameter = 0;
                         parameter < state.model.size(); ++parameter)
                        fittedModel[parameter] = state.model[parameter]
                            + scale * delta[parameter];
                    if (std::all_of(fittedModel.begin(), fittedModel.end(),
                            [&](double value) {
                                return value >= state.lower
                                    && value <= state.upper;
                            }))
                        break;
                    scale *= 0.5;
                }
                for (double &value : fittedModel)
                    value = std::clamp(value, state.lower, state.upper);
            }
            fitStates.push_back(index);
            fitOptions.push_back(state.options);
            fitModels.push_back(std::move(fittedModel));
            fitAlphas.push_back(fittedAlpha);
            fitScales.push_back(scale);
            fitFractions.push_back(fraction);
        }
        if (!fitStates.empty()) {
            const auto alphaForwardStart = Clock::now();
            auto predictions = forwardLogModelsBatch(
                fitOptions, fitModels, maximumThreads, cancelled);
            const double alphaForwardSeconds
                = elapsedSeconds(alphaForwardStart);
            for (std::size_t candidate = 0;
                 candidate < fitStates.size(); ++candidate) {
                const std::size_t index = fitStates[candidate];
                auto &state = states[index];
                auto &iteration = work[index];
                state.result.timing.alphaForwardSeconds
                    += alphaForwardSeconds;
                ++state.result.timing.alphaTrialForwards;
                const double fittedRms = rms(
                    state.observed, predictions[candidate], state.weights);
                if (std::isfinite(fittedRms)) {
                    iteration.trials.push_back({
                        fitAlphas[candidate], fittedRms,
                        std::move(fitModels[candidate]),
                        std::move(predictions[candidate])});
                    iteration.targetFitApplied = true;
                }
                if (progress) {
                    const std::string detail = fitFractions[candidate] >= 0.0
                        ? "Single-step RMS=1 fit from first-alpha undershoot, fraction "
                            + fixedTwoDecimals(fitFractions[candidate])
                            + ", actual RMS " + fixedTwoDecimals(fittedRms)
                        : "Single parabolic RMS=1 fit, alpha "
                            + fixedTwoDecimals(fitAlphas[candidate])
                            + ", actual RMS " + fixedTwoDecimals(fittedRms);
                    progress(index, state.iteration + 1, fittedRms, detail);
                }
            }
        }

        for (const std::size_t index : active) {
            auto &state = states[index];
            auto &iteration = work[index];
            if (iteration.trials.empty()) {
                state.result.message
                    = "No regularized update improved the data fit";
                state.done = true;
                continue;
            }
            std::size_t selected = 0;
            if (iteration.reachedTarget) {
                selected = iteration.trials.size() - 1;
            } else {
                for (std::size_t trial = 1;
                     trial < iteration.trials.size(); ++trial)
                    if (iteration.trials[trial].rms
                        < iteration.trials[selected].rms)
                        selected = trial;
            }
            if (!(iteration.trials[selected].rms < state.currentRms)) {
                state.result.message
                    = "No regularized update improved the data fit";
                state.done = true;
                continue;
            }
            if (state.options.jacobianUpdateMethod
                == JacobianUpdateMethod::BroydenRankOne) {
                const auto broydenStart = Clock::now();
                Matrix updated = iteration.jacobian;
                state.broydenJacobianValid = applyBroydenRankOneUpdate(
                    updated, state.model, iteration.trials[selected].model,
                    state.predicted,
                    iteration.trials[selected].prediction);
                state.result.timing.broydenUpdateSeconds
                    += elapsedSeconds(broydenStart);
                if (state.broydenJacobianValid) {
                    state.broydenJacobian = std::move(updated);
                    ++state.broydenAge;
                    ++state.result.timing.broydenUpdates;
                } else {
                    state.broydenAge
                        = state.options.broydenRefreshInterval;
                    if (progress)
                        progress(index, state.iteration + 1,
                            iteration.trials[selected].rms,
                            "Broyden guard requested a full Jacobian");
                }
            }
            state.model = std::move(iteration.trials[selected].model);
            state.predicted
                = std::move(iteration.trials[selected].prediction);
            state.currentRms = iteration.trials[selected].rms;
            state.alphaStart = boundedAlpha(
                iteration.trials[selected].alpha
                * weakerAlphaFactor(state.options.alphaLogStep));
            ++state.iteration;
            state.result.iterations = state.iteration;
            if (iteration.reachedTarget) {
                state.result.message = iteration.targetFitApplied
                    ? "Single-step RMS=1 target fit applied"
                    : "RMS rounds to 1.00; converged";
                state.done = true;
            }
        }
        finalizeReady();
    }
    finalizeReady();
    std::vector<InversionResult> results;
    results.reserve(states.size());
    for (auto &state : states)
        results.push_back(std::move(state.result));
    return results;
}

void TemSolver::clearJacobianMemoryCache()
{
    JacobianCache::clearMemory();
}

bool TemSolver::gpuAvailable(std::string *reason)
{
    return cudaForwardAvailable(reason);
}

} // namespace pytem
