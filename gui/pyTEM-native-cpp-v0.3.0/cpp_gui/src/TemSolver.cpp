#include "TemSolver.h"
#include "JacobianCache.h"
#include "TransformWeights.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <type_traits>

namespace pytem {
namespace {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double mu0 = 4.0e-7 * pi;
using Complex = std::complex<double>;
using Matrix = std::vector<std::vector<double>>;

bool wasCancelled(const CancelCallback &callback)
{
    return callback && callback();
}

Complex reflectionCoefficient(double lambda, Complex omega,
                              const std::vector<double> &thicknesses,
                              const std::vector<double> &resistivities,
                              std::vector<Complex> &gammaLayer)
{
    const std::size_t count = resistivities.size();
    gammaLayer.resize(count);
    const Complex s = Complex(0.0, 1.0) * omega;
    for (std::size_t layer = 0; layer < count; ++layer)
        gammaLayer[layer] = std::sqrt(lambda * lambda + s * mu0 / resistivities[layer]);

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

Complex circleFrequencyKernel(Complex omega, const ForwardModel &model,
                              std::vector<Complex> &gammaLayer)
{
    Complex field = 0.0;
    for (std::size_t index = 0; index < weights::hankel_base_101.size(); ++index) {
        const double lambda = weights::hankel_base_101[index] / model.txSize;
        double extra = 1.0;
        if (model.geometry == Geometry::CircleOffset)
            extra = std::cyl_bessel_j(0.0, lambda * model.rxX);
        field += reflectionCoefficient(lambda, omega, model.thicknesses, model.resistivities, gammaLayer)
               * lambda * weights::hankel_j1_101[index] * extra;
    }
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
void parallelFor(std::size_t count, unsigned preferredThreads, const Function &function)
{
    const unsigned available = preferredThreads > 0
        ? preferredThreads : std::max(1u, std::thread::hardware_concurrency());
    const unsigned threadCount = std::min<unsigned>(available, static_cast<unsigned>(count));
    if (threadCount <= 1 || count < 24) {
        for (std::size_t index = 0; index < count; ++index)
            function(index);
        return;
    }
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
}

std::vector<double> stepResponse(const ForwardModel &model,
                                 const std::vector<double> &evaluationTimes,
                                 const CancelCallback &cancelled)
{
    std::vector<double> response(evaluationTimes.size());
    auto evaluate = [&](std::size_t gate) {
        if (wasCancelled(cancelled))
            throw std::runtime_error("Inversion cancelled");
        const double time = evaluationTimes[gate];
        if (time <= 0.0)
            throw std::invalid_argument("Gate times must be positive");
        std::vector<Complex> gammaLayer(model.resistivities.size());
        if (model.transform == TransformMethod::DigitalLinearFilter) {
            double transformSum = 0.0;
            for (std::size_t frequency = 0; frequency < weights::fourier_base_81.size(); ++frequency) {
                const double omega = weights::fourier_base_81[frequency] / time;
                const Complex field = circleFrequencyKernel(omega, model, gammaLayer) * systemTransfer(omega, model);
                transformSum += mu0 * field.imag() * weights::fourier_sin_81[frequency];
            }
            response[gate] = -transformSum / time * 2.0 / pi;
        } else {
            constexpr double eulerA = 18.4;
            const auto eta = eulerWeights(model.eulerOrder);
            const double realShift = eulerA / (2.0 * time);
            const double spacing = pi / time;
            double transformSum = 0.0;
            for (std::size_t k = 0; k < eta.size(); ++k) {
                // s = A/(2t) + i*k*pi/t and omega = s/i.
                const Complex omega(static_cast<double>(k) * spacing, -realShift);
                const Complex field = circleFrequencyKernel(omega, model, gammaLayer) * systemTransfer(omega, model);
                const double sign = (k & 1U) ? -1.0 : 1.0;
                transformSum += eta[k] * sign * (mu0 * field).real();
            }
            response[gate] = std::exp(eulerA / 2.0) * transformSum / time;
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
    for (std::size_t pivot = 0; pivot < count; ++pivot) {
        std::size_t best = pivot;
        for (std::size_t row = pivot + 1; row < count; ++row)
            if (std::abs(matrix[row][pivot]) > std::abs(matrix[best][pivot]))
                best = row;
        if (std::abs(matrix[best][pivot]) < 1.0e-14)
            matrix[best][pivot] += 1.0e-10;
        std::swap(matrix[pivot], matrix[best]);
        std::swap(rhs[pivot], rhs[best]);

        const double divisor = matrix[pivot][pivot];
        for (std::size_t column = pivot; column < count; ++column)
            matrix[pivot][column] /= divisor;
        rhs[pivot] /= divisor;
        for (std::size_t row = 0; row < count; ++row) {
            if (row == pivot)
                continue;
            const double factor = matrix[row][pivot];
            for (std::size_t column = pivot; column < count; ++column)
                matrix[row][column] -= factor * matrix[pivot][column];
            rhs[row] -= factor * rhs[pivot];
        }
    }
    return rhs;
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

} // namespace

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
}

InversionResult TemSolver::invert(const InversionOptions &options,
                                  const ProgressCallback &progress,
                                  const CancelCallback &cancelled)
{
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
    auto predicted = forwardLogModel(options, model, cancelled);
    double currentRms = rms(observed, predicted, weights);
    double alphaStart = 1.0;

    for (int iteration = 0; iteration < options.maxIterations; ++iteration) {
        if (wasCancelled(cancelled))
            throw std::runtime_error("Inversion cancelled");
        result.rmsHistory.push_back(currentRms);
        if (progress)
            progress(iteration + 1, currentRms,
                     iteration == 0 && options.cacheFirstJacobian
                         ? "Loading or building shared first Jacobian"
                         : (options.jacobianMethod == JacobianMethod::Analytical
                             ? "Building analytical Jacobian" : "Building finite-difference Jacobian"));
        if (currentRms <= 1.0) {
            result.converged = true;
            break;
        }

        Matrix jacobian;
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

        double bestRms = std::numeric_limits<double>::infinity();
        double bestAlpha = alphaStart;
        std::vector<double> bestModel;
        std::vector<double> bestPrediction;

        for (int trialIndex = 0; trialIndex < options.alphaSteps; ++trialIndex) {
            if (wasCancelled(cancelled))
                throw std::runtime_error("Inversion cancelled");
            const double alpha = alphaStart * std::pow(10.0, -options.alphaLogStep * trialIndex);
            const auto delta = gaussNewtonStep(jacobian, weightedResidual, weights,
                                               regularization, model, alpha,
                                               options.model.thicknesses);
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
            const auto trialPrediction = forwardLogModel(options, trial, cancelled);
            const double trialRms = rms(observed, trialPrediction, weights);
            if (progress)
                progress(iteration + 1, trialRms,
                         "Alpha " + std::to_string(alpha) + ", step " + std::to_string(scale));
            if (trialRms < bestRms) {
                bestRms = trialRms;
                bestAlpha = alpha;
                bestModel = trial;
                bestPrediction = trialPrediction;
            }
            if (trialRms <= 1.0)
                break;
        }

        if (!(bestRms < currentRms) || bestModel.empty()) {
            result.message = "No regularized update improved the data fit";
            break;
        }
        model = std::move(bestModel);
        predicted = std::move(bestPrediction);
        currentRms = bestRms;
        alphaStart = bestAlpha * std::pow(10.0, options.alphaLogStep);
        result.iterations = iteration + 1;
    }

    if (result.rmsHistory.empty() || std::abs(result.rmsHistory.back() - currentRms) > 1.0e-12)
        result.rmsHistory.push_back(currentRms);
    result.converged = result.converged || currentRms <= 1.0;
    result.predicted = predicted;
    result.resistivities.resize(model.size());
    std::transform(model.begin(), model.end(), result.resistivities.begin(),
                   [](double value) { return std::exp(value); });

    if (options.calculateSensitivity && !wasCancelled(cancelled)) {
        const Matrix jacobian = buildJacobian(options, model, predicted, cancelled);
        result.sensitivity.assign(model.size(), 0.0);
        for (std::size_t parameter = 0; parameter < model.size(); ++parameter) {
            for (const auto &row : jacobian)
                result.sensitivity[parameter] += row[parameter] * row[parameter];
            result.sensitivity[parameter] = std::sqrt(result.sensitivity[parameter]);
        }
    }
    if (result.message.empty())
        result.message = result.converged ? "Target RMS reached" : "Maximum iterations reached";
    return result;
}

void TemSolver::clearJacobianMemoryCache()
{
    JacobianCache::clearMemory();
}

} // namespace pytem
