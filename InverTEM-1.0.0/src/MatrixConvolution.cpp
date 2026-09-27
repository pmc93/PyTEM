#include "MatrixConvolution.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pytem {
namespace {

double splineS1(double x)
{
    return 1.0 - 2.25 * x * x
        * (1.0 - (2.0 / 9.0) * x
        * (1.0 + 2.5 * x * (1.0 - 0.4 * x)));
}

double splineS2(double x)
{
    return -0.5 * x
        * (1.0 - (11.0 / 6.0) * x
        * (1.0 - (2.0 / 11.0) * x
        * (1.0 + 2.5 * x * (1.0 - 0.4 * x))));
}

double quinticSpline(double x)
{
    if (x >= -2.0 && x <= -1.0) return splineS2(-x - 1.0);
    if (x > -1.0 && x <= 0.0) return splineS1(-x);
    if (x > 0.0 && x <= 1.0) return splineS1(x);
    if (x > 1.0 && x <= 2.0) return splineS2(x - 1.0);
    return 0.0;
}

std::vector<double> simpsonWeights(int count, double start, double end)
{
    std::vector<double> weights(static_cast<std::size_t>(count), 0.0);
    if (count < 2) return weights;
    const double spacing = (end - start) / static_cast<double>(count - 1);
    if ((count - 1) % 2 == 0) {
        weights.front() = weights.back() = spacing / 3.0;
        for (int i = 1; i + 1 < count; ++i)
            weights[static_cast<std::size_t>(i)] = spacing / 3.0 * (i % 2 ? 4.0 : 2.0);
    } else {
        const int simpsonCount = count - 1;
        weights.front() += spacing / 3.0;
        weights[static_cast<std::size_t>(simpsonCount - 1)] += spacing / 3.0;
        for (int i = 1; i + 1 < simpsonCount; ++i)
            weights[static_cast<std::size_t>(i)] += spacing / 3.0 * (i % 2 ? 4.0 : 2.0);
        weights[static_cast<std::size_t>(simpsonCount - 1)] += spacing / 2.0;
        weights[static_cast<std::size_t>(simpsonCount)] += spacing / 2.0;
    }
    return weights;
}

struct KernelSample { double time; double weight; };

std::vector<KernelSample> waveformKernel(const std::vector<double> &times,
                                         const std::vector<double> &amplitudes,
                                         int samplesPerSegment)
{
    std::vector<KernelSample> result;
    for (std::size_t segment = 0; segment + 1 < times.size(); ++segment) {
        const double duration = times[segment + 1] - times[segment];
        if (std::abs(duration) < 1.0e-30) continue;
        const double slope = (amplitudes[segment + 1] - amplitudes[segment]) / duration;
        if (std::abs(slope) < 1.0e-30) continue;
        const auto weights = simpsonWeights(samplesPerSegment, times[segment], times[segment + 1]);
        for (int i = 0; i < samplesPerSegment; ++i) {
            const double fraction = i / static_cast<double>(samplesPerSegment - 1);
            result.push_back({times[segment] + fraction * duration,
                              -slope * weights[static_cast<std::size_t>(i)]});
        }
    }
    return result;
}

std::vector<std::vector<double>> waveformMatrix(
    const std::vector<KernelSample> &kernel,
    const std::vector<double> &stepTimes,
    const std::vector<double> &outputTimes)
{
    const double deltaLog = std::log(stepTimes[1]) - std::log(stepTimes[0]);
    const double logStart = std::log(stepTimes[0]);
    std::vector<std::vector<double>> matrix(
        outputTimes.size(), std::vector<double>(stepTimes.size(), 0.0));
    for (const auto &sample : kernel) {
        for (std::size_t row = 0; row < outputTimes.size(); ++row) {
            const double shifted = outputTimes[row] - sample.time;
            if (shifted <= 0.0) continue;
            const double position = (std::log(shifted) - logStart) / deltaLog;
            const int base = static_cast<int>(std::floor(position));
            for (int offset = -2; offset <= 3; ++offset) {
                const int column = base + offset;
                if (column < 0 || column >= static_cast<int>(stepTimes.size())) continue;
                const double x = position - column;
                if (x >= -2.0 && x <= 2.0)
                    matrix[row][static_cast<std::size_t>(column)] += sample.weight * quinticSpline(x);
            }
        }
    }
    return matrix;
}

} // namespace

std::vector<double> MatrixConvolution::apply(const std::vector<double> &stepResponse) const
{
    if (stepResponse.size() != stepTimes.size())
        throw std::invalid_argument("Step response does not match convolution matrix");
    std::vector<double> result(matrix.size(), 0.0);
    for (std::size_t row = 0; row < matrix.size(); ++row)
        for (std::size_t column = 0; column < stepResponse.size(); ++column)
            result[row] += matrix[row][column] * stepResponse[column];
    return result;
}

MatrixConvolution MatrixConvolution::build(
    const std::vector<double> &gateCenters,
    const std::vector<double> &gateOpen,
    const std::vector<double> &gateClose,
    const std::vector<double> &waveformTimes,
    const std::vector<double> &waveformAmplitudes,
    int stepCount, int samplesPerSegment, double padDecades)
{
    return buildShared({{gateCenters, gateOpen, gateClose, waveformTimes, waveformAmplitudes}},
                       stepCount, samplesPerSegment, padDecades).front();
}

std::vector<MatrixConvolution> MatrixConvolution::buildShared(
    const std::vector<System> &systems, int stepCount, int samplesPerSegment, double padDecades)
{
    stepCount = std::max(stepCount, 20);
    samplesPerSegment = std::max(samplesPerSegment, 3);

    std::vector<double> positiveDelays;
    for (const auto &system : systems) {
        const auto &centers = system.gateCenters;
        if (centers.empty() || system.gateOpen.size() != centers.size()
            || system.gateClose.size() != centers.size())
            throw std::invalid_argument("Gate center/open/close arrays are inconsistent");
        if (system.waveformTimes.size() < 2
            || system.waveformTimes.size() != system.waveformAmplitudes.size())
            throw std::invalid_argument("Waveform time/amplitude arrays are inconsistent");
        std::vector<double> endpoints;
        for (std::size_t i = 0; i + 1 < system.waveformTimes.size(); ++i) {
            if (system.waveformAmplitudes[i] != system.waveformAmplitudes[i + 1]) {
                endpoints.push_back(system.waveformTimes[i]);
                endpoints.push_back(system.waveformTimes[i + 1]);
            }
        }
        for (std::size_t gate = 0; gate < centers.size(); ++gate)
            for (double output : {centers[gate], system.gateOpen[gate], system.gateClose[gate]})
                for (double endpoint : endpoints)
                    if (output - endpoint > 0.0) positiveDelays.push_back(output - endpoint);
    }
    if (positiveDelays.empty())
        throw std::invalid_argument("Waveform has no causal overlap with receiver gates");
    const auto [minimum, maximum] = std::minmax_element(positiveDelays.begin(), positiveDelays.end());

    std::vector<double> stepTimes(static_cast<std::size_t>(stepCount));
    const double lo = std::log10(*minimum) - padDecades;
    const double hi = std::log10(*maximum) + padDecades;
    for (int i = 0; i < stepCount; ++i)
        stepTimes[static_cast<std::size_t>(i)] =
            std::pow(10.0, lo + (hi - lo) * i / static_cast<double>(stepCount - 1));

    // Twelve-point GL integrates the waveform matrix over each finite receiver gate.
    static constexpr double nodes[] = {
        -0.9815606342467192, -0.9041172563704749, -0.7699026741943047, -0.5873179542866175,
        -0.3678314989981802, -0.1252334085114689,  0.1252334085114689,  0.3678314989981802,
         0.5873179542866175,  0.7699026741943047,  0.9041172563704749,  0.9815606342467192
    };
    static constexpr double weights[] = {
        0.0471753363865118, 0.1069393259953184, 0.1600783285433462, 0.2031674267230659,
        0.2334925365383548, 0.2491470458134028, 0.2491470458134028, 0.2334925365383548,
        0.2031674267230659, 0.1600783285433462, 0.1069393259953184, 0.0471753363865118
    };
    std::vector<MatrixConvolution> results;
    for (const auto &system : systems) {
        const std::size_t gates = system.gateCenters.size();
        std::vector<double> quadratureTimes;
        quadratureTimes.reserve(gates * 12);
        for (std::size_t gate = 0; gate < gates; ++gate) {
            if (system.gateClose[gate] <= system.gateOpen[gate])
                throw std::invalid_argument("Gate close time must exceed gate open time");
            const double half = 0.5 * (system.gateClose[gate] - system.gateOpen[gate]);
            const double middle = 0.5 * (system.gateClose[gate] + system.gateOpen[gate]);
            for (double node : nodes) quadratureTimes.push_back(middle + half * node);
        }
        const auto kernel = waveformKernel(system.waveformTimes, system.waveformAmplitudes,
                                           samplesPerSegment);
        const auto sampled = waveformMatrix(kernel, stepTimes, quadratureTimes);
        MatrixConvolution result;
        result.stepTimes = stepTimes;
        result.matrix.assign(gates, std::vector<double>(stepTimes.size(), 0.0));
        for (std::size_t gate = 0; gate < gates; ++gate)
            for (int q = 0; q < 12; ++q)
                for (std::size_t column = 0; column < stepTimes.size(); ++column)
                    result.matrix[gate][column] += 0.5 * weights[q] * sampled[gate * 12 + q][column];
        results.push_back(std::move(result));
    }
    return results;
}

} // namespace pytem

