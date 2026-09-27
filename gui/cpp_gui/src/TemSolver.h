#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace pytem {

enum class Geometry {
    CircleCentral,
    CircleOffset,
};

enum class TransformMethod {
    DigitalLinearFilter,
    Euler,
};

enum class JacobianMethod {
    Analytical,
    FiniteDifference,
};

enum class RegularizationNorm {
    L2Smooth,
    L1Blocky,
};

struct ForwardModel {
    std::vector<double> thicknesses;
    std::vector<double> resistivities;
    std::vector<double> times;
    Geometry geometry = Geometry::CircleCentral;
    TransformMethod transform = TransformMethod::DigitalLinearFilter;
    int eulerOrder = 11;
    double txSize = 12.5;
    double rxX = 0.0;
    std::vector<double> stepTimes;
    std::vector<std::vector<double>> responseMatrix;
    std::vector<double> lowPassFrequencies;
    std::vector<int> lowPassOrders;
    std::vector<double> highPassFrequencies;
    std::vector<int> highPassOrders;
    // Uses the optional CUDA DLF backend when compiled and available.
    bool useGpu = false;
    // Zero selects all available cores. Batch mode assigns a per-job limit.
    unsigned maxThreads = 0;
};

struct InversionDataSet {
    ForwardModel model;
    std::vector<double> observed;
    std::vector<double> noiseStd;
};

struct InversionOptions {
    ForwardModel model;
    std::vector<double> observed;
    std::vector<double> noiseStd;
    // Additional moments are concatenated with the primary dataset and share
    // the same thicknesses/resistivities during a joint inversion.
    std::vector<InversionDataSet> additionalDataSets;
    double rhoMin = 0.1;
    double rhoMax = 1.0e5;
    int maxIterations = 20;
    int alphaSteps = 8;
    double alphaLogStep = 1.0 / 9.0;
    double jacobianStep = 1.0e-4;
    JacobianMethod jacobianMethod = JacobianMethod::Analytical;
    RegularizationNorm regularizationNorm = RegularizationNorm::L2Smooth;
    bool cacheFirstJacobian = false;
    std::string jacobianCacheDirectory;
    bool calculateSensitivity = true;
};

struct InversionResult {
    struct Timing {
        double totalSeconds = 0.0;
        double initialForwardSeconds = 0.0;
        double jacobianSeconds = 0.0;
        double linearSolveSeconds = 0.0;
        double alphaForwardSeconds = 0.0;
        double sensitivitySeconds = 0.0;
        int jacobianEvaluations = 0;
        int alphaTrialForwards = 0;
    } timing;
    std::vector<double> resistivities;
    std::vector<double> predicted;
    std::vector<double> rmsHistory;
    std::vector<double> sensitivity;
    int iterations = 0;
    bool converged = false;
    std::string message;
    std::string firstJacobianSource = "not used";
};

using ProgressCallback = std::function<void(int, double, const std::string &)>;
using CancelCallback = std::function<bool()>;

class TemSolver final
{
public:
    // N-1 finite layers sum to maxDepth. The first is 1 m and subsequent
    // thicknesses follow one geometric ratio; layer N is the half-space.
    static std::vector<double> geometricThicknesses(int layerCount,
                                                    double maxDepth);

    // Returns the positive step-off magnitude -dB/dt [T/s].
    static std::vector<double> forward(const ForwardModel &model,
                                       const CancelCallback &cancelled = {});

    // Returns d(log response) / d(log resistivity).
    static std::vector<std::vector<double>> logJacobian(
        const ForwardModel &model,
        JacobianMethod method = JacobianMethod::Analytical,
        double finiteDifferenceStep = 1.0e-4,
        const CancelCallback &cancelled = {});

    static InversionResult invert(const InversionOptions &options,
                                  const ProgressCallback &progress = {},
                                  const CancelCallback &cancelled = {});

    static bool gpuAvailable(std::string *reason = nullptr);

    static void validate(const InversionOptions &options);

    static void clearJacobianMemoryCache();
};

} // namespace pytem
