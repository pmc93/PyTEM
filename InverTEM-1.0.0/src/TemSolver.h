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

enum class JacobianUpdateMethod {
    FullEveryIteration,
    BroydenRankOne,
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
    // Tx + Rx height above ground [m]: exp(-lambda * altitude) per wavenumber.
    double altitude = 0.0;
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

// Fast SCI, after Lupus (TEMcompany): all soundings in one system with
// vertical constraints between layers and spatial constraints between
// Delaunay neighbours; Marquardt damping is set by a step-length limit.
struct SciSettings {
    double verticalFactor = 3.0;      // resistivity factor between adjacent layers (1 std)
    double lateralFactor = 1.5;       // factor between neighbours at the reference distance (1 std)
    double referenceDistance = 100.0; // [m]
    double distancePower = 0.5;       // lateral std scales as (distance / reference)^power
    int maxIterations = 30;
    // Largest |d ln rho| accepted per iteration: grows each iteration, shrinks
    // when the objective stops improving, and the inversion ends at stepMin.
    double stepMax = 3.0, stepGrowth = 1.2, stepShrink = 1.8, stepMin = 1.1;
    double relativeChangeThreshold = 0.007;
    bool logDataSpace = false; // misfit in ln(data) instead of data
    // Lateral constraints compare each layer with the neighbour's layers it
    // overlaps at the same elevation (weighted by the overlap, using each
    // sounding's ground elevation) instead of the same depth below the surface.
    bool elevationConstraints = true;
    // SCI adaptive: the constraint strength is chosen by the discrepancy
    // principle (see TemSolver::invertSci).
    bool adaptive = false;
    // Stop once the median RMS reaches 1 (half the soundings fit), before the
    // rest over-fit. SCI adaptive runs to convergence instead.
    bool stopAtHalfFit = true;
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
    // Dynamically reverses toward stronger regularization when the ordinary
    // descending alpha sweep produces an invalid or non-improving first
    // trial. A guarded sounding falls back to the fixed sweep thereafter.
    bool adaptiveAlphaSearch = true;
    double jacobianStep = 1.0e-4;
    JacobianMethod jacobianMethod = JacobianMethod::Analytical;
    JacobianUpdateMethod jacobianUpdateMethod
        = JacobianUpdateMethod::FullEveryIteration;
    // A value of three computes a full Jacobian on iterations 1, 4, 7, ...
    // while guarded Broyden updates supply the intervening iterations.
    int broydenRefreshInterval = 3;
    RegularizationNorm regularizationNorm = RegularizationNorm::L2Smooth;
    bool cacheFirstJacobian = false;
    std::string jacobianCacheDirectory;
    bool calculateSensitivity = true;
    // Selects TemSolver::invertJoint (pyTEM invert_joint) in the GUI worker.
    bool pytemJoint = false;
    // invertJoint DOI: cumulative-sensitivity thresholds and layer refinement.
    double doiThreshold = 1.2;
    double doiConservativeThreshold = 5.0;
    int doiRefinement = 10;
    // invertJoint: AVX2 wavenumber-vectorised recursion (used only when
    // pytem::vectorKernelAvailable(); the GUI always requests it).
    bool vectorizedKernel = false;
    // invertJoint / invertSci start: best homogeneous half-space instead of the given model.
    bool halfSpaceStart = false;
    // Invert the whole batch together with TemSolver::invertSci.
    bool spatialConstraints = false;
    SciSettings sci;
};

struct InversionResult {
    struct Timing {
        double totalSeconds = 0.0;
        double initialForwardSeconds = 0.0;
        double jacobianSeconds = 0.0;
        double broydenUpdateSeconds = 0.0;
        double linearSolveSeconds = 0.0;
        double alphaForwardSeconds = 0.0;
        double sensitivitySeconds = 0.0;
        int jacobianEvaluations = 0;
        int broydenUpdates = 0;
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
    // Filled by invertJoint; negative when not calculated.
    double alphaFinal = -1.0;
    double doiStandard = -1.0;
    double doiConservative = -1.0;
    bool doiStandardCapped = false;
    bool doiConservativeCapped = false;
    SciSettings sci; // invertSci: the settings this result was inverted with
};

using ProgressCallback = std::function<void(int, double, const std::string &)>;
using BatchProgressCallback = std::function<void(
    std::size_t, int, double, const std::string &)>;
using BatchResultCallback = std::function<void(
    std::size_t, const InversionResult &)>;
using CancelCallback = std::function<bool()>;

class TemSolver final
{
public:
    // N-1 finite layers sum to maxDepth. The first is 1 m and subsequent
    // thicknesses follow one geometric ratio; layer N is the half-space.
    // N-1 finite layers whose bases are log-spaced from firstDepth to
    // maxDepth (pyTEM notebook grid); layer N is the half-space.
    static std::vector<double> logSpacedThicknesses(int layerCount, double firstDepth,
                                                    double maxDepth);

    static std::vector<double> geometricThicknesses(int layerCount,
                                                    double maxDepth);

    // Returns the depth containing the requested fraction of cumulative
    // finite-layer sensitivity information. The half-space is deliberately
    // excluded because it has no finite bottom boundary.
    static double depthOfInvestigation(
        const std::vector<double> &thicknesses,
        const std::vector<double> &sensitivity,
        double cumulativeFraction = 0.95);

    // Returns the positive step-off magnitude -dB/dt [T/s].
    static std::vector<double> forward(const ForwardModel &model,
                                       const CancelCallback &cancelled = {});

    // Returns d(log response) / d(log resistivity).
    static std::vector<std::vector<double>> logJacobian(
        const ForwardModel &model,
        JacobianMethod method = JacobianMethod::Analytical,
        double finiteDifferenceStep = 1.0e-4,
        const CancelCallback &cancelled = {});

    // Port of pyTEM invert_joint (tunoe notebook): every dataset shares one
    // step-time grid, so each model needs a single step response, followed by
    // a 5-trial alpha ladder with RMS backtracking and cumulative-sensitivity DOI.
    static InversionResult invertJoint(const InversionOptions &options,
                                       const ProgressCallback &progress = {},
                                       const CancelCallback &cancelled = {});

    // Fast SCI over every sounding at once (see SciSettings); soundings share
    // the layer grid. Positions are planar coordinates and ground elevations in metres.
    // Starts from the best half-space when halfSpaceStart is set, otherwise from
    // the given models. With sci.adaptive, the smoothest constraints that fit the
    // data: the log factors are scaled from 1/4 to 4 times, each run starting from
    // the previous models and running to convergence, and the first run whose
    // total RMS over all soundings is <= 1 is kept.
    static std::vector<InversionResult> invertSci(
        const std::vector<InversionOptions> &soundings,
        const std::vector<double> &eastings, const std::vector<double> &northings,
        const std::vector<double> &elevations,
        unsigned maximumThreads = 0,
        const BatchProgressCallback &progress = {},
        const CancelCallback &cancelled = {});

    static InversionResult invert(const InversionOptions &options,
                                  const ProgressCallback &progress = {},
                                  const CancelCallback &cancelled = {});

    // Inverts each sounding independently while synchronizing expensive
    // forward/Jacobian stages across the active batch. No lateral constraints
    // or model coupling are introduced.
    static std::vector<InversionResult> invertIndependentBatch(
        const std::vector<InversionOptions> &options,
        unsigned maximumThreads = 0,
        const BatchProgressCallback &progress = {},
        const CancelCallback &cancelled = {},
        const BatchResultCallback &completed = {});

    static bool gpuAvailable(std::string *reason = nullptr);

    static void validate(const InversionOptions &options);

    static void clearJacobianMemoryCache();
};

} // namespace pytem
