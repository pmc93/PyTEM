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

struct ForwardModel {
    std::vector<double> thicknesses;
    std::vector<double> resistivities;
    std::vector<double> times;
    Geometry geometry = Geometry::CircleCentral;
    double txSize = 12.5;
    double rxX = 0.0;
    std::vector<double> stepTimes;
    std::vector<std::vector<double>> responseMatrix;
    std::vector<double> lowPassFrequencies;
    std::vector<int> lowPassOrders;
    std::vector<double> highPassFrequencies;
    std::vector<int> highPassOrders;
};

struct InversionOptions {
    ForwardModel model;
    std::vector<double> observed;
    std::vector<double> noiseStd;
    double rhoMin = 0.1;
    double rhoMax = 1.0e5;
    int maxIterations = 20;
    int alphaSteps = 8;
    double alphaLogStep = 0.25;
    double jacobianStep = 1.0e-4;
    bool calculateSensitivity = true;
};

struct InversionResult {
    std::vector<double> resistivities;
    std::vector<double> predicted;
    std::vector<double> rmsHistory;
    std::vector<double> sensitivity;
    int iterations = 0;
    bool converged = false;
    std::string message;
};

using ProgressCallback = std::function<void(int, double, const std::string &)>;
using CancelCallback = std::function<bool()>;

class TemSolver final
{
public:
    // Returns the positive step-off magnitude -dB/dt [T/s].
    static std::vector<double> forward(const ForwardModel &model,
                                       const CancelCallback &cancelled = {});

    static InversionResult invert(const InversionOptions &options,
                                  const ProgressCallback &progress = {},
                                  const CancelCallback &cancelled = {});

    static void validate(const InversionOptions &options);
};

} // namespace pytem
