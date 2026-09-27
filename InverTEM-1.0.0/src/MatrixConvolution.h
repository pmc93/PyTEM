#pragma once

#include <vector>

namespace pytem {

struct MatrixConvolution {
    std::vector<double> stepTimes;
    std::vector<std::vector<double>> matrix;

    std::vector<double> apply(const std::vector<double> &stepResponse) const;

    static MatrixConvolution build(
        const std::vector<double> &gateCenters,
        const std::vector<double> &gateOpen,
        const std::vector<double> &gateClose,
        const std::vector<double> &waveformTimes,
        const std::vector<double> &waveformAmplitudes,
        int stepCount = 300,
        int samplesPerSegment = 81,
        double padDecades = 1.0);

    struct System {
        std::vector<double> gateCenters, gateOpen, gateClose;
        std::vector<double> waveformTimes, waveformAmplitudes;
    };
    // One log-spaced step grid spanning every system's gates (pyTEM
    // setup_shared_gate_matrices), with one gate matrix per system on it.
    static std::vector<MatrixConvolution> buildShared(
        const std::vector<System> &systems, int stepCount = 75,
        int samplesPerSegment = 81, double padDecades = 1.0);
};

} // namespace pytem

