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
};

} // namespace pytem

