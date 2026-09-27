#pragma once

#include <string>
#include <vector>

namespace pytem {

struct UsfMoment {
    std::string name;
    int channel = 0;
    int stackCount = 0;
    double meanCurrent = 0.0;
    double meanFrequency = 0.0;
    std::vector<double> times;
    std::vector<double> voltages;
    std::vector<double> standardErrors;
    std::vector<bool> qualityAccepted;
    std::vector<double> gateOpen;
    std::vector<double> gateClose;
    std::vector<double> waveformTimes;
    std::vector<double> waveformAmplitudes;
    std::vector<double> lowPassFrequencies;
    std::vector<int> lowPassOrders;
    std::vector<double> highPassFrequencies;
    std::vector<int> highPassOrders;
    // Transmitter/receiver heights above ground [m] (/TX_HEIGHT, /RX_HEIGHT).
    double txHeight = 0.0;
    double rxHeight = 0.0;
};

struct UsfSounding {
    std::string soundingName;
    std::string date;
    std::string voltageUnits;
    int epsg = 0;
    int soundingNumber = 0;
    double sourceX = 0.0;
    double sourceY = 0.0;
    double longitude = 0.0;
    double latitude = 0.0;
    double elevation = 0.0;
    double loopX = 0.0;
    double loopY = 0.0;
    double coilX = 0.0;
    double coilY = 0.0;
    std::vector<UsfMoment> moments;

    double equivalentCircularRadius() const;
    double radialReceiverOffset() const;
};

class UsfReader final
{
public:
    static UsfSounding read(const std::string &path);

    // Project batches may contain many positions, but all soundings must use
    // one transmitter/receiver geometry. Coordinates and elevation are
    // deliberately not part of this comparison.
    static bool sameSystemGeometry(const UsfSounding &first,
                                   const UsfSounding &second,
                                   double relativeTolerance = 1.0e-6);

    // Automatic gate selection for one moment: accepted quality, positive
    // voltage and SNR >= 3, nothing after the first negative voltage; then
    // gates that break a smooth, decaying curve in log-log (leave-one-out
    // quadratic through the 6 nearest gates, tolerance max(0.2, 3 x relative
    // error) in ln V; rising voltages) are culled one at a time, worst first.
    // Late times where most gates fail are cut entirely, and a moment where
    // most gates fail is dropped.
    static std::vector<bool> autoSelectGates(const UsfMoment &moment);
};

} // namespace pytem
