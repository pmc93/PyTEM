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
    std::vector<double> gateOpen;
    std::vector<double> gateClose;
    std::vector<double> waveformTimes;
    std::vector<double> waveformAmplitudes;
    std::vector<double> lowPassFrequencies;
    std::vector<int> lowPassOrders;
    std::vector<double> highPassFrequencies;
    std::vector<int> highPassOrders;
};

struct UsfSounding {
    std::string soundingName;
    std::string date;
    std::string voltageUnits;
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
};

} // namespace pytem
