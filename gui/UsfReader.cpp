#include "UsfReader.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace pytem {
namespace {

struct Sweep {
    int channel = 0;
    bool noise = false;
    double current = 0.0;
    double frequency = 0.0;
    std::vector<double> time;
    std::vector<double> voltage;
    std::vector<double> gateOpen;
    std::vector<double> gateClose;
    std::vector<double> rampTime;
    std::vector<double> rampAmplitude;
    std::vector<double> lowPassFrequencies;
    std::vector<int> lowPassOrders;
    std::vector<double> highPassFrequencies;
    std::vector<int> highPassOrders;
};

std::string trim(std::string value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string afterColon(const std::string &line)
{
    const auto position = line.find(':');
    return position == std::string::npos ? std::string{} : trim(line.substr(position + 1));
}

std::vector<double> numbers(std::string text)
{
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream stream(text);
    std::vector<double> result;
    double value = 0.0;
    while (stream >> value)
        result.push_back(value);
    return result;
}

double mean(const std::vector<double> &values)
{
    return values.empty() ? 0.0
        : std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

UsfMoment stackChannel(int channel, const std::vector<Sweep> &sweeps)
{
    if (sweeps.empty())
        throw std::runtime_error("Cannot stack an empty USF channel");
    const std::size_t gateCount = sweeps.front().voltage.size();
    UsfMoment result;
    result.channel = channel;
    result.stackCount = static_cast<int>(sweeps.size());
    result.times = sweeps.front().time;
    result.gateOpen = sweeps.front().gateOpen;
    result.gateClose = sweeps.front().gateClose;
    result.waveformTimes = sweeps.front().rampTime;
    result.waveformAmplitudes = sweeps.front().rampAmplitude;
    result.lowPassFrequencies = sweeps.front().lowPassFrequencies;
    result.lowPassOrders = sweeps.front().lowPassOrders;
    result.highPassFrequencies = sweeps.front().highPassFrequencies;
    result.highPassOrders = sweeps.front().highPassOrders;
    result.voltages.assign(gateCount, 0.0);
    result.standardErrors.assign(gateCount, 0.0);
    std::vector<double> currents, frequencies;

    for (const Sweep &sweep : sweeps) {
        if (sweep.voltage.size() != gateCount)
            throw std::runtime_error("Inconsistent gate count within a USF channel");
        currents.push_back(sweep.current);
        frequencies.push_back(sweep.frequency);
        for (std::size_t gate = 0; gate < gateCount; ++gate)
            result.voltages[gate] += sweep.voltage[gate];
    }
    for (double &voltage : result.voltages)
        voltage /= static_cast<double>(sweeps.size());

    if (sweeps.size() > 1) {
        for (const Sweep &sweep : sweeps) {
            for (std::size_t gate = 0; gate < gateCount; ++gate) {
                const double delta = sweep.voltage[gate] - result.voltages[gate];
                result.standardErrors[gate] += delta * delta;
            }
        }
        for (double &error : result.standardErrors)
            error = std::sqrt(error / static_cast<double>(sweeps.size() - 1))
                  / std::sqrt(static_cast<double>(sweeps.size()));
    }
    result.meanCurrent = mean(currents);
    result.meanFrequency = mean(frequencies);
    return result;
}

} // namespace

double UsfSounding::equivalentCircularRadius() const
{
    constexpr double pi = 3.14159265358979323846;
    return loopX > 0.0 && loopY > 0.0 ? std::sqrt(loopX * loopY / pi) : 0.0;
}

double UsfSounding::radialReceiverOffset() const
{
    return std::hypot(coilX, coilY);
}

UsfSounding UsfReader::read(const std::string &path)
{
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Could not open USF file: " + path);

    UsfSounding sounding;
    std::map<int, std::vector<Sweep>> byChannel;
    Sweep current;
    bool haveSweep = false;
    bool inTable = false;
    std::string rawLine;

    auto finishSweep = [&]() {
        if (haveSweep && !current.noise && current.channel != 0 && !current.voltage.empty())
            byChannel[current.channel].push_back(current);
        current = Sweep{};
        haveSweep = false;
        inTable = false;
    };

    while (std::getline(input, rawLine)) {
        const std::string line = trim(rawLine);
        if (line.empty())
            continue;
        if (line.rfind("/SOUNDING_NAME:", 0) == 0)
            sounding.soundingName = afterColon(line);
        else if (line.rfind("/DATE:", 0) == 0)
            sounding.date = afterColon(line);
        else if (line.rfind("/VOLTAGE_UNITS:", 0) == 0)
            sounding.voltageUnits = afterColon(line);
        else if (line.rfind("/LOOP_SIZE:", 0) == 0) {
            const auto values = numbers(afterColon(line));
            if (values.size() >= 2) { sounding.loopX = values[0]; sounding.loopY = values[1]; }
        } else if (line.rfind("/COIL_LOCATION:", 0) == 0) {
            const auto values = numbers(afterColon(line));
            if (values.size() >= 2) { sounding.coilX = values[0]; sounding.coilY = values[1]; }
        } else if (line.rfind("/LOCATION:", 0) == 0) {
            const auto values = numbers(afterColon(line));
            if (values.size() >= 3) {
                sounding.longitude = values[0]; sounding.latitude = values[1]; sounding.elevation = values[2];
            }
        } else if (line.rfind("/SWEEP_NUMBER:", 0) == 0) {
            finishSweep();
            haveSweep = true;
        } else if (haveSweep && line.rfind("/CURRENT:", 0) == 0)
            current.current = std::stod(afterColon(line));
        else if (haveSweep && line.rfind("/FREQUENCY:", 0) == 0)
            current.frequency = std::stod(afterColon(line));
        else if (haveSweep && line.rfind("/SWEEP_IS_NOISE:", 0) == 0)
            current.noise = std::stoi(afterColon(line)) != 0;
        else if (haveSweep && line.rfind("/CHANNEL:", 0) == 0)
            current.channel = std::stoi(afterColon(line));
        else if (haveSweep && (line.rfind("/LOW_PASS:", 0) == 0
                               || line.rfind("/HIGH_PASS:", 0) == 0)) {
            const bool lowPass = line.rfind("/LOW_PASS:", 0) == 0;
            const auto values = numbers(afterColon(line));
            for (std::size_t i = 0; i + 1 < values.size(); i += 2) {
                (lowPass ? current.lowPassFrequencies : current.highPassFrequencies).push_back(values[i]);
                (lowPass ? current.lowPassOrders : current.highPassOrders).push_back(
                    std::clamp(static_cast<int>(std::lround(values[i + 1])), 1, 2));
            }
        }
        else if (haveSweep && line.rfind("/TX_RAMP:", 0) == 0) {
            const auto values = numbers(afterColon(line));
            for (std::size_t i = 0; i + 1 < values.size(); i += 2) {
                current.rampTime.push_back(values[i]);
                current.rampAmplitude.push_back(values[i + 1]);
            }
        } else if (haveSweep && line.rfind("TIME,", 0) == 0) {
            inTable = true;
        } else if (haveSweep && inTable && line.rfind("/END", 0) == 0) {
            finishSweep();
        } else if (haveSweep && inTable && line.front() != '/') {
            const auto values = numbers(line);
            if (values.size() >= 6) {
                current.time.push_back(values[0]);
                current.voltage.push_back(values[1]);
                current.gateOpen.push_back(values[4]);
                current.gateClose.push_back(values[5]);
            }
        }
    }
    finishSweep();

    if (byChannel.empty())
        throw std::runtime_error("No non-noise sweep tables found in USF file");
    for (const auto &[channel, sweeps] : byChannel)
        sounding.moments.push_back(stackChannel(channel, sweeps));
    std::sort(sounding.moments.begin(), sounding.moments.end(),
              [](const UsfMoment &a, const UsfMoment &b) { return a.meanFrequency > b.meanFrequency; });
    for (std::size_t index = 0; index < sounding.moments.size(); ++index)
        sounding.moments[index].name = index == 0 ? "LM" : (index == 1 ? "HM" : "Moment " + std::to_string(index + 1));
    return sounding;
}

} // namespace pytem
