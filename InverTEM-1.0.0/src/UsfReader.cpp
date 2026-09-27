#include "UsfReader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace pytem {

bool UsfReader::sameSystemGeometry(const UsfSounding &first,
                                   const UsfSounding &second,
                                   double relativeTolerance)
{
    const auto close = [relativeTolerance](double left, double right) {
        if (!std::isfinite(left) || !std::isfinite(right))
            return false;
        const double scale = std::max({1.0, std::abs(left), std::abs(right)});
        return std::abs(left - right) <= relativeTolerance * scale;
    };
    return close(first.loopX, second.loopX)
        && close(first.loopY, second.loopY)
        && close(first.coilX, second.coilX)
        && close(first.coilY, second.coilY);
}
namespace {

struct Sweep {
    int channel = 0;
    bool noise = false;
    double current = 0.0;
    double frequency = 0.0;
    std::vector<double> time;
    std::vector<double> voltage;
    std::vector<bool> qualityAccepted;
    std::vector<double> gateOpen;
    std::vector<double> gateClose;
    std::vector<double> rampTime;
    std::vector<double> rampAmplitude;
    std::vector<double> lowPassFrequencies;
    std::vector<int> lowPassOrders;
    std::vector<double> highPassFrequencies;
    std::vector<int> highPassOrders;
    std::vector<double> errorBar;
    double txHeight = 0.0;
    double rxHeight = 0.0;
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

std::vector<std::string> columns(const std::string &line)
{
    std::vector<std::string> result;
    std::istringstream stream(line);
    std::string column;
    while (std::getline(stream, column, ',')) {
        column = trim(column);
        std::transform(column.begin(), column.end(), column.begin(),
                       [](unsigned char character) {
                           return static_cast<char>(std::toupper(character));
                       });
        result.push_back(std::move(column));
    }
    return result;
}

int columnIndex(const std::vector<std::string> &tableColumns,
                const std::string &name)
{
    const auto found = std::find(tableColumns.begin(), tableColumns.end(), name);
    return found == tableColumns.end()
        ? -1 : static_cast<int>(std::distance(tableColumns.begin(), found));
}

void deriveGateEdges(const std::vector<double> &centres,
                     std::vector<double> &opens,
                     std::vector<double> &closes)
{
    opens.resize(centres.size());
    closes.resize(centres.size());
    if (centres.empty())
        return;
    if (centres.size() == 1) {
        opens[0] = 0.9 * centres[0];
        closes[0] = 1.1 * centres[0];
        return;
    }
    std::vector<double> boundaries(centres.size() + 1, 0.0);
    for (std::size_t i = 1; i < centres.size(); ++i) {
        boundaries[i] = centres[i - 1] > 0.0 && centres[i] > 0.0
            ? std::sqrt(centres[i - 1] * centres[i])
            : 0.5 * (centres[i - 1] + centres[i]);
    }
    if (centres[0] > 0.0 && boundaries[1] > 0.0)
        boundaries[0] = centres[0] * centres[0] / boundaries[1];
    else
        boundaries[0] = std::max(0.0, centres[0] - (boundaries[1] - centres[0]));
    if (centres.back() > 0.0 && boundaries[centres.size() - 1] > 0.0)
        boundaries.back() = centres.back() * centres.back()
            / boundaries[centres.size() - 1];
    else
        boundaries.back() = centres.back()
            + (centres.back() - boundaries[centres.size() - 1]);
    for (std::size_t i = 0; i < centres.size(); ++i) {
        opens[i] = boundaries[i];
        closes[i] = boundaries[i + 1];
    }
}

std::pair<double, double> utmToLongitudeLatitude(double easting,
                                                 double northing,
                                                 int zone,
                                                 bool northernHemisphere)
{
    constexpr double semiMajor = 6378137.0;
    constexpr double flattening = 1.0 / 298.257223563;
    constexpr double scale = 0.9996;
    constexpr double pi = 3.14159265358979323846;
    const double eccentricitySquared = flattening * (2.0 - flattening);
    const double secondEccentricitySquared = eccentricitySquared
        / (1.0 - eccentricitySquared);
    const double x = easting - 500000.0;
    const double y = northernHemisphere ? northing : northing - 10000000.0;
    const double meridionalArc = y / scale;
    const double mu = meridionalArc / (semiMajor
        * (1.0 - eccentricitySquared / 4.0
           - 3.0 * eccentricitySquared * eccentricitySquared / 64.0
           - 5.0 * eccentricitySquared * eccentricitySquared
             * eccentricitySquared / 256.0));
    const double e1 = (1.0 - std::sqrt(1.0 - eccentricitySquared))
        / (1.0 + std::sqrt(1.0 - eccentricitySquared));
    const double phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * std::pow(e1, 3) / 32.0) * std::sin(2.0 * mu)
        + (21.0 * e1 * e1 / 16.0 - 55.0 * std::pow(e1, 4) / 32.0)
            * std::sin(4.0 * mu)
        + 151.0 * std::pow(e1, 3) / 96.0 * std::sin(6.0 * mu)
        + 1097.0 * std::pow(e1, 4) / 512.0 * std::sin(8.0 * mu);
    const double sinPhi = std::sin(phi1);
    const double cosPhi = std::cos(phi1);
    const double tanPhi = std::tan(phi1);
    const double n1 = semiMajor
        / std::sqrt(1.0 - eccentricitySquared * sinPhi * sinPhi);
    const double r1 = semiMajor * (1.0 - eccentricitySquared)
        / std::pow(1.0 - eccentricitySquared * sinPhi * sinPhi, 1.5);
    const double t1 = tanPhi * tanPhi;
    const double c1 = secondEccentricitySquared * cosPhi * cosPhi;
    const double d = x / (n1 * scale);
    const double latitude = phi1 - (n1 * tanPhi / r1)
        * (d * d / 2.0
           - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 * c1
              - 9.0 * secondEccentricitySquared) * std::pow(d, 4) / 24.0
           + (61.0 + 90.0 * t1 + 298.0 * c1
              + 45.0 * t1 * t1 - 252.0 * secondEccentricitySquared
              - 3.0 * c1 * c1) * std::pow(d, 6) / 720.0);
    const double longitudeOffset = (d
        - (1.0 + 2.0 * t1 + c1) * std::pow(d, 3) / 6.0
        + (5.0 - 2.0 * c1 + 28.0 * t1
           - 3.0 * c1 * c1 + 8.0 * secondEccentricitySquared
           + 24.0 * t1 * t1) * std::pow(d, 5) / 120.0) / cosPhi;
    const double centralMeridian = (zone * 6.0 - 183.0) * pi / 180.0;
    return {(centralMeridian + longitudeOffset) * 180.0 / pi,
            latitude * 180.0 / pi};
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
    result.txHeight = sweeps.front().txHeight;
    result.rxHeight = sweeps.front().rxHeight;
    result.voltages.assign(gateCount, 0.0);
    result.standardErrors.assign(gateCount, 0.0);
    result.qualityAccepted.assign(gateCount, true);
    std::vector<double> currents, frequencies;

    for (const Sweep &sweep : sweeps) {
        if (sweep.voltage.size() != gateCount)
            throw std::runtime_error("Inconsistent gate count within a USF channel");
        currents.push_back(sweep.current);
        frequencies.push_back(sweep.frequency);
        for (std::size_t gate = 0; gate < gateCount; ++gate) {
            result.voltages[gate] += sweep.voltage[gate];
            if (gate < sweep.qualityAccepted.size())
                result.qualityAccepted[gate] = result.qualityAccepted[gate]
                    && sweep.qualityAccepted[gate];
        }
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
    } else if (sweeps.front().errorBar.size() == gateCount) {
        // A single (already processed) sweep: ERROR_BAR is its relative error [%].
        for (std::size_t gate = 0; gate < gateCount; ++gate)
            result.standardErrors[gate] = std::abs(result.voltages[gate]) * sweeps.front().errorBar[gate] / 100.0;
    }
    result.meanCurrent = mean(currents);
    result.meanFrequency = mean(frequencies);
    if (result.gateOpen.size() != gateCount
        || result.gateClose.size() != gateCount)
        deriveGateEdges(result.times, result.gateOpen, result.gateClose);
    if (result.waveformTimes.size() < 2
        || result.waveformTimes.size() != result.waveformAmplitudes.size()) {
        const double firstPositiveTime = result.times.empty()
            ? 1.0e-6 : std::max(1.0e-12, result.times.front());
        const double idealStepWidth = std::max(1.0e-12,
                                               firstPositiveTime * 1.0e-6);
        result.waveformTimes = {-idealStepWidth, 0.0};
        result.waveformAmplitudes = {1.0, 0.0};
    }
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
    std::vector<std::string> tableColumns;
    std::string rawLine;

    auto finishSweep = [&]() {
        if (haveSweep && !current.noise && current.channel != 0 && !current.voltage.empty())
            byChannel[current.channel].push_back(current);
        current = Sweep{};
        haveSweep = false;
        inTable = false;
        tableColumns.clear();
    };

    while (std::getline(input, rawLine)) {
        const std::string line = trim(rawLine);
        if (line.empty())
            continue;
        if (line.rfind("//EPSG:", 0) == 0 || line.rfind("/EPSG:", 0) == 0)
            sounding.epsg = std::stoi(afterColon(line));
        else if (line.rfind("/SOUNDING_NAME:", 0) == 0)
            sounding.soundingName = afterColon(line);
        else if (line.rfind("/SOUNDING_NUMBER:", 0) == 0)
            sounding.soundingNumber = std::stoi(afterColon(line));
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
                sounding.sourceX = values[0];
                sounding.sourceY = values[1];
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
        else if (haveSweep && line.rfind("/TX_HEIGHT:", 0) == 0)
            current.txHeight = std::stod(afterColon(line));
        else if (haveSweep && line.rfind("/RX_HEIGHT:", 0) == 0)
            current.rxHeight = std::stod(afterColon(line));
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
            tableColumns = columns(line);
            inTable = true;
        } else if (haveSweep && inTable && line.rfind("/END", 0) == 0) {
            finishSweep();
        } else if (haveSweep && inTable && line.front() != '/') {
            const auto values = numbers(line);
            const int timeColumn = columnIndex(tableColumns, "TIME");
            const int voltageColumn = columnIndex(tableColumns, "VOLTAGE");
            const int qualityColumn = columnIndex(tableColumns, "QUALITY");
            const int openColumn = columnIndex(tableColumns, "TIMEOPEN");
            const int closeColumn = columnIndex(tableColumns, "TIMECLOSE");
            const int errorColumn = columnIndex(tableColumns, "ERROR_BAR");
            const auto available = [&](int index) {
                return index >= 0 && static_cast<std::size_t>(index) < values.size();
            };
            if (available(timeColumn) && available(voltageColumn)) {
                current.time.push_back(values[static_cast<std::size_t>(timeColumn)]);
                current.voltage.push_back(values[static_cast<std::size_t>(voltageColumn)]);
                current.qualityAccepted.push_back(!available(qualityColumn)
                    || values[static_cast<std::size_t>(qualityColumn)] > 0.0);
                if (available(errorColumn))
                    current.errorBar.push_back(values[static_cast<std::size_t>(errorColumn)]);
                if (available(openColumn) && available(closeColumn)) {
                    current.gateOpen.push_back(
                        values[static_cast<std::size_t>(openColumn)]);
                    current.gateClose.push_back(
                        values[static_cast<std::size_t>(closeColumn)]);
                }
            }
        }
    }
    finishSweep();

    if ((sounding.epsg >= 32601 && sounding.epsg <= 32660)
        || (sounding.epsg >= 32701 && sounding.epsg <= 32760)) {
        const bool northern = sounding.epsg < 32700;
        const int zone = sounding.epsg % 100;
        const auto [longitude, latitude] = utmToLongitudeLatitude(
            sounding.longitude, sounding.latitude, zone, northern);
        sounding.longitude = longitude;
        sounding.latitude = latitude;
    }

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

std::vector<bool> UsfReader::autoSelectGates(const UsfMoment &m)
{
    const std::size_t n = m.times.size();
    std::vector<bool> keep(n, false);
    for (std::size_t g = 0; g < n; ++g) {
        const double v = m.voltages[g], e = m.standardErrors[g];
        keep[g] = (g >= m.qualityAccepted.size() || m.qualityAccepted[g]) && v > 0.0 && m.times[g] > 0.0
            && (e <= 0.0 || v / e >= 3.0);
    }
    // A step-off dB/dt does not change sign: after the first negative
    // accepted gate the rest of the decay is not usable (e.g. IP effects).
    for (std::size_t g = 0; g < n; ++g)
        if ((g >= m.qualityAccepted.size() || m.qualityAccepted[g]) && m.voltages[g] < 0.0) {
            std::fill(keep.begin() + static_cast<std::ptrdiff_t>(g), keep.end(), false);
            break;
        }
    const std::vector<bool> base = keep;
    // |ln V - smooth fit| relative to the gate's tolerance; > 1 is not smooth.
    auto score = [&](std::size_t g) {
        std::vector<std::size_t> near;
        for (std::size_t k = 0; k < n; ++k)
            if (k != g && keep[k])
                near.push_back(k);
        std::sort(near.begin(), near.end(), [g](std::size_t a, std::size_t b) {
            return (a > g ? a - g : g - a) < (b > g ? b - g : g - b);
        });
        if (near.size() < 4)
            return 0.0;
        near.resize(std::min<std::size_t>(near.size(), 6));
        const double x0 = std::log(m.times[g]);
        double a[3][4] = {}; // normal equations of y = c0 + c1 dx + c2 dx^2
        for (std::size_t k : near) {
            const double dx = std::log(m.times[k]) - x0, y = std::log(m.voltages[k]);
            const double basis[3] = {1.0, dx, dx * dx};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c)
                    a[r][c] += basis[r] * basis[c];
                a[r][3] += basis[r] * y;
            }
        }
        for (int p = 0; p < 3; ++p) // Gauss-Jordan on the 3x3 system
            for (int r = 0; r < 3; ++r)
                if (r != p && a[p][p] != 0.0) {
                    const double f = a[r][p] / a[p][p];
                    for (int c = p; c < 4; ++c)
                        a[r][c] -= f * a[p][c];
                }
        if (a[0][0] == 0.0)
            return 0.0;
        const double e = m.standardErrors[g] > 0.0 ? m.standardErrors[g] / m.voltages[g] : 0.0;
        double result = std::abs(std::log(m.voltages[g]) - a[0][3] / a[0][0]) / std::max(0.2, 3.0 * e);
        for (std::size_t p = g; p-- > 0;) // a step-off decay only falls
            if (keep[p]) {
                if (m.voltages[g] > 1.05 * m.voltages[p])
                    result = std::max(result, 2.0);
                break;
            }
        return result;
    };
    for (std::size_t pass = 0; pass < n; ++pass) {
        std::size_t worst = n;
        double worstScore = 1.0;
        for (std::size_t g = 0; g < n; ++g)
            if (keep[g]) {
                const double value = score(g);
                if (value > worstScore)
                    worstScore = value, worst = g;
            }
        if (worst == n)
            break;
        keep[worst] = false;
    }
    std::size_t baseCount = 0, culled = 0;
    for (std::size_t g = 0; g < n; ++g)
        if (base[g])
            ++baseCount, culled += keep[g] ? 0 : 1;
    // Late-time noise: from the first culled gate after which most gates are culled, cut the rest.
    for (std::size_t g = 0; g < n; ++g) {
        if (!base[g])
            continue;
        std::size_t after = 0, culledAfter = 0;
        for (std::size_t k = g; k < n; ++k)
            if (base[k])
                ++after, culledAfter += keep[k] ? 0 : 1;
        if (!keep[g] && after >= 2 && 2 * culledAfter >= after) {
            std::fill(keep.begin() + static_cast<std::ptrdiff_t>(g), keep.end(), false);
            break;
        }
    }
    if (baseCount >= 4 && 2 * culled > baseCount) // mostly not smooth: drop the moment
        std::fill(keep.begin(), keep.end(), false);
    return keep;
}

} // namespace pytem
