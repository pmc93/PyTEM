#include "UsfReader.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <tuple>
#include <cstdio>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <system_error>
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

// Comma- or space-separated numbers up to the first non-number (locale-independent).
std::vector<double> numbers(const std::string &text)
{
    std::vector<double> result;
    const char *p = text.data(), *end = p + text.size();
    while (true) {
        while (p < end && (*p == ' ' || *p == ',' || *p == '\t' || *p == '+'))
            ++p;
        double value = 0.0;
        const auto parsed = std::from_chars(p, end, value);
        if (parsed.ec != std::errc())
            return result;
        result.push_back(value);
        p = parsed.ptr;
    }
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
    std::array<int, 6> tableIndices{}; // TIME, VOLTAGE, QUALITY, TIMEOPEN, TIMECLOSE, ERROR_BAR columns
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
            tableIndices = {columnIndex(tableColumns, "TIME"), columnIndex(tableColumns, "VOLTAGE"),
                            columnIndex(tableColumns, "QUALITY"), columnIndex(tableColumns, "TIMEOPEN"),
                            columnIndex(tableColumns, "TIMECLOSE"), columnIndex(tableColumns, "ERROR_BAR")};
            inTable = true;
        } else if (haveSweep && inTable && line.rfind("/END", 0) == 0) {
            finishSweep();
        } else if (haveSweep && inTable && line.front() != '/') {
            const auto values = numbers(line);
            const auto [timeColumn, voltageColumn, qualityColumn, openColumn, closeColumn, errorColumn] = tableIndices;
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

std::pair<double, double> UsfReader::toUtm(double longitude, double latitude, int zone)
{
    constexpr double a = 6378137.0, f = 1.0 / 298.257223563, k0 = 0.9996, pi = 3.14159265358979323846;
    const double e2 = f * (2.0 - f), ep2 = e2 / (1.0 - e2);
    const double phi = latitude * pi / 180.0;
    const double lambda0 = (zone * 6.0 - 183.0) * pi / 180.0;
    const double n = a / std::sqrt(1.0 - e2 * std::sin(phi) * std::sin(phi));
    const double t = std::tan(phi) * std::tan(phi), c = ep2 * std::cos(phi) * std::cos(phi);
    const double A = std::cos(phi) * (longitude * pi / 180.0 - lambda0);
    const double m = a * ((1.0 - e2 / 4.0 - 3.0 * e2 * e2 / 64.0 - 5.0 * e2 * e2 * e2 / 256.0) * phi
        - (3.0 * e2 / 8.0 + 3.0 * e2 * e2 / 32.0 + 45.0 * e2 * e2 * e2 / 1024.0) * std::sin(2.0 * phi)
        + (15.0 * e2 * e2 / 256.0 + 45.0 * e2 * e2 * e2 / 1024.0) * std::sin(4.0 * phi)
        - 35.0 * e2 * e2 * e2 / 3072.0 * std::sin(6.0 * phi));
    const double easting = 500000.0 + k0 * n * (A + (1.0 - t + c) * std::pow(A, 3) / 6.0
        + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * ep2) * std::pow(A, 5) / 120.0);
    const double northing = k0 * (m + n * std::tan(phi) * (A * A / 2.0
        + (5.0 - t + 9.0 * c + 4.0 * c * c) * std::pow(A, 4) / 24.0
        + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * ep2) * std::pow(A, 6) / 720.0));
    return {easting, latitude < 0.0 ? northing + 10000000.0 : northing};
}

std::string UsfReader::findSystemFile(const std::string &path)
{
    namespace fs = std::filesystem;
    const fs::path data = fs::absolute(path);
    {   // a TEMcompany data file describes its own system
        std::ifstream own(data);
        std::string line;
        std::getline(own, line);
        if (line.find("TEMcompany") != std::string::npos)
            return data.string();
    }
    if (fs::exists(fs::path(data).replace_extension(".gex")))
        return fs::path(data).replace_extension(".gex").string();
    std::vector<fs::path> gex, temcompany;
    std::error_code error;
    for (const auto &entry : fs::directory_iterator(data.parent_path(), error)) {
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension == ".gex")
            gex.push_back(entry.path());
        else if (extension == ".xyz" && entry.path() != data) {
            std::ifstream header(entry.path());
            std::string line;
            for (int i = 0; i < 40 && std::getline(header, line); ++i)
                if (line.find("[RxTxSpecs]") != std::string::npos) {
                    temcompany.push_back(entry.path());
                    break;
                }
        }
    }
    std::sort(temcompany.begin(), temcompany.end());
    return gex.size() == 1 ? gex.front().string() : temcompany.empty() ? std::string() : temcompany.front().string();
}

namespace {

// A TEM system: tTEM defaults, replaced by what a .gex or TEMcompany (stb2xyz)
// file defines. TEMcompany gate times are shifted by the moment's GateTimeShift.
struct TemSystem {
    struct Moment { std::vector<double> times, amplitudes, lowPass, open, close, centres; double frequency, shift = 0.0, factor = 1.0; };
    std::map<std::string, Moment> moments;
    double loopX = 2.0, loopY = 4.0, coilX = -9.0, coilY = 0.0, txHeight = 0.5, rxHeight = 0.5;

    // Gate edges, waveform, filters and heights of one moment (named LM/HM, times set).
    void apply(UsfMoment &moment) const
    {
        const Moment &settings = moments.at(moment.name);
        moment.meanFrequency = settings.frequency;
        if (settings.open.size() == moment.times.size() && settings.close.size() == moment.times.size())
            for (std::size_t gate = 0; gate < moment.times.size(); ++gate) {
                moment.gateOpen.push_back(settings.open[gate] + settings.shift);
                moment.gateClose.push_back(settings.close[gate] + settings.shift);
            }
        else
            deriveGateEdges(moment.times, moment.gateOpen, moment.gateClose);
        moment.waveformTimes = settings.times;
        moment.waveformAmplitudes = settings.amplitudes;
        moment.lowPassFrequencies = settings.lowPass;
        moment.lowPassOrders.assign(settings.lowPass.size(), 1);
        moment.txHeight = txHeight;
        moment.rxHeight = rxHeight;
    }
};

TemSystem readSystem(const std::string &systemPath)
{
    using Moment = TemSystem::Moment;
    const auto pulse = [](double on, double off, double frequency) {
        Moment moment{{}, {}, {670.0e3}, {}, {}, {}, frequency};
        for (double sign : {-1.0, 1.0}) { // the previous, opposite pulse and this one
            const double shift = sign < 0.0 ? -1.0 / frequency : 0.0;
            for (int k = 0; k <= 8; ++k) {
                moment.times.push_back(shift - on + on * k / 8.0);
                moment.amplitudes.push_back(sign * (1.0 - std::exp(-3.0 * k / 8.0)) / (1.0 - std::exp(-3.0)));
            }
            moment.times.push_back(shift + off);
            moment.amplitudes.push_back(0.0);
        }
        return moment;
    };
    TemSystem result{{{"LM", pulse(200.0e-6, 2.5e-6, 2110.0)}, {"HM", pulse(450.0e-6, 4.0e-6, 660.0)}}};
    auto &system = result.moments;
    double &loopX = result.loopX, &loopY = result.loopY, &coilX = result.coilX, &coilY = result.coilY,
           &txHeight = result.txHeight, &rxHeight = result.rxHeight;
    if (!systemPath.empty()) {
        std::ifstream systemInput(systemPath);
        if (!systemInput)
            throw std::runtime_error("Could not open system file: " + systemPath);
        std::vector<double> loopXs, loopYs, rx, tx, filters;
        std::map<std::string, Moment> waveforms;
        std::string section, raw;
        std::map<std::string, std::string> sectionMoment;
        std::map<std::string, std::vector<double>> sectionFilters;
        while (std::getline(systemInput, raw)) {
            const std::string line = trim(raw);
            if (line.size() > 2 && line.front() == '[') { section = line; continue; }
            const auto equals = line.find('=');
            if (equals == std::string::npos)
                continue;
            const std::string key = trim(line.substr(0, equals));
            const auto values = numbers(line.substr(equals + 1));
            const std::string moment = key.substr(0, 2); // TEMcompany keys start with LM_ / HM_
            if (key.rfind("TxLoopPoint", 0) == 0 && values.size() >= 2) {         // .gex
                loopXs.push_back(values[0]);
                loopYs.push_back(values[1]);
            } else if (key == "RxCoilPosition1" || key == "RxCoil_XYZPos") {
                rx = values;
                if (key == "RxCoilPosition1" && values.size() >= 3) // .gex: z down, so heights are -z
                    rxHeight = -values[2];
            } else if (key == "TxCoilPosition1" && values.size() >= 3)
                txHeight = -values[2];
            else if ((key.rfind("GateTimeLM", 0) == 0 || key.rfind("GateTimeHM", 0) == 0) && values.size() >= 3) {
                system[key.substr(8, 2)].open.push_back(values[1]); // .gex: centre, open, close
                system[key.substr(8, 2)].close.push_back(values[2]);
            } else if ((key == "GateTimeShift" || key == "RepFreq") && sectionMoment.count(section) && !values.empty())
                // .gex [ChannelN], after its TransmitterMoment
                (key == "RepFreq" ? system[sectionMoment[section]].frequency : system[sectionMoment[section]].shift) = values[0];
            else if (key == "TxLoop_XYZPos")
                tx = values;
            else if (key == "TxLoop_XYLength" && values.size() >= 2) {
                loopX = values[0];
                loopY = values[1];
            } else if ((key.rfind("WaveformLMPoint", 0) == 0 || key.rfind("WaveformHMPoint", 0) == 0) && values.size() >= 2) {
                auto &waveform = waveforms[key.substr(8, 2)];
                waveform.times.push_back(values[0]);
                waveform.amplitudes.push_back(values[1]);
            } else if (key == moment + "_Waveform_Time")
                waveforms[moment].times = values;
            else if (key == moment + "_Waveform_Amplitude")
                waveforms[moment].amplitudes = values;
            else if (key == moment + "_OpenTime")
                system[moment].open = values;
            else if (key == moment + "_CloseTime")
                system[moment].close = values;
            else if (key == moment + "_GateTimeShift" && !values.empty())
                system[moment].shift = values[0];
            else if (key == moment + "_CenterTime")
                system[moment].centres = values;
            else if (key == moment + "_DataFactor" && !values.empty())
                system[moment].factor = values[0];
            else if (key == "TransmitterMoment")
                sectionMoment[section] = trim(line.substr(equals + 1));
            else if ((key.find("LowPassFilter") != std::string::npos || key.find("LPFilter") != std::string::npos)
                     && !values.empty())
                (section.rfind("[Channel", 0) == 0 ? sectionFilters[section] : filters)
                    .push_back(*std::max_element(values.begin(), values.end()));
        }
        if (loopXs.size() >= 3) {
            const auto [minX, maxX] = std::minmax_element(loopXs.begin(), loopXs.end());
            const auto [minY, maxY] = std::minmax_element(loopYs.begin(), loopYs.end());
            loopX = *maxX - *minX;
            loopY = *maxY - *minY;
            tx = {0.5 * (*maxX + *minX), 0.5 * (*maxY + *minY)};
        }
        if (rx.size() >= 2) {
            coilX = rx[0] - (tx.size() >= 2 ? tx[0] : 0.0);
            coilY = rx[1] - (tx.size() >= 2 ? tx[1] : 0.0);
        }
        if (tx.size() >= 3 && rx.size() >= 3) { // TEMcompany TxLoop_XYZPos / RxCoil_XYZPos: heights above ground
            txHeight = tx[2];
            rxHeight = rx[2];
        }
        for (auto &[name, waveform] : waveforms)
            if (waveform.times.size() >= 2 && waveform.times.size() == waveform.amplitudes.size()) {
                system[name].times = waveform.times;
                system[name].amplitudes = waveform.amplitudes;
            }
        for (auto &[name, moment] : system)
            if (!filters.empty())
                moment.lowPass = filters;
        for (const auto &[name, moment] : sectionMoment)
            if (!sectionFilters[name].empty() && system.count(moment)) { // .gex: the coil's filters, then the channel's
                system[moment].lowPass = filters;
                system[moment].lowPass.insert(system[moment].lowPass.end(), sectionFilters[name].begin(), sectionFilters[name].end());
            }
    }

    return result;
}

// Seconds since 1970 of a date (dd-mm-yyyy, yyyy-mm-dd or yyyymmdd) and a time (hh:mm:ss[.s]).
double timestamp(const std::string &date, const std::string &time)
{
    std::string digits;
    for (char c : date)
        if (std::isdigit(static_cast<unsigned char>(c))) digits += c;
    if (digits.size() != 8)
        return std::nan("");
    const bool dayFirst = date.size() > 2 && !std::isdigit(static_cast<unsigned char>(date[2]));
    int y = std::stoi(dayFirst ? digits.substr(4, 4) : digits.substr(0, 4));
    const int m = std::stoi(dayFirst ? digits.substr(2, 2) : digits.substr(4, 2));
    const int d = std::stoi(dayFirst ? digits.substr(0, 2) : digits.substr(6, 2));
    y -= m <= 2; // days from civil (Howard Hinnant)
    const int era = (y >= 0 ? y : y - 399) / 400, yoe = y - era * 400;
    const int doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const long days = era * 146097L + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468;
    double h = 0, mi = 0, sec = 0;
    std::sscanf(time.c_str(), "%lf:%lf:%lf", &h, &mi, &sec);
    return days * 86400.0 + h * 3600.0 + mi * 60.0 + sec;
}

// A line file (.lin): "date time line lat lon ! Start|End" pairs. Data recorded
// outside every Start-End interval (turns) are not imported.
struct LineIntervals {
    std::vector<std::tuple<double, double, int>> intervals; // start, end, line
    explicit LineIntervals(const std::string &path)
    {
        if (path.empty())
            return;
        std::ifstream input(path);
        if (!input)
            throw std::runtime_error("Could not open line file: " + path);
        double start = std::nan("");
        for (std::string raw; std::getline(input, raw);) {
            std::istringstream stream(raw);
            std::string date, time, line;
            if (!(stream >> date >> time >> line))
                continue;
            if (raw.find("Start") != std::string::npos)
                start = timestamp(date, time);
            else if (raw.find("End") != std::string::npos && std::isfinite(start)) {
                intervals.emplace_back(start, timestamp(date, time), std::stoi(line));
                start = std::nan("");
            }
        }
        std::sort(intervals.begin(), intervals.end());
    }
    // The line recording at this time: 0 when there is no line file, -1 outside every line.
    int lineAt(double time) const
    {
        if (intervals.empty())
            return 0;
        auto after = std::upper_bound(intervals.begin(), intervals.end(), std::make_tuple(time, INFINITY, 0));
        return after != intervals.begin() && time <= std::get<1>(*std::prev(after)) ? std::get<2>(*std::prev(after)) : -1;
    }
};

// TEMcompany stb2xyz data: one row per recorded LM or HM stack, with the system
// in its own header. Consecutive LM and HM records form one sounding; dB/dt in
// V/m^2 is divided by the current and multiplied by the moment's DataFactor.
std::vector<UsfSounding> readTemcompanyXyz(const std::string &path, const LineIntervals &lines, int &skipped)
{
    const TemSystem system = readSystem(path);
    std::map<int, int> perLine;
    std::vector<int> lineOf; // line of each sounding
    std::ifstream input(path);
    std::map<std::string, std::size_t> header;
    std::vector<UsfSounding> soundings;
    std::string raw;
    while (std::getline(input, raw)) {
        std::istringstream stream(raw);
        std::vector<std::string> cells{std::istream_iterator<std::string>(stream), std::istream_iterator<std::string>()};
        if (cells.empty())
            continue;
        if (cells[0] == "Date") {
            for (const auto &name : cells)
                header.emplace(name, header.size());
            continue;
        }
        if (header.empty() || cells.size() < header.size())
            continue;
        const auto value = [&](const std::string &name) {
            const auto index = header.find(name);
            return index == header.end() ? std::nan("") : std::strtod(cells[index->second].c_str(), nullptr);
        };
        const int line = lines.lineAt(timestamp(cells[0], cells[1]));
        if (line < 0) {
            ++skipped;
            continue;
        }
        UsfMoment moment;
        moment.name = value("Moment") == 0.0 ? "LM" : "HM";
        const auto &settings = system.moments.at(moment.name);
        if (settings.centres.empty())
            throw std::runtime_error("No " + moment.name + "_CenterTime in the TEMcompany header");
        moment.channel = moment.name == "LM" ? 1 : 2;
        moment.stackCount = 1;
        moment.meanCurrent = value("TxCurrent");
        for (std::size_t gate = 0; gate < settings.centres.size(); ++gate) {
            char suffix[24];
            std::snprintf(suffix, sizeof(suffix), "%03zu", gate + 1);
            const double data = value(std::string("dbdtDat") + suffix) / moment.meanCurrent * settings.factor;
            const double relative = value(std::string("dbdtStd") + suffix);
            const bool valid = std::isfinite(data) && std::isfinite(relative);
            moment.times.push_back(settings.centres[gate] + settings.shift);
            moment.voltages.push_back(valid ? data : 0.0);
            moment.standardErrors.push_back(valid ? std::abs(data) * relative : 0.0);
            moment.qualityAccepted.push_back(valid);
        }
        system.apply(moment);
        const bool newSounding = soundings.empty() || lineOf.back() != line || std::any_of(soundings.back().moments.begin(),
            soundings.back().moments.end(), [&](const UsfMoment &m) { return m.name == moment.name; });
        if (newSounding) {
            UsfSounding sounding;
            sounding.soundingNumber = ++perLine[line];
            sounding.soundingName = line > 0 ? "Line" + std::to_string(line) + "_" + std::to_string(sounding.soundingNumber)
                                             : "Sounding_" + std::to_string(sounding.soundingNumber);
            lineOf.push_back(line);
            sounding.date = cells[0];
            sounding.time = timestamp(cells[0], cells[1]);
            sounding.voltageUnits = "V/AM2";
            sounding.longitude = value("Longitude");
            sounding.latitude = value("Latitude");
            sounding.elevation = value("Elevation");
            const int zone = static_cast<int>(std::floor((sounding.longitude + 180.0) / 6.0)) + 1;
            sounding.epsg = (sounding.latitude >= 0.0 ? 32600 : 32700) + zone;
            std::tie(sounding.sourceX, sounding.sourceY) = UsfReader::toUtm(sounding.longitude, sounding.latitude, zone);
            sounding.loopX = system.loopX;
            sounding.loopY = system.loopY;
            sounding.coilX = system.coilX;
            sounding.coilY = system.coilY;
            soundings.push_back(std::move(sounding));
        }
        auto &moments = soundings.back().moments;
        moments.push_back(std::move(moment));
        std::sort(moments.begin(), moments.end(),
                  [](const UsfMoment &a, const UsfMoment &b) { return a.meanFrequency > b.meanFrequency; });
    }
    if (soundings.empty())
        throw std::runtime_error("No soundings found in TEMcompany XYZ file");
    return soundings;
}

} // namespace

std::vector<UsfSounding> UsfReader::readWorkbenchXyz(const std::string &path, const std::string &systemPath,
                                                     const std::string &linePath, int *skippedRecords)
{
    const LineIntervals lines(linePath);
    int skipped = 0;
    if (skippedRecords)
        *skippedRecords = 0;
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Could not open XYZ file: " + path);
    std::string first;
    std::getline(input, first);
    if (first.find("TEMcompany") != std::string::npos) {
        auto soundings = readTemcompanyXyz(path, lines, skipped);
        if (skippedRecords)
            *skippedRecords = skipped;
        return soundings;
    }
    input.seekg(0);

    const TemSystem system = readSystem(systemPath);
    double dummy = 9999.0;
    int epsg = 0;
    std::map<int, std::vector<double>> gateTimes;
    std::map<std::string, std::size_t> header;
    std::vector<UsfSounding> soundings;
    std::map<std::string, std::size_t> byStamp;
    std::map<int, int> perLine;
    std::string raw, previous;
    while (std::getline(input, raw)) {
        const std::string line = trim(raw);
        if (line.empty())
            continue;
        if (line.front() == '/') {
            const std::string body = trim(line.substr(1));
            if (previous == "DUMMY")
                dummy = std::stod(body);
            else if (previous == "COORDINATE SYSTEM" && body.find("epsg:") != std::string::npos)
                epsg = std::stoi(body.substr(body.find("epsg:") + 5));
            else if (body.rfind("Gates for channel", 0) == 0)
                gateTimes[std::stoi(body.substr(17))] = numbers(afterColon(body));
            else if (body.rfind("DATE", 0) == 0)
                for (const auto &name : columns(body))
                    header.emplace(name, header.size());
            previous = body;
            continue;
        }
        if (header.empty())
            throw std::runtime_error("No column header found in XYZ file");
        std::vector<std::string> cells;
        std::istringstream stream(line);
        for (std::string cell; std::getline(stream, cell, ',');)
            cells.push_back(trim(cell));
        const auto value = [&](const std::string &name) {
            const auto index = header.find(name);
            return index != header.end() && index->second < cells.size() && !cells[index->second].empty()
                ? std::stod(cells[index->second]) : dummy;
        };
        const int channel = static_cast<int>(value("CHANNEL_NO"));
        const auto times = gateTimes.find(channel);
        if (times == gateTimes.end())
            continue;
        const std::string stamp = cells[0] + ' ' + (cells.size() > 1 ? cells[1] : std::string());
        const int lineFromFile = lines.lineAt(timestamp(cells[0], cells.size() > 1 ? cells[1] : std::string()));
        if (lineFromFile < 0) {
            ++skipped;
            continue;
        }
        auto found = byStamp.find(stamp);
        if (found == byStamp.end()) {
            const int lineNumber = lineFromFile > 0 ? lineFromFile : static_cast<int>(value("LINE_NO"));
            UsfSounding sounding;
            sounding.soundingNumber = ++perLine[lineNumber];
            sounding.soundingName = "Line" + std::to_string(lineNumber) + "_" + std::to_string(sounding.soundingNumber);
            sounding.date = cells[0];
            sounding.time = timestamp(cells[0], cells.size() > 1 ? cells[1] : std::string());
            sounding.voltageUnits = "V/AM2";
            sounding.epsg = epsg;
            sounding.sourceX = sounding.longitude = value("X");
            sounding.sourceY = sounding.latitude = value("Y");
            sounding.elevation = value("ELEVATION");
            if ((epsg >= 32601 && epsg <= 32660) || (epsg >= 32701 && epsg <= 32760))
                std::tie(sounding.longitude, sounding.latitude)
                    = utmToLongitudeLatitude(sounding.sourceX, sounding.sourceY, epsg % 100, epsg < 32700);
            sounding.loopX = system.loopX;
            sounding.loopY = system.loopY;
            sounding.coilX = system.coilX;
            sounding.coilY = system.coilY;
            found = byStamp.emplace(stamp, soundings.size()).first;
            soundings.push_back(std::move(sounding));
        }
        UsfMoment moment;
        moment.channel = channel;
        moment.stackCount = 1;
        moment.meanCurrent = value("CURRENT");
        moment.name = moment.meanCurrent < 10.0 ? "LM" : "HM";
        moment.times = times->second;
        const double area = value("TX_AREA") != dummy ? value("TX_AREA") : system.loopX * system.loopY;
        for (std::size_t gate = 1; gate <= moment.times.size(); ++gate) {
            const std::string suffix = "CH" + std::to_string(channel) + "GT" + std::to_string(gate);
            const double data = value("DBDT_" + suffix), relative = value("DBDT_STD_" + suffix);
            const bool valid = data != dummy && relative != dummy;
            moment.voltages.push_back(valid ? data * area : 0.0);
            moment.standardErrors.push_back(valid ? std::abs(data * area) * relative : 0.0);
            moment.qualityAccepted.push_back(valid);
        }
        system.apply(moment);
        auto &moments = soundings[found->second].moments;
        moments.push_back(std::move(moment));
        std::sort(moments.begin(), moments.end(),
                  [](const UsfMoment &a, const UsfMoment &b) { return a.meanFrequency > b.meanFrequency; });
    }
    if (skippedRecords)
        *skippedRecords = skipped;
    if (soundings.empty())
        throw std::runtime_error("No soundings found in XYZ file");
    return soundings;
}

} // namespace pytem
