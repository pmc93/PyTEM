#pragma once

#include <limits>
#include <string>
#include <utility>
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
    double time = std::numeric_limits<double>::quiet_NaN(); // recording time, seconds since 1970 (XYZ data)
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

    // Aarhus Workbench XYZ data export (e.g. tTEM): every row is one moment,
    // and rows sharing a DATE and TIME form one sounding (Line<line>_<n>).
    // dB/dt [V/Am^4] is converted to V/Am^2 with TX_AREA. The XYZ holds no
    // system description: it comes from systemPath, a .gex or a TEMcompany
    // stb2xyz file ([RxTxSpecs]: loop, coil position and heights, filters,
    // waveforms, gate edges and time shifts), or with no system file from
    // tTEM defaults: 2 x 4 m loop, receiver 9 m behind, 0.5 m heights,
    // 670 kHz receiver filter, LM 200 us on / 2.5 us off at 2110 Hz, HM
    // 450 us on / 4 us off at 660 Hz.
    // A TEMcompany stb2xyz data file is read too (its header is the system):
    // one row per LM or HM stack, consecutive LM and HM rows forming one
    // sounding (Sounding_<n>), dB/dt [V/m^2] divided by the current and scaled
    // by the moment's DataFactor, gate centres shifted by its GateTimeShift.
    // A line file (.lin: Start/End times per line) drops the records outside
    // every line (turns); their count goes to skippedRecords, and soundings take
    // the line numbers of the file.
    static std::vector<UsfSounding> readWorkbenchXyz(const std::string &path, const std::string &systemPath = {},
                                                     const std::string &linePath = {}, int *skippedRecords = nullptr);
    // The system file for a Workbench XYZ: a .gex of the same name, else the
    // only .gex in its folder, else a TEMcompany .xyz there; empty if none.
    static std::string findSystemFile(const std::string &path);

    // WGS84 longitude/latitude to UTM easting/northing in the given zone.
    static std::pair<double, double> toUtm(double longitude, double latitude, int zone);

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
