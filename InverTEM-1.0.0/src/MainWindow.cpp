#include "MainWindow.h"
#include "InversionWorker.h"
#include "MapTileLoader.h"
#include "MatrixConvolution.h"
#include "VectorKernel.h"
#include "PlotWidget.h"

#include <QApplication>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGuiApplication>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QInputDialog>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QProgressDialog>
#include <QPushButton>
#include <QRadioButton>
#include <QSaveFile>
#include <QSettings>
#include <QShortcut>
#include <QScreen>
#include <QSizePolicy>
#include <QSpinBox>
#include <QStackedWidget>
#include <QStandardPaths>
#include <QStringList>
#include <QSplitter>
#include <QTextStream>
#include <QThread>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace {

constexpr int inversionIterations = 15;
constexpr int inversionAlphaTrials = 7;
// pyTEM invert_joint settings used by the tunoe notebook.
constexpr int pytemJointIterations = 50;
constexpr int pytemJointAlphaTrials = 5;
constexpr int pytemJointStepCount = 75;
constexpr double pytemJointMaxRelativeError = 0.5; // notebook MAX_NOISE
constexpr int transectPointIdOffset = 1000000;
constexpr int fastSciLin = 2, fastSciLog = 3; // Model norm entries that select TemSolver::invertSci

int detectedCpuCoreCount()
{
    const int qtCount = QThread::idealThreadCount();
    const unsigned standardCount = std::thread::hardware_concurrency();
    if (qtCount > 0)
        return qtCount;
    return static_cast<int>(std::max(1u, standardCount));
}

bool validMapCoordinate(const pytem::UsfSounding &sounding)
{
    return std::isfinite(sounding.longitude)
        && std::isfinite(sounding.latitude)
        && (sounding.longitude != 0.0 || sounding.latitude != 0.0);
}

double soundingDistanceMetres(const pytem::UsfSounding &first,
                              const pytem::UsfSounding &second)
{
    if (!validMapCoordinate(first) || !validMapCoordinate(second))
        return 1.0;
    constexpr double radians = 3.14159265358979323846 / 180.0;
    constexpr double earthRadius = 6371000.0;
    const double latitude1 = first.latitude * radians;
    const double latitude2 = second.latitude * radians;
    const double deltaLatitude = (second.latitude - first.latitude) * radians;
    const double deltaLongitude = (second.longitude - first.longitude) * radians;
    const double a = std::sin(deltaLatitude * 0.5)
            * std::sin(deltaLatitude * 0.5)
        + std::cos(latitude1) * std::cos(latitude2)
            * std::sin(deltaLongitude * 0.5)
            * std::sin(deltaLongitude * 0.5);
    const double distance = 2.0 * earthRadius
        * std::atan2(std::sqrt(a), std::sqrt(std::max(0.0, 1.0 - a)));
    return std::isfinite(distance) && distance > 0.0 ? distance : 1.0;
}

QDoubleSpinBox *makeDoubleSpin(double minimum, double maximum, double value,
                               int decimals = 2, double step = 1.0)
{
    auto *spin = new QDoubleSpinBox;
    spin->setRange(minimum, maximum);
    spin->setValue(value);
    spin->setDecimals(decimals);
    spin->setSingleStep(step);
    spin->setKeyboardTracking(false);
    return spin;
}

QJsonArray jsonArray(const std::vector<double> &values)
{
    QJsonArray array;
    for (double value : values) array.append(value);
    return array;
}

QJsonArray jsonArray(const std::vector<std::size_t> &values)
{
    QJsonArray array;
    for (std::size_t value : values)
        array.append(static_cast<double>(value));
    return array;
}

QJsonArray jsonArray(const std::vector<int> &values)
{
    QJsonArray array;
    for (int value : values) array.append(value);
    return array;
}

QJsonArray jsonArray(const std::vector<std::string> &values)
{
    QJsonArray array;
    for (const auto &value : values)
        array.append(QString::fromStdString(value));
    return array;
}

std::vector<double> doubleVector(const QJsonArray &array)
{
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(array.size()));
    for (const auto &value : array) values.push_back(value.toDouble());
    return values;
}

std::vector<int> intVector(const QJsonArray &array)
{
    std::vector<int> values;
    values.reserve(static_cast<std::size_t>(array.size()));
    for (const auto &value : array) values.push_back(value.toInt());
    return values;
}

// Parsed (stacked) USF sounding, embedded in projects so they load without the USF files.
QJsonObject usfJson(const pytem::UsfSounding &sounding)
{
    QJsonObject object;
    object["name"] = QString::fromStdString(sounding.soundingName);
    object["date"] = QString::fromStdString(sounding.date);
    object["voltage_units"] = QString::fromStdString(sounding.voltageUnits);
    object["epsg"] = sounding.epsg;
    object["number"] = sounding.soundingNumber;
    object["source_xy"] = jsonArray(std::vector<double>{sounding.sourceX, sounding.sourceY});
    object["lon_lat_elevation"] = jsonArray(std::vector<double>{sounding.longitude, sounding.latitude, sounding.elevation});
    object["loop_xy"] = jsonArray(std::vector<double>{sounding.loopX, sounding.loopY});
    object["coil_xy"] = jsonArray(std::vector<double>{sounding.coilX, sounding.coilY});
    QJsonArray moments;
    for (const auto &m : sounding.moments) {
        QJsonObject moment;
        moment["name"] = QString::fromStdString(m.name);
        moment["channel"] = m.channel;
        moment["stack_count"] = m.stackCount;
        moment["mean_current"] = m.meanCurrent;
        moment["mean_frequency"] = m.meanFrequency;
        moment["times"] = jsonArray(m.times);
        moment["voltages"] = jsonArray(m.voltages);
        moment["standard_errors"] = jsonArray(m.standardErrors);
        moment["quality"] = jsonArray(std::vector<int>(m.qualityAccepted.begin(), m.qualityAccepted.end()));
        moment["gate_open"] = jsonArray(m.gateOpen);
        moment["gate_close"] = jsonArray(m.gateClose);
        moment["waveform_times"] = jsonArray(m.waveformTimes);
        moment["waveform_amplitudes"] = jsonArray(m.waveformAmplitudes);
        moment["low_pass_frequencies"] = jsonArray(m.lowPassFrequencies);
        moment["low_pass_orders"] = jsonArray(m.lowPassOrders);
        moment["high_pass_frequencies"] = jsonArray(m.highPassFrequencies);
        moment["high_pass_orders"] = jsonArray(m.highPassOrders);
        moment["tx_height"] = m.txHeight;
        moment["rx_height"] = m.rxHeight;
        moments.append(moment);
    }
    object["moments"] = moments;
    return object;
}

pytem::UsfSounding usfFromJson(const QJsonObject &object)
{
    pytem::UsfSounding sounding;
    sounding.soundingName = object["name"].toString().toStdString();
    sounding.date = object["date"].toString().toStdString();
    sounding.voltageUnits = object["voltage_units"].toString().toStdString();
    sounding.epsg = object["epsg"].toInt();
    sounding.soundingNumber = object["number"].toInt();
    const auto source = doubleVector(object["source_xy"].toArray());
    const auto position = doubleVector(object["lon_lat_elevation"].toArray());
    const auto loop = doubleVector(object["loop_xy"].toArray());
    const auto coil = doubleVector(object["coil_xy"].toArray());
    if (source.size() != 2 || position.size() != 3 || loop.size() != 2 || coil.size() != 2)
        throw std::runtime_error("embedded sounding data is incomplete");
    sounding.sourceX = source[0];
    sounding.sourceY = source[1];
    sounding.longitude = position[0];
    sounding.latitude = position[1];
    sounding.elevation = position[2];
    sounding.loopX = loop[0];
    sounding.loopY = loop[1];
    sounding.coilX = coil[0];
    sounding.coilY = coil[1];
    for (const auto &entry : object["moments"].toArray()) {
        const auto moment = entry.toObject();
        pytem::UsfMoment m;
        m.name = moment["name"].toString().toStdString();
        m.channel = moment["channel"].toInt();
        m.stackCount = moment["stack_count"].toInt();
        m.meanCurrent = moment["mean_current"].toDouble();
        m.meanFrequency = moment["mean_frequency"].toDouble();
        m.times = doubleVector(moment["times"].toArray());
        m.voltages = doubleVector(moment["voltages"].toArray());
        m.standardErrors = doubleVector(moment["standard_errors"].toArray());
        for (int quality : intVector(moment["quality"].toArray()))
            m.qualityAccepted.push_back(quality != 0);
        m.gateOpen = doubleVector(moment["gate_open"].toArray());
        m.gateClose = doubleVector(moment["gate_close"].toArray());
        m.waveformTimes = doubleVector(moment["waveform_times"].toArray());
        m.waveformAmplitudes = doubleVector(moment["waveform_amplitudes"].toArray());
        m.lowPassFrequencies = doubleVector(moment["low_pass_frequencies"].toArray());
        m.lowPassOrders = intVector(moment["low_pass_orders"].toArray());
        m.highPassFrequencies = doubleVector(moment["high_pass_frequencies"].toArray());
        m.highPassOrders = intVector(moment["high_pass_orders"].toArray());
        m.txHeight = moment["tx_height"].toDouble();
        m.rxHeight = moment["rx_height"].toDouble();
        const std::size_t gates = m.times.size();
        if (gates == 0 || m.voltages.size() != gates || m.standardErrors.size() != gates
            || m.qualityAccepted.size() != gates || m.gateOpen.size() != gates || m.gateClose.size() != gates)
            throw std::runtime_error("embedded sounding data is inconsistent");
        sounding.moments.push_back(std::move(m));
    }
    if (sounding.moments.empty())
        throw std::runtime_error("embedded sounding data has no moments");
    return sounding;
}

std::vector<std::size_t> sizeVector(const QJsonArray &array)
{
    std::vector<std::size_t> values;
    values.reserve(static_cast<std::size_t>(array.size()));
    for (const auto &value : array)
        values.push_back(static_cast<std::size_t>(value.toDouble()));
    return values;
}

std::vector<std::string> stringVector(const QJsonArray &array)
{
    std::vector<std::string> values;
    values.reserve(static_cast<std::size_t>(array.size()));
    for (const auto &value : array)
        values.push_back(value.toString().toStdString());
    return values;
}

QJsonObject forwardModelJson(const pytem::ForwardModel &model)
{
    QJsonObject object;
    object["thicknesses"] = jsonArray(model.thicknesses);
    object["resistivities"] = jsonArray(model.resistivities);
    object["times"] = jsonArray(model.times);
    object["geometry"] = static_cast<int>(model.geometry);
    object["transform"] = static_cast<int>(model.transform);
    object["euler_order"] = model.eulerOrder;
    object["tx_size"] = model.txSize;
    object["rx_x"] = model.rxX;
    object["altitude"] = model.altitude;
    object["step_times"] = jsonArray(model.stepTimes);
    QJsonArray matrix;
    for (const auto &row : model.responseMatrix) matrix.append(jsonArray(row));
    object["response_matrix"] = matrix;
    object["low_pass_frequencies"] = jsonArray(model.lowPassFrequencies);
    object["low_pass_orders"] = jsonArray(model.lowPassOrders);
    object["high_pass_frequencies"] = jsonArray(model.highPassFrequencies);
    object["high_pass_orders"] = jsonArray(model.highPassOrders);
    object["use_gpu"] = model.useGpu;
    object["max_threads"] = static_cast<double>(model.maxThreads);
    return object;
}

pytem::ForwardModel forwardModelFromJson(const QJsonObject &object)
{
    pytem::ForwardModel model;
    model.thicknesses = doubleVector(object["thicknesses"].toArray());
    model.resistivities = doubleVector(object["resistivities"].toArray());
    model.times = doubleVector(object["times"].toArray());
    model.geometry = static_cast<pytem::Geometry>(object["geometry"].toInt());
    model.transform = static_cast<pytem::TransformMethod>(object["transform"].toInt());
    model.eulerOrder = object["euler_order"].toInt(11);
    model.txSize = object["tx_size"].toDouble(12.5);
    model.rxX = object["rx_x"].toDouble();
    model.altitude = object["altitude"].toDouble(0.0);
    model.stepTimes = doubleVector(object["step_times"].toArray());
    for (const auto &row : object["response_matrix"].toArray())
        model.responseMatrix.push_back(doubleVector(row.toArray()));
    model.lowPassFrequencies = doubleVector(object["low_pass_frequencies"].toArray());
    model.lowPassOrders = intVector(object["low_pass_orders"].toArray());
    model.highPassFrequencies = doubleVector(object["high_pass_frequencies"].toArray());
    model.highPassOrders = intVector(object["high_pass_orders"].toArray());
    model.useGpu = object["use_gpu"].toBool();
    model.maxThreads = static_cast<unsigned>(object["max_threads"].toDouble());
    return model;
}

QJsonObject optionsJson(const pytem::InversionOptions &options)
{
    QJsonObject object;
    object["model"] = forwardModelJson(options.model);
    object["observed"] = jsonArray(options.observed);
    object["noise_std"] = jsonArray(options.noiseStd);
    QJsonArray additional;
    for (const auto &dataSet : options.additionalDataSets) {
        QJsonObject data;
        data["model"] = forwardModelJson(dataSet.model);
        data["observed"] = jsonArray(dataSet.observed);
        data["noise_std"] = jsonArray(dataSet.noiseStd);
        additional.append(data);
    }
    object["additional_datasets"] = additional;
    object["rho_min"] = options.rhoMin;
    object["rho_max"] = options.rhoMax;
    object["max_iterations"] = options.maxIterations;
    object["alpha_steps"] = options.alphaSteps;
    object["alpha_log_step"] = options.alphaLogStep;
    object["adaptive_alpha_search"] = options.adaptiveAlphaSearch;
    object["jacobian_step"] = options.jacobianStep;
    object["jacobian_method"] = static_cast<int>(options.jacobianMethod);
    object["jacobian_update_method"]
        = static_cast<int>(options.jacobianUpdateMethod);
    object["broyden_refresh_interval"] = options.broydenRefreshInterval;
    object["regularization_norm"] = static_cast<int>(options.regularizationNorm);
    object["cache_first_jacobian"] = options.cacheFirstJacobian;
    object["jacobian_cache_directory"] = QString::fromStdString(options.jacobianCacheDirectory);
    object["calculate_sensitivity"] = options.calculateSensitivity;
    object["pytem_joint"] = options.pytemJoint;
    object["doi_threshold"] = options.doiThreshold;
    object["doi_conservative_threshold"] = options.doiConservativeThreshold;
    object["doi_refinement"] = options.doiRefinement;
    object["vectorized_kernel"] = options.vectorizedKernel;
    object["spatial_constraints"] = options.spatialConstraints;
    object["sci_log_data_space"] = options.sci.logDataSpace;
    return object;
}

pytem::InversionOptions optionsFromJson(const QJsonObject &object)
{
    pytem::InversionOptions options;
    options.model = forwardModelFromJson(object["model"].toObject());
    options.observed = doubleVector(object["observed"].toArray());
    options.noiseStd = doubleVector(object["noise_std"].toArray());
    for (const auto &entry : object["additional_datasets"].toArray()) {
        const auto data = entry.toObject();
        pytem::InversionDataSet dataSet;
        dataSet.model = forwardModelFromJson(data["model"].toObject());
        dataSet.observed = doubleVector(data["observed"].toArray());
        dataSet.noiseStd = doubleVector(data["noise_std"].toArray());
        options.additionalDataSets.push_back(std::move(dataSet));
    }
    options.rhoMin = object["rho_min"].toDouble(0.1);
    options.rhoMax = object["rho_max"].toDouble(1.0e5);
    options.maxIterations = object["max_iterations"].toInt(20);
    options.alphaSteps = object["alpha_steps"].toInt(8);
    options.alphaLogStep = object["alpha_log_step"].toDouble(1.0 / 9.0);
    options.adaptiveAlphaSearch = object["adaptive_alpha_search"].toBool(true);
    options.jacobianStep = object["jacobian_step"].toDouble(1.0e-4);
    options.jacobianMethod = static_cast<pytem::JacobianMethod>(
        object["jacobian_method"].toInt());
    options.jacobianUpdateMethod = static_cast<pytem::JacobianUpdateMethod>(
        object["jacobian_update_method"].toInt(
            static_cast<int>(pytem::JacobianUpdateMethod::FullEveryIteration)));
    options.broydenRefreshInterval
        = object["broyden_refresh_interval"].toInt(3);
    options.regularizationNorm = static_cast<pytem::RegularizationNorm>(
        object["regularization_norm"].toInt());
    options.cacheFirstJacobian = object["cache_first_jacobian"].toBool();
    options.jacobianCacheDirectory = object["jacobian_cache_directory"].toString().toStdString();
    options.calculateSensitivity = object["calculate_sensitivity"].toBool(true);
    options.pytemJoint = object["pytem_joint"].toBool(false);
    options.doiThreshold = object["doi_threshold"].toDouble(options.doiThreshold);
    options.doiConservativeThreshold
        = object["doi_conservative_threshold"].toDouble(options.doiConservativeThreshold);
    options.doiRefinement = object["doi_refinement"].toInt(options.doiRefinement);
    options.vectorizedKernel = object["vectorized_kernel"].toBool(false);
    options.spatialConstraints = object["spatial_constraints"].toBool(false);
    options.sci.logDataSpace = object["sci_log_data_space"].toBool(false);
    return options;
}

QJsonObject resultJson(const pytem::InversionResult &result)
{
    QJsonObject object;
    object["resistivities"] = jsonArray(result.resistivities);
    object["predicted"] = jsonArray(result.predicted);
    object["rms_history"] = jsonArray(result.rmsHistory);
    object["sensitivity"] = jsonArray(result.sensitivity);
    object["iterations"] = result.iterations;
    object["converged"] = result.converged;
    object["message"] = QString::fromStdString(result.message);
    object["first_jacobian_source"] = QString::fromStdString(result.firstJacobianSource);
    object["alpha_final"] = result.alphaFinal;
    object["doi_standard"] = result.doiStandard;
    object["doi_conservative"] = result.doiConservative;
    object["doi_standard_capped"] = result.doiStandardCapped;
    object["doi_conservative_capped"] = result.doiConservativeCapped;
    QJsonObject timing;
    timing["total"] = result.timing.totalSeconds;
    timing["initial_forward"] = result.timing.initialForwardSeconds;
    timing["jacobian"] = result.timing.jacobianSeconds;
    timing["broyden_update"] = result.timing.broydenUpdateSeconds;
    timing["linear_solve"] = result.timing.linearSolveSeconds;
    timing["alpha_forward"] = result.timing.alphaForwardSeconds;
    timing["sensitivity"] = result.timing.sensitivitySeconds;
    timing["jacobian_evaluations"] = result.timing.jacobianEvaluations;
    timing["broyden_updates"] = result.timing.broydenUpdates;
    timing["alpha_trial_forwards"] = result.timing.alphaTrialForwards;
    object["timing"] = timing;
    return object;
}

pytem::InversionResult resultFromJson(const QJsonObject &object)
{
    pytem::InversionResult result;
    result.resistivities = doubleVector(object["resistivities"].toArray());
    result.predicted = doubleVector(object["predicted"].toArray());
    result.rmsHistory = doubleVector(object["rms_history"].toArray());
    result.sensitivity = doubleVector(object["sensitivity"].toArray());
    result.iterations = object["iterations"].toInt();
    result.converged = object["converged"].toBool();
    result.message = object["message"].toString().toStdString();
    result.firstJacobianSource = object["first_jacobian_source"].toString().toStdString();
    result.alphaFinal = object["alpha_final"].toDouble(-1.0);
    result.doiStandard = object["doi_standard"].toDouble(-1.0);
    result.doiConservative = object["doi_conservative"].toDouble(-1.0);
    result.doiStandardCapped = object["doi_standard_capped"].toBool();
    result.doiConservativeCapped = object["doi_conservative_capped"].toBool();
    const auto timing = object["timing"].toObject();
    result.timing.totalSeconds = timing["total"].toDouble();
    result.timing.initialForwardSeconds = timing["initial_forward"].toDouble();
    result.timing.jacobianSeconds = timing["jacobian"].toDouble();
    result.timing.broydenUpdateSeconds
        = timing["broyden_update"].toDouble();
    result.timing.linearSolveSeconds = timing["linear_solve"].toDouble();
    result.timing.alphaForwardSeconds = timing["alpha_forward"].toDouble();
    result.timing.sensitivitySeconds = timing["sensitivity"].toDouble();
    result.timing.jacobianEvaluations = timing["jacobian_evaluations"].toInt();
    result.timing.broydenUpdates = timing["broyden_updates"].toInt();
    result.timing.alphaTrialForwards = timing["alpha_trial_forwards"].toInt();
    return result;
}

QColor momentColor(const std::string &name, std::size_t fallbackIndex)
{
    const QString upper = QString::fromStdString(name).toUpper();
    if (upper.contains("HM") || upper.contains("HIGH"))
        return QColor("#d73027");
    if (upper.contains("LM") || upper.contains("LOW"))
        return QColor("#2166ac");
    const QColor fallback[] = {QColor("#2166ac"), QColor("#d73027"),
                               QColor("#1b9e77"), QColor("#984ea3")};
    return fallback[fallbackIndex % 4];
}

QString formatDuration(qint64 milliseconds)
{
    qint64 seconds = std::max<qint64>(0, milliseconds / 1000);
    const qint64 hours = seconds / 3600;
    seconds %= 3600;
    const qint64 minutes = seconds / 60;
    seconds %= 60;
    return hours > 0
        ? QString("%1:%2:%3").arg(hours).arg(minutes, 2, 10, QChar('0')).arg(seconds, 2, 10, QChar('0'))
        : QString("%1:%2").arg(minutes).arg(seconds, 2, 10, QChar('0'));
}

double finiteModelDepth(const pytem::InversionOptions &options)
{
    return std::accumulate(options.model.thicknesses.begin(),
                           options.model.thicknesses.end(), 0.0);
}

QString savedModelKey(const pytem::InversionOptions &options, int momentChoice)
{
    return QString("%1|%2|%3|%4")
        .arg(momentChoice)
        .arg(options.spatialConstraints ? (options.sci.logDataSpace ? fastSciLog : fastSciLin)
                                        : static_cast<int>(options.regularizationNorm))
        .arg(options.model.resistivities.size())
        .arg(finiteModelDepth(options), 0, 'g', 12);
}

// "L1 (40 layers, 120.00 m)", or "L1 (40 layers)" for plot legends.
QString savedModelLabel(const pytem::InversionOptions &options, bool withDepth = true)
{
    const QString norm = options.spatialConstraints ? (options.sci.logDataSpace ? "SCI" : "SCI lin")
        : options.regularizationNorm == pytem::RegularizationNorm::L1Blocky ? "L1" : "L2";
    const QString depth = withDepth ? QString(", %1 m").arg(finiteModelDepth(options), 0, 'f', 2) : QString();
    return QString("%1 (%2 layers%3)").arg(norm).arg(options.model.resistivities.size()).arg(depth);
}

double sensitivityDoi(const pytem::InversionOptions &options,
                      const pytem::InversionResult &result)
{
    if (result.doiStandard >= 0.0)
        return result.doiStandard;
    return pytem::TemSolver::depthOfInvestigation(
        options.model.thicknesses, result.sensitivity);
}

struct SplitModelCurve {
    QVector<QPointF> withinDoi;
    QVector<QPointF> beyondDoi;
};

SplitModelCurve splitModelCurveAtDoi(const QVector<QPointF> &points,
                                     double doiDepth)
{
    if (!(doiDepth > 0.0) || !std::isfinite(doiDepth))
        return {points, {}};

    SplitModelCurve result;
    auto appendDistinct = [](QVector<QPointF> &destination,
                             const QPointF &point) {
        if (destination.isEmpty() || destination.back() != point)
            destination.push_back(point);
    };
    for (int index = 0; index < points.size(); ++index) {
        const QPointF point = points[index];
        if (index > 0) {
            const QPointF previous = points[index - 1];
            if (previous.y() < doiDepth && point.y() > doiDepth) {
                const double fraction = (doiDepth - previous.y())
                    / (point.y() - previous.y());
                const QPointF crossing(
                    previous.x() + fraction * (point.x() - previous.x()),
                    doiDepth);
                appendDistinct(result.withinDoi, crossing);
                appendDistinct(result.beyondDoi, crossing);
            }
        }
        if (point.y() <= doiDepth)
            appendDistinct(result.withinDoi, point);
        if (point.y() >= doiDepth)
            appendDistinct(result.beyondDoi, point);
    }
    return result;
}

int xyzLineNumber(const pytem::UsfSounding &sounding)
{
    std::string upper = sounding.soundingName;
    std::transform(upper.begin(), upper.end(), upper.begin(),
                   [](unsigned char character) {
                       return static_cast<char>(std::toupper(character));
                   });
    const auto line = upper.find("LINE");
    if (line != std::string::npos) {
        std::size_t firstDigit = line + 4;
        while (firstDigit < upper.size()
               && !std::isdigit(static_cast<unsigned char>(upper[firstDigit])))
            ++firstDigit;
        std::size_t endDigit = firstDigit;
        while (endDigit < upper.size()
               && std::isdigit(static_cast<unsigned char>(upper[endDigit])))
            ++endDigit;
        if (endDigit > firstDigit)
            return std::stoi(upper.substr(firstDigit, endDigit - firstDigit));
    }
    return sounding.soundingNumber > 0 ? sounding.soundingNumber : 1;
}

} // namespace

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    qRegisterMetaType<pytem::InversionResult>("pytem::InversionResult");
    readSolverSettings();
    buildInterface();
    if (const auto *screen = QGuiApplication::primaryScreen()) {
        const QRect available = screen->availableGeometry();
        const int width = std::min(available.width(),
            std::min(1600, std::max(640, available.width() - 40)));
        const int height = std::min(available.height(),
            std::min(950, std::max(480, available.height() - 60)));
        resize(width, height);
        move(available.center() - rect().center());
    } else {
        resize(1400, 850);
    }
    setRunning(false);
    updateProjectTitle();
}

MainWindow::~MainWindow()
{
    if (m_worker) m_worker->requestCancel();
    if (m_workerThread) {
        m_workerThread->quit();
        m_workerThread->wait();
    }
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if (m_worker) {
        const auto answer = QMessageBox::question(this, "Inversion is running",
            "Stop the inversion and close InverTEM? Completed models will be kept.",
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (answer != QMessageBox::Yes) {
            event->ignore();
            return;
        }
        m_worker->requestCancel();
    }
    if (!confirmProjectReplacement()) {
        event->ignore();
        return;
    }
    event->accept();
}

void MainWindow::buildInterface()
{
    auto *central = new QWidget;
    auto *root = new QVBoxLayout(central);
    setCentralWidget(central);

    auto *informationSidebar = new QWidget;
    informationSidebar->setMinimumWidth(230);
    informationSidebar->setMaximumWidth(320);
    auto *informationLayout = new QVBoxLayout(informationSidebar);
    m_projectLabel = new QLabel("Project: Untitled");
    m_projectLabel->setWordWrap(true);
    auto *projectButtons = new QHBoxLayout;
    projectButtons->setContentsMargins(0, 0, 0, 0);
    m_newProjectButton = new QPushButton("New");
    m_loadProjectButton = new QPushButton("Load");
    m_saveProjectButton = new QPushButton("Save");
    m_newProjectButton->setToolTip("Create a new InverTEM project");
    m_loadProjectButton->setToolTip("Load an InverTEM project");
    m_saveProjectButton->setToolTip("Save this project, including inversion results");
    projectButtons->addWidget(m_newProjectButton);
    projectButtons->addWidget(m_loadProjectButton);
    projectButtons->addWidget(m_saveProjectButton);
    informationLayout->addWidget(m_projectLabel);
    informationLayout->addLayout(projectButtons);
    m_importButton = new QPushButton("Import one or more .usf files...");
    m_importButton->setToolTip(
        "Add soundings that match the project geometry");
    m_geometryLabel = new QLabel("Geometry: no sounding loaded");
    m_geometryLabel->setWordWrap(true);
    m_geometryLabel->setMinimumWidth(0);
    m_geometryLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    m_systemLabel = new QLabel("Filters and gates are read from the selected moment(s).");
    m_systemLabel->setWordWrap(true);
    m_systemLabel->setMinimumWidth(0);
    m_systemLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    m_maxDepth = makeDoubleSpin(0.2, 100000.0, 120.0, 1, 10.0);
    m_maxDepth->setSuffix(" m");
    m_maxDepth->setToolTip(
        "Depth to the top of layer N, the half-space (the base of layer N-1).");
    m_firstDepth = makeDoubleSpin(0.1, 10000.0, 2.0, 2, 0.5);
    m_firstDepth->setSuffix(" m");
    m_firstDepth->setToolTip(
        "Thickness of layer 1 (depth to its base). Layer boundaries are log-spaced from here "
        "to the top of layer N, as in the pyTEM notebook (2 m to 120 m).");
    m_layerCount = new QSpinBox;
    m_layerCount->setRange(3, 100);
    m_layerCount->setValue(15);
    m_autoResistivityAxis = new QCheckBox("Auto resistivity axis");
    m_autoResistivityAxis->setChecked(true);
    m_resistivityAxisMinimum = makeDoubleSpin(
        0.01, 1000000.0, 1.0, 2, 1.0);
    m_resistivityAxisMaximum = makeDoubleSpin(
        0.02, 10000000.0, 1000.0, 2, 100.0);
    m_resistivityAxisMinimum->setSuffix(" Ohm m");
    m_resistivityAxisMaximum->setSuffix(" Ohm m");
    m_resistivityAxisMinimum->setEnabled(false);
    m_resistivityAxisMaximum->setEnabled(false);

    m_previousButton = new QPushButton("◀");
    m_previousButton->setToolTip("Previous sounding and recovered model");
    m_previousButton->setEnabled(false);
    m_previousButton->setMaximumWidth(52);
    m_soundingSelector = new QComboBox;
    m_soundingSelector->setEnabled(false);
    m_soundingSelector->setMinimumContentsLength(8);
    m_soundingSelector->setSizeAdjustPolicy(
        QComboBox::AdjustToMinimumContentsLengthWithIcon);
    m_soundingSelector->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    m_nextButton = new QPushButton("▶");
    m_nextButton->setToolTip("Next sounding and recovered model");
    m_nextButton->setEnabled(false);
    m_nextButton->setMaximumWidth(52);
    m_moment = new QComboBox;
    m_moment->setEnabled(false);
    m_moment->setMinimumContentsLength(8);
    m_moment->setSizeAdjustPolicy(
        QComboBox::AdjustToMinimumContentsLengthWithIcon);
    m_moment->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Fixed);
    auto *soundingNavigation = new QHBoxLayout;
    soundingNavigation->setContentsMargins(0, 0, 0, 0);
    soundingNavigation->addWidget(m_previousButton);
    soundingNavigation->addWidget(m_soundingSelector, 1);
    soundingNavigation->addWidget(m_nextButton);
    informationLayout->addWidget(m_importButton);
    informationLayout->addLayout(soundingNavigation);
    informationLayout->addWidget(new QLabel("Moment"));
    informationLayout->addWidget(m_moment);
    m_plotMode = new QComboBox;
    m_plotMode->addItem("Single sounding");
    m_plotMode->addItem("Sounding transect");
    m_transectSize = new QSpinBox;
    m_transectSize->setRange(2, 500);
    m_transectSize->setValue(30);
    m_transectSize->setSuffix(" soundings");
    m_transectSize->setToolTip("Number of soundings shown on each transect page");
    informationLayout->addWidget(new QLabel("Data view"));
    auto *plotModeRow = new QHBoxLayout;
    plotModeRow->setContentsMargins(0, 0, 0, 0);
    plotModeRow->addWidget(m_plotMode, 1);
    plotModeRow->addWidget(m_transectSize);
    informationLayout->addLayout(plotModeRow);
    auto *transectNavigation = new QHBoxLayout;
    transectNavigation->setContentsMargins(0, 0, 0, 0);
    m_previousTransectButton = new QPushButton("◀");
    m_nextTransectButton = new QPushButton("▶");
    m_transectRangeLabel = new QLabel("0 / 0");
    m_transectRangeLabel->setAlignment(Qt::AlignCenter);
    transectNavigation->addWidget(m_previousTransectButton);
    transectNavigation->addWidget(m_transectRangeLabel, 1);
    transectNavigation->addWidget(m_nextTransectButton);
    informationLayout->addLayout(transectNavigation);
    m_previousTransectButton->setVisible(false);
    m_nextTransectButton->setVisible(false);
    m_transectRangeLabel->setVisible(false);
    m_transectSize->setVisible(false);

    auto *geometryColumn = new QGroupBox;
    auto *geometryColumnLayout = new QVBoxLayout(geometryColumn);
    geometryColumnLayout->addWidget(m_geometryLabel);
    geometryColumnLayout->addStretch();
    auto *systemColumn = new QGroupBox;
    auto *systemColumnLayout = new QVBoxLayout(systemColumn);
    systemColumnLayout->addWidget(m_systemLabel);
    systemColumnLayout->addStretch();
    informationLayout->addWidget(geometryColumn);
    informationLayout->addWidget(systemColumn);
    auto *layerColumn = new QGroupBox("Layer model");
    auto *layerForm = new QFormLayout(layerColumn);
    layerForm->addRow("Number of layers", m_layerCount);
    layerForm->addRow("Thickness of layer 1", m_firstDepth);
    layerForm->addRow("Depth to top of layer N", m_maxDepth);
    informationLayout->addStretch();

    auto *workspaceSplitter = new QSplitter(Qt::Horizontal);
    auto *leftWorkspace = new QWidget;
    auto *leftWorkspaceLayout = new QVBoxLayout(leftWorkspace);
    leftWorkspaceLayout->setContentsMargins(0, 0, 0, 0);
    auto *plotSplitter = new QSplitter(Qt::Horizontal);
    m_soundingPlot = new PlotWidget;
    m_soundingPlot->setAxes("TEM data",
                            "Time [s]", "|dB/dt| [V/Am²]", true, true);
    m_soundingPlot->setScientificX(true);
    const QString gateHelp = "Right-click a gate: toggle it\nRight-drag a box: remove the gates in it\n"
                             "Shift + right-drag: restore them\nCtrl+Z: undo the last gate or inclusion edit";
    m_soundingPlot->setToolTip(gateHelp);
    m_modelPlot = new PlotWidget;
    m_modelPlot->setAxes("Model", "Resistivity [Ohm m]",
                         "Depth [m]", true, false, true);
    m_modelPlot->setMinimumY(0.0);
    m_modelPlot->setNiceLogXTicks(true);
    m_modelPlot->setLegendBackground(true);
    auto *singlePlotSplitter = new QSplitter(Qt::Horizontal);
    singlePlotSplitter->addWidget(m_soundingPlot);
    singlePlotSplitter->addWidget(m_modelPlot);
    singlePlotSplitter->setStretchFactor(0, 1);
    singlePlotSplitter->setStretchFactor(1, 1);
    singlePlotSplitter->setSizes({360, 360});
    auto *transectPage = new QWidget;
    auto *transectLayout = new QVBoxLayout(transectPage);
    transectLayout->setContentsMargins(0, 0, 0, 0);
    m_transectLowPlot = new PlotWidget;
    m_transectLowPlot->setAxes(
        "LM transect",
        "", "|dB/dt| [V/Am²]", false, true); // shares the distance axis label of the HM plot below
    m_transectHighPlot = new PlotWidget;
    m_transectHighPlot->setAxes(
        "HM transect",
        "Distance [m]", "|dB/dt| [V/Am²]", false, true);
    m_transectLowPlot->setToolTip(gateHelp);
    m_transectLowPlot->setYTickIntervals(3); // four y labels on the short transect panels
    m_transectHighPlot->setYTickIntervals(3);
    m_transectHighPlot->setToolTip(gateHelp);
    transectLayout->addWidget(m_transectLowPlot, 1);
    transectLayout->addWidget(m_transectHighPlot, 1);
    m_plotModeStack = new QStackedWidget;
    m_plotModeStack->addWidget(singlePlotSplitter);
    m_plotModeStack->addWidget(transectPage);
    auto *mapPanel = new QWidget;
    mapPanel->setMinimumWidth(460);
    mapPanel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    auto *mapPanelLayout = new QVBoxLayout(mapPanel);
    mapPanelLayout->setContentsMargins(0, 0, 0, 0);
    auto *mapControls = new QHBoxLayout;
    m_noMap = new QRadioButton("No map");
    m_openStreetMap = new QRadioButton("OpenStreetMap");
    m_satelliteMap = new QRadioButton("Satellite map");
    m_mapPointSize = new QComboBox;
    m_mapPointSize->addItem("Small", 3.2);
    m_mapPointSize->addItem("Medium", 4.8);
    m_mapPointSize->addItem("Large", 6.5);
    m_mapPointSize->addItem("Extra large", 8.5);
    m_mapPointSize->setCurrentIndex(1);
    m_mapPointSize->setToolTip(
        "Marker size for sounding points on the map.");
    m_noMap->setChecked(true);
    m_mapProgress = new QProgressBar;
    m_mapProgress->setRange(0, 49);
    m_mapProgress->setFormat("Map tiles %v/%m");
    m_mapProgress->setFixedWidth(120);
    QSizePolicy mapProgressPolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    mapProgressPolicy.setRetainSizeWhenHidden(true);
    m_mapProgress->setSizePolicy(mapProgressPolicy);
    m_mapProgress->setVisible(false);
    mapControls->addWidget(m_noMap);
    mapControls->addWidget(m_openStreetMap);
    mapControls->addWidget(m_satelliteMap);
    mapControls->addSpacing(10);
    mapControls->addWidget(new QLabel("Point size"));
    mapControls->addWidget(m_mapPointSize);
    mapControls->addWidget(m_mapProgress, 1);
    m_mapPlot = new PlotWidget;
    m_mapPlot->setAxes("Sounding map (left-click selects; right-click includes/excludes)",
                       "Easting", "Northing", false, false);
    m_mapPlot->setEqualAspect(true);
    m_mapPlot->setGeographicScaleBar(true);
    m_mapPlot->setLegendBackground(true);
    m_mapPlot->setTickLabelsVisible(false);
    m_mapPlot->setMarkerRadius(m_mapPointSize->currentData().toDouble());
    mapPanelLayout->addWidget(m_mapPlot, 1);
    mapPanelLayout->addLayout(mapControls);
    plotSplitter->addWidget(informationSidebar);
    plotSplitter->addWidget(m_plotModeStack);
    plotSplitter->setStretchFactor(0, 0);
    plotSplitter->setStretchFactor(1, 1);
    plotSplitter->setSizes({250, 720});
    leftWorkspaceLayout->addWidget(plotSplitter, 1);
    workspaceSplitter->addWidget(leftWorkspace);
    workspaceSplitter->addWidget(mapPanel);
    workspaceSplitter->setStretchFactor(0, 5);
    workspaceSplitter->setStretchFactor(1, 4);
    workspaceSplitter->setSizes({880, 720});

    auto *settingsGroup = new QGroupBox;
    auto *settings = new QVBoxLayout(settingsGroup);
    m_errorFloor = makeDoubleSpin(0.01, 100.0, 5.0, 2, 0.5);
    m_errorFloor->setSuffix(" %");
    m_parallelJobs = new QSpinBox;
    const int detectedCores = detectedCpuCoreCount();
    m_parallelJobs->setRange(1, detectedCores);
    m_parallelJobs->setValue(detectedCores);
    m_parallelJobs->setToolTip(
        "Maximum CPU workers used by the independent batch scheduler. "
        "Soundings remain mathematically independent.");
    m_regularizationNorm = new QComboBox;
    m_regularizationNorm->addItem("L1 blocky",
        static_cast<int>(pytem::RegularizationNorm::L1Blocky));
    m_regularizationNorm->addItem("L2 smooth",
        static_cast<int>(pytem::RegularizationNorm::L2Smooth));
    // Fast SCI in log data space; the linear-data-space variant (fastSciLin) is hidden for now.
    m_regularizationNorm->addItem("SCI", fastSciLog);
    m_regularizationNorm->setItemData(2,
        "Spatially constrained inversion: all included soundings together, with vertical constraints and "
        "constraints between map neighbours (Delaunay), Marquardt damping by step length, as in Lupus. "
        "Constraint strengths are set in invertem_solver.txt.", Qt::ToolTipRole);
    m_cacheJacobian = new QCheckBox("Reuse/cache first Jacobian");
    m_cacheJacobian->setChecked(true);
    m_cacheJacobian->setToolTip("Shared in memory across parallel jobs and saved on disk for compatible future runs.");
    m_sensitivity = new QCheckBox("Calculate final sensitivity and DOI");
    m_sensitivity->setChecked(true);
    m_adaptiveAlpha = new QCheckBox("Adaptive alpha search");
    m_adaptiveAlpha->setChecked(!m_pytemJoint);
    m_adaptiveAlpha->setToolTip(m_pytemJoint
        ? "When an alpha trial undershoots RMS 1, run one extra trial at the alpha interpolated "
          "to RMS 1, so each iteration keeps the smoothest model that fits (RMS up to 1.005). "
          "Off reproduces the pyTEM notebook, which accepts the first trial below RMS 1."
        : "Adjust alpha independently for each sounding. Invalid or unstable "
          "trials retain the best valid model and fall back to the fixed alpha "
          "sweep. Uncheck for the original reproducible fixed search.");
    std::string gpuReason;
    const bool gpuAvailable = pytem::TemSolver::gpuAvailable(&gpuReason);
    m_useGpu = new QCheckBox("Use NVIDIA GPU (CUDA)");
    const bool dlf = m_transform == pytem::TransformMethod::DigitalLinearFilter;
    m_useGpu->setChecked(gpuAvailable && dlf && !m_pytemJoint);
    m_useGpu->setEnabled(gpuAvailable && dlf && !m_pytemJoint);
    m_useGpu->setToolTip(!gpuAvailable
        ? QString("CPU only: %1").arg(QString::fromStdString(gpuReason))
        : (dlf && !m_pytemJoint ? QString("Accelerates DLF forward responses and alpha trials on the GPU.")
               : QString("GPU acceleration needs method = adaptive and transform = dlf in invertem_solver.txt.")));

    auto *columns = new QHBoxLayout;
    columns->addWidget(layerColumn, 1);

    auto *inversionColumn = new QGroupBox("Solver");
    auto *inversionForm = new QFormLayout(inversionColumn);
    inversionForm->addRow("Error floor", m_errorFloor);
    inversionForm->addRow("Regularisation", m_regularizationNorm);
    inversionForm->addRow(m_adaptiveAlpha);
    inversionForm->addRow(m_useGpu);
    columns->addWidget(inversionColumn, 1);

    auto *batchColumn = new QGroupBox("Batch and output");
    auto *batchForm = new QFormLayout(batchColumn);
    auto *coreCount = new QLabel(QString("%1 available").arg(detectedCores));
    coreCount->setToolTip(
        "Logical CPU processors reported by the operating system. "
        "The batch-worker setting cannot exceed this value.");
    batchForm->addRow("CPU cores/threads", coreCount);
    batchForm->addRow("Batch workers", m_parallelJobs);
    batchForm->addRow(m_cacheJacobian);
    batchForm->addRow(m_sensitivity);
    batchForm->addRow(m_autoResistivityAxis);
    batchForm->addRow("Axis minimum", m_resistivityAxisMinimum);
    batchForm->addRow("Axis maximum", m_resistivityAxisMaximum);
    columns->addWidget(batchColumn, 1);
    settings->addLayout(columns);

    m_runButton = new QPushButton("Run batch inversion");
    m_runButton->setDefault(true);
    m_runButton->setEnabled(false);
    m_killButton = new QPushButton("Kill inversion");
    m_killButton->setToolTip(
        "Stop active and queued inversions while keeping completed models.");
    m_killButton->setEnabled(false);
    m_exportModelledButton = new QPushButton("Export modelled data (.xyz)...");
    m_exportModelledButton->setToolTip(
        "Export one saved model configuration for all completed soundings in Aarhus Workbench XYZ format.");
    m_exportModelledButton->setEnabled(false);
    m_autoFilterButton = new QPushButton("Auto-filter gates");
    m_autoFilterButton->setToolTip(
        "Re-select the gates of every sounding automatically: quality, SNR >= 3, nothing after a sign "
        "change, and culling of gates that break a smooth decay (late-time noise is cut; a moment that "
        "is mostly not smooth is dropped). Replaces manual gate edits.");
    m_autoFilterButton->setEnabled(false);
    m_moveKeptButton = new QPushButton("Move kept USF files...");
    m_moveKeptButton->setToolTip(
        "Move the USF files of every sounding included in the batch (not excluded on the map) "
        "to a new folder. The project follows the files; gate edits stay in the project.");
    m_moveKeptButton->setEnabled(false);
    m_clearResultsButton = new QPushButton("Clear saved results");
    m_clearResultsButton->setToolTip(
        "Remove all completed inversion models from memory; Jacobian cache files are kept.");
    m_clearResultsButton->setEnabled(false);
    m_progress = new QProgressBar;
    m_progress->setRange(0, inversionIterations);
    m_progress->setFormat("%p%");
    m_timeEstimate = new QLabel("Elapsed 00:00 — ETA estimating…");
    m_timeEstimate->setFixedWidth(190);
    m_status = new QLabel("0/0 completed");
    m_status->setMinimumWidth(0);
    m_status->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    auto *actions = new QHBoxLayout;
    actions->addWidget(m_runButton);
    actions->addWidget(m_killButton);
    actions->addWidget(m_progress, 1);
    actions->addWidget(m_timeEstimate);
    actions->addWidget(m_status, 1);
    settings->addLayout(actions);
    auto *resultActions = new QHBoxLayout;
    resultActions->addWidget(m_exportModelledButton);
    resultActions->addWidget(m_autoFilterButton);
    resultActions->addWidget(m_moveKeptButton);
    resultActions->addWidget(m_clearResultsButton);
    resultActions->addStretch();
    settings->addLayout(resultActions);
    m_log = new QPlainTextEdit;
    m_log->setReadOnly(true);
    m_log->setMaximumBlockCount(1000);
    m_log->setMaximumHeight(90);
    settings->addWidget(m_log);
    leftWorkspaceLayout->addWidget(settingsGroup);
    root->addWidget(workspaceSplitter, 1);

    m_elapsedTimer = new QTimer(this);
    m_elapsedTimer->setInterval(1000);
    connect(m_elapsedTimer, &QTimer::timeout,
            this, &MainWindow::updateBatchTiming);
    // Coalesces per-iteration map repaints (live RMS colours) during a batch.
    m_mapRefreshTimer = new QTimer(this);
    m_mapRefreshTimer->setSingleShot(true);
    m_mapRefreshTimer->setInterval(250);
    connect(m_mapRefreshTimer, &QTimer::timeout, this, &MainWindow::updateMapPlot);

    connect(m_newProjectButton, &QPushButton::clicked,
            this, &MainWindow::newProject);
    connect(m_loadProjectButton, &QPushButton::clicked,
            this, &MainWindow::loadProject);
    connect(m_saveProjectButton, &QPushButton::clicked,
            this, &MainWindow::saveProject);
    connect(m_importButton, &QPushButton::clicked, this, &MainWindow::loadUsf);
    connect(m_soundingSelector, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &MainWindow::selectSounding);
    connect(m_previousButton, &QPushButton::clicked, this, &MainWindow::previousSounding);
    connect(m_nextButton, &QPushButton::clicked, this, &MainWindow::nextSounding);
    connect(m_moment, qOverload<int>(&QComboBox::currentIndexChanged), this, &MainWindow::selectMoment);
    connect(m_mapPointSize, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this](int) {
                m_mapPlot->setMarkerRadius(
                    m_mapPointSize->currentData().toDouble());
                markProjectModified();
            });
    connect(m_layerCount, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::setLayerCount);
    connect(m_autoResistivityAxis, &QCheckBox::toggled, this,
            [this](bool automatic) {
                m_resistivityAxisMinimum->setEnabled(!automatic);
                m_resistivityAxisMaximum->setEnabled(!automatic);
                updateResistivityAxis();
            });
    connect(m_resistivityAxisMinimum,
            qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            [this](double minimum) {
                if (minimum >= m_resistivityAxisMaximum->value())
                    m_resistivityAxisMaximum->setValue(minimum * 10.0);
                updateResistivityAxis();
            });
    connect(m_resistivityAxisMaximum,
            qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            [this](double) { updateResistivityAxis(); });
    const auto projectChanged = [this] { markProjectModified(); };
    connect(m_firstDepth, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, projectChanged);
    connect(m_maxDepth, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, projectChanged);
    connect(m_errorFloor, qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, projectChanged);
    connect(m_parallelJobs, qOverload<int>(&QSpinBox::valueChanged),
            this, projectChanged);
    connect(m_regularizationNorm, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this, projectChanged](int) {
                setRunning(m_worker != nullptr);
                projectChanged();
            });
    connect(m_useGpu, &QCheckBox::toggled, this, projectChanged);
    connect(m_cacheJacobian, &QCheckBox::toggled, this, projectChanged);
    connect(m_adaptiveAlpha, &QCheckBox::toggled, this, projectChanged);
    connect(m_sensitivity, &QCheckBox::toggled, this, projectChanged);
    connect(m_moment, qOverload<int>(&QComboBox::currentIndexChanged),
            this, projectChanged);
    connect(m_autoResistivityAxis, &QCheckBox::toggled,
            this, projectChanged);
    connect(m_resistivityAxisMinimum,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, projectChanged);
    connect(m_resistivityAxisMaximum,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this, projectChanged);
    connect(m_noMap, &QRadioButton::toggled, this,
            [projectChanged](bool checked) { if (checked) projectChanged(); });
    connect(m_openStreetMap, &QRadioButton::toggled, this,
            [projectChanged](bool checked) { if (checked) projectChanged(); });
    connect(m_satelliteMap, &QRadioButton::toggled, this,
            [projectChanged](bool checked) { if (checked) projectChanged(); });
    connect(m_runButton, &QPushButton::clicked, this, &MainWindow::runInversion);
    connect(m_killButton, &QPushButton::clicked, this, &MainWindow::killInversion);
    connect(m_exportModelledButton, &QPushButton::clicked,
            this, &MainWindow::exportModelledData);
    connect(m_moveKeptButton, &QPushButton::clicked, this, &MainWindow::moveKeptUsfFiles);
    connect(m_autoFilterButton, &QPushButton::clicked, this, [this] {
        if (m_worker || QMessageBox::question(this, "Auto-filter gates",
                "Re-select the gates of all soundings automatically? Manual gate edits are replaced.")
                != QMessageBox::Yes)
            return;
        std::size_t kept = 0, total = 0;
        std::vector<int> all(m_soundings.size());
        std::iota(all.begin(), all.end(), 0);
        pushGateUndo(all);
        for (auto &state : m_soundings) {
            auto &target = state.gateEnabled;
            target.clear();
            state.resultStale = state.haveResult;
            for (const auto &moment : state.sounding.moments) {
                target.push_back(pytem::UsfReader::autoSelectGates(moment));
                kept += std::count(target.back().begin(), target.back().end(), true);
                total += moment.times.size();
            }
        }
        m_log->appendPlainText(QString("Auto-filter: %1 of %2 gates kept").arg(kept).arg(total));
        markProjectModified();
        if (const auto *state = activeSounding())
            state->haveResult ? displayResult(*state) : displayInput(*state);
        updateTransectPlot();
    });
    connect(m_clearResultsButton, &QPushButton::clicked,
            this, &MainWindow::clearSavedResults);
    connect(m_soundingPlot, &PlotWidget::pointRightClicked,
            this, &MainWindow::toggleDataPoint);
    connect(m_soundingPlot, &PlotWidget::pointsRightDragged,
            this, &MainWindow::editDataPoints);
    auto *undo = new QShortcut(QKeySequence::Undo, this);
    connect(undo, &QShortcut::activated, this, &MainWindow::undoGateEdit);
    connect(m_transectSize, qOverload<int>(&QSpinBox::valueChanged), this, [this] {
        if (m_activeSoundingIndex >= 0)
            m_transectStart = (m_activeSoundingIndex / m_transectSize->value()) * m_transectSize->value();
        updateTransectPlot();
        updateTransectNavigation();
        updateMapPlot();
        markProjectModified();
    });
    connect(m_transectLowPlot, &PlotWidget::pointRightClicked,
            this, &MainWindow::toggleDataPoint);
    connect(m_transectHighPlot, &PlotWidget::pointRightClicked,
            this, &MainWindow::toggleDataPoint);
    connect(m_transectLowPlot, &PlotWidget::pointsRightDragged,
            this, &MainWindow::editDataPoints);
    connect(m_transectHighPlot, &PlotWidget::pointsRightDragged,
            this, &MainWindow::editDataPoints);
    connect(m_plotMode, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &MainWindow::selectPlotMode);
    connect(m_previousTransectButton, &QPushButton::clicked,
            this, &MainWindow::previousTransectPage);
    connect(m_nextTransectButton, &QPushButton::clicked,
            this, &MainWindow::nextTransectPage);
    connect(m_mapPlot, &PlotWidget::pointClicked, this, &MainWindow::navigateToSounding);
    connect(m_mapPlot, &PlotWidget::pointRightClicked, this, &MainWindow::toggleSoundingData);
    connect(m_noMap, &QRadioButton::toggled, this,
            [this](bool checked) { if (checked) changeMapBackground(); });
    connect(m_openStreetMap, &QRadioButton::toggled, this,
            [this](bool checked) { if (checked) changeMapBackground(); });
    connect(m_satelliteMap, &QRadioButton::toggled, this,
            [this](bool checked) { if (checked) changeMapBackground(); });

    m_mapTileLoader = new MapTileLoader(this);
    connect(m_mapTileLoader, &MapTileLoader::progress, this,
            [this](int completed, int total) {
                m_mapProgress->setRange(0, total);
                m_mapProgress->setValue(completed);
                m_mapProgress->setVisible(true);
            });
    connect(m_mapTileLoader, &MapTileLoader::loaded, this,
            [this](const QImage &image, const QRectF &bounds,
                   const QString &, int failedTiles) {
                m_mapPlot->setBackgroundImage(image, bounds);
                m_mapProgress->setVisible(false);
                m_log->appendPlainText(failedTiles == 0
                    ? "Background map loaded"
                    : QString("Background map loaded; %1/49 tiles unavailable")
                          .arg(failedTiles));
            });
    connect(m_mapTileLoader, &MapTileLoader::failed, this,
            [this](const QString &message) {
                m_mapProgress->setVisible(false);
                m_mapPlot->clearBackgroundImage();
                QMessageBox::warning(this, "Background map", message);
            });
}

MainWindow::SoundingState MainWindow::readSoundingState(
    const QString &path, const QJsonObject &embedded) const
{
    SoundingState state;
    state.sounding = embedded.isEmpty() ? pytem::UsfReader::read(path.toStdString())
                                        : usfFromJson(embedded);
    state.path = QFileInfo(path).absoluteFilePath();
    state.txRadius = state.sounding.equivalentCircularRadius();
    state.rxOffset = state.sounding.radialReceiverOffset();
    state.geometry = state.rxOffset > 0.0
        ? pytem::Geometry::CircleOffset : pytem::Geometry::CircleCentral;
    state.momentChoice = state.sounding.moments.size() > 1 ? -1 : 0;
    for (const auto &moment : state.sounding.moments)
        state.gateEnabled.push_back(pytem::UsfReader::autoSelectGates(moment));
    return state;
}

void MainWindow::markProjectModified()
{
    if (!m_projectModified) {
        m_projectModified = true;
        updateProjectTitle();
    }
}

void MainWindow::updateProjectTitle()
{
    const QString modified = m_projectModified ? " *" : QString();
    setWindowTitle(QString("InverTEM %1 — %2%3")
        .arg(QApplication::applicationVersion(), m_projectName, modified));
    if (m_projectLabel)
        m_projectLabel->setText(QString("Project: %1%2")
            .arg(m_projectName, modified));
}

void MainWindow::rebuildSoundingSelector(int selectedIndex)
{
    m_soundingSelector->blockSignals(true);
    m_soundingSelector->clear();
    for (std::size_t i = 0; i < m_soundings.size(); ++i) {
        const auto &state = m_soundings[i];
        m_soundingSelector->addItem(QString("%1/%2  %3 — %4")
            .arg(i + 1).arg(m_soundings.size())
            .arg(QString::fromStdString(state.sounding.soundingName),
                 QFileInfo(state.path).fileName()));
    }
    m_soundingSelector->blockSignals(false);
    const bool haveSoundings = !m_soundings.empty();
    m_soundingSelector->setEnabled(haveSoundings);
    m_runButton->setEnabled(haveSoundings && !m_worker);
    m_moveKeptButton->setEnabled(haveSoundings && !m_worker);
    m_autoFilterButton->setEnabled(haveSoundings && !m_worker);
    if (!haveSoundings) {
        m_activeSoundingIndex = -1;
        m_moment->clear();
        m_moment->setEnabled(false);
        m_previousButton->setEnabled(false);
        m_nextButton->setEnabled(false);
        m_geometryLabel->setText("Geometry: no sounding loaded");
        m_systemLabel->setText(
            "Filters and gates are read from the selected moment(s).");
        m_soundingPlot->clear();
        m_modelPlot->clear();
        m_transectLowPlot->clear();
        m_transectHighPlot->clear();
        m_mapPlot->clear();
        m_transectStart = 0;
        updateTransectNavigation();
        return;
    }
    selectedIndex = std::clamp(selectedIndex, 0,
                               static_cast<int>(m_soundings.size()) - 1);
    m_activeSoundingIndex = -1;
    selectSounding(selectedIndex);
    m_soundingSelector->setCurrentIndex(selectedIndex);
}

// Inversion settings start from their defaults in every new or loaded project;
// display settings (map, transect, resistivity axis) still come from the project.
void MainWindow::resetInversionSettings()
{
    m_layerCount->setValue(15);
    m_firstDepth->setValue(2.0);
    m_maxDepth->setValue(120.0);
    m_errorFloor->setValue(5.0);
    m_parallelJobs->setValue(detectedCpuCoreCount());
    m_regularizationNorm->setCurrentIndex(0); // L1 blocky
    m_adaptiveAlpha->setChecked(!m_pytemJoint);
    m_useGpu->setChecked(m_useGpu->isEnabled());
    m_cacheJacobian->setChecked(true);
    m_sensitivity->setChecked(true);
}

void MainWindow::resetProjectState()
{
    resetInversionSettings();
    m_soundings.clear();
    m_transectPointReferences.clear();
    m_transectStart = 0;
    m_projectGeometry = {};
    m_projectPath.clear();
    m_projectName = "Untitled";
    m_projectModified = false;
    m_completedJobs = 0;
    m_successfulJobs = 0;
    m_totalJobs = 0;
    m_status->setText("0/0 completed");
    m_log->clear();
    m_exportModelledButton->setEnabled(false);
    m_clearResultsButton->setEnabled(false);
    rebuildSoundingSelector();
    updateProjectTitle();
}

bool MainWindow::confirmProjectReplacement()
{
    // A project without soundings (e.g. the untitled start-up project with only
    // changed settings) has nothing worth saving.
    if (!m_projectModified || m_soundings.empty())
        return true;
    QMessageBox prompt(this);
    prompt.setWindowTitle("Save project changes?");
    prompt.setText(QString("Save changes to %1 before continuing?")
        .arg(m_projectName));
    prompt.setStandardButtons(QMessageBox::Save | QMessageBox::Discard
                              | QMessageBox::Cancel);
    prompt.setDefaultButton(QMessageBox::Save);
    const auto choice = static_cast<QMessageBox::StandardButton>(prompt.exec());
    if (choice == QMessageBox::Cancel)
        return false;
    if (choice == QMessageBox::Discard)
        return true;
    QString path = m_projectPath;
    if (path.isEmpty()) {
        path = QFileDialog::getSaveFileName(this, "Save InverTEM project",
            QSettings().value("lastProjectDirectory").toString()
                + "/InverTEM_project.ivtp",
            "InverTEM project (*.ivtp)");
        if (path.isEmpty())
            return false;
        if (!path.endsWith(".ivtp", Qt::CaseInsensitive))
            path += ".ivtp";
    }
    return writeProject(path);
}

void MainWindow::newProject()
{
    if (m_worker || !confirmProjectReplacement())
        return;
    QString path = QFileDialog::getSaveFileName(this, "Create InverTEM project",
        QSettings().value("lastProjectDirectory").toString()
            + "/InverTEM_project.ivtp",
        "InverTEM project (*.ivtp)");
    if (path.isEmpty())
        return;
    if (!path.endsWith(".ivtp", Qt::CaseInsensitive))
        path += ".ivtp";
    resetProjectState();
    m_projectPath = QFileInfo(path).absoluteFilePath();
    m_projectName = QFileInfo(path).completeBaseName();
    writeProject(m_projectPath);
    m_log->appendPlainText(
        "New project created; import USF soundings to lock its geometry");
}

void MainWindow::saveProject()
{
    if (m_worker)
        return;
    QString path = m_projectPath;
    if (path.isEmpty()) {
        path = QFileDialog::getSaveFileName(this, "Save InverTEM project",
            QSettings().value("lastProjectDirectory").toString()
                + "/InverTEM_project.ivtp",
            "InverTEM project (*.ivtp)");
        if (path.isEmpty())
            return;
        if (!path.endsWith(".ivtp", Qt::CaseInsensitive))
            path += ".ivtp";
    }
    if (writeProject(path))
        m_log->appendPlainText(
            "Project saved: " + QFileInfo(path).fileName());
}

void MainWindow::loadProject()
{
    if (m_worker || !confirmProjectReplacement())
        return;
    const QString path = QFileDialog::getOpenFileName(this,
        "Load InverTEM project",
        QSettings().value("lastProjectDirectory").toString(),
        "InverTEM project (*.ivtp);;JSON files (*.json)");
    if (!path.isEmpty())
        readProject(path);
}

bool MainWindow::writeProject(const QString &path)
{
    saveActiveSoundingEdits();
    const QFileInfo projectInfo(path);
    const QDir projectDirectory(projectInfo.absolutePath());
    QJsonObject root;
    root["format"] = "InverTEM project";
    root["version"] = 1;
    root["name"] = projectInfo.completeBaseName();
    root["active_sounding"] = m_activeSoundingIndex;

    QJsonObject geometry;
    geometry["locked"] = m_projectGeometry.locked;
    geometry["loop_x"] = m_projectGeometry.loopX;
    geometry["loop_y"] = m_projectGeometry.loopY;
    geometry["coil_x"] = m_projectGeometry.coilX;
    geometry["coil_y"] = m_projectGeometry.coilY;
    root["system_geometry"] = geometry;

    QJsonObject settings;
    settings["layer_count"] = m_layerCount->value();
    settings["maximum_depth"] = m_maxDepth->value();
    settings["first_boundary_depth"] = m_firstDepth->value();
    settings["error_floor"] = m_errorFloor->value();
    settings["parallel_soundings"] = m_parallelJobs->value();
    settings["model_norm"] = m_regularizationNorm->currentData().toInt();
    settings["use_gpu"] = m_useGpu->isChecked();
    settings["cache_first_jacobian"] = m_cacheJacobian->isChecked();
    settings["adaptive_alpha"] = m_adaptiveAlpha->isChecked();
    settings["calculate_sensitivity"] = m_sensitivity->isChecked();
    settings["transect_soundings"] = m_transectSize->value();
    settings["auto_resistivity_axis"] = m_autoResistivityAxis->isChecked();
    settings["resistivity_axis_minimum"] = m_resistivityAxisMinimum->value();
    settings["resistivity_axis_maximum"] = m_resistivityAxisMaximum->value();
    settings["map_background"] = m_openStreetMap->isChecked() ? "osm"
        : (m_satelliteMap->isChecked() ? "satellite" : "none");
    settings["map_point_size"] = m_mapPointSize->currentData().toDouble();
    root["settings"] = settings;

    auto gateArrays = [](const std::vector<std::vector<bool>> &moments) {
        QJsonArray outer;
        for (const auto &gates : moments) {
            QJsonArray inner;
            for (bool enabled : gates) inner.append(enabled);
            outer.append(inner);
        }
        return outer;
    };
    QJsonArray soundings;
    for (const auto &state : m_soundings) {
        QJsonObject sounding;
        sounding["path"] = state.path;
        sounding["relative_path"] = projectDirectory.relativeFilePath(state.path);
        sounding["data"] = usfJson(state.sounding);
        sounding["moment_choice"] = state.momentChoice;
        sounding["include_in_batch"] = state.includeInBatch;
        sounding["gate_enabled"] = gateArrays(state.gateEnabled);
        if (state.haveResult)
            sounding["active_model_key"] = savedModelKey(
                state.options, state.momentChoice);
        QJsonArray savedModels;
        for (const auto &saved : state.savedModels) {
            QJsonObject model;
            model["key"] = saved.key;
            model["label"] = saved.label;
            model["moment_choice"] = saved.momentChoice;
            model["options"] = optionsJson(saved.options);
            model["result"] = resultJson(saved.result);
            model["moment_names"] = jsonArray(saved.momentNames);
            model["moment_indices"] = jsonArray(saved.momentIndices);
            QJsonArray gateIndices;
            for (const auto &indices : saved.gateIndices)
                gateIndices.append(jsonArray(indices));
            model["gate_indices"] = gateIndices;
            model["doi_depth"] = saved.doiDepth;
            savedModels.append(model);
        }
        sounding["saved_models"] = savedModels;
        soundings.append(sounding);
    }
    root["soundings"] = soundings;

    QSaveFile file(projectInfo.absoluteFilePath());
    if (!file.open(QIODevice::WriteOnly)) {
        QMessageBox::critical(this, "Project save failed", file.errorString());
        return false;
    }
    if (file.write(QJsonDocument(root).toJson(QJsonDocument::Indented)) < 0
        || !file.commit()) {
        QMessageBox::critical(this, "Project save failed", file.errorString());
        return false;
    }
    m_projectPath = projectInfo.absoluteFilePath();
    m_projectName = projectInfo.completeBaseName();
    m_projectModified = false;
    QSettings appSettings;
    appSettings.setValue("lastProjectDirectory", projectInfo.absolutePath());
    updateProjectTitle();
    return true;
}

bool MainWindow::readProject(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, "Project load failed", file.errorString());
        return false;
    }
    QJsonParseError parseError;
    const QJsonDocument document = QJsonDocument::fromJson(
        file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError
        || !document.isObject()) {
        QMessageBox::critical(this, "Project load failed",
            "This is not a valid InverTEM project: " + parseError.errorString());
        return false;
    }
    const QJsonObject root = document.object();
    if (root["format"].toString() != "InverTEM project"
        || root["version"].toInt() != 1) {
        QMessageBox::critical(this, "Project load failed",
            "Unsupported InverTEM project format or version.");
        return false;
    }

    ProjectGeometry projectGeometry;
    const auto geometry = root["system_geometry"].toObject();
    projectGeometry.locked = geometry["locked"].toBool();
    projectGeometry.loopX = geometry["loop_x"].toDouble();
    projectGeometry.loopY = geometry["loop_y"].toDouble();
    projectGeometry.coilX = geometry["coil_x"].toDouble();
    projectGeometry.coilY = geometry["coil_y"].toDouble();
    pytem::UsfSounding lockedGeometry;
    lockedGeometry.loopX = projectGeometry.loopX;
    lockedGeometry.loopY = projectGeometry.loopY;
    lockedGeometry.coilX = projectGeometry.coilX;
    lockedGeometry.coilY = projectGeometry.coilY;

    const QJsonArray soundingObjects = root["soundings"].toArray();
    std::vector<SoundingState> loaded;
    QStringList failures;
    QProgressDialog progress("Loading project soundings...", "Cancel", 0,
                             soundingObjects.size(), this);
    progress.setWindowTitle("Loading InverTEM project");
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    const QDir projectDirectory(QFileInfo(path).absolutePath());
    for (int index = 0; index < soundingObjects.size(); ++index) {
        if (progress.wasCanceled())
            return false;
        const auto object = soundingObjects[index].toObject();
        QString soundingPath = projectDirectory.filePath(
            object["relative_path"].toString());
        if (!QFileInfo::exists(soundingPath))
            soundingPath = object["path"].toString();
        progress.setLabelText(QString("Loading %1 (%2 of %3)...")
            .arg(QFileInfo(soundingPath).fileName())
            .arg(index + 1).arg(soundingObjects.size()));
        QApplication::processEvents();
        try {
            // Projects embed the sounding data; older ones re-read their USF files.
            const QJsonObject embedded = object["data"].toObject();
            if (embedded.isEmpty() && !QFileInfo::exists(soundingPath))
                throw std::runtime_error("source USF file was not found");
            SoundingState state = readSoundingState(soundingPath, embedded);
            if (!projectGeometry.locked) {
                projectGeometry.locked = true;
                projectGeometry.loopX = state.sounding.loopX;
                projectGeometry.loopY = state.sounding.loopY;
                projectGeometry.coilX = state.sounding.coilX;
                projectGeometry.coilY = state.sounding.coilY;
                lockedGeometry = state.sounding;
            } else if (!pytem::UsfReader::sameSystemGeometry(
                           lockedGeometry, state.sounding)) {
                throw std::runtime_error(
                    "geometry does not match the project geometry");
            }
            state.momentChoice = object["moment_choice"].toInt(
                state.momentChoice);
            state.includeInBatch = object["include_in_batch"].toBool(true);
            auto applyGates = [](const QJsonArray &source,
                                 std::vector<std::vector<bool>> &destination) {
                for (int moment = 0;
                     moment < source.size()
                     && moment < static_cast<int>(destination.size()); ++moment) {
                    const auto gates = source[moment].toArray();
                    for (int gate = 0;
                         gate < gates.size()
                         && gate < static_cast<int>(destination[moment].size()); ++gate)
                        destination[moment][gate] = gates[gate].toBool();
                }
            };
            applyGates(object["gate_enabled"].toArray(), state.gateEnabled);
            // Older projects blanked an excluded sounding's gates and kept the
            // real selection here; exclusion no longer touches the gates.
            const auto before = object["gates_before_exclusion"].toArray();
            if (!before.isEmpty() && !state.includeInBatch)
                applyGates(before, state.gateEnabled);
            for (const auto &entry : object["saved_models"].toArray()) {
                const auto modelObject = entry.toObject();
                SoundingState::SavedModel saved;
                saved.key = modelObject["key"].toString();
                saved.label = modelObject["label"].toString();
                saved.momentChoice = modelObject["moment_choice"].toInt(-1);
                saved.options = optionsFromJson(
                    modelObject["options"].toObject());
                saved.result = resultFromJson(
                    modelObject["result"].toObject());
                saved.momentNames = stringVector(
                    modelObject["moment_names"].toArray());
                saved.momentIndices = sizeVector(
                    modelObject["moment_indices"].toArray());
                for (const auto &indices : modelObject["gate_indices"].toArray())
                    saved.gateIndices.push_back(sizeVector(indices.toArray()));
                saved.doiDepth = modelObject["doi_depth"].toDouble(-1.0);
                state.savedModels.push_back(std::move(saved));
            }
            const QString activeKey = object["active_model_key"].toString();
            const auto active = std::find_if(
                state.savedModels.begin(), state.savedModels.end(),
                [&activeKey](const SoundingState::SavedModel &saved) {
                    return saved.key == activeKey;
                });
            if (active != state.savedModels.end())
                activateSavedModel(state, *active);
            loaded.push_back(std::move(state));
        } catch (const std::exception &error) {
            failures << QString("%1: %2")
                .arg(QFileInfo(soundingPath).fileName(),
                     QString::fromUtf8(error.what()));
        }
        progress.setValue(index + 1);
        QApplication::processEvents();
    }

    m_soundings = std::move(loaded);
    m_projectGeometry = projectGeometry;
    m_projectPath = QFileInfo(path).absoluteFilePath();
    m_projectName = root["name"].toString(
        QFileInfo(path).completeBaseName());
    const auto settings = root["settings"].toObject();
    resetInversionSettings();
    m_transectSize->setValue(settings["transect_soundings"].toInt(30));
    m_autoResistivityAxis->setChecked(settings["auto_resistivity_axis"].toBool(true));
    m_resistivityAxisMinimum->setValue(
        settings["resistivity_axis_minimum"].toDouble(1.0));
    m_resistivityAxisMaximum->setValue(
        settings["resistivity_axis_maximum"].toDouble(1000.0));
    const QString map = settings["map_background"].toString("none");
    if (map == "osm") m_openStreetMap->setChecked(true);
    else if (map == "satellite") m_satelliteMap->setChecked(true);
    else m_noMap->setChecked(true);
    const double savedMapPointSize
        = settings["map_point_size"].toDouble(4.8);
    int mapPointIndex = 0;
    double smallestDifference = std::numeric_limits<double>::infinity();
    for (int index = 0; index < m_mapPointSize->count(); ++index) {
        const double difference = std::abs(
            m_mapPointSize->itemData(index).toDouble()
            - savedMapPointSize);
        if (difference < smallestDifference) {
            smallestDifference = difference;
            mapPointIndex = index;
        }
    }
    m_mapPointSize->setCurrentIndex(mapPointIndex);
    rebuildSoundingSelector(root["active_sounding"].toInt());
    if (!m_noMap->isChecked() && !m_soundings.empty())
        loadMapTiles();
    m_projectModified = false;
    QSettings appSettings;
    appSettings.setValue("lastProjectDirectory",
                         QFileInfo(path).absolutePath());
    updateProjectTitle();
    m_completedJobs = 0;
    m_successfulJobs = 0;
    m_totalJobs = 0;
    m_status->setText("0/0 completed");
    m_log->appendPlainText(QString("Project loaded: %1 (%2 soundings)")
        .arg(QFileInfo(path).fileName()).arg(m_soundings.size()));
    if (!failures.isEmpty())
        QMessageBox::warning(this, "Some project soundings were skipped",
                             failures.join("\n"));
    return true;
}

void MainWindow::loadUsf()
{
    QSettings settings;
    const QStringList paths = QFileDialog::getOpenFileNames(this, "Open USF soundings",
        settings.value("lastUsfDirectory").toString(), "Universal Sounding Format (*.usf *.USF);;All files (*)");
    if (paths.isEmpty()) return;
    std::vector<SoundingState> loaded;
    ProjectGeometry importGeometry = m_projectGeometry;
    QStringList failures;
    QProgressDialog loading("Loading USF files...", "Cancel", 0, paths.size(), this);
    loading.setWindowTitle("Importing TEM data");
    loading.setWindowModality(Qt::WindowModal);
    loading.setMinimumDuration(0);
    loading.setAutoClose(false);
    loading.setValue(0);
    loading.show();
    QApplication::processEvents();
    for (int pathIndex = 0; pathIndex < paths.size(); ++pathIndex) {
        if (loading.wasCanceled())
            break;
        const QString &path = paths[pathIndex];
        loading.setLabelText(QString("Loading %1 (%2 of %3)...")
            .arg(QFileInfo(path).fileName()).arg(pathIndex + 1).arg(paths.size()));
        QApplication::processEvents();
        try {
            const QString canonical = QFileInfo(path).canonicalFilePath();
            const bool duplicate = std::any_of(
                m_soundings.begin(), m_soundings.end(),
                [&canonical](const SoundingState &existing) {
                    return QFileInfo(existing.path).canonicalFilePath()
                        .compare(canonical, Qt::CaseInsensitive) == 0;
                }) || std::any_of(
                    loaded.begin(), loaded.end(),
                    [&canonical](const SoundingState &existing) {
                        return QFileInfo(existing.path).canonicalFilePath()
                            .compare(canonical, Qt::CaseInsensitive) == 0;
                    });
            if (duplicate) {
                failures << QString("%1: already present in this project")
                    .arg(QFileInfo(path).fileName());
            } else {
                SoundingState state = readSoundingState(path);
                if (!importGeometry.locked) {
                    importGeometry.locked = true;
                    importGeometry.loopX = state.sounding.loopX;
                    importGeometry.loopY = state.sounding.loopY;
                    importGeometry.coilX = state.sounding.coilX;
                    importGeometry.coilY = state.sounding.coilY;
                }
                pytem::UsfSounding locked;
                locked.loopX = importGeometry.loopX;
                locked.loopY = importGeometry.loopY;
                locked.coilX = importGeometry.coilX;
                locked.coilY = importGeometry.coilY;
                if (!pytem::UsfReader::sameSystemGeometry(
                        locked, state.sounding)) {
                    failures << QString(
                        "%1: geometry does not match this project "
                        "(expected loop %2 × %3 m, coil (%4, %5) m; "
                        "found loop %6 × %7 m, coil (%8, %9) m)")
                        .arg(QFileInfo(path).fileName())
                        .arg(importGeometry.loopX, 0, 'f', 2)
                        .arg(importGeometry.loopY, 0, 'f', 2)
                        .arg(importGeometry.coilX, 0, 'f', 2)
                        .arg(importGeometry.coilY, 0, 'f', 2)
                        .arg(state.sounding.loopX, 0, 'f', 2)
                        .arg(state.sounding.loopY, 0, 'f', 2)
                        .arg(state.sounding.coilX, 0, 'f', 2)
                        .arg(state.sounding.coilY, 0, 'f', 2);
                } else {
                    loaded.push_back(std::move(state));
                }
            }
        } catch (const std::exception &error) {
            failures << QString("%1: %2").arg(QFileInfo(path).fileName(),
                                               QString::fromUtf8(error.what()));
        }
        loading.setValue(pathIndex + 1);
        QApplication::processEvents();
    }
    loading.close();
    if (loaded.empty()) {
        if (loading.wasCanceled())
            return;
        QMessageBox::warning(this, "No USF files imported", failures.join("\n"));
        return;
    }
    saveActiveSoundingEdits();
    const int firstNewIndex = static_cast<int>(m_soundings.size());
    for (auto &state : loaded)
        m_soundings.push_back(std::move(state));
    m_projectGeometry = importGeometry;
    settings.setValue("lastUsfDirectory", QFileInfo(paths.front()).absolutePath());
    markProjectModified();
    rebuildSoundingSelector(firstNewIndex);
    if (!m_noMap->isChecked())
        loadMapTiles();
    if (!failures.isEmpty())
        QMessageBox::warning(this, "Some USF files were skipped", failures.join("\n"));
}

MainWindow::SoundingState *MainWindow::activeSounding()
{
    return m_activeSoundingIndex >= 0
        && m_activeSoundingIndex < static_cast<int>(m_soundings.size())
        ? &m_soundings[static_cast<std::size_t>(m_activeSoundingIndex)] : nullptr;
}

const MainWindow::SoundingState *MainWindow::activeSounding() const
{
    return m_activeSoundingIndex >= 0
        && m_activeSoundingIndex < static_cast<int>(m_soundings.size())
        ? &m_soundings[static_cast<std::size_t>(m_activeSoundingIndex)] : nullptr;
}

void MainWindow::activateSavedModel(
    SoundingState &state, const SoundingState::SavedModel &saved)
{
    state.options = saved.options;
    state.result = saved.result;
    state.resultMomentNames = saved.momentNames;
    state.resultMomentIndices = saved.momentIndices;
    state.resultGateIndices = saved.gateIndices;
    state.haveResult = true;
    state.resultStale = false;
}

void MainWindow::saveActiveSoundingEdits()
{
    auto *state = activeSounding();
    if (!state)
        return;
    state->momentChoice = m_moment->currentData().toInt();
}

void MainWindow::selectSounding(int index)
{
    if (index < 0 || index >= static_cast<int>(m_soundings.size()))
        return;
    saveActiveSoundingEdits();
    m_activeSoundingIndex = index;
    auto &state = m_soundings[static_cast<std::size_t>(index)];
    m_rowMomentIndices.clear();
    m_rowGateIndices.clear();
    QStringList heights;
    for (const auto &moment : state.sounding.moments)
        if (moment.txHeight != 0.0 || moment.rxHeight != 0.0)
            heights << QString("%1 %2/%3 m").arg(QString::fromStdString(moment.name))
                           .arg(moment.txHeight, 0, 'f', 1).arg(moment.rxHeight, 0, 'f', 1);
    m_geometryLabel->setText(QString(
        "Geometry: transmitter loop %1 × %2 m; "
        "receiver coil location (%3, %4) m; ground elevation %5 m%6")
        .arg(state.sounding.loopX, 0, 'f', 2)
        .arg(state.sounding.loopY, 0, 'f', 2)
        .arg(state.sounding.coilX, 0, 'f', 2)
        .arg(state.sounding.coilY, 0, 'f', 2)
        .arg(state.sounding.elevation, 0, 'f', 1)
        .arg(heights.isEmpty() ? QString() : "; Tx/Rx height " + heights.join(", ")));
    m_moment->blockSignals(true);
    m_moment->clear();
    if (state.sounding.moments.size() > 1)
        m_moment->addItem("LM + HM", -1);
    for (std::size_t momentIndex = 0; momentIndex < state.sounding.moments.size(); ++momentIndex) {
        const auto &moment = state.sounding.moments[momentIndex];
        m_moment->addItem(QString("%1 — channel %2, %3 stacks, %4 Hz")
            .arg(QString::fromStdString(moment.name)).arg(moment.channel).arg(moment.stackCount)
            .arg(moment.meanFrequency, 0, 'f', 2), static_cast<int>(momentIndex));
    }
    int momentIndex = m_moment->findData(state.momentChoice);
    if (momentIndex < 0) momentIndex = 0;
    m_moment->setCurrentIndex(momentIndex);
    m_moment->blockSignals(false);
    m_moment->setEnabled(m_worker == nullptr);
    selectMoment(momentIndex);
    m_previousButton->setEnabled(index > 0);
    m_nextButton->setEnabled(index + 1 < static_cast<int>(m_soundings.size()));
    m_exportModelledButton->setEnabled(state.haveResult && m_worker == nullptr);
    m_clearResultsButton->setEnabled(m_worker == nullptr && std::any_of(
        m_soundings.begin(), m_soundings.end(),
        [](const SoundingState &candidate) {
            return !candidate.savedModels.empty();
        }));
    if (state.haveResult)
        displayResult(state);
    else
        displayInput(state);
    if (m_plotMode && m_plotMode->currentIndex() == 1) {
        m_transectStart = (index / m_transectSize->value()) * m_transectSize->value();
        updateTransectPlot();
        updateTransectNavigation();
    }
    updateMapPlot();
}

void MainWindow::previousSounding()
{
    if (m_activeSoundingIndex > 0)
        m_soundingSelector->setCurrentIndex(m_activeSoundingIndex - 1);
}

void MainWindow::nextSounding()
{
    if (m_activeSoundingIndex + 1 < static_cast<int>(m_soundings.size()))
        m_soundingSelector->setCurrentIndex(m_activeSoundingIndex + 1);
}

void MainWindow::selectPlotMode(int index)
{
    if (!m_plotModeStack)
        return;
    const int mode = index == 1 ? 1 : 0;
    m_plotModeStack->setCurrentIndex(mode);
    const bool transect = mode == 1;
    m_previousTransectButton->setVisible(transect);
    m_nextTransectButton->setVisible(transect);
    m_transectRangeLabel->setVisible(transect);
    m_transectSize->setVisible(transect);
    if (transect && m_activeSoundingIndex >= 0)
        m_transectStart = (m_activeSoundingIndex / m_transectSize->value())
            * m_transectSize->value();
    updateTransectPlot();
    updateTransectNavigation();
    updateMapPlot();
}

void MainWindow::previousTransectPage()
{
    m_transectStart = std::max(0, m_transectStart - m_transectSize->value());
    updateTransectPlot();
    updateTransectNavigation();
    updateMapPlot();
}

void MainWindow::nextTransectPage()
{
    if (m_transectStart + m_transectSize->value()
        >= static_cast<int>(m_soundings.size()))
        return;
    m_transectStart += m_transectSize->value();
    updateTransectPlot();
    updateTransectNavigation();
    updateMapPlot();
}

void MainWindow::updateTransectNavigation()
{
    if (!m_transectRangeLabel)
        return;
    const int count = static_cast<int>(m_soundings.size());
    if (count == 0) {
        m_transectRangeLabel->setText("0 / 0");
    } else {
        m_transectStart = std::clamp(m_transectStart, 0,
            ((count - 1) / m_transectSize->value()) * m_transectSize->value());
        const int last = std::min(count, m_transectStart + m_transectSize->value());
        m_transectRangeLabel->setText(QString("%1–%2 / %3")
            .arg(m_transectStart + 1).arg(last).arg(count));
    }
    const bool available = m_plotMode && m_plotMode->isEnabled() && count > 0;
    m_previousTransectButton->setEnabled(
        available && m_transectStart > 0);
    m_nextTransectButton->setEnabled(
        available && m_transectStart + m_transectSize->value() < count);
}

void MainWindow::updateTransectPlot()
{
    if (!m_transectLowPlot || !m_transectHighPlot)
        return;
    m_transectPointReferences.clear();
    if (m_soundings.empty()) {
        m_transectLowPlot->clear();
        m_transectHighPlot->clear();
        return;
    }
    const int first = std::clamp(m_transectStart, 0,
        static_cast<int>(m_soundings.size()) - 1);
    const int last = std::min(static_cast<int>(m_soundings.size()),
                              first + m_transectSize->value());
    QVector<double> distances(last - first, 0.0);
    for (int index = first + 1; index < last; ++index) {
        distances[index - first] = distances[index - first - 1]
            + soundingDistanceMetres(
                m_soundings[static_cast<std::size_t>(index - 1)].sounding,
                m_soundings[static_cast<std::size_t>(index)].sounding);
    }

    auto buildMomentCurves = [&](bool highMoment) {
        QVector<PlotWidget::Curve> curves;
        std::size_t maximumGateCount = 0;
        for (int index = first; index < last; ++index) {
            const auto &state = m_soundings[static_cast<std::size_t>(index)];
            for (std::size_t momentIndex = 0;
                 momentIndex < state.sounding.moments.size(); ++momentIndex) {
                const QString upper = QString::fromStdString(
                    state.sounding.moments[momentIndex].name).toUpper();
                const bool isHigh = upper.startsWith("H")
                    || upper.contains("HIGH");
                if (isHigh == highMoment)
                    maximumGateCount = std::max(maximumGateCount,
                        state.sounding.moments[momentIndex].times.size());
            }
        }
        for (std::size_t gate = 0; gate < maximumGateCount; ++gate) {
            PlotWidget::Curve enabled;
            enabled.color = QColor::fromHsvF(
                maximumGateCount > 1
                    ? static_cast<double>(gate) / maximumGateCount : 0.58,
                0.70, 0.82);
            enabled.markers = true;
            enabled.selectable = true;
            enabled.line = true;
            PlotWidget::Curve disabled;
            disabled.color = QColor("#303030");
            disabled.markers = true;
            disabled.selectable = true;
            disabled.line = false;
            for (int index = first; index < last; ++index) {
                auto &state = m_soundings[static_cast<std::size_t>(index)];
                for (std::size_t momentIndex = 0;
                     momentIndex < state.sounding.moments.size(); ++momentIndex) {
                    const auto &moment = state.sounding.moments[momentIndex];
                    const QString upper = QString::fromStdString(moment.name).toUpper();
                    const bool isHigh = upper.startsWith("H")
                        || upper.contains("HIGH");
                    if (isHigh != highMoment || gate >= moment.voltages.size()
                        || momentIndex >= state.gateEnabled.size()
                        || gate >= state.gateEnabled[momentIndex].size())
                        continue;
                    const double value = std::abs(moment.voltages[gate]);
                    if (!(value > 0.0) || !std::isfinite(value))
                        continue;
                    const int reference = static_cast<int>(
                        m_transectPointReferences.size());
                    m_transectPointReferences.push_back(
                        {index, momentIndex, gate});
                    const QPointF point(distances[index - first], value);
                    if (state.includeInBatch && state.gateEnabled[momentIndex][gate]) {
                        enabled.points.push_back(point);
                        enabled.pointIds.push_back(
                            transectPointIdOffset + reference);
                    } else {
                        disabled.points.push_back(point);
                        disabled.pointIds.push_back(
                            transectPointIdOffset + reference);
                    }
                }
            }
            if (!enabled.points.isEmpty())
                curves.push_back(std::move(enabled));
            if (!disabled.points.isEmpty())
                curves.push_back(std::move(disabled));
        }
        return curves;
    };
    const auto lowCurves = buildMomentCurves(false);
    const auto highCurves = buildMomentCurves(true);
    m_transectLowPlot->setCurves(lowCurves);
    m_transectHighPlot->setCurves(highCurves);
    m_transectHighPlot->setVisible(!highCurves.isEmpty());
}

std::vector<const pytem::UsfMoment *> MainWindow::selectedMoments(
    const SoundingState &state) const
{
    std::vector<const pytem::UsfMoment *> moments;
    const int selected = state.momentChoice;
    if (selected < 0) {
        for (const auto &moment : state.sounding.moments)
            moments.push_back(&moment);
    } else if (selected < static_cast<int>(state.sounding.moments.size())) {
        moments.push_back(&state.sounding.moments[static_cast<std::size_t>(selected)]);
    }
    return moments;
}

// The selected moments that have at least one used gate: a moment the gate
// filter emptied is left out of the inversion instead of blocking the sounding.
std::vector<const pytem::UsfMoment *> MainWindow::fittedMoments(const SoundingState &state) const
{
    auto moments = selectedMoments(state);
    moments.erase(std::remove_if(moments.begin(), moments.end(), [&](const pytem::UsfMoment *moment) {
        const std::size_t index = static_cast<std::size_t>(moment - state.sounding.moments.data());
        for (std::size_t gate = 0; gate < moment->times.size(); ++gate)
            if (gateUsed(state, index, gate))
                return false;
        return true;
    }), moments.end());
    return moments;
}

void MainWindow::selectMoment(int)
{
    saveActiveSoundingEdits();
    auto *state = activeSounding();
    if (!state) return;
    state->momentChoice = m_moment->currentData().toInt();
    const auto moments = selectedMoments(*state);
    if (moments.empty()) return;
    m_rowMomentIndices.clear();
    m_rowGateIndices.clear();
    for (const auto *moment : moments) {
        const std::size_t momentIndex = static_cast<std::size_t>(moment - state->sounding.moments.data());
        for (std::size_t gate = 0; gate < moment->times.size(); ++gate) {
            m_rowMomentIndices.push_back(momentIndex);
            m_rowGateIndices.push_back(gate);
        }
    }
    updateSystemSummary();
    m_log->appendPlainText(QString("Loaded %1 from %2")
        .arg(moments.size() > 1 ? "joint LM + HM dataset" : QString::fromStdString(moments.front()->name),
             QFileInfo(state->path).fileName()));
    const auto matching = std::find_if(
        state->savedModels.rbegin(), state->savedModels.rend(),
        [state](const SoundingState::SavedModel &saved) {
            return saved.momentChoice == state->momentChoice;
        });
    if (matching != state->savedModels.rend())
        activateSavedModel(*state, *matching);
    else
        state->haveResult = false;
    if (state->haveResult)
        displayResult(*state);
    else {
        updateInputPlot();
        m_modelPlot->clear();
    }
    m_exportModelledButton->setEnabled(state->haveResult && m_worker == nullptr);
}

void MainWindow::setLayerCount(int)
{
    markProjectModified();
}

// Right-click toggles one gate; a right-drag removes every gate in the box,
// or restores them with Shift held. Ctrl+Z undoes the last edit.
void MainWindow::toggleDataPoint(int pointId)
{
    editGates({pointId}, GateEdit::Toggle);
}

void MainWindow::editDataPoints(const QVector<int> &pointIds, bool restore)
{
    editGates(pointIds, restore ? GateEdit::Restore : GateEdit::Remove);
}

void MainWindow::pushGateUndo(const std::vector<int> &soundingIndices)
{
    std::vector<GateSnapshot> snapshot;
    for (int index : soundingIndices) {
        const auto &state = m_soundings[static_cast<std::size_t>(index)];
        snapshot.push_back({index, state.gateEnabled, state.includeInBatch});
    }
    m_gateUndo.push_back(std::move(snapshot));
    if (m_gateUndo.size() > 100)
        m_gateUndo.erase(m_gateUndo.begin());
}

void MainWindow::undoGateEdit()
{
    if (m_worker || m_gateUndo.empty())
        return;
    for (const auto &entry : m_gateUndo.back()) {
        if (entry.sounding >= static_cast<int>(m_soundings.size()))
            continue;
        auto &state = m_soundings[static_cast<std::size_t>(entry.sounding)];
        state.gateEnabled = entry.gates;
        state.includeInBatch = entry.included;
        state.resultStale = state.haveResult;
    }
    m_gateUndo.pop_back();
    m_log->appendPlainText("Undid the last gate or inclusion edit");
    markProjectModified();
    if (const auto *state = activeSounding())
        state->haveResult ? displayResult(*state) : updateInputPlot();
    updateTransectPlot();
    updateMapPlot();
}

void MainWindow::editGates(const QVector<int> &pointIds, GateEdit edit)
{
    if (m_worker || pointIds.isEmpty())
        return;
    QVector<TransectPointReference> references;
    for (const int pointId : pointIds) {
        TransectPointReference reference;
        if (pointId >= transectPointIdOffset) {
            const int index = pointId - transectPointIdOffset;
            if (index < 0
                || index >= static_cast<int>(m_transectPointReferences.size()))
                continue;
            reference = m_transectPointReferences[static_cast<std::size_t>(index)];
        } else {
            if (pointId < 0
                || static_cast<std::size_t>(pointId) >= m_rowMomentIndices.size()
                || m_rowGateIndices.size() != m_rowMomentIndices.size()
                || m_activeSoundingIndex < 0)
                continue;
            reference.soundingIndex = m_activeSoundingIndex;
            reference.momentIndex = m_rowMomentIndices[static_cast<std::size_t>(pointId)];
            reference.gateIndex = m_rowGateIndices[static_cast<std::size_t>(pointId)];
        }
        if (reference.soundingIndex < 0
            || reference.soundingIndex >= static_cast<int>(m_soundings.size()))
            continue;
        auto &state = m_soundings[static_cast<std::size_t>(reference.soundingIndex)];
        if (!state.includeInBatch // excluded soundings keep their gates as they are
            || reference.momentIndex >= state.gateEnabled.size()
            || reference.gateIndex
                >= state.gateEnabled[reference.momentIndex].size())
            continue;
        const bool duplicate = std::any_of(references.begin(), references.end(),
            [&reference](const TransectPointReference &candidate) {
                return candidate.soundingIndex == reference.soundingIndex
                    && candidate.momentIndex == reference.momentIndex
                    && candidate.gateIndex == reference.gateIndex;
            });
        if (!duplicate)
            references.push_back(reference);
    }
    if (references.isEmpty())
        return;
    std::vector<int> touched;
    for (const auto &reference : references)
        if (std::find(touched.begin(), touched.end(), reference.soundingIndex) == touched.end())
            touched.push_back(reference.soundingIndex);
    pushGateUndo(touched);
    bool activeChanged = false;
    for (const auto &reference : references) {
        auto &state = m_soundings[static_cast<std::size_t>(reference.soundingIndex)];
        auto &gates = state.gateEnabled[reference.momentIndex];
        gates[reference.gateIndex] = edit == GateEdit::Toggle ? !gates[reference.gateIndex] : edit == GateEdit::Restore;
        // The model stays on screen, marked out of date, until it is re-inverted.
        state.resultStale = state.haveResult;
        activeChanged = activeChanged || reference.soundingIndex == m_activeSoundingIndex;
    }
    markProjectModified();
    if (activeChanged) {
        const auto *active = activeSounding();
        active->haveResult ? displayResult(*active) : updateInputPlot();
        updateSystemSummary();
    }
    updateTransectPlot();
    updateMapPlot();
}

void MainWindow::toggleSoundingData(int soundingIndex)
{
    if (m_worker || soundingIndex < 0
        || soundingIndex >= static_cast<int>(m_soundings.size()))
        return;
    pushGateUndo({soundingIndex});
    auto &state = m_soundings[static_cast<std::size_t>(soundingIndex)];
    state.includeInBatch = !state.includeInBatch; // gates are left untouched
    markProjectModified();
    updateTransectPlot();
    updateMapPlot();
    if (soundingIndex != m_activeSoundingIndex) {
        m_soundingSelector->setCurrentIndex(soundingIndex);
    } else {
        state.haveResult ? displayResult(state) : updateInputPlot();
        updateSystemSummary();
    }
    m_log->appendPlainText(state.includeInBatch
        ? QString("Sounding enabled for batch inversion — %1")
              .arg(QFileInfo(state.path).fileName())
        : QString("Sounding excluded from batch inversion — %1")
              .arg(QFileInfo(state.path).fileName()));
}

void MainWindow::navigateToSounding(int soundingIndex)
{
    if (soundingIndex < 0
        || soundingIndex >= static_cast<int>(m_soundings.size()))
        return;
    if (m_soundingSelector->currentIndex() == soundingIndex)
        selectSounding(soundingIndex);
    else
        m_soundingSelector->setCurrentIndex(soundingIndex);
}

void MainWindow::changeMapBackground()
{
    if (m_noMap->isChecked()) {
        m_mapTileLoader->cancel();
        m_mapProgress->setVisible(false);
        m_mapPlot->clearBackgroundImage();
        m_log->appendPlainText("Background map disabled");
        return;
    }
    loadMapTiles();
}

void MainWindow::loadMapTiles()
{
    double minimumLongitude = 180.0;
    double maximumLongitude = -180.0;
    double minimumLatitude = 90.0;
    double maximumLatitude = -90.0;
    double latitudeSum = 0.0;
    int coordinateCount = 0;
    bool haveCoordinates = false;
    for (const auto &state : m_soundings) {
        const double longitude = state.sounding.longitude;
        const double latitude = state.sounding.latitude;
        if (!std::isfinite(longitude) || !std::isfinite(latitude)
            || (longitude == 0.0 && latitude == 0.0))
            continue;
        minimumLongitude = std::min(minimumLongitude, longitude);
        maximumLongitude = std::max(maximumLongitude, longitude);
        minimumLatitude = std::min(minimumLatitude, latitude);
        maximumLatitude = std::max(maximumLatitude, latitude);
        latitudeSum += latitude;
        ++coordinateCount;
        haveCoordinates = true;
    }
    if (!haveCoordinates) {
        m_mapPlot->clearBackgroundImage();
        m_mapProgress->setVisible(false);
        m_log->appendPlainText(
            "No valid coordinates are available for a background map");
        return;
    }
    const double centerLongitude = 0.5 * (minimumLongitude + maximumLongitude);
    const double centerLatitude = 0.5 * (minimumLatitude + maximumLatitude);
    double longitudeSpan = std::max(0.002, maximumLongitude - minimumLongitude) * 1.12;
    double latitudeSpan = std::max(0.002, maximumLatitude - minimumLatitude) * 1.16;
    constexpr double degreesToRadians = 3.14159265358979323846 / 180.0;
    const double meanLatitude = latitudeSum / coordinateCount;
    const double longitudeScale = std::max(
        0.05, std::abs(std::cos(meanLatitude * degreesToRadians)));
    const double plotWidth = std::max(40, m_mapPlot->width() - 100);
    const double plotHeight = std::max(40, m_mapPlot->height() - 105);
    const double plotAspect = plotWidth / plotHeight;
    if (longitudeSpan * longitudeScale / latitudeSpan < plotAspect)
        longitudeSpan = latitudeSpan * plotAspect / longitudeScale;
    else
        latitudeSpan = longitudeSpan * longitudeScale / plotAspect;
    minimumLongitude = centerLongitude - 0.5 * longitudeSpan;
    maximumLongitude = centerLongitude + 0.5 * longitudeSpan;
    minimumLatitude = centerLatitude - 0.5 * latitudeSpan;
    maximumLatitude = centerLatitude + 0.5 * latitudeSpan;
    m_mapPlot->clearBackgroundImage();
    m_mapProgress->setRange(0, 25);
    m_mapProgress->setValue(0);
    m_mapProgress->setVisible(true);
    const auto style = m_openStreetMap->isChecked()
        ? MapTileLoader::Style::OpenStreetMap
        : MapTileLoader::Style::Satellite;
    m_mapTileLoader->load(style, minimumLongitude, maximumLongitude,
                          minimumLatitude, maximumLatitude);
}

bool MainWindow::validateInputs(const SoundingState &state, QString &error) const
{
    const auto moments = fittedMoments(state);
    if (moments.empty()) { error = "The sounding has no selectable moments."; return false; }
    for (const auto *moment : moments) {
        const std::size_t momentIndex = static_cast<std::size_t>(moment - state.sounding.moments.data());
        int used = 0;
        double previous = 0.0;
        for (std::size_t gate = 0; gate < moment->times.size(); ++gate) {
            if (!state.gateEnabled[momentIndex][gate]) continue;
            if (!(moment->times[gate] > previous && moment->voltages[gate] > 0.0)) {
                error = QString("%1 has an invalid selected gate at row %2.")
                    .arg(QString::fromStdString(moment->name)).arg(gate + 1);
                return false;
            }
            previous = moment->times[gate];
            ++used;
        }
        if (used < 3) {
            error = QString("Select at least three positive %1 gates.")
                .arg(QString::fromStdString(moment->name));
            return false;
        }
    }
    if (m_maxDepth->value() <= m_firstDepth->value()) {
        error = "The depth to the top of layer N must exceed the thickness of layer 1.";
        return false;
    }
    return true;
}

bool MainWindow::gateUsed(const SoundingState &state, std::size_t momentIndex, std::size_t gate) const
{
    if (!state.gateEnabled[momentIndex][gate])
        return false;
    // pyTEM joint drops gates whose relative error exceeds 50%, as the notebook does.
    const auto &moment = state.sounding.moments[momentIndex];
    return !m_pytemJoint
        || std::abs(moment.standardErrors[gate]) <= pytemJointMaxRelativeError * std::abs(moment.voltages[gate]);
}

pytem::InversionOptions MainWindow::buildOptions(const SoundingState &state) const
{
    pytem::InversionOptions options;
    pytem::ForwardModel commonModel;
    commonModel.geometry = state.geometry;
    commonModel.transform = m_transform;
    commonModel.useGpu = m_useGpu->isChecked();
    commonModel.txSize = state.txRadius;
    commonModel.rxX = state.rxOffset;
    commonModel.thicknesses = pytem::TemSolver::logSpacedThicknesses(
        m_layerCount->value(), m_firstDepth->value(), m_maxDepth->value());
    commonModel.resistivities.assign(static_cast<std::size_t>(m_layerCount->value()), 100.0);

    const double errorFloor = m_errorFloor->value() / 100.0;
    const bool joint = m_pytemJoint;
    const auto moments = fittedMoments(state);
    // pyTEM joint: one step grid over every gate of the selected moments, so a
    // single step response serves LM and HM (and all soundings share the grid).
    std::vector<pytem::MatrixConvolution> shared;
    if (joint) {
        // Soundings from one system share gates and waveform, so the (slow)
        // B-spline gate matrices are built once and reused across the batch.
        std::vector<pytem::MatrixConvolution::System> systems;
        std::vector<double> key;
        for (const auto *moment : moments) {
            systems.push_back({moment->times, moment->gateOpen, moment->gateClose,
                               moment->waveformTimes, moment->waveformAmplitudes});
            for (const auto *part : {&moment->times, &moment->gateOpen, &moment->gateClose,
                                     &moment->waveformTimes, &moment->waveformAmplitudes}) {
                key.push_back(static_cast<double>(part->size()));
                key.insert(key.end(), part->begin(), part->end());
            }
        }
        auto cached = m_convolutionCache.find(key);
        if (cached == m_convolutionCache.end())
            cached = m_convolutionCache.emplace(std::move(key),
                pytem::MatrixConvolution::buildShared(systems, pytemJointStepCount)).first;
        shared = cached->second;
    }
    std::vector<pytem::InversionDataSet> dataSets;
    for (const auto *moment : moments) {
        pytem::InversionDataSet dataSet;
        dataSet.model = commonModel;
        std::vector<double> gateOpen;
        std::vector<double> gateClose;
        const std::size_t momentIndex = static_cast<std::size_t>(moment - state.sounding.moments.data());
        for (std::size_t gate = 0; gate < moment->times.size(); ++gate) {
            if (!gateUsed(state, momentIndex, gate)) continue;
            const double value = std::abs(moment->voltages[gate]);
            const double sem = std::abs(moment->standardErrors[gate]);
            dataSet.model.times.push_back(moment->times[gate]);
            dataSet.observed.push_back(value);
            dataSet.noiseStd.push_back(std::max(sem, errorFloor * value));
            gateOpen.push_back(moment->gateOpen[gate]);
            gateClose.push_back(moment->gateClose[gate]);
            if (!shared.empty())
                dataSet.model.responseMatrix.push_back(shared[dataSets.size()].matrix[gate]);
        }
        if (!shared.empty()) {
            dataSet.model.stepTimes = shared[dataSets.size()].stepTimes;
        } else {
            const auto convolution = pytem::MatrixConvolution::build(
                dataSet.model.times, gateOpen, gateClose,
                moment->waveformTimes, moment->waveformAmplitudes, 300, 81, 1.0);
            dataSet.model.stepTimes = convolution.stepTimes;
            dataSet.model.responseMatrix = convolution.matrix;
        }
        dataSet.model.altitude = moment->txHeight + moment->rxHeight;
        dataSet.model.lowPassFrequencies = moment->lowPassFrequencies;
        dataSet.model.lowPassOrders = moment->lowPassOrders;
        dataSet.model.highPassFrequencies = moment->highPassFrequencies;
        dataSet.model.highPassOrders = moment->highPassOrders;
        dataSets.push_back(std::move(dataSet));
    }
    options.model = std::move(dataSets.front().model);
    options.observed = std::move(dataSets.front().observed);
    options.noiseStd = std::move(dataSets.front().noiseStd);
    for (std::size_t i = 1; i < dataSets.size(); ++i)
        options.additionalDataSets.push_back(std::move(dataSets[i]));
    options.pytemJoint = joint;
    options.vectorizedKernel = true; // AVX2 kernel whenever this CPU and build support it
    options.halfSpaceStart = true;   // L1/L2 (pyTEM joint) and SCI start from the best half-space
    options.maxIterations = joint ? pytemJointIterations : inversionIterations;
    options.alphaSteps = joint ? pytemJointAlphaTrials : inversionAlphaTrials;
    options.adaptiveAlphaSearch = m_adaptiveAlpha->isChecked();
    options.jacobianMethod = m_jacobian;
    options.jacobianUpdateMethod = m_jacobianUpdate;
    options.broydenRefreshInterval = m_broydenRefresh;
    const int norm = m_regularizationNorm->currentData().toInt();
    options.spatialConstraints = norm == fastSciLin || norm == fastSciLog;
    options.sci = m_sci;
    options.sci.logDataSpace = norm == fastSciLog;
    options.regularizationNorm = options.spatialConstraints
        ? pytem::RegularizationNorm::L2Smooth : static_cast<pytem::RegularizationNorm>(norm);
    options.cacheFirstJacobian = m_cacheJacobian->isChecked();
    if (options.spatialConstraints)
        options.maxIterations = m_sci.maxIterations;
    if (options.cacheFirstJacobian) {
        QString cacheRoot = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
        if (cacheRoot.isEmpty())
            cacheRoot = QDir::tempPath() + "/invertem-cache";
        options.jacobianCacheDirectory = (cacheRoot + "/jacobians").toStdString();
    }
    options.calculateSensitivity = m_sensitivity->isChecked();
    return options;
}

// Solver choices without GUI controls live in invertem_solver.txt beside the
// executable. It is read at start-up and again before every batch, and any
// setting missing from it is appended with its current value, so the file
// always lists everything that can be changed.
void MainWindow::readSolverSettings()
{
    QFile file(QDir(QApplication::applicationDirPath()).filePath("invertem_solver.txt"));
    QStringList seen;
    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&file);
        while (!in.atEnd()) {
            const QStringList pair = in.readLine().section('#', 0, 0).split('=');
            if (pair.size() != 2)
                continue;
            const QString key = pair[0].trimmed().toLower(), value = pair[1].trimmed().toLower();
            seen << key;
            if (key == "transform")
                m_transform = value == "dlf" ? pytem::TransformMethod::DigitalLinearFilter : pytem::TransformMethod::Euler;
            else if (key == "jacobian")
                m_jacobian = value == "finite_difference" ? pytem::JacobianMethod::FiniteDifference : pytem::JacobianMethod::Analytical;
            else if (key == "jacobian_update")
                m_jacobianUpdate = value == "broyden" ? pytem::JacobianUpdateMethod::BroydenRankOne : pytem::JacobianUpdateMethod::FullEveryIteration;
            else if (key == "method")
                m_pytemJoint = value != "adaptive";
            else if (key == "broyden_refresh")
                m_broydenRefresh = std::clamp(value.toInt(), 2, 10);
            else if (key == "sci_vertical_factor")
                m_sci.verticalFactor = std::max(1.01, value.toDouble());
            else if (key == "sci_lateral_factor")
                m_sci.lateralFactor = std::max(1.01, value.toDouble());
            else if (key == "sci_reference_distance")
                m_sci.referenceDistance = std::max(1.0, value.toDouble());
            else if (key == "sci_distance_power")
                m_sci.distancePower = std::max(0.0, value.toDouble());
            else if (key == "sci_constraints")
                m_sci.elevationConstraints = value != "depth";
            else if (key == "sci_max_iterations")
                m_sci.maxIterations = std::clamp(value.toInt(), 1, 200);
        }
        file.close();
    }
    const QList<std::pair<QString, QString>> settings{
        {"method", "method = pytem_joint       # pytem_joint | adaptive (InverTEM adaptive Gauss-Newton)"},
        {"transform", "transform = euler          # euler | dlf (DLF enables the CUDA option)"},
        {"jacobian", "jacobian = analytical      # analytical | finite_difference (not used by pyTEM joint)"},
        {"jacobian_update", "jacobian_update = full     # full | broyden (not used by pyTEM joint)"},
        {"broyden_refresh", "broyden_refresh = 3        # iterations between full Jacobians with broyden (2-10)"},
        {"sci_vertical_factor", QString("sci_vertical_factor = %1  # Fast SCI: resistivity factor allowed between adjacent layers (larger = looser)").arg(m_sci.verticalFactor)},
        {"sci_lateral_factor", QString("sci_lateral_factor = %1   # Fast SCI: factor allowed between neighbours at the reference distance (larger = looser)").arg(m_sci.lateralFactor)},
        {"sci_reference_distance", QString("sci_reference_distance = %1  # Fast SCI: distance [m] at which sci_lateral_factor applies").arg(m_sci.referenceDistance)},
        {"sci_distance_power", QString("sci_distance_power = %1   # Fast SCI: lateral constraint loosens as (distance / reference)^power").arg(m_sci.distancePower)},
        {"sci_constraints", QString("sci_constraints = %1   # SCI: compare neighbours at the same elevation | depth below the surface")
             .arg(m_sci.elevationConstraints ? "elevation" : "depth")},
        {"sci_max_iterations", QString("sci_max_iterations = %1   # Fast SCI: iteration limit").arg(m_sci.maxIterations)}};
    QStringList missing;
    for (const auto &[key, line] : settings)
        if (!seen.contains(key))
            missing << line;
    if (!missing.isEmpty() && file.open(QIODevice::Append | QIODevice::Text)) {
        QTextStream out(&file);
        if (seen.isEmpty())
            out << "# InverTEM solver settings, read at start-up and before every batch\n";
        out << missing.join('\n') << '\n';
    }
}

void MainWindow::runInversion()
{
    readSolverSettings(); // picks up edits made while InverTEM is open
    saveActiveSoundingEdits();
    if (m_soundings.empty()) return;
    QElapsedTimer preparationTimer;
    preparationTimer.start();
    QStringList errors;
    QStringList skippedForTooFewGates;
    std::vector<InversionJob> jobs;
    QProgressDialog preparing("Preparing inversions...", QString(), 0,
                              static_cast<int>(m_soundings.size()), this);
    preparing.setWindowTitle("Batch inversion");
    preparing.setWindowModality(Qt::WindowModal);
    preparing.setMinimumDuration(0);
    preparing.setValue(0);
    for (std::size_t i = 0; i < m_soundings.size(); ++i) {
        preparing.setValue(static_cast<int>(i)); // also processes events so the dialog stays live
        auto &state = m_soundings[i];
        if (!state.includeInBatch)
            continue;
        QStringList sparseMoments;
        if (fittedMoments(state).empty())
            sparseMoments << "no enabled gates";
        for (const auto *moment : fittedMoments(state)) {
            const std::size_t momentIndex = static_cast<std::size_t>(
                moment - state.sounding.moments.data());
            int enabledCount = 0;
            for (std::size_t gate = 0; gate < moment->times.size(); ++gate)
                enabledCount += gateUsed(state, momentIndex, gate) ? 1 : 0;
            if (enabledCount < 3)
                sparseMoments << QString("%1: %2 gates")
                    .arg(QString::fromStdString(moment->name))
                    .arg(enabledCount);
        }
        if (!sparseMoments.isEmpty()) {
            skippedForTooFewGates << QString("%1 (%2)")
                .arg(QFileInfo(state.path).fileName(), sparseMoments.join(", "));
            continue;
        }
        QString error;
        if (!validateInputs(state, error)) {
            errors << QString("%1: %2").arg(QFileInfo(state.path).fileName(), error);
            continue;
        }
        try {
            state.options = buildOptions(state);
            state.resultMomentNames.clear();
            state.resultMomentIndices.clear();
            state.resultGateIndices.clear();
            for (const auto *moment : fittedMoments(state)) {
                const std::size_t momentIndex = static_cast<std::size_t>(
                    moment - state.sounding.moments.data());
                state.resultMomentNames.push_back(moment->name);
                state.resultMomentIndices.push_back(momentIndex);
                std::vector<std::size_t> gates;
                for (std::size_t gate = 0; gate < moment->times.size(); ++gate) {
                    if (gateUsed(state, momentIndex, gate))
                        gates.push_back(gate);
                }
                state.resultGateIndices.push_back(std::move(gates));
            }
            state.haveResult = false;
            state.error.clear();
            // Local planar metres from longitude/latitude, for the SCI neighbour graph.
            constexpr double degree = 3.14159265358979323846 / 180.0;
            jobs.push_back({static_cast<int>(i), QFileInfo(state.path).fileName(), state.options,
                            state.sounding.longitude * 111320.0 * std::cos(state.sounding.latitude * degree),
                            state.sounding.latitude * 110540.0, state.sounding.elevation});
        } catch (const std::exception &exception) {
            errors << QString("%1: %2").arg(QFileInfo(state.path).fileName(),
                                             QString::fromUtf8(exception.what()));
        }
    }
    preparing.reset();
    m_preparationSeconds = preparationTimer.elapsed() / 1000.0; // before any dialog waits for the user
    // Long lists are summarised in dialogs; the full list goes to the run log.
    m_log->clear();
    if (!skippedForTooFewGates.isEmpty())
        m_log->appendPlainText("Skipped for fewer than three enabled gates:\n  "
                               + skippedForTooFewGates.join("\n  ") + "\n");
    const QString skippedList = skippedForTooFewGates.size() > 10
        ? QString("%1 stations do not have enough data to invert (see the run log).").arg(skippedForTooFewGates.size())
        : skippedForTooFewGates.join("\n");
    if (!errors.isEmpty()) {
        QString message = errors.join("\n");
        if (!skippedForTooFewGates.isEmpty())
            message += QStringLiteral("\n\nAlso skipped for fewer than three enabled gates:\n") + skippedList;
        QMessageBox::warning(this, "Cannot start batch inversion", message);
        return;
    }
    if (jobs.empty()) {
        if (!skippedForTooFewGates.isEmpty()) {
            QMessageBox::warning(this, "Soundings skipped",
                QStringLiteral("No sounding has at least three enabled gates for every selected moment.\n\n")
                + skippedList);
            return;
        }
        QMessageBox::warning(this, "Cannot start batch inversion",
            "Enable at least one sounding by clicking its point on the map.");
        return;
    }
    if (!skippedForTooFewGates.isEmpty())
        QMessageBox::warning(this, "Soundings skipped", skippedForTooFewGates.size() > 10
            ? QString("%1 stations do not have enough data to invert, so they will be skipped. "
                      "They are listed in the run log.").arg(skippedForTooFewGates.size())
            : QStringLiteral("The following soundings have fewer than three enabled gates and will be skipped:\n\n")
                + skippedList);
    const int firstJobIndex = jobs.front().index;
    const QDir reportDirectory(QFileInfo(
        m_soundings[static_cast<std::size_t>(firstJobIndex)].path).absolutePath());
    m_timingReportPath = reportDirectory.filePath(
        "invertem_timing_"
        + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".txt");
    const auto &firstOptions = jobs.front().options;
    const QString transform = firstOptions.model.transform == pytem::TransformMethod::Euler
        ? "Euler-11" : "DLF";
    const QString jacobian = firstOptions.jacobianMethod == pytem::JacobianMethod::Analytical
        ? "analytical" : "finite difference";
    const QString norm = firstOptions.regularizationNorm == pytem::RegularizationNorm::L1Blocky
        ? "L1 blocky" : "L2 smooth";
    const QString backend = firstOptions.model.useGpu
        ? "CUDA forward" : "CPU forward";
#ifdef INVERTEM_USE_OPENMP
    const QString scheduler = "OpenMP scheduler";
#else
    const QString scheduler = "C++ thread scheduler";
#endif
    if (firstOptions.spatialConstraints)
        m_log->appendPlainText(QString(
            "Starting Fast SCI over %1 sounding(s): %2, %7 data space, vertical factor %3, lateral factor %4 at %5 m "
            "(distance power %6), Delaunay neighbours at the same %8, step-length Marquardt damping")
            .arg(jobs.size()).arg(transform).arg(m_sci.verticalFactor).arg(m_sci.lateralFactor)
            .arg(m_sci.referenceDistance).arg(m_sci.distancePower)
            .arg(firstOptions.sci.logDataSpace ? "log" : "linear")
            .arg(m_sci.elevationConstraints ? "elevation" : "depth"));
    else if (firstOptions.pytemJoint)
        m_log->appendPlainText(QString(
            "Starting %1 independent pyTEM joint inversion(s): %2, %5%6, shared %3-point step grid, %4 kernel")
            .arg(jobs.size()).arg(transform).arg(pytemJointStepCount)
            .arg(firstOptions.vectorizedKernel && pytem::vectorKernelAvailable() ? "AVX2 vectorised" : "scalar")
            .arg(norm,
                 firstOptions.adaptiveAlphaSearch ? ", RMS=1 alpha fit" : ""));
    else
        m_log->appendPlainText(QString(
            "Starting %1 independent inversion(s) in one batch: %2, %3 Jacobian, %4, %5, %6")
            .arg(jobs.size()).arg(transform, jacobian, norm, backend, scheduler));
    m_jobIterations.assign(m_soundings.size(), 0);
    m_liveRms.assign(m_soundings.size(), std::numeric_limits<double>::quiet_NaN());
    m_completedJobs = 0;
    m_successfulJobs = 0;
    m_totalJobs = static_cast<int>(jobs.size());
    m_status->setText(QString("0/%1 completed").arg(m_totalJobs));
    m_batchKilled = false;
    m_batchTimer.restart();
    m_lastEtaUpdateMs = -1;
    m_estimatedRemainingMs = -1;
    m_progress->setRange(0, static_cast<int>(jobs.size()) * firstOptions.maxIterations);
    m_progress->setValue(0);
    m_timeEstimate->setText("Elapsed 00:00 — ETA estimating…");
    m_elapsedTimer->start();
    setRunning(true);
    m_workerThread = new QThread(this);
    m_worker = new InversionWorker(std::move(jobs),
                                   static_cast<unsigned>(m_parallelJobs->value()));
    m_worker->moveToThread(m_workerThread);
    connect(m_workerThread, &QThread::started, m_worker, &InversionWorker::run);
    connect(m_worker, &InversionWorker::progress, this, &MainWindow::showProgress);
    connect(m_worker, &InversionWorker::resultReady, this, &MainWindow::inversionComplete);
    connect(m_worker, &InversionWorker::failed, this, &MainWindow::inversionFailed);
    connect(m_worker, &InversionWorker::allFinished, this, &MainWindow::batchFinished);
    m_workerThread->start();
}

void MainWindow::killInversion()
{
    if (!m_worker)
        return;
    m_batchKilled = true;
    m_worker->requestCancel();
    m_killButton->setEnabled(false);
    m_log->appendPlainText(
        "Kill requested: stopping active and queued inversions; completed models are retained.");
}

void MainWindow::showProgress(int jobIndex, int iteration, double rms, const QString &detail)
{
    if (jobIndex < 0 || jobIndex >= static_cast<int>(m_soundings.size())) return;
    m_jobIterations[static_cast<std::size_t>(jobIndex)] = std::max(
        m_jobIterations[static_cast<std::size_t>(jobIndex)], iteration);
    int totalProgress = 0;
    for (int value : m_jobIterations) totalProgress += value;
    m_progress->setValue(totalProgress);
    m_liveRms[static_cast<std::size_t>(jobIndex)] = rms;
    if (!m_mapRefreshTimer->isActive())
        m_mapRefreshTimer->start();
    updateBatchTiming();
    const QString name = QFileInfo(m_soundings[static_cast<std::size_t>(jobIndex)].path).fileName();
    if (!m_batchKilled)
        m_status->setText(QString("%1/%2 completed")
            .arg(m_successfulJobs).arg(m_totalJobs));
    m_log->appendPlainText(QString("[%1] iteration %2  RMS=%3  %4")
        .arg(name).arg(iteration).arg(rms, 0, 'f', 2).arg(detail));
}

void MainWindow::inversionComplete(int jobIndex, const pytem::InversionResult &result)
{
    if (jobIndex < 0 || jobIndex >= static_cast<int>(m_soundings.size())) return;
    auto &state = m_soundings[static_cast<std::size_t>(jobIndex)];
    state.result = result;
    state.haveResult = true;
    state.resultStale = false;
    SoundingState::SavedModel saved;
    saved.key = savedModelKey(state.options, state.momentChoice);
    saved.label = savedModelLabel(state.options);
    saved.momentChoice = state.momentChoice;
    saved.options = state.options;
    saved.result = result;
    saved.momentNames = state.resultMomentNames;
    saved.momentIndices = state.resultMomentIndices;
    saved.gateIndices = state.resultGateIndices;
    saved.doiDepth = sensitivityDoi(state.options, result);
    const auto existing = std::find_if(
        state.savedModels.begin(), state.savedModels.end(),
        [&saved](const SoundingState::SavedModel &candidate) {
            return candidate.key == saved.key;
        });
    if (existing == state.savedModels.end())
        state.savedModels.push_back(std::move(saved));
    else
        *existing = std::move(saved);
    markProjectModified();
    ++m_completedJobs;
    ++m_successfulJobs;
    m_jobIterations[static_cast<std::size_t>(jobIndex)] = state.options.maxIterations;
    m_progress->setValue(std::accumulate(m_jobIterations.begin(), m_jobIterations.end(), 0));
    updateBatchTiming();
    m_log->appendPlainText(QString("[%1] finished: %2; first Jacobian: %3")
        .arg(QFileInfo(state.path).fileName(), QString::fromStdString(result.message),
             QString::fromStdString(result.firstJacobianSource)));
    m_log->appendPlainText(QString("[%1] timing: total %2 s, Jacobian %3 s, alpha forwards %4 s")
        .arg(QFileInfo(state.path).fileName())
        .arg(result.timing.totalSeconds, 0, 'f', 2)
        .arg(result.timing.jacobianSeconds, 0, 'f', 2)
        .arg(result.timing.alphaForwardSeconds, 0, 'f', 2));
    if (jobIndex == m_activeSoundingIndex)
        displayResult(state);
    updateTransectPlot();
    updateMapPlot();
    // Result signals arrive while the worker continues with the remaining
    // soundings. Repaint now so each newly completed marker fills immediately
    // instead of waiting for the batch to finish or another window event.
    m_mapPlot->repaint();
    m_status->setText(QString("%1/%2 completed")
        .arg(m_successfulJobs).arg(m_totalJobs));
}

void MainWindow::inversionFailed(int jobIndex, const QString &message)
{
    if (jobIndex < 0 || jobIndex >= static_cast<int>(m_soundings.size())) return;
    auto &state = m_soundings[static_cast<std::size_t>(jobIndex)];
    state.error = message;
    if (!state.savedModels.empty())
        activateSavedModel(state, state.savedModels.back());
    else
        state.haveResult = false;
    ++m_completedJobs;
    m_jobIterations[static_cast<std::size_t>(jobIndex)] = state.options.maxIterations;
    m_progress->setValue(std::accumulate(m_jobIterations.begin(), m_jobIterations.end(), 0));
    updateBatchTiming();
    m_log->appendPlainText(QString("[%1] stopped: %2")
        .arg(QFileInfo(state.path).fileName(), message));
    m_status->setText(QString("%1/%2 completed")
        .arg(m_successfulJobs).arg(m_totalJobs));
}

void MainWindow::batchFinished()
{
    const bool wasKilled = m_batchKilled;
    const int successful = m_successfulJobs;
    finishWorker();
    m_liveRms.clear();
    updateMapPlot();
    writeTimingReport();
    if (const auto *state = activeSounding()) {
        m_exportModelledButton->setEnabled(state->haveResult);
        if (state->haveResult) displayResult(*state);
    }
    m_status->setText(QString("%1/%2 completed")
        .arg(successful).arg(m_totalJobs));
    if (m_batchTimer.isValid())
        m_timeEstimate->setText(QString("Elapsed %1 — %2")
            .arg(formatDuration(m_batchTimer.elapsed()),
                 wasKilled ? QString("stopped") : QString("complete")));
    if (wasKilled)
        m_log->appendPlainText(QString("Inversion killed: %1 completed model(s) kept.")
            .arg(successful));
    m_batchKilled = false;
}

void MainWindow::clearSavedResults()
{
    if (m_worker)
        return;
    for (auto &state : m_soundings) {
        std::vector<SoundingState::SavedModel>().swap(state.savedModels);
        state.options = {};
        state.result = {};
        state.resultMomentNames.clear();
        state.resultMomentIndices.clear();
        state.resultGateIndices.clear();
        state.haveResult = false;
        state.error.clear();
    }
    m_modelPlot->clear();
    updateInputPlot();
    updateMapPlot();
    m_exportModelledButton->setEnabled(false);
    m_clearResultsButton->setEnabled(false);
    markProjectModified();
    m_log->appendPlainText("Saved inversion results cleared from memory");
}

void MainWindow::writeTimingReport()
{
    if (m_timingReportPath.isEmpty())
        return;
    QFile file(m_timingReportPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QString fallbackRoot = QStandardPaths::writableLocation(
            QStandardPaths::DocumentsLocation);
        if (fallbackRoot.isEmpty())
            fallbackRoot = QStandardPaths::writableLocation(
                QStandardPaths::AppDataLocation);
        const QString fallbackDirectory = fallbackRoot + "/InverTEM";
        QDir().mkpath(fallbackDirectory);
        m_timingReportPath = QDir(fallbackDirectory).filePath(
            QFileInfo(m_timingReportPath).fileName());
        file.setFileName(m_timingReportPath);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            m_log->appendPlainText("Could not write timing report: "
                                   + file.errorString());
            return;
        }
    }
    QTextStream out(&file);
    out << "InverTEM 1.0.0 inversion timing report\n";
    out << "Generated\t" << QDateTime::currentDateTime().toString(Qt::ISODate) << '\n';
    out << "Batch preparation seconds\t"
        << QString::number(m_preparationSeconds, 'f', 2) << '\n';
    out << "Batch wall seconds\t"
        << QString::number(m_batchTimer.isValid() ? m_batchTimer.elapsed() / 1000.0 : 0.0,
                           'f', 2) << '\n';
    out << "Independent batch workers\t" << m_parallelJobs->value() << '\n';
#ifdef INVERTEM_USE_OPENMP
    out << "Batch scheduler\tOpenMP\n";
#else
    out << "Batch scheduler\tstandard C++ threads\n";
#endif
    out << "Forward backend requested\t"
        << (m_useGpu->isChecked() ? "NVIDIA CUDA" : "CPU") << '\n';
    out << "Layers\t" << m_layerCount->value() << '\n';
    const bool joint = m_pytemJoint;
    out << "Inversion method\t" << (joint ? "pyTEM joint" : "InverTEM adaptive") << '\n';
    if (joint) {
        out << "Forward kernel\t" << (pytem::vectorKernelAvailable() ? "AVX2 vectorised" : "scalar") << '\n';
    }
    out << "Fixed maximum iterations\t"
        << (joint ? pytemJointIterations : inversionIterations) << '\n';
    out << "Transform\t" << (m_transform == pytem::TransformMethod::Euler ? "Euler-11" : "DLF (Key)") << '\n';
    out << "Jacobian\t" << (m_jacobian == pytem::JacobianMethod::Analytical ? "analytical adjoint" : "finite difference") << '\n';
    const bool broyden = m_jacobianUpdate == pytem::JacobianUpdateMethod::BroydenRankOne;
    out << "Jacobian update\t"
        << (broyden ? "guarded Broyden rank-one"
                    : "full every iteration") << '\n';
    if (broyden)
        out << "Full Jacobian refresh interval\t"
            << m_broydenRefresh << '\n';
    out << "Regularisation\t" << m_regularizationNorm->currentText() << '\n';
    out << "Alpha search\t"
        << (!m_adaptiveAlpha->isChecked() ? "fixed descending sweep"
            : joint ? "RMS=1 target fit" : "adaptive with fixed fallback")
        << '\n';
    out << "Maximum alpha trials\t"
        << (joint ? pytemJointAlphaTrials : inversionAlphaTrials) << '\n';
    out << "Alpha log10 step\t" << QString::number(1.0 / 9.0, 'f', 2)
        << "\n\n";
    out << "sounding\tstatus\titerations\ttotal_s\tinitial_forward_s\t"
           "jacobian_s\tbroyden_update_s\tlinear_solve_s\talpha_forward_s\tsensitivity_s\t"
           "other_s\tjacobian_evaluations\tbroyden_updates\talpha_trial_forwards\t"
           "first_jacobian_source\n";
    for (const auto &state : m_soundings) {
        if (!state.includeInBatch)
            continue;
        out << QFileInfo(state.path).fileName() << '\t';
        if (!state.haveResult) {
            out << "failed: " << state.error << "\n";
            continue;
        }
        const auto &timing = state.result.timing;
        const double measuredStages = timing.initialForwardSeconds
            + timing.jacobianSeconds + timing.broydenUpdateSeconds
            + timing.linearSolveSeconds
            + timing.alphaForwardSeconds + timing.sensitivitySeconds;
        const double otherSeconds = std::max(0.0,
            timing.totalSeconds - measuredStages);
        out << "completed\t" << state.result.iterations << '\t'
            << QString::number(timing.totalSeconds, 'f', 2) << '\t'
            << QString::number(timing.initialForwardSeconds, 'f', 2) << '\t'
            << QString::number(timing.jacobianSeconds, 'f', 2) << '\t'
            << QString::number(timing.broydenUpdateSeconds, 'f', 2) << '\t'
            << QString::number(timing.linearSolveSeconds, 'f', 2) << '\t'
            << QString::number(timing.alphaForwardSeconds, 'f', 2) << '\t'
            << QString::number(timing.sensitivitySeconds, 'f', 2) << '\t'
            << QString::number(otherSeconds, 'f', 2) << '\t'
            << timing.jacobianEvaluations << '\t'
            << timing.broydenUpdates << '\t'
            << timing.alphaTrialForwards << '\t'
            << QString::fromStdString(state.result.firstJacobianSource) << '\n';
    }
    file.close();
    m_log->appendPlainText("Timing report: " + QDir::toNativeSeparators(m_timingReportPath));
}

void MainWindow::updateBatchTiming()
{
    if (!m_batchTimer.isValid())
        return;
    const qint64 elapsed = m_batchTimer.elapsed();
    const int completedUnits = m_progress->value();
    const int totalUnits = m_progress->maximum();
    const bool complete = totalUnits > 0 && completedUnits >= totalUnits;
    constexpr qint64 etaRefreshIntervalMs = 10000;
    if (completedUnits <= 0 || totalUnits <= 0) {
        m_timeEstimate->setText(QString("Elapsed %1 — ETA estimating…")
            .arg(formatDuration(elapsed)));
        return;
    }
    if (complete) {
        m_estimatedRemainingMs = 0;
    } else if (m_lastEtaUpdateMs < 0
               || elapsed - m_lastEtaUpdateMs >= etaRefreshIntervalMs) {
        m_lastEtaUpdateMs = elapsed;
        m_estimatedRemainingMs = static_cast<qint64>(
            static_cast<double>(elapsed) * (totalUnits - completedUnits)
            / completedUnits);
    }
    if (m_estimatedRemainingMs < 0) {
        m_timeEstimate->setText(QString("Elapsed %1 — ETA estimating…")
            .arg(formatDuration(elapsed)));
        return;
    }
    m_timeEstimate->setText(QString("Elapsed %1 — ETA %2")
        .arg(formatDuration(elapsed), formatDuration(m_estimatedRemainingMs)));
}

void MainWindow::updateResistivityAxis()
{
    if (!m_modelPlot)
        return;
    if (m_autoResistivityAxis->isChecked()) {
        m_modelPlot->clearXRange();
        return;
    }
    m_modelPlot->setXRange(m_resistivityAxisMinimum->value(),
                           m_resistivityAxisMaximum->value());
}

void MainWindow::finishWorker()
{
    if (m_elapsedTimer)
        m_elapsedTimer->stop();
    setRunning(false);
    if (!m_workerThread) return;
    m_workerThread->quit();
    m_workerThread->wait();
    delete m_worker;
    m_worker = nullptr;
    delete m_workerThread;
    m_workerThread = nullptr;
}

void MainWindow::displayResult(const SoundingState &state)
{
    const auto &result = state.result;
    const auto &options = state.options;
    updateInputPlot(); // current gates, plus this result's predicted response
    auto modelCurve = [](const pytem::InversionOptions &modelOptions,
                         const std::vector<double> &rho) {
        QVector<QPointF> points;
        double bottom = finiteModelDepth(modelOptions);
        bottom += std::max(10.0, bottom * 0.35);
        double top = 0.0;
        for (std::size_t i = 0; i < rho.size(); ++i) {
            const double base = i < modelOptions.model.thicknesses.size()
                ? top + modelOptions.model.thicknesses[i] : bottom;
            points.push_back({rho[i], top}); points.push_back({rho[i], base});
            if (i + 1 < rho.size()) points.push_back({rho[i + 1], base});
            top = base;
        }
        return points;
    };
    QVector<const SoundingState::SavedModel *> displayedModels;
    for (const auto &saved : state.savedModels) {
        if (saved.momentChoice == state.momentChoice)
            displayedModels.push_back(&saved);
    }
    QVector<PlotWidget::Curve> modelCurves;
    auto rmsText = [](const pytem::InversionResult &r) {
        return r.rmsHistory.empty() ? QString("RMS n/a") : QString("RMS %1").arg(r.rmsHistory.back(), 0, 'f', 2);
    };
    // One distinct colour per shown model; legend "L1 (40 layers) RMS 0.98",
    // with the depth added only when two shown models would otherwise match.
    static const char *palette[] = {"#1b75bc", "#d95f02", "#1a9850", "#7b3294",
                                    "#e7298a", "#8c564b", "#17a2b8", "#555555"};
    for (int index = 0; index < displayedModels.size(); ++index) {
        const auto *saved = displayedModels[index];
        const QColor color(palette[index % 8]);
        const auto split = splitModelCurveAtDoi(
            modelCurve(saved->options, saved->result.resistivities),
            saved->doiDepth);
        QString name = savedModelLabel(saved->options, false);
        if (std::count_if(displayedModels.begin(), displayedModels.end(), [&](const auto *other) {
                return savedModelLabel(other->options, false) == name; }) > 1)
            name = savedModelLabel(saved->options);
        modelCurves.push_back({name + " " + rmsText(saved->result),
                              split.withinDoi, color, false, false});
        if (!split.beyondDoi.isEmpty()) {
            PlotWidget::Curve uncertain;
            uncertain.points = split.beyondDoi;
            uncertain.color = color;
            uncertain.dashed = true;
            modelCurves.push_back(std::move(uncertain));
        }
    }
    if (modelCurves.isEmpty()) {
        const double doiDepth = sensitivityDoi(options, result);
        const auto split = splitModelCurveAtDoi(
            modelCurve(options, result.resistivities), doiDepth);
        modelCurves.push_back({rmsText(result), split.withinDoi,
                               QColor("#1b75bc"), false, false});
        if (!split.beyondDoi.isEmpty()) {
            PlotWidget::Curve uncertain;
            uncertain.points = split.beyondDoi;
            uncertain.color = QColor("#1b75bc");
            uncertain.dashed = true;
            modelCurves.push_back(std::move(uncertain));
        }
    }
    // The model plot was configured with logX=true; resistivity is always logarithmic.
    m_modelPlot->setAxes("Model, " + rmsText(result) + (state.resultStale ? " (gates changed)" : ""),
                         "Resistivity [Ohm m]",
                         "Depth [m]", true, false, true);
    m_modelPlot->setCurves(modelCurves);
}

void MainWindow::displayInput(const SoundingState &)
{
    updateInputPlot();
    m_modelPlot->setAxes("Model", "Resistivity [Ohm m]", "Depth [m]", true, false, true);
    m_modelPlot->clear();
}

void MainWindow::updateInputPlot()
{
    const auto *state = activeSounding();
    if (!state) {
        m_soundingPlot->clear();
        return;
    }
    QVector<PlotWidget::Curve> curves;
    for (std::size_t momentIndex = 0;
         momentIndex < state->sounding.moments.size(); ++momentIndex) {
        QVector<QPointF> used, rejected;
        QVector<int> usedRows, rejectedRows;
        QVector<double> usedErrors, rejectedErrors;
        for (std::size_t row = 0; row < m_rowMomentIndices.size(); ++row) {
            if (m_rowMomentIndices[row] != momentIndex)
                continue;
            const std::size_t gate = m_rowGateIndices[row];
            const auto &moment = state->sounding.moments[momentIndex];
            if (gate >= moment.times.size() || gate >= moment.voltages.size())
                continue;
            const double time = moment.times[gate];
            const double value = std::abs(moment.voltages[gate]);
            const double standardError = gate < moment.standardErrors.size()
                ? std::abs(moment.standardErrors[gate]) : 0.0;
            if (time <= 0.0 || value <= 0.0)
                continue;
            if (state->includeInBatch && state->gateEnabled[momentIndex][gate]) {
                used.push_back({time, value});
                usedRows.push_back(static_cast<int>(row));
                usedErrors.push_back(standardError);
            } else {
                rejected.push_back({time, value});
                rejectedRows.push_back(static_cast<int>(row));
                rejectedErrors.push_back(standardError);
            }
        }
        if (used.isEmpty() && rejected.isEmpty())
            continue;
        const auto &moment = state->sounding.moments[momentIndex];
        const QString name = QString::fromStdString(moment.name);
        const QColor color = momentColor(moment.name, momentIndex);
        curves.push_back({name + " data", used, color, true, false, true,
                          usedRows, false, usedErrors});
        QColor disabled = color.lighter(180);
        disabled.setAlpha(150);
        curves.push_back({{}, rejected, disabled, true, false,
                          true, rejectedRows, false, rejectedErrors});
    }
    // Predicted response of the shown result, dashed when gates changed since.
    if (state->haveResult) {
        std::size_t offset = 0, dataSet = 0;
        auto addPrediction = [&](const pytem::ForwardModel &model) {
            QVector<QPointF> points;
            for (std::size_t i = 0; i < model.times.size() && offset + i < state->result.predicted.size(); ++i)
                points.push_back({model.times[i], std::abs(state->result.predicted[offset + i])});
            offset += model.times.size();
            const std::size_t momentIndex = dataSet < state->resultMomentIndices.size()
                ? state->resultMomentIndices[dataSet] : dataSet;
            PlotWidget::Curve predicted;
            predicted.points = points;
            predicted.color = momentColor(momentIndex < state->sounding.moments.size()
                ? state->sounding.moments[momentIndex].name : std::string(), momentIndex);
            predicted.dashed = state->resultStale;
            curves.push_back(std::move(predicted));
            ++dataSet;
        };
        addPrediction(state->options.model);
        for (const auto &extra : state->options.additionalDataSets)
            addPrediction(extra.model);
    }
    m_soundingPlot->setCurves(curves);
}

void MainWindow::updateMapPlot()
{
    QVector<QPointF> includedPoints;
    QVector<int> includedIds;
    QVector<QPointF> excludedPoints;
    QVector<int> excludedIds;
    QVector<QPointF> unusablePoints;
    QVector<int> unusableIds;
    QVector<QPointF> activePoint;
    QVector<QPointF> transectPoints;
    QVector<int> transectIds;
    struct CompletedPoint {
        QPointF point;
        int id = -1;
        double rms = 0.0;
        bool running = false;
    };
    QVector<CompletedPoint> completedPoints;
    double latitudeSum = 0.0;
    int coordinateCount = 0;
    for (std::size_t i = 0; i < m_soundings.size(); ++i) {
        const auto &sounding = m_soundings[i].sounding;
        if (!std::isfinite(sounding.longitude) || !std::isfinite(sounding.latitude)
            || (sounding.longitude == 0.0 && sounding.latitude == 0.0))
            continue;
        const QPointF point(sounding.longitude, sounding.latitude);
        latitudeSum += sounding.latitude;
        ++coordinateCount;
        // Included soundings the batch would skip (a fitted moment with fewer
        // than three used gates) are drawn grey like excluded ones.
        const auto moments = fittedMoments(m_soundings[i]);
        const bool usable = !moments.empty() && std::all_of(
            moments.begin(), moments.end(), [&](const pytem::UsfMoment *moment) {
                const std::size_t momentIndex = static_cast<std::size_t>(
                    moment - sounding.moments.data());
                int used = 0;
                for (std::size_t gate = 0; gate < moment->times.size(); ++gate)
                    used += gateUsed(m_soundings[i], momentIndex, gate) ? 1 : 0;
                return used >= 3;
            });
        if (m_soundings[i].includeInBatch && usable) {
            includedPoints.push_back(point);
            includedIds.push_back(static_cast<int>(i));
        } else if (m_soundings[i].includeInBatch) {
            unusablePoints.push_back(point);
            unusableIds.push_back(static_cast<int>(i));
        } else {
            excludedPoints.push_back(point);
            excludedIds.push_back(static_cast<int>(i));
        }
        if (static_cast<int>(i) == m_activeSoundingIndex)
            activePoint.push_back(point);
        if (m_plotMode && m_plotMode->currentIndex() == 1
            && static_cast<int>(i) >= m_transectStart
            && static_cast<int>(i) < m_transectStart + m_transectSize->value()) {
            transectPoints.push_back(point);
            transectIds.push_back(static_cast<int>(i));
        }
        if (m_soundings[i].includeInBatch && usable && m_soundings[i].haveResult
            && !m_soundings[i].result.rmsHistory.empty()) {
            completedPoints.push_back({
                point, static_cast<int>(i),
                m_soundings[i].result.rmsHistory.back()});
        } else if (i < m_liveRms.size() && std::isfinite(m_liveRms[i])) {
            // Still inverting: ring coloured by the RMS after its latest iteration.
            completedPoints.push_back({point, static_cast<int>(i), m_liveRms[i], true});
        }
    }
    QVector<PlotWidget::Curve> curves;
    if (!includedPoints.isEmpty())
        curves.push_back({"Included soundings", includedPoints, QColor("#4c78a8"),
                          true, false, true, includedIds, false});
    if (!excludedPoints.isEmpty())
        curves.push_back({"Excluded soundings", excludedPoints, QColor("#b8b8b8"),
                          true, false, true, excludedIds, false});
    if (!unusablePoints.isEmpty())
        curves.push_back({"Too few gates", unusablePoints, QColor("#dcdcdc"),
                          true, false, true, unusableIds, false});
    for (const auto &completed : completedPoints) {
        PlotWidget::Curve curve;
        curve.points = {completed.point};
        curve.color = PlotWidget::colorScaleColor(completed.rms, 0.0, 5.0);
        curve.markers = true;
        curve.selectable = true;
        curve.pointIds = {completed.id};
        curve.line = false;
        curve.filledMarkers = !completed.running;
        curves.push_back(std::move(curve));
    }
    if (!transectPoints.isEmpty()) {
        PlotWidget::Curve visible;
        visible.name = "Visible transect";
        visible.points = transectPoints;
        visible.color = QColor("#8e44ad");
        visible.markers = true;
        visible.selectable = true;
        visible.pointIds = transectIds;
        visible.line = false;
        curves.push_back(std::move(visible));
    }
    if (!activePoint.isEmpty())
        curves.push_back({"Current sounding", activePoint, QColor("#f28e2b"),
                          true, false, false, {}, false});
    if (coordinateCount > 0) {
        constexpr double degreesToRadians = 3.14159265358979323846 / 180.0;
        const double meanLatitude = latitudeSum / coordinateCount;
        const double longitudeScale = std::max(
            0.05, std::abs(std::cos(meanLatitude * degreesToRadians)));
        m_mapPlot->setEqualAspect(true, longitudeScale);
    }
    if (completedPoints.isEmpty())
        m_mapPlot->clearColorScale();
    else
        m_mapPlot->setColorScale("RMS", 0.0, 5.0);
    m_mapPlot->setCurves(curves);
}

void MainWindow::updateSystemSummary()
{
    const auto *state = activeSounding();
    if (!state)
        return;
    const auto moments = selectedMoments(*state);
    QStringList stageSummaries;
    int selected = 0;
    int total = 0;
    for (const auto *moment : moments) {
        const std::size_t momentIndex = static_cast<std::size_t>(
            moment - state->sounding.moments.data());
        QStringList stages;
        for (std::size_t i = 0; i < moment->lowPassFrequencies.size(); ++i)
            stages << QString("LP %1 kHz (order %2)")
                .arg(moment->lowPassFrequencies[i] / 1000.0)
                .arg(moment->lowPassOrders[i]);
        for (std::size_t i = 0; i < moment->highPassFrequencies.size(); ++i)
            stages << QString("HP %1 Hz (order %2)")
                .arg(moment->highPassFrequencies[i])
                .arg(moment->highPassOrders[i]);
        stageSummaries << QString("%1: %2")
            .arg(QString::fromStdString(moment->name))
            .arg(stages.isEmpty() ? "no filters" : stages.join(" × "));
        if (momentIndex >= state->gateEnabled.size())
            continue;
        total += static_cast<int>(moment->times.size());
        for (std::size_t gate = 0;
             gate < moment->times.size()
             && gate < state->gateEnabled[momentIndex].size(); ++gate)
            selected += state->gateEnabled[momentIndex][gate] ? 1 : 0;
    }
    QString summary = QString("%1; %2/%3 gates selected")
        .arg(stageSummaries.join(" | ")).arg(selected).arg(total);
    if (!state->includeInBatch)
        summary += "; sounding excluded from batch";
    m_systemLabel->setText(summary);
}

void MainWindow::exportModelledData()
{
    const auto *active = activeSounding();
    if (!active || active->savedModels.empty())
        return;
    QVector<const SoundingState::SavedModel *> choices;
    QStringList labels;
    for (const auto &saved : active->savedModels) {
        if (saved.momentChoice != active->momentChoice)
            continue;
        choices.push_back(&saved);
        labels.push_back(saved.label);
    }
    if (choices.isEmpty())
        return;
    int selectedIndex = 0;
    if (choices.size() > 1) {
        bool accepted = false;
        const QString selected = QInputDialog::getItem(
            this, "Export modelled data", "Saved model configuration",
            labels, labels.size() - 1, false, &accepted);
        if (!accepted)
            return;
        selectedIndex = labels.indexOf(selected);
        if (selectedIndex < 0)
            return;
    }
    const auto &prototype = *choices[selectedIndex];
    const std::size_t layerCount = prototype.result.resistivities.size();
    if (layerCount == 0)
        return;

    QSettings settings;
    const QString defaultName = "invertem_modelled_data.xyz";
    QString path = QFileDialog::getSaveFileName(
        this, "Export modelled data",
        settings.value("lastExportDirectory").toString() + "/" + defaultName,
        "Aarhus Workbench XYZ (*.xyz)");
    if (path.isEmpty())
        return;
    if (!path.endsWith(".xyz", Qt::CaseInsensitive))
        path += ".xyz";
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Export failed", file.errorString());
        return;
    }
    QTextStream out(&file);
    out << "Aarhus Workbench XYZ export\n"
        << "DATA TYPE\n"
        << "tTEM/DT\n"
        << "COORDINATE SYSTEM\n"
        << "epsg:" << (active->sounding.epsg > 0
                            ? active->sounding.epsg : 4326) << '\n'
        << "/ LINE_NO UTMX UTMY ELEVATION";
    for (std::size_t layer = 0; layer < layerCount; ++layer)
        out << " RHO" << layer + 1;
    for (std::size_t layer = 0; layer < layerCount; ++layer)
        out << " RHO" << layer + 1 << "_STD";
    for (std::size_t layer = 0; layer < layerCount; ++layer)
        out << " DEP_BOT" << layer + 1;
    for (std::size_t layer = 0; layer < layerCount; ++layer)
        out << " DEP_BOT" << layer + 1 << "_STD";
    out << " DOI_CONSERVATIVE DOI_STANDARD RESDATA\n";

    int exported = 0;
    for (const auto &state : m_soundings) {
        const auto saved = std::find_if(
            state.savedModels.begin(), state.savedModels.end(),
            [&prototype](const SoundingState::SavedModel &candidate) {
                return candidate.key == prototype.key;
            });
        if (saved == state.savedModels.end()
            || saved->result.resistivities.size() != layerCount)
            continue;
        const double x = state.sounding.sourceX != 0.0
            ? state.sounding.sourceX : state.sounding.longitude;
        const double y = state.sounding.sourceY != 0.0
            ? state.sounding.sourceY : state.sounding.latitude;
        out << xyzLineNumber(state.sounding) << ' '
            << QString::number(x, 'g', 12) << ' '
            << QString::number(y, 'g', 12) << ' '
            << QString::number(state.sounding.elevation, 'g', 12);
        for (double resistivity : saved->result.resistivities)
            out << ' ' << QString::number(resistivity, 'g', 12);
        for (std::size_t layer = 0; layer < layerCount; ++layer)
            out << " nan";
        double depth = 0.0;
        for (std::size_t layer = 0; layer < layerCount; ++layer) {
            if (layer < saved->options.model.thicknesses.size())
                depth += saved->options.model.thicknesses[layer];
            out << ' ' << QString::number(depth, 'g', 12);
        }
        for (std::size_t layer = 0; layer < layerCount; ++layer)
            out << " nan";
        if (saved->doiDepth > 0.0)
            out << ' ' << QString::number(saved->result.doiConservative >= 0.0
                                              ? saved->result.doiConservative : saved->doiDepth, 'g', 12)
                << ' ' << QString::number(saved->doiDepth, 'g', 12);
        else
            out << " nan nan";
        const double rms = saved->result.rmsHistory.empty()
            ? 0.0 : saved->result.rmsHistory.back();
        out << ' ' << QString::number(rms, 'g', 12) << '\n';
        ++exported;
    }
    file.close();
    settings.setValue("lastExportDirectory", QFileInfo(path).absolutePath());
    m_log->appendPlainText(QString("Exported %1 model(s) to %2")
        .arg(exported).arg(QFileInfo(path).fileName()));
}

void MainWindow::setRunning(bool running)
{
    m_importButton->setEnabled(!running);
    m_newProjectButton->setEnabled(!running);
    m_loadProjectButton->setEnabled(!running);
    m_saveProjectButton->setEnabled(!running);
    m_runButton->setEnabled(!running && !m_soundings.empty());
    m_moveKeptButton->setEnabled(!running && !m_soundings.empty());
    m_autoFilterButton->setEnabled(!running && !m_soundings.empty());
    m_killButton->setEnabled(running);
    const bool dlf = m_transform == pytem::TransformMethod::DigitalLinearFilter;
    const bool joint = m_pytemJoint;
    m_useGpu->setEnabled(!running && dlf && !joint
                         && pytem::TemSolver::gpuAvailable());
    // Controls that only one inversion method uses are hidden for the other.
    m_useGpu->setVisible(!joint);
    m_cacheJacobian->setVisible(!joint);
    for (QWidget *control : std::initializer_list<QWidget *>{m_regularizationNorm, m_adaptiveAlpha, m_cacheJacobian})
        control->setEnabled(!running);
    m_adaptiveAlpha->setEnabled(!running && m_regularizationNorm->currentData().toInt() < fastSciLin);
    m_soundingSelector->setEnabled(!m_soundings.empty());
    m_moment->setEnabled(!running && activeSounding());
    m_plotMode->setEnabled(!running);
    m_previousButton->setEnabled(m_activeSoundingIndex > 0);
    m_nextButton->setEnabled(m_activeSoundingIndex + 1 < static_cast<int>(m_soundings.size()));
    updateTransectNavigation();
    if (running) {
        m_previousTransectButton->setEnabled(false);
        m_nextTransectButton->setEnabled(false);
    }
    const auto *state = activeSounding();
    m_exportModelledButton->setEnabled(!running && state && state->haveResult);
    m_clearResultsButton->setEnabled(!running && std::any_of(
        m_soundings.begin(), m_soundings.end(),
        [](const SoundingState &candidate) {
            return !candidate.savedModels.empty();
        }));
}

void MainWindow::moveKeptUsfFiles()
{
    if (m_worker)
        return;
    std::vector<SoundingState *> kept;
    for (auto &state : m_soundings)
        if (state.includeInBatch)
            kept.push_back(&state);
    if (kept.empty()) {
        QMessageBox::information(this, "Move kept USF files", "No sounding is included in the batch.");
        return;
    }
    QSettings settings;
    const QString folder = QFileDialog::getExistingDirectory(this, "Move kept USF files to",
        settings.value("lastMoveDirectory", QFileInfo(kept.front()->path).absolutePath()).toString());
    if (folder.isEmpty())
        return;
    settings.setValue("lastMoveDirectory", folder);
    if (QMessageBox::question(this, "Move kept USF files",
            QString("Move %1 USF file(s) to\n%2?\n\n%3 excluded sounding(s) stay where they are.")
                .arg(kept.size()).arg(QDir::toNativeSeparators(folder))
                .arg(m_soundings.size() - kept.size())) != QMessageBox::Yes)
        return;
    QStringList failures;
    int moved = 0;
    for (auto *state : kept) {
        const QString target = QDir(folder).filePath(QFileInfo(state->path).fileName());
        if (QFileInfo(target) == QFileInfo(state->path))
            continue;
        QString problem;
        if (!QFileInfo::exists(state->path))
            problem = "source file not found";
        else if (QFileInfo::exists(target))
            problem = "a file with that name already exists in the folder";
        // rename fails across drives; fall back to copy and delete.
        else if (!QFile::rename(state->path, target)
                 && !(QFile::copy(state->path, target) && QFile::remove(state->path)))
            problem = "could not move the file";
        if (!problem.isEmpty()) {
            failures << QString("%1: %2").arg(QFileInfo(state->path).fileName(), problem);
            continue;
        }
        state->path = QFileInfo(target).absoluteFilePath();
        ++moved;
    }
    if (moved > 0)
        markProjectModified();
    m_log->appendPlainText(QString("Moved %1 kept USF file(s) to %2").arg(moved).arg(QDir::toNativeSeparators(folder)));
    if (!failures.isEmpty())
        QMessageBox::warning(this, "Some USF files were not moved", failures.join("\n"));
}
