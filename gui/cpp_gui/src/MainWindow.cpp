#include "MainWindow.h"
#include "InversionWorker.h"
#include "MapTileLoader.h"
#include "MatrixConvolution.h"
#include "PlotWidget.h"

#include <QApplication>
#include <QCheckBox>
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
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QProgressDialog>
#include <QPushButton>
#include <QRadioButton>
#include <QSettings>
#include <QScreen>
#include <QSizePolicy>
#include <QSpinBox>
#include <QStandardPaths>
#include <QStringList>
#include <QSplitter>
#include <QTextStream>
#include <QThread>
#include <QVBoxLayout>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <numeric>
#include <thread>

namespace {

constexpr int inversionIterations = 15;
constexpr int inversionAlphaTrials = 7;

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

} // namespace

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    qRegisterMetaType<pytem::InversionResult>("pytem::InversionResult");
    buildInterface();
    if (const auto *screen = QGuiApplication::primaryScreen()) {
        const QRect available = screen->availableGeometry();
        const int width = std::min(available.width(),
            std::min(1600, std::max(640, available.width() - 40)));
        const int height = std::min(available.height(),
            std::min(950, std::max(480, available.height() - 60)));
        setMaximumSize(available.size());
        resize(width, height);
        move(available.center() - rect().center());
    } else {
        resize(1400, 850);
    }
    setWindowTitle("pyTEM Native C++ Inversion v0.4.0");
}

MainWindow::~MainWindow()
{
    if (m_worker) m_worker->requestCancel();
    if (m_workerThread) {
        m_workerThread->quit();
        m_workerThread->wait();
    }
}

void MainWindow::buildInterface()
{
    auto *central = new QWidget;
    auto *root = new QVBoxLayout(central);
    root->setSizeConstraint(QLayout::SetNoConstraint);
    setCentralWidget(central);

    auto *informationSidebar = new QWidget;
    informationSidebar->setMinimumWidth(230);
    informationSidebar->setMaximumWidth(320);
    auto *informationLayout = new QVBoxLayout(informationSidebar);
    auto *openButton = new QPushButton("Open one or more .usf files...");
    m_geometryLabel = new QLabel("Geometry: no sounding loaded");
    m_geometryLabel->setWordWrap(true);
    m_geometryLabel->setMinimumWidth(0);
    m_geometryLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    m_systemLabel = new QLabel("Waveform, gate widths, and filters are read from the selected moment(s).");
    m_systemLabel->setWordWrap(true);
    m_systemLabel->setMinimumWidth(0);
    m_systemLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);

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
    informationLayout->addWidget(openButton);
    informationLayout->addLayout(soundingNavigation);
    informationLayout->addWidget(new QLabel("Moment"));
    informationLayout->addWidget(m_moment);

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
    informationLayout->addStretch();

    auto *workspaceSplitter = new QSplitter(Qt::Horizontal);
    auto *leftWorkspace = new QWidget;
    auto *leftWorkspaceLayout = new QVBoxLayout(leftWorkspace);
    leftWorkspaceLayout->setContentsMargins(0, 0, 0, 0);
    auto *plotSplitter = new QSplitter(Qt::Horizontal);
    m_soundingPlot = new PlotWidget;
    m_soundingPlot->setAxes("TEM data (right-click a point to enable/disable)",
                            "Time [s]", "|dB/dt| [V/Am²]", true, true);
    m_soundingPlot->setScientificX(true);
    m_modelPlot = new PlotWidget;
    m_modelPlot->setAxes("Layered-earth model", "Resistivity [Ohm m]",
                         "Depth [m]", true, false, true);
    m_modelPlot->setMinimumY(0.0);
    auto *mapPanel = new QWidget;
    mapPanel->setMinimumWidth(460);
    mapPanel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Expanding);
    auto *mapPanelLayout = new QVBoxLayout(mapPanel);
    mapPanelLayout->setContentsMargins(0, 0, 0, 0);
    auto *mapControls = new QHBoxLayout;
    m_noMap = new QRadioButton("No map");
    m_openStreetMap = new QRadioButton("OpenStreetMap");
    m_satelliteMap = new QRadioButton("Satellite map");
    m_noMap->setChecked(true);
    m_mapProgress = new QProgressBar;
    m_mapProgress->setRange(0, 25);
    m_mapProgress->setFormat("Map tiles %v/%m");
    m_mapProgress->setFixedWidth(120);
    QSizePolicy mapProgressPolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
    mapProgressPolicy.setRetainSizeWhenHidden(true);
    m_mapProgress->setSizePolicy(mapProgressPolicy);
    m_mapProgress->setVisible(false);
    mapControls->addWidget(m_noMap);
    mapControls->addWidget(m_openStreetMap);
    mapControls->addWidget(m_satelliteMap);
    mapControls->addWidget(m_mapProgress, 1);
    m_mapPlot = new PlotWidget;
    m_mapPlot->setAxes("Sounding map (left-click selects; right-click includes/excludes)",
                       "Easting", "Northing", false, false);
    m_mapPlot->setEqualAspect(true);
    m_mapPlot->setGeographicScaleBar(true);
    m_mapPlot->setLegendBackground(true);
    m_mapPlot->setTickLabelsVisible(false);
    mapPanelLayout->addWidget(m_mapPlot, 1);
    mapPanelLayout->addLayout(mapControls);
    plotSplitter->addWidget(informationSidebar);
    plotSplitter->addWidget(m_soundingPlot);
    plotSplitter->addWidget(m_modelPlot);
    plotSplitter->setStretchFactor(0, 0);
    plotSplitter->setStretchFactor(1, 1);
    plotSplitter->setStretchFactor(2, 1);
    plotSplitter->setSizes({250, 360, 360});
    leftWorkspaceLayout->addWidget(plotSplitter, 1);
    workspaceSplitter->addWidget(leftWorkspace);
    workspaceSplitter->addWidget(mapPanel);
    workspaceSplitter->setStretchFactor(0, 5);
    workspaceSplitter->setStretchFactor(1, 4);
    workspaceSplitter->setSizes({880, 720});

    auto *settingsGroup = new QGroupBox;
    auto *settings = new QVBoxLayout(settingsGroup);
    m_maxDepth = makeDoubleSpin(14.0, 100000.0, 100.0, 1, 10.0);
    m_maxDepth->setSuffix(" m");
    m_maxDepth->setToolTip("Depth to the base of layer N-1; layer N is the half-space.");
    m_layerCount = new QSpinBox;
    m_layerCount->setRange(3, 100);
    m_layerCount->setValue(15);

    m_errorFloor = makeDoubleSpin(0.01, 100.0, 5.0, 2, 0.5);
    m_errorFloor->setSuffix(" %");
    m_parallelJobs = new QSpinBox;
    m_parallelJobs->setRange(1, 128);
    m_parallelJobs->setValue(static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
    m_parallelJobs->setToolTip("Maximum soundings inverted concurrently; CPU cores are divided among active jobs.");
    m_transform = new QComboBox;
    m_transform->addItem("Digital linear filter (Key)",
                         static_cast<int>(pytem::TransformMethod::DigitalLinearFilter));
    m_transform->addItem("Euler inverse Laplace (order 11)",
                         static_cast<int>(pytem::TransformMethod::Euler));
    m_jacobian = new QComboBox;
    m_jacobian->addItem("Analytical adjoint",
                        static_cast<int>(pytem::JacobianMethod::Analytical));
    m_jacobian->addItem("Finite difference",
                        static_cast<int>(pytem::JacobianMethod::FiniteDifference));
    m_regularizationNorm = new QComboBox;
    m_regularizationNorm->addItem("L2 smooth",
        static_cast<int>(pytem::RegularizationNorm::L2Smooth));
    m_regularizationNorm->addItem("L1 blocky (IRLS)",
        static_cast<int>(pytem::RegularizationNorm::L1Blocky));
    m_cacheJacobian = new QCheckBox("Reuse/cache first Jacobian");
    m_cacheJacobian->setChecked(true);
    m_cacheJacobian->setToolTip("Shared in memory across parallel jobs and saved on disk for compatible future runs.");
    m_sensitivity = new QCheckBox("Calculate final sensitivity");
    m_sensitivity->setChecked(true);
    m_useWaveform = new QCheckBox("Model waveform");
    m_useWaveform->setChecked(true);
    m_useFilters = new QCheckBox("Model bandpass filters");
    m_useFilters->setChecked(true);
    std::string gpuReason;
    const bool gpuAvailable = pytem::TemSolver::gpuAvailable(&gpuReason);
    m_useGpu = new QCheckBox("Use NVIDIA GPU (CUDA)");
    m_useGpu->setChecked(gpuAvailable);
    m_useGpu->setEnabled(gpuAvailable);
    m_useGpu->setToolTip(gpuAvailable
        ? "Accelerates DLF forward responses and alpha trials on the GPU."
        : QString("CPU only: %1").arg(QString::fromStdString(gpuReason)));

    auto *columns = new QHBoxLayout;
    auto *layerColumn = new QGroupBox("Layer model");
    auto *layerForm = new QFormLayout(layerColumn);
    layerForm->addRow("Number of layers", m_layerCount);
    layerForm->addRow("Depth to base N-1", m_maxDepth);
    columns->addWidget(layerColumn, 1);

    auto *forwardColumn = new QGroupBox("Forward response");
    auto *forwardForm = new QFormLayout(forwardColumn);
    forwardForm->addRow("Transform", m_transform);
    forwardForm->addRow(m_useWaveform);
    forwardForm->addRow(m_useFilters);
    forwardForm->addRow(m_useGpu);
    columns->addWidget(forwardColumn, 1);

    auto *inversionColumn = new QGroupBox("Solver");
    auto *inversionForm = new QFormLayout(inversionColumn);
    inversionForm->addRow("Error floor", m_errorFloor);
    inversionForm->addRow("Jacobian", m_jacobian);
    inversionForm->addRow("Model norm", m_regularizationNorm);
    columns->addWidget(inversionColumn, 1);

    auto *batchColumn = new QGroupBox("Batch and output");
    auto *batchForm = new QFormLayout(batchColumn);
    batchForm->addRow("Parallel soundings", m_parallelJobs);
    batchForm->addRow(m_cacheJacobian);
    batchForm->addRow(m_sensitivity);
    columns->addWidget(batchColumn, 1);
    settings->addLayout(columns);

    m_runButton = new QPushButton("Run batch inversion");
    m_runButton->setDefault(true);
    m_runButton->setEnabled(false);
    m_killButton = new QPushButton("Kill inversion");
    m_killButton->setToolTip(
        "Stop active and queued inversions while keeping completed models.");
    m_killButton->setEnabled(false);
    m_exportButton = new QPushButton("Export current...");
    m_exportButton->setEnabled(false);
    m_progress = new QProgressBar;
    m_progress->setRange(0, inversionIterations);
    m_progress->setFormat("%p%");
    m_timeEstimate = new QLabel("Elapsed 00:00 — ETA estimating…");
    m_timeEstimate->setFixedWidth(190);
    m_status = new QLabel("Open a USF sounding to begin.");
    m_status->setMinimumWidth(0);
    m_status->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    auto *actions = new QHBoxLayout;
    actions->addWidget(m_runButton);
    actions->addWidget(m_killButton);
    actions->addWidget(m_exportButton);
    actions->addWidget(m_progress, 1);
    actions->addWidget(m_timeEstimate);
    actions->addWidget(m_status, 1);
    settings->addLayout(actions);
    m_log = new QPlainTextEdit;
    m_log->setReadOnly(true);
    m_log->setMaximumBlockCount(1000);
    m_log->setMaximumHeight(90);
    settings->addWidget(m_log);
    leftWorkspaceLayout->addWidget(settingsGroup);
    root->addWidget(workspaceSplitter, 1);

    connect(openButton, &QPushButton::clicked, this, &MainWindow::loadUsf);
    connect(m_soundingSelector, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &MainWindow::selectSounding);
    connect(m_previousButton, &QPushButton::clicked, this, &MainWindow::previousSounding);
    connect(m_nextButton, &QPushButton::clicked, this, &MainWindow::nextSounding);
    connect(m_moment, qOverload<int>(&QComboBox::currentIndexChanged), this, &MainWindow::selectMoment);
    connect(m_layerCount, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::setLayerCount);
    connect(m_transform, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this, gpuAvailable, gpuReason](int) {
                const bool dlf = static_cast<pytem::TransformMethod>(
                    m_transform->currentData().toInt())
                    == pytem::TransformMethod::DigitalLinearFilter;
                m_useGpu->setEnabled(gpuAvailable && dlf && !m_worker);
                if (!dlf)
                    m_useGpu->setChecked(false);
                m_useGpu->setToolTip(!gpuAvailable
                    ? QString("CPU only: %1").arg(QString::fromStdString(gpuReason))
                    : (dlf
                        ? QString("Accelerates DLF forward responses and alpha trials on the GPU.")
                        : QString("GPU acceleration is currently available for the DLF transform only.")));
            });
    connect(m_runButton, &QPushButton::clicked, this, &MainWindow::runInversion);
    connect(m_killButton, &QPushButton::clicked, this, &MainWindow::killInversion);
    connect(m_exportButton, &QPushButton::clicked, this, &MainWindow::exportResult);
    connect(m_soundingPlot, &PlotWidget::pointRightClicked,
            this, &MainWindow::toggleDataPoint);
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
                m_status->setText(failedTiles == 0
                    ? "Background map loaded"
                    : QString("Background map loaded; %1/25 tiles unavailable")
                          .arg(failedTiles));
            });
    connect(m_mapTileLoader, &MapTileLoader::failed, this,
            [this](const QString &message) {
                m_mapProgress->setVisible(false);
                m_mapPlot->clearBackgroundImage();
                QMessageBox::warning(this, "Background map", message);
            });
}

void MainWindow::loadUsf()
{
    QSettings settings;
    const QStringList paths = QFileDialog::getOpenFileNames(this, "Open TEMcompany USF soundings",
        settings.value("lastUsfDirectory").toString(), "Universal Sounding Format (*.usf *.USF);;All files (*)");
    if (paths.isEmpty()) return;
    std::vector<SoundingState> loaded;
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
            SoundingState state;
            state.sounding = pytem::UsfReader::read(path.toStdString());
            state.path = path;
            state.txRadius = state.sounding.equivalentCircularRadius();
            state.rxOffset = state.sounding.radialReceiverOffset();
            state.geometry = state.rxOffset > 0.0
                ? pytem::Geometry::CircleOffset : pytem::Geometry::CircleCentral;
            state.momentChoice = state.sounding.moments.size() > 1 ? -1 : 0;
            state.gateEnabled.resize(state.sounding.moments.size());
            for (std::size_t momentIndex = 0; momentIndex < state.sounding.moments.size(); ++momentIndex) {
                const auto &moment = state.sounding.moments[momentIndex];
                auto &enabled = state.gateEnabled[momentIndex];
                enabled.resize(moment.times.size(), false);
                for (std::size_t gate = 0; gate < moment.times.size(); ++gate) {
                    const double voltage = moment.voltages[gate];
                    const double error = moment.standardErrors[gate];
                    const bool qualityAccepted = gate >= moment.qualityAccepted.size()
                        || moment.qualityAccepted[gate];
                    enabled[gate] = qualityAccepted && voltage > 0.0
                        && (error <= 0.0 || voltage / error >= 3.0);
                }
            }
            loaded.push_back(std::move(state));
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
        QMessageBox::critical(this, "USF import failed", failures.join("\n"));
        return;
    }
    saveActiveSoundingEdits();
    m_soundings = std::move(loaded);
    m_activeSoundingIndex = -1;
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
    m_soundingSelector->setEnabled(true);
    settings.setValue("lastUsfDirectory", QFileInfo(paths.front()).absolutePath());
    selectSounding(0);
    m_soundingSelector->setCurrentIndex(0);
    m_runButton->setEnabled(true);
    m_exportButton->setEnabled(false);
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
    m_geometryLabel->setText(QString(
        "Geometry: transmitter loop %1 × %2 m; receiver coil location (%3, %4) m")
        .arg(state.sounding.loopX, 0, 'g', 6)
        .arg(state.sounding.loopY, 0, 'g', 6)
        .arg(state.sounding.coilX, 0, 'g', 6)
        .arg(state.sounding.coilY, 0, 'g', 6));
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
    m_previousButton->setEnabled(index > 0 && m_worker == nullptr);
    m_nextButton->setEnabled(index + 1 < static_cast<int>(m_soundings.size()) && m_worker == nullptr);
    m_exportButton->setEnabled(state.haveResult && m_worker == nullptr);
    if (state.haveResult)
        displayResult(state);
    else
        displayInput(state);
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
    m_status->setText(QString("Loaded %1 from %2")
        .arg(moments.size() > 1 ? "joint LM + HM dataset" : QString::fromStdString(moments.front()->name),
             QFileInfo(state->path).fileName()));
    if (state->haveResult)
        displayResult(*state);
    else
        updateInputPlot();
}

void MainWindow::setLayerCount(int count)
{
    const double minimumDepth = static_cast<double>(count - 1);
    m_maxDepth->setMinimum(minimumDepth);
    if (m_maxDepth->value() < minimumDepth)
        m_maxDepth->setValue(minimumDepth);
}

void MainWindow::toggleDataPoint(int tableRow)
{
    if (tableRow < 0
        || static_cast<std::size_t>(tableRow) >= m_rowMomentIndices.size()
        || m_rowGateIndices.size() != m_rowMomentIndices.size())
        return;
    auto *state = activeSounding();
    if (!state)
        return;
    const std::size_t row = static_cast<std::size_t>(tableRow);
    const std::size_t momentIndex = m_rowMomentIndices[row];
    const std::size_t gateIndex = m_rowGateIndices[row];
    if (momentIndex >= state->gateEnabled.size()
        || gateIndex >= state->gateEnabled[momentIndex].size())
        return;
    state->gateEnabled[momentIndex][gateIndex]
        = !state->gateEnabled[momentIndex][gateIndex];
    state->haveResult = false;
    m_exportButton->setEnabled(false);
    m_modelPlot->clear();
    updateInputPlot();
    updateSystemSummary();
}

void MainWindow::toggleSoundingData(int soundingIndex)
{
    if (m_worker || soundingIndex < 0
        || soundingIndex >= static_cast<int>(m_soundings.size()))
        return;
    auto &state = m_soundings[static_cast<std::size_t>(soundingIndex)];
    if (state.includeInBatch) {
        state.gateEnabledBeforeExclusion = state.gateEnabled;
        for (auto &momentGates : state.gateEnabled)
            std::fill(momentGates.begin(), momentGates.end(), false);
        state.includeInBatch = false;
    } else {
        if (state.gateEnabledBeforeExclusion.size() == state.gateEnabled.size())
            state.gateEnabled = std::move(state.gateEnabledBeforeExclusion);
        state.gateEnabledBeforeExclusion.clear();
        state.includeInBatch = true;
    }
    state.haveResult = false;
    updateMapPlot();
    if (soundingIndex != m_activeSoundingIndex) {
        m_soundingSelector->setCurrentIndex(soundingIndex);
    } else {
        m_exportButton->setEnabled(false);
        m_modelPlot->clear();
        updateInputPlot();
        updateSystemSummary();
    }
    m_status->setText(state.includeInBatch
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
        m_status->setText("Background map disabled");
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
        m_status->setText("No valid coordinates are available for a background map");
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
    const auto moments = selectedMoments(state);
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
    if (m_maxDepth->value() < m_layerCount->value() - 1) {
        error = "The depth to layer N-1 is too shallow for a 1 m first layer and non-decreasing thicknesses.";
        return false;
    }
    return true;
}

pytem::InversionOptions MainWindow::buildOptions(const SoundingState &state) const
{
    pytem::InversionOptions options;
    pytem::ForwardModel commonModel;
    commonModel.geometry = state.geometry;
    commonModel.transform = static_cast<pytem::TransformMethod>(m_transform->currentData().toInt());
    commonModel.useGpu = m_useGpu->isChecked();
    commonModel.txSize = state.txRadius;
    commonModel.rxX = state.rxOffset;
    commonModel.thicknesses = pytem::TemSolver::geometricThicknesses(
        m_layerCount->value(), m_maxDepth->value());
    commonModel.resistivities.assign(static_cast<std::size_t>(m_layerCount->value()), 100.0);

    const double errorFloor = m_errorFloor->value() / 100.0;
    std::vector<pytem::InversionDataSet> dataSets;
    for (const auto *moment : selectedMoments(state)) {
        pytem::InversionDataSet dataSet;
        dataSet.model = commonModel;
        std::vector<double> gateOpen;
        std::vector<double> gateClose;
        const std::size_t momentIndex = static_cast<std::size_t>(moment - state.sounding.moments.data());
        for (std::size_t gate = 0; gate < moment->times.size(); ++gate) {
            if (!state.gateEnabled[momentIndex][gate]) continue;
            const double value = std::abs(moment->voltages[gate]);
            const double sem = std::abs(moment->standardErrors[gate]);
            dataSet.model.times.push_back(moment->times[gate]);
            dataSet.observed.push_back(value);
            dataSet.noiseStd.push_back(std::max(sem, errorFloor * value));
            gateOpen.push_back(moment->gateOpen[gate]);
            gateClose.push_back(moment->gateClose[gate]);
        }
        if (m_useWaveform->isChecked()) {
            const auto convolution = pytem::MatrixConvolution::build(
                dataSet.model.times, gateOpen, gateClose,
                moment->waveformTimes, moment->waveformAmplitudes, 300, 81, 1.0);
            dataSet.model.stepTimes = convolution.stepTimes;
            dataSet.model.responseMatrix = convolution.matrix;
        }
        if (m_useFilters->isChecked()) {
            dataSet.model.lowPassFrequencies = moment->lowPassFrequencies;
            dataSet.model.lowPassOrders = moment->lowPassOrders;
            dataSet.model.highPassFrequencies = moment->highPassFrequencies;
            dataSet.model.highPassOrders = moment->highPassOrders;
        }
        dataSets.push_back(std::move(dataSet));
    }
    options.model = std::move(dataSets.front().model);
    options.observed = std::move(dataSets.front().observed);
    options.noiseStd = std::move(dataSets.front().noiseStd);
    for (std::size_t i = 1; i < dataSets.size(); ++i)
        options.additionalDataSets.push_back(std::move(dataSets[i]));
    options.maxIterations = inversionIterations;
    options.alphaSteps = inversionAlphaTrials;
    options.jacobianMethod = static_cast<pytem::JacobianMethod>(m_jacobian->currentData().toInt());
    options.regularizationNorm = static_cast<pytem::RegularizationNorm>(
        m_regularizationNorm->currentData().toInt());
    options.cacheFirstJacobian = m_cacheJacobian->isChecked();
    if (options.cacheFirstJacobian) {
        QString cacheRoot = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
        if (cacheRoot.isEmpty())
            cacheRoot = QDir::tempPath() + "/pytem-cache";
        options.jacobianCacheDirectory = (cacheRoot + "/jacobians").toStdString();
    }
    options.calculateSensitivity = m_sensitivity->isChecked();
    return options;
}

void MainWindow::runInversion()
{
    saveActiveSoundingEdits();
    if (m_soundings.empty()) return;
    QElapsedTimer preparationTimer;
    preparationTimer.start();
    QStringList errors;
    QStringList skippedForTooFewGates;
    std::vector<InversionJob> jobs;
    for (std::size_t i = 0; i < m_soundings.size(); ++i) {
        auto &state = m_soundings[i];
        if (!state.includeInBatch)
            continue;
        QStringList sparseMoments;
        for (const auto *moment : selectedMoments(state)) {
            const std::size_t momentIndex = static_cast<std::size_t>(
                moment - state.sounding.moments.data());
            const int enabledCount = static_cast<int>(std::count(
                state.gateEnabled[momentIndex].begin(),
                state.gateEnabled[momentIndex].end(), true));
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
            for (const auto *moment : selectedMoments(state)) {
                const std::size_t momentIndex = static_cast<std::size_t>(
                    moment - state.sounding.moments.data());
                state.resultMomentNames.push_back(moment->name);
                state.resultMomentIndices.push_back(momentIndex);
                std::vector<std::size_t> gates;
                for (std::size_t gate = 0; gate < moment->times.size(); ++gate) {
                    if (state.gateEnabled[momentIndex][gate])
                        gates.push_back(gate);
                }
                state.resultGateIndices.push_back(std::move(gates));
            }
            state.haveResult = false;
            state.error.clear();
            jobs.push_back({static_cast<int>(i), QFileInfo(state.path).fileName(), state.options});
        } catch (const std::exception &exception) {
            errors << QString("%1: %2").arg(QFileInfo(state.path).fileName(),
                                             QString::fromUtf8(exception.what()));
        }
    }
    if (!errors.isEmpty()) {
        QString message = errors.join("\n");
        if (!skippedForTooFewGates.isEmpty())
            message += QStringLiteral("\n\nAlso skipped for fewer than three enabled gates:\n")
                + skippedForTooFewGates.join("\n");
        QMessageBox::warning(this, "Cannot start batch inversion", message);
        return;
    }
    if (jobs.empty()) {
        if (!skippedForTooFewGates.isEmpty()) {
            QMessageBox::warning(this, "Soundings skipped",
                QStringLiteral("No sounding has at least three enabled gates for every selected moment.\n\n")
                + skippedForTooFewGates.join("\n"));
            return;
        }
        QMessageBox::warning(this, "Cannot start batch inversion",
            "Enable at least one sounding by clicking its point on the map.");
        return;
    }
    if (!skippedForTooFewGates.isEmpty()) {
        QMessageBox::warning(this, "Soundings skipped",
            QStringLiteral("The following soundings have fewer than three enabled gates and will be skipped:\n\n")
            + skippedForTooFewGates.join("\n"));
    }
    m_preparationSeconds = preparationTimer.elapsed() / 1000.0;
    const int firstJobIndex = jobs.front().index;
    const QDir reportDirectory(QFileInfo(
        m_soundings[static_cast<std::size_t>(firstJobIndex)].path).absolutePath());
    m_timingReportPath = reportDirectory.filePath(
        "pytem_inversion_timing_"
        + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".txt");
    m_log->clear();
    const auto &firstOptions = jobs.front().options;
    const QString transform = firstOptions.model.transform == pytem::TransformMethod::Euler
        ? "Euler-11" : "DLF";
    const QString jacobian = firstOptions.jacobianMethod == pytem::JacobianMethod::Analytical
        ? "analytical" : "finite difference";
    const QString norm = firstOptions.regularizationNorm == pytem::RegularizationNorm::L1Blocky
        ? "L1 blocky" : "L2 smooth";
    const QString backend = firstOptions.model.useGpu
        ? "CUDA forward" : "CPU forward";
    m_log->appendPlainText(QString("Starting %1 parallel sounding inversion(s): %2, %3 Jacobian, %4, %5")
        .arg(jobs.size()).arg(transform, jacobian, norm, backend));
    m_jobIterations.assign(m_soundings.size(), 0);
    m_completedJobs = 0;
    m_successfulJobs = 0;
    m_totalJobs = static_cast<int>(jobs.size());
    m_batchKilled = false;
    m_batchTimer.restart();
    m_lastEtaUpdateMs = -1;
    m_progress->setRange(0, static_cast<int>(jobs.size()) * firstOptions.maxIterations);
    m_progress->setValue(0);
    m_timeEstimate->setText("Elapsed 00:00 — ETA estimating…");
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
    m_status->setText("Stopping inversion — completed models will be kept...");
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
    updateBatchTiming();
    const QString name = QFileInfo(m_soundings[static_cast<std::size_t>(jobIndex)].path).fileName();
    if (!m_batchKilled)
        m_status->setText(QString("%1/%2 completed")
            .arg(m_completedJobs).arg(m_totalJobs));
    m_log->appendPlainText(QString("[%1] iteration %2  RMS=%3  %4")
        .arg(name).arg(iteration).arg(rms, 0, 'g', 6).arg(detail));
}

void MainWindow::inversionComplete(int jobIndex, const pytem::InversionResult &result)
{
    if (jobIndex < 0 || jobIndex >= static_cast<int>(m_soundings.size())) return;
    auto &state = m_soundings[static_cast<std::size_t>(jobIndex)];
    state.result = result;
    state.haveResult = true;
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
        .arg(result.timing.totalSeconds, 0, 'f', 3)
        .arg(result.timing.jacobianSeconds, 0, 'f', 3)
        .arg(result.timing.alphaForwardSeconds, 0, 'f', 3));
    if (jobIndex == m_activeSoundingIndex)
        displayResult(state);
    if (m_batchKilled)
        m_status->setText("Stopping inversion — completed models will be kept...");
}

void MainWindow::inversionFailed(int jobIndex, const QString &message)
{
    if (jobIndex < 0 || jobIndex >= static_cast<int>(m_soundings.size())) return;
    auto &state = m_soundings[static_cast<std::size_t>(jobIndex)];
    state.error = message;
    ++m_completedJobs;
    m_jobIterations[static_cast<std::size_t>(jobIndex)] = state.options.maxIterations;
    m_progress->setValue(std::accumulate(m_jobIterations.begin(), m_jobIterations.end(), 0));
    updateBatchTiming();
    m_log->appendPlainText(QString("[%1] stopped: %2")
        .arg(QFileInfo(state.path).fileName(), message));
}

void MainWindow::batchFinished()
{
    const bool wasKilled = m_batchKilled;
    const int successful = m_successfulJobs;
    finishWorker();
    writeTimingReport();
    if (const auto *state = activeSounding()) {
        m_exportButton->setEnabled(state->haveResult);
        if (state->haveResult) displayResult(*state);
    }
    m_status->setText(wasKilled
        ? QString("Inversion killed — %1 completed model(s) kept").arg(successful)
        : QString("Batch finished — %1/%2 soundings completed")
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
        const QString fallbackDirectory = fallbackRoot + "/pyTEM";
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
    out << "pyTEM native C++ inversion timing report\n";
    out << "Generated\t" << QDateTime::currentDateTime().toString(Qt::ISODate) << '\n';
    out << "Batch preparation seconds\t"
        << QString::number(m_preparationSeconds, 'f', 6) << '\n';
    out << "Batch wall seconds\t"
        << QString::number(m_batchTimer.isValid() ? m_batchTimer.elapsed() / 1000.0 : 0.0,
                           'f', 6) << '\n';
    out << "Parallel soundings\t" << m_parallelJobs->value() << '\n';
    out << "Forward backend requested\t"
        << (m_useGpu->isChecked() ? "NVIDIA CUDA" : "CPU") << '\n';
    out << "Layers\t" << m_layerCount->value() << '\n';
    out << "Fixed maximum iterations\t" << inversionIterations << '\n';
    out << "Fixed alpha trials\t" << inversionAlphaTrials << '\n';
    out << "Alpha log10 step\t" << QString::number(1.0 / 9.0, 'f', 9)
        << "\n\n";
    out << "sounding\tstatus\titerations\ttotal_s\tinitial_forward_s\t"
           "jacobian_s\tlinear_solve_s\talpha_forward_s\tsensitivity_s\t"
           "other_s\tjacobian_evaluations\talpha_trial_forwards\t"
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
            + timing.jacobianSeconds + timing.linearSolveSeconds
            + timing.alphaForwardSeconds + timing.sensitivitySeconds;
        const double otherSeconds = std::max(0.0,
            timing.totalSeconds - measuredStages);
        out << "completed\t" << state.result.iterations << '\t'
            << QString::number(timing.totalSeconds, 'f', 6) << '\t'
            << QString::number(timing.initialForwardSeconds, 'f', 6) << '\t'
            << QString::number(timing.jacobianSeconds, 'f', 6) << '\t'
            << QString::number(timing.linearSolveSeconds, 'f', 6) << '\t'
            << QString::number(timing.alphaForwardSeconds, 'f', 6) << '\t'
            << QString::number(timing.sensitivitySeconds, 'f', 6) << '\t'
            << QString::number(otherSeconds, 'f', 6) << '\t'
            << timing.jacobianEvaluations << '\t'
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
    if (!complete && m_lastEtaUpdateMs >= 0
        && elapsed - m_lastEtaUpdateMs < etaRefreshIntervalMs)
        return;
    m_lastEtaUpdateMs = elapsed;
    if (completedUnits <= 0 || totalUnits <= 0) {
        m_timeEstimate->setText(QString("Elapsed %1 — ETA estimating…")
            .arg(formatDuration(elapsed)));
        return;
    }
    const qint64 remaining = static_cast<qint64>(
        static_cast<double>(elapsed) * (totalUnits - completedUnits)
        / completedUnits);
    m_timeEstimate->setText(QString("Elapsed %1 — ETA %2")
        .arg(formatDuration(elapsed), formatDuration(remaining)));
}

void MainWindow::finishWorker()
{
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
    QVector<PlotWidget::Curve> curves;
    std::size_t predictionOffset = 0;
    auto addDataSet = [&](const pytem::ForwardModel &model,
                          const std::vector<double> &observed,
                          const QString &name, std::size_t dataSetIndex) {
        QVector<QPointF> observedPoints, predictedPoints;
        QVector<int> pointRows;
        QVector<double> observedErrors;
        for (std::size_t i = 0; i < model.times.size(); ++i) {
            observedPoints.push_back({model.times[i], observed[i]});
            predictedPoints.push_back({model.times[i], std::abs(result.predicted[predictionOffset + i])});
            int matchingRow = -1;
            double standardError = 0.0;
            if (dataSetIndex < state.resultMomentIndices.size()
                && dataSetIndex < state.resultGateIndices.size()
                && i < state.resultGateIndices[dataSetIndex].size()) {
                const std::size_t momentIndex = state.resultMomentIndices[dataSetIndex];
                const std::size_t gateIndex = state.resultGateIndices[dataSetIndex][i];
                if (momentIndex < state.sounding.moments.size()
                    && gateIndex < state.sounding.moments[momentIndex].standardErrors.size())
                    standardError = std::abs(
                        state.sounding.moments[momentIndex].standardErrors[gateIndex]);
                for (std::size_t row = 0; row < m_rowMomentIndices.size(); ++row) {
                    if (m_rowMomentIndices[row] == momentIndex
                        && m_rowGateIndices[row] == gateIndex) {
                        matchingRow = static_cast<int>(row);
                        break;
                    }
                }
            }
            pointRows.push_back(matchingRow);
            observedErrors.push_back(standardError);
        }
        const QColor color = momentColor(name.toStdString(), dataSetIndex);
        curves.push_back({name + " data", observedPoints, color, true, false,
                          true, pointRows, false, observedErrors});
        curves.push_back({{}, predictedPoints, color, false, false});
        predictionOffset += model.times.size();
    };
    addDataSet(options.model, options.observed,
               state.resultMomentNames.empty() ? "Moment"
                   : QString::fromStdString(state.resultMomentNames.front()), 0);
    for (std::size_t i = 0; i < options.additionalDataSets.size(); ++i) {
        const auto &dataSet = options.additionalDataSets[i];
        addDataSet(dataSet.model, dataSet.observed,
                   i + 1 < state.resultMomentNames.size()
                       ? QString::fromStdString(state.resultMomentNames[i + 1])
                                          : QString("Moment %1").arg(i + 2),
                   i + 1);
    }
    m_soundingPlot->setCurves(curves);
    double bottom = 0.0;
    for (double h : options.model.thicknesses) bottom += h;
    bottom += std::max(10.0, bottom * 0.35);
    auto curve = [&](const std::vector<double> &rho) {
        QVector<QPointF> points;
        double top = 0.0;
        for (std::size_t i = 0; i < rho.size(); ++i) {
            const double base = i < options.model.thicknesses.size() ? top + options.model.thicknesses[i] : bottom;
            points.push_back({rho[i], top}); points.push_back({rho[i], base});
            if (i + 1 < rho.size()) points.push_back({rho[i + 1], base});
            top = base;
        }
        return points;
    };
    // The model plot was configured with logX=true; resistivity is always logarithmic.
    m_modelPlot->setCurves({{"", curve(result.resistivities),
                             QColor("#1b75bc"), false, false}});
    const double finalRms = result.rmsHistory.empty() ? 0.0 : result.rmsHistory.back();
    m_status->setText(QString("Finished — RMS %1; %2").arg(finalRms, 0, 'g', 4).arg(QString::fromStdString(result.message)));
}

void MainWindow::displayInput(const SoundingState &)
{
    updateInputPlot();
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
    int selectedCount = 0;
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
            if (state->gateEnabled[momentIndex][gate]) {
                used.push_back({time, value});
                usedRows.push_back(static_cast<int>(row));
                usedErrors.push_back(standardError);
                ++selectedCount;
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
    m_soundingPlot->setCurves(curves);
    m_status->setText(QString("%1/%2 visible gates enabled — right-click a data point to toggle")
        .arg(selectedCount).arg(m_rowMomentIndices.size()));
}

void MainWindow::updateMapPlot()
{
    QVector<QPointF> includedPoints;
    QVector<int> includedIds;
    QVector<QPointF> excludedPoints;
    QVector<int> excludedIds;
    QVector<QPointF> activePoint;
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
        if (m_soundings[i].includeInBatch) {
            includedPoints.push_back(point);
            includedIds.push_back(static_cast<int>(i));
        } else {
            excludedPoints.push_back(point);
            excludedIds.push_back(static_cast<int>(i));
        }
        if (static_cast<int>(i) == m_activeSoundingIndex)
            activePoint.push_back(point);
    }
    QVector<PlotWidget::Curve> curves;
    if (!includedPoints.isEmpty())
        curves.push_back({"Included soundings", includedPoints, QColor("#4c78a8"),
                          true, false, true, includedIds, false});
    if (!excludedPoints.isEmpty())
        curves.push_back({"Excluded soundings", excludedPoints, QColor("#b8b8b8"),
                          true, false, true, excludedIds, false});
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
        stageSummaries << QString("%1: %2 waveform points, %3")
            .arg(QString::fromStdString(moment->name))
            .arg(moment->waveformTimes.size())
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

void MainWindow::exportResult()
{
    const auto *state = activeSounding();
    if (!state || !state->haveResult) return;
    const auto &options = state->options;
    const auto &result = state->result;
    QSettings settings;
    const QString defaultName = QFileInfo(state->path).completeBaseName() + "_pytem_cpp_result.json";
    QString path = QFileDialog::getSaveFileName(this, "Export native inversion",
        settings.value("lastExportDirectory").toString() + "/" + defaultName,
        "JSON result (*.json);;Layer model CSV (*.csv)");
    if (path.isEmpty()) return;
    if (!path.endsWith(".json", Qt::CaseInsensitive) && !path.endsWith(".csv", Qt::CaseInsensitive)) path += ".json";
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) { QMessageBox::critical(this, "Export failed", file.errorString()); return; }
    if (path.endsWith(".json", Qt::CaseInsensitive)) {
        QJsonObject object;
        object["source_usf"] = state->path;
        object["moment"] = state->resultMomentNames.size() > 1 ? "LM + HM joint"
            : QString::fromStdString(state->resultMomentNames.front());
        object["first_jacobian_source"] = QString::fromStdString(result.firstJacobianSource);
        object["regularization_norm"] = options.regularizationNorm == pytem::RegularizationNorm::L1Blocky
            ? "L1" : "L2";
        std::vector<double> allTimes = options.model.times;
        std::vector<double> allObserved = options.observed;
        for (const auto &dataSet : options.additionalDataSets) {
            allTimes.insert(allTimes.end(), dataSet.model.times.begin(), dataSet.model.times.end());
            allObserved.insert(allObserved.end(), dataSet.observed.begin(), dataSet.observed.end());
        }
        object["times"] = jsonArray(allTimes);
        object["observed"] = jsonArray(allObserved);
        object["predicted"] = jsonArray(result.predicted);
        QJsonArray dataSetsJson;
        std::size_t predictionOffset = 0;
        auto appendDataSet = [&](const pytem::ForwardModel &model,
                                 const std::vector<double> &observed,
                                 const QString &name) {
            QJsonObject dataSetObject;
            dataSetObject["name"] = name;
            dataSetObject["times"] = jsonArray(model.times);
            dataSetObject["observed"] = jsonArray(observed);
            std::vector<double> prediction(
                result.predicted.begin() + static_cast<std::ptrdiff_t>(predictionOffset),
                result.predicted.begin() + static_cast<std::ptrdiff_t>(predictionOffset + model.times.size()));
            dataSetObject["predicted"] = jsonArray(prediction);
            predictionOffset += model.times.size();
            dataSetsJson.append(dataSetObject);
        };
        appendDataSet(options.model, options.observed,
                      state->resultMomentNames.empty() ? "Moment"
                          : QString::fromStdString(state->resultMomentNames.front()));
        for (std::size_t i = 0; i < options.additionalDataSets.size(); ++i) {
            const auto &dataSet = options.additionalDataSets[i];
            appendDataSet(dataSet.model, dataSet.observed,
                          i + 1 < state->resultMomentNames.size()
                              ? QString::fromStdString(state->resultMomentNames[i + 1])
                                                 : QString("Moment %1").arg(i + 2));
        }
        object["datasets"] = dataSetsJson;
        object["thicknesses"] = jsonArray(options.model.thicknesses);
        object["resistivities"] = jsonArray(result.resistivities);
        object["sensitivity"] = jsonArray(result.sensitivity);
        object["rms_history"] = jsonArray(result.rmsHistory);
        file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    } else {
        QTextStream out(&file);
        out << "layer,top_depth_m,bottom_depth_m,resistivity_ohm_m,sensitivity\n";
        double top = 0.0;
        for (std::size_t i = 0; i < result.resistivities.size(); ++i) {
            out << i + 1 << ',' << QString::number(top, 'g', 12) << ',';
            if (i < options.model.thicknesses.size()) { top += options.model.thicknesses[i]; out << QString::number(top, 'g', 12); }
            else out << "inf";
            out << ',' << QString::number(result.resistivities[i], 'g', 12) << ',';
            if (i < result.sensitivity.size()) out << QString::number(result.sensitivity[i], 'g', 12);
            out << '\n';
        }
    }
    settings.setValue("lastExportDirectory", QFileInfo(path).absolutePath());
}

void MainWindow::setRunning(bool running)
{
    m_runButton->setEnabled(!running && !m_soundings.empty());
    m_killButton->setEnabled(running);
    const bool dlf = static_cast<pytem::TransformMethod>(
        m_transform->currentData().toInt())
        == pytem::TransformMethod::DigitalLinearFilter;
    m_useGpu->setEnabled(!running && dlf
                         && pytem::TemSolver::gpuAvailable());
    m_soundingSelector->setEnabled(!running && !m_soundings.empty());
    m_moment->setEnabled(!running && activeSounding());
    m_previousButton->setEnabled(!running && m_activeSoundingIndex > 0);
    m_nextButton->setEnabled(!running && m_activeSoundingIndex + 1 < static_cast<int>(m_soundings.size()));
    const auto *state = activeSounding();
    m_exportButton->setEnabled(!running && state && state->haveResult);
}
