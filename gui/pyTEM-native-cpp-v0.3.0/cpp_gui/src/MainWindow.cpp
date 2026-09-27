#include "MainWindow.h"
#include "InversionWorker.h"
#include "MatrixConvolution.h"
#include "PlotWidget.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSettings>
#include <QSpinBox>
#include <QStandardPaths>
#include <QStringList>
#include <QSplitter>
#include <QTableWidget>
#include <QTextStream>
#include <QThread>
#include <QVBoxLayout>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <thread>

namespace {

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

QTableWidgetItem *numberItem(double value)
{
    return new QTableWidgetItem(QString::number(value, 'g', 11));
}

QJsonArray jsonArray(const std::vector<double> &values)
{
    QJsonArray array;
    for (double value : values) array.append(value);
    return array;
}

} // namespace

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    qRegisterMetaType<pytem::InversionResult>("pytem::InversionResult");
    buildInterface();
    resize(1280, 850);
    setWindowTitle("pyTEM Native C++ Circular-Loop Inversion");
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
    auto *root = new QHBoxLayout(central);
    auto *splitter = new QSplitter(Qt::Horizontal);
    root->addWidget(splitter);
    setCentralWidget(central);

    auto *controls = new QWidget;
    auto *controlsLayout = new QVBoxLayout(controls);
    auto *dataGroup = new QGroupBox("TEMcompany USF soundings");
    auto *dataLayout = new QVBoxLayout(dataGroup);
    auto *openButton = new QPushButton("Open one or more .usf files...");
    m_fileLabel = new QLabel("No sounding loaded");
    m_fileLabel->setWordWrap(true);
    auto *soundingRow = new QHBoxLayout;
    m_previousButton = new QPushButton("◀");
    m_previousButton->setToolTip("Previous sounding and recovered model");
    m_previousButton->setEnabled(false);
    m_soundingSelector = new QComboBox;
    m_soundingSelector->setEnabled(false);
    m_nextButton = new QPushButton("▶");
    m_nextButton->setToolTip("Next sounding and recovered model");
    m_nextButton->setEnabled(false);
    soundingRow->addWidget(m_previousButton);
    soundingRow->addWidget(m_soundingSelector, 1);
    soundingRow->addWidget(m_nextButton);
    m_moment = new QComboBox;
    m_moment->setEnabled(false);
    m_systemLabel = new QLabel("Waveform, gate widths, and filters are read from the selected moment(s).");
    m_systemLabel->setWordWrap(true);
    m_dataTable = new QTableWidget(0, 4);
    m_dataTable->setHorizontalHeaderLabels({"Use", "Time [s]", "Stacked V/Am²", "SEM [V/Am²]"});
    m_dataTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_dataTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_dataTable->setMinimumHeight(190);
    dataLayout->addWidget(openButton);
    dataLayout->addWidget(m_fileLabel);
    dataLayout->addLayout(soundingRow);
    dataLayout->addWidget(m_moment);
    dataLayout->addWidget(m_systemLabel);
    dataLayout->addWidget(m_dataTable);
    controlsLayout->addWidget(dataGroup);

    auto *surveyGroup = new QGroupBox("Equivalent circular system");
    auto *surveyForm = new QFormLayout(surveyGroup);
    m_geometry = new QComboBox;
    m_geometry->addItem("Central receiver", static_cast<int>(pytem::Geometry::CircleCentral));
    m_geometry->addItem("Radial offset receiver", static_cast<int>(pytem::Geometry::CircleOffset));
    m_txRadius = makeDoubleSpin(0.01, 100000.0, 1.0, 3, 0.1);
    m_txRadius->setSuffix(" m");
    m_rxOffset = makeDoubleSpin(0.0, 100000.0, 0.0, 2, 0.5);
    m_rxOffset->setSuffix(" m");
    m_useWaveform = new QCheckBox("Use waveform + finite-gate matrix");
    m_useWaveform->setChecked(true);
    m_useFilters = new QCheckBox("Apply USF band-pass stages");
    m_useFilters->setChecked(true);
    surveyForm->addRow("Receiver geometry", m_geometry);
    surveyForm->addRow("Equivalent loop radius", m_txRadius);
    surveyForm->addRow("Radial receiver offset", m_rxOffset);
    surveyForm->addRow(m_useWaveform);
    surveyForm->addRow(m_useFilters);
    controlsLayout->addWidget(surveyGroup);

    auto *modelGroup = new QGroupBox("Fixed-thickness starting model");
    auto *modelLayout = new QVBoxLayout(modelGroup);
    constexpr int defaultLayers = 15;
    auto *layerCountRow = new QHBoxLayout;
    layerCountRow->addWidget(new QLabel("Number of layers"));
    m_layerCount = new QSpinBox;
    m_layerCount->setRange(2, 100);
    m_layerCount->setValue(defaultLayers);
    layerCountRow->addWidget(m_layerCount, 1);
    m_layerTable = new QTableWidget(defaultLayers, 2);
    m_layerTable->setHorizontalHeaderLabels({"Thickness [m]", "Resistivity [Ohm m]"});
    m_layerTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    const double logMinimum = std::log10(2.0);
    const double logMaximum = std::log10(25.0);
    for (int row = 0; row < defaultLayers; ++row) {
        const double fraction = static_cast<double>(row) / static_cast<double>(defaultLayers - 2);
        const double thickness = std::pow(10.0, logMinimum + fraction * (logMaximum - logMinimum));
        m_layerTable->setItem(row, 0, row < defaultLayers - 1
            ? numberItem(thickness) : new QTableWidgetItem(""));
        m_layerTable->setItem(row, 1, numberItem(50.0));
    }
    auto *layerButtons = new QHBoxLayout;
    auto *addButton = new QPushButton("Add layer");
    auto *removeButton = new QPushButton("Remove layer");
    layerButtons->addWidget(addButton);
    layerButtons->addWidget(removeButton);
    modelLayout->addLayout(layerCountRow);
    modelLayout->addWidget(m_layerTable);
    modelLayout->addLayout(layerButtons);
    controlsLayout->addWidget(modelGroup);

    auto *optionsGroup = new QGroupBox("Inversion options");
    auto *optionsForm = new QFormLayout(optionsGroup);
    m_errorFloor = makeDoubleSpin(0.01, 100.0, 5.0, 2, 0.5);
    m_errorFloor->setSuffix(" %");
    m_rhoMin = makeDoubleSpin(0.001, 1e9, 0.1, 3, 1.0);
    m_rhoMax = makeDoubleSpin(0.01, 1e9, 100000.0, 1, 100.0);
    m_maxIterations = new QSpinBox;
    m_maxIterations->setRange(1, 100);
    m_maxIterations->setValue(15);
    m_alphaSteps = new QSpinBox;
    m_alphaSteps->setRange(2, 30);
    m_alphaSteps->setValue(7);
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
    optionsForm->addRow("Relative error floor", m_errorFloor);
    optionsForm->addRow("Time transform", m_transform);
    optionsForm->addRow("Jacobian", m_jacobian);
    optionsForm->addRow("Model norm", m_regularizationNorm);
    optionsForm->addRow("Minimum resistivity", m_rhoMin);
    optionsForm->addRow("Maximum resistivity", m_rhoMax);
    optionsForm->addRow("Maximum iterations", m_maxIterations);
    optionsForm->addRow("Alpha trials / iteration", m_alphaSteps);
    optionsForm->addRow("Parallel soundings", m_parallelJobs);
    optionsForm->addRow(m_cacheJacobian);
    optionsForm->addRow(m_sensitivity);
    controlsLayout->addWidget(optionsGroup);

    auto *runRow = new QHBoxLayout;
    m_runButton = new QPushButton("Run batch inversion");
    m_runButton->setDefault(true);
    m_runButton->setEnabled(false);
    m_cancelButton = new QPushButton("Cancel");
    m_cancelButton->setEnabled(false);
    m_exportButton = new QPushButton("Export current...");
    m_exportButton->setEnabled(false);
    runRow->addWidget(m_runButton);
    runRow->addWidget(m_cancelButton);
    runRow->addWidget(m_exportButton);
    m_progress = new QProgressBar;
    m_progress->setRange(0, m_maxIterations->value());
    m_status = new QLabel("Open a USF sounding to begin.");
    controlsLayout->addLayout(runRow);
    controlsLayout->addWidget(m_progress);
    controlsLayout->addWidget(m_status);
    controlsLayout->addStretch();

    auto *scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setWidget(controls);
    scroll->setMinimumWidth(465);
    splitter->addWidget(scroll);

    auto *results = new QWidget;
    auto *resultsLayout = new QVBoxLayout(results);
    m_soundingPlot = new PlotWidget;
    m_soundingPlot->setAxes("Stacked sounding", "Time [s]", "|dB/dt| [V/Am²]", true, true);
    m_modelPlot = new PlotWidget;
    m_modelPlot->setAxes("Layered-earth model", "Resistivity [Ohm m]", "Depth [m]", true, false, true);
    m_log = new QPlainTextEdit;
    m_log->setReadOnly(true);
    m_log->setMaximumBlockCount(1000);
    m_log->setMinimumHeight(125);
    resultsLayout->addWidget(m_soundingPlot, 2);
    resultsLayout->addWidget(m_modelPlot, 2);
    resultsLayout->addWidget(m_log, 1);
    splitter->addWidget(results);
    splitter->setStretchFactor(1, 1);

    connect(openButton, &QPushButton::clicked, this, &MainWindow::loadUsf);
    connect(m_soundingSelector, qOverload<int>(&QComboBox::currentIndexChanged),
            this, &MainWindow::selectSounding);
    connect(m_previousButton, &QPushButton::clicked, this, &MainWindow::previousSounding);
    connect(m_nextButton, &QPushButton::clicked, this, &MainWindow::nextSounding);
    connect(m_moment, qOverload<int>(&QComboBox::currentIndexChanged), this, &MainWindow::selectMoment);
    connect(addButton, &QPushButton::clicked, this, &MainWindow::addLayer);
    connect(removeButton, &QPushButton::clicked, this, &MainWindow::removeLayer);
    connect(m_layerCount, qOverload<int>(&QSpinBox::valueChanged), this, &MainWindow::setLayerCount);
    connect(m_runButton, &QPushButton::clicked, this, &MainWindow::runInversion);
    connect(m_cancelButton, &QPushButton::clicked, this, &MainWindow::cancelInversion);
    connect(m_exportButton, &QPushButton::clicked, this, &MainWindow::exportResult);
    connect(m_geometry, qOverload<int>(&QComboBox::currentIndexChanged), this, &MainWindow::updateGeometryControls);
    connect(m_dataTable, &QTableWidget::cellChanged, this, &MainWindow::updateInputPlot);
}

void MainWindow::loadUsf()
{
    QSettings settings;
    const QStringList paths = QFileDialog::getOpenFileNames(this, "Open TEMcompany USF soundings",
        settings.value("lastUsfDirectory").toString(), "Universal Sounding Format (*.usf *.USF);;All files (*)");
    if (paths.isEmpty()) return;
    std::vector<SoundingState> loaded;
    QStringList failures;
    for (const QString &path : paths) {
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
                    enabled[gate] = voltage > 0.0
                        && (error <= 0.0 || voltage / error >= 3.0);
                }
            }
            loaded.push_back(std::move(state));
        } catch (const std::exception &error) {
            failures << QString("%1: %2").arg(QFileInfo(path).fileName(),
                                               QString::fromUtf8(error.what()));
        }
    }
    if (loaded.empty()) {
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
    state->geometry = static_cast<pytem::Geometry>(m_geometry->currentData().toInt());
    state->txRadius = m_txRadius->value();
    state->rxOffset = m_rxOffset->value();
    if (m_rowMomentIndices.size() != static_cast<std::size_t>(m_dataTable->rowCount())
        || m_rowGateIndices.size() != m_rowMomentIndices.size())
        return;
    for (int row = 0; row < m_dataTable->rowCount(); ++row) {
        const std::size_t momentIndex = m_rowMomentIndices[static_cast<std::size_t>(row)];
        const std::size_t gateIndex = m_rowGateIndices[static_cast<std::size_t>(row)];
        if (momentIndex >= state->sounding.moments.size()
            || gateIndex >= state->sounding.moments[momentIndex].times.size())
            continue;
        state->gateEnabled[momentIndex][gateIndex]
            = m_dataTable->item(row, 0)->checkState() == Qt::Checked;
        bool timeOk = false, voltageOk = false, errorOk = false;
        const double time = m_dataTable->item(row, 1)->text().toDouble(&timeOk);
        const double voltage = m_dataTable->item(row, 2)->text().toDouble(&voltageOk);
        const double error = m_dataTable->item(row, 3)->text().toDouble(&errorOk);
        auto &moment = state->sounding.moments[momentIndex];
        if (timeOk) moment.times[gateIndex] = time;
        if (voltageOk) moment.voltages[gateIndex] = voltage;
        if (errorOk) moment.standardErrors[gateIndex] = error;
    }
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
    m_dataTable->setRowCount(0);
    m_fileLabel->setText(QString("%1 — %2\n%3, %4; elevation %5 m")
        .arg(QString::fromStdString(state.sounding.soundingName), QFileInfo(state.path).fileName())
        .arg(state.sounding.longitude, 0, 'f', 6).arg(state.sounding.latitude, 0, 'f', 6)
        .arg(state.sounding.elevation, 0, 'f', 2));
    m_txRadius->setValue(state.txRadius);
    m_rxOffset->setValue(state.rxOffset);
    const int geometryIndex = m_geometry->findData(static_cast<int>(state.geometry));
    if (geometryIndex >= 0) m_geometry->setCurrentIndex(geometryIndex);
    m_moment->blockSignals(true);
    m_moment->clear();
    if (state.sounding.moments.size() > 1)
        m_moment->addItem("LM + HM — joint inversion", -1);
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
    std::size_t totalRows = 0;
    for (const auto *moment : moments)
        totalRows += moment->times.size();
    m_dataTable->blockSignals(true);
    m_dataTable->setRowCount(static_cast<int>(totalRows));
    int selected = 0;
    int tableRow = 0;
    QStringList stageSummaries;
    for (const auto *moment : moments) {
        const std::size_t momentIndex = static_cast<std::size_t>(moment - state->sounding.moments.data());
        QStringList stages;
        for (std::size_t i = 0; i < moment->lowPassFrequencies.size(); ++i)
            stages << QString("LP %1 kHz (order %2)").arg(moment->lowPassFrequencies[i] / 1000.0).arg(moment->lowPassOrders[i]);
        for (std::size_t i = 0; i < moment->highPassFrequencies.size(); ++i)
            stages << QString("HP %1 Hz (order %2)").arg(moment->highPassFrequencies[i]).arg(moment->highPassOrders[i]);
        stageSummaries << QString("%1: %2 waveform points, %3")
            .arg(QString::fromStdString(moment->name)).arg(moment->waveformTimes.size())
            .arg(stages.isEmpty() ? "no filters" : stages.join(" × "));
        for (std::size_t gate = 0; gate < moment->times.size(); ++gate, ++tableRow) {
            auto *useItem = new QTableWidgetItem;
            useItem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsUserCheckable | Qt::ItemIsSelectable);
            const double voltage = moment->voltages[gate];
            const double error = moment->standardErrors[gate];
            const bool enabled = state->gateEnabled[momentIndex][gate];
            useItem->setCheckState(enabled ? Qt::Checked : Qt::Unchecked);
            selected += enabled ? 1 : 0;
            m_dataTable->setItem(tableRow, 0, useItem);
            m_dataTable->setItem(tableRow, 1, numberItem(moment->times[gate]));
            m_dataTable->setItem(tableRow, 2, numberItem(voltage));
            m_dataTable->setItem(tableRow, 3, numberItem(error));
            m_dataTable->setVerticalHeaderItem(tableRow,
                new QTableWidgetItem(QString("%1 %2").arg(QString::fromStdString(moment->name)).arg(gate + 1)));
            m_rowMomentIndices.push_back(momentIndex);
            m_rowGateIndices.push_back(gate);
        }
    }
    m_dataTable->blockSignals(false);
    m_systemLabel->setText(QString("%1; %2/%3 gates selected")
        .arg(stageSummaries.join(" | ")).arg(selected).arg(totalRows));
    m_status->setText(QString("Loaded %1 from %2")
        .arg(moments.size() > 1 ? "joint LM + HM dataset" : QString::fromStdString(moments.front()->name),
             QFileInfo(state->path).fileName()));
    if (state->haveResult)
        displayResult(*state);
    else
        updateInputPlot();
}

void MainWindow::addLayer()
{
    m_layerCount->setValue(m_layerCount->value() + 1);
}

void MainWindow::removeLayer()
{
    m_layerCount->setValue(m_layerCount->value() - 1);
}

void MainWindow::setLayerCount(int count)
{
    m_layerTable->blockSignals(true);
    m_layerTable->setRowCount(count);
    const double logMinimum = std::log10(2.0);
    const double logMaximum = std::log10(25.0);
    for (int row = 0; row < count; ++row) {
        if (row < count - 1) {
            const double fraction = count == 2 ? 0.0
                : static_cast<double>(row) / static_cast<double>(count - 2);
            const double thickness = std::pow(
                10.0, logMinimum + fraction * (logMaximum - logMinimum));
            if (!m_layerTable->item(row, 0))
                m_layerTable->setItem(row, 0, numberItem(thickness));
            else
                m_layerTable->item(row, 0)->setText(QString::number(thickness, 'g', 11));
        } else {
            if (!m_layerTable->item(row, 0))
                m_layerTable->setItem(row, 0, new QTableWidgetItem(""));
            else
                m_layerTable->item(row, 0)->setText("");
        }
        if (!m_layerTable->item(row, 1))
            m_layerTable->setItem(row, 1, numberItem(50.0));
    }
    m_layerTable->blockSignals(false);
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
    for (int row = 0; row < m_layerTable->rowCount(); ++row) {
        bool okR = false;
        if (!m_layerTable->item(row, 1) || m_layerTable->item(row, 1)->text().toDouble(&okR) <= 0.0 || !okR) {
            error = QString("Invalid resistivity in model row %1.").arg(row + 1); return false;
        }
        if (row + 1 < m_layerTable->rowCount()) {
            bool okH = false;
            if (!m_layerTable->item(row, 0) || m_layerTable->item(row, 0)->text().toDouble(&okH) <= 0.0 || !okH) {
                error = QString("Invalid thickness in model row %1.").arg(row + 1); return false;
            }
        }
    }
    if (m_rhoMin->value() >= m_rhoMax->value()) { error = "Minimum resistivity must be below maximum resistivity."; return false; }
    return true;
}

pytem::InversionOptions MainWindow::buildOptions(const SoundingState &state) const
{
    pytem::InversionOptions options;
    pytem::ForwardModel commonModel;
    commonModel.geometry = state.geometry;
    commonModel.transform = static_cast<pytem::TransformMethod>(m_transform->currentData().toInt());
    commonModel.txSize = state.txRadius;
    commonModel.rxX = state.rxOffset;
    for (int row = 0; row < m_layerTable->rowCount(); ++row) {
        if (row + 1 < m_layerTable->rowCount())
            commonModel.thicknesses.push_back(m_layerTable->item(row, 0)->text().toDouble());
        commonModel.resistivities.push_back(m_layerTable->item(row, 1)->text().toDouble());
    }

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
    options.rhoMin = m_rhoMin->value();
    options.rhoMax = m_rhoMax->value();
    options.maxIterations = m_maxIterations->value();
    options.alphaSteps = m_alphaSteps->value();
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
    QStringList errors;
    std::vector<InversionJob> jobs;
    for (std::size_t i = 0; i < m_soundings.size(); ++i) {
        auto &state = m_soundings[i];
        QString error;
        if (!validateInputs(state, error)) {
            errors << QString("%1: %2").arg(QFileInfo(state.path).fileName(), error);
            continue;
        }
        try {
            state.options = buildOptions(state);
            state.resultMomentNames.clear();
            for (const auto *moment : selectedMoments(state))
                state.resultMomentNames.push_back(moment->name);
            state.haveResult = false;
            state.error.clear();
            jobs.push_back({static_cast<int>(i), QFileInfo(state.path).fileName(), state.options});
        } catch (const std::exception &exception) {
            errors << QString("%1: %2").arg(QFileInfo(state.path).fileName(),
                                             QString::fromUtf8(exception.what()));
        }
    }
    if (!errors.isEmpty()) {
        QMessageBox::warning(this, "Cannot start batch inversion", errors.join("\n"));
        return;
    }
    m_log->clear();
    const auto &firstOptions = jobs.front().options;
    const QString transform = firstOptions.model.transform == pytem::TransformMethod::Euler
        ? "Euler-11" : "DLF";
    const QString jacobian = firstOptions.jacobianMethod == pytem::JacobianMethod::Analytical
        ? "analytical" : "finite difference";
    const QString norm = firstOptions.regularizationNorm == pytem::RegularizationNorm::L1Blocky
        ? "L1 blocky" : "L2 smooth";
    m_log->appendPlainText(QString("Starting %1 parallel sounding inversion(s): %2, %3 Jacobian, %4")
        .arg(jobs.size()).arg(transform, jacobian, norm));
    m_jobIterations.assign(m_soundings.size(), 0);
    m_completedJobs = 0;
    m_progress->setRange(0, static_cast<int>(jobs.size()) * firstOptions.maxIterations);
    m_progress->setValue(0);
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

void MainWindow::cancelInversion()
{
    if (m_worker) { m_worker->requestCancel(); m_status->setText("Cancelling..."); }
}

void MainWindow::showProgress(int jobIndex, int iteration, double rms, const QString &detail)
{
    if (jobIndex < 0 || jobIndex >= static_cast<int>(m_soundings.size())) return;
    m_jobIterations[static_cast<std::size_t>(jobIndex)] = std::max(
        m_jobIterations[static_cast<std::size_t>(jobIndex)], iteration);
    int totalProgress = 0;
    for (int value : m_jobIterations) totalProgress += value;
    m_progress->setValue(totalProgress);
    const QString name = QFileInfo(m_soundings[static_cast<std::size_t>(jobIndex)].path).fileName();
    m_status->setText(QString("%1 — iteration %2, RMS %3; %4/%5 completed")
        .arg(name).arg(iteration).arg(rms, 0, 'g', 4)
        .arg(m_completedJobs).arg(m_soundings.size()));
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
    m_log->appendPlainText(QString("[%1] finished: %2; first Jacobian: %3")
        .arg(QFileInfo(state.path).fileName(), QString::fromStdString(result.message),
             QString::fromStdString(result.firstJacobianSource)));
    if (jobIndex == m_activeSoundingIndex)
        displayResult(state);
}

void MainWindow::inversionFailed(int jobIndex, const QString &message)
{
    if (jobIndex < 0 || jobIndex >= static_cast<int>(m_soundings.size())) return;
    auto &state = m_soundings[static_cast<std::size_t>(jobIndex)];
    state.error = message;
    ++m_completedJobs;
    m_log->appendPlainText(QString("[%1] stopped: %2")
        .arg(QFileInfo(state.path).fileName(), message));
}

void MainWindow::batchFinished()
{
    const int successful = static_cast<int>(std::count_if(
        m_soundings.begin(), m_soundings.end(),
        [](const SoundingState &state) { return state.haveResult; }));
    finishWorker();
    if (const auto *state = activeSounding()) {
        m_exportButton->setEnabled(state->haveResult);
        if (state->haveResult) displayResult(*state);
    }
    m_status->setText(QString("Batch finished — %1/%2 soundings completed")
        .arg(successful).arg(m_soundings.size()));
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
    const QColor colors[] = {QColor("#1b75bc"), QColor("#7b3294"), QColor("#008837")};
    std::size_t predictionOffset = 0;
    auto addDataSet = [&](const pytem::ForwardModel &model,
                          const std::vector<double> &observed,
                          const QString &name, std::size_t dataSetIndex) {
        QVector<QPointF> observedPoints, predictedPoints;
        for (std::size_t i = 0; i < model.times.size(); ++i) {
            observedPoints.push_back({model.times[i], observed[i]});
            predictedPoints.push_back({model.times[i], std::abs(result.predicted[predictionOffset + i])});
        }
        const QColor color = colors[dataSetIndex % 3];
        curves.push_back({name + " observed", observedPoints, color, true, false});
        curves.push_back({name + " predicted", predictedPoints, color, false, false});
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
    m_modelPlot->setCurves({{"Starting", curve(options.model.resistivities), QColor("#999999"), false, false},
                            {"Recovered", curve(result.resistivities), QColor("#1b75bc"), false, false}});
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
    QVector<QPointF> used, rejected;
    for (int row = 0; row < m_dataTable->rowCount(); ++row) {
        bool okT = false, okV = false;
        const double time = m_dataTable->item(row, 1)->text().toDouble(&okT);
        const double value = std::abs(m_dataTable->item(row, 2)->text().toDouble(&okV));
        if (!okT || !okV || time <= 0.0 || value <= 0.0) continue;
        (m_dataTable->item(row, 0)->checkState() == Qt::Checked ? used : rejected).push_back({time, value});
    }
    m_soundingPlot->setCurves({{"Used", used, QColor("#202124"), true, false},
                               {"Rejected", rejected, QColor("#bbbbbb"), true, false}});
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
    m_cancelButton->setEnabled(running);
    m_soundingSelector->setEnabled(!running && !m_soundings.empty());
    m_moment->setEnabled(!running && activeSounding());
    m_previousButton->setEnabled(!running && m_activeSoundingIndex > 0);
    m_nextButton->setEnabled(!running && m_activeSoundingIndex + 1 < static_cast<int>(m_soundings.size()));
    const auto *state = activeSounding();
    m_exportButton->setEnabled(!running && state && state->haveResult);
}

void MainWindow::updateGeometryControls()
{
    m_rxOffset->setEnabled(m_geometry->currentData().toInt() == static_cast<int>(pytem::Geometry::CircleOffset));
}
