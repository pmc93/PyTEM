#include "MainWindow.h"
#include "InversionWorker.h"
#include "MatrixConvolution.h"
#include "PlotWidget.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
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
#include <QStringList>
#include <QSplitter>
#include <QTableWidget>
#include <QTextStream>
#include <QThread>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

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
    auto *dataGroup = new QGroupBox("TEMcompany USF sounding");
    auto *dataLayout = new QVBoxLayout(dataGroup);
    auto *openButton = new QPushButton("Open .usf...");
    m_fileLabel = new QLabel("No sounding loaded");
    m_fileLabel->setWordWrap(true);
    m_moment = new QComboBox;
    m_moment->setEnabled(false);
    m_systemLabel = new QLabel("Waveform, gate widths, and filters are read from the selected moment.");
    m_systemLabel->setWordWrap(true);
    m_dataTable = new QTableWidget(0, 4);
    m_dataTable->setHorizontalHeaderLabels({"Use", "Time [s]", "Stacked V/Am²", "SEM [V/Am²]"});
    m_dataTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_dataTable->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    m_dataTable->setMinimumHeight(190);
    dataLayout->addWidget(openButton);
    dataLayout->addWidget(m_fileLabel);
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
    m_layerTable = new QTableWidget(6, 2);
    m_layerTable->setHorizontalHeaderLabels({"Thickness [m]", "Resistivity [Ohm m]"});
    m_layerTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    const double defaultThickness[] = {2.0, 3.0, 5.0, 8.0, 12.0};
    for (int row = 0; row < 6; ++row) {
        m_layerTable->setItem(row, 0, row < 5 ? numberItem(defaultThickness[row]) : new QTableWidgetItem(""));
        m_layerTable->setItem(row, 1, numberItem(50.0));
    }
    auto *layerButtons = new QHBoxLayout;
    auto *addButton = new QPushButton("Add layer");
    auto *removeButton = new QPushButton("Remove layer");
    layerButtons->addWidget(addButton);
    layerButtons->addWidget(removeButton);
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
    m_sensitivity = new QCheckBox("Calculate final sensitivity");
    m_sensitivity->setChecked(true);
    optionsForm->addRow("Relative error floor", m_errorFloor);
    optionsForm->addRow("Minimum resistivity", m_rhoMin);
    optionsForm->addRow("Maximum resistivity", m_rhoMax);
    optionsForm->addRow("Maximum iterations", m_maxIterations);
    optionsForm->addRow("Alpha trials / iteration", m_alphaSteps);
    optionsForm->addRow(m_sensitivity);
    controlsLayout->addWidget(optionsGroup);

    auto *runRow = new QHBoxLayout;
    m_runButton = new QPushButton("Run native inversion");
    m_runButton->setDefault(true);
    m_runButton->setEnabled(false);
    m_cancelButton = new QPushButton("Cancel");
    m_cancelButton->setEnabled(false);
    m_exportButton = new QPushButton("Export...");
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
    connect(m_moment, qOverload<int>(&QComboBox::currentIndexChanged), this, &MainWindow::selectMoment);
    connect(addButton, &QPushButton::clicked, this, &MainWindow::addLayer);
    connect(removeButton, &QPushButton::clicked, this, &MainWindow::removeLayer);
    connect(m_runButton, &QPushButton::clicked, this, &MainWindow::runInversion);
    connect(m_cancelButton, &QPushButton::clicked, this, &MainWindow::cancelInversion);
    connect(m_exportButton, &QPushButton::clicked, this, &MainWindow::exportResult);
    connect(m_geometry, qOverload<int>(&QComboBox::currentIndexChanged), this, &MainWindow::updateGeometryControls);
    connect(m_dataTable, &QTableWidget::cellChanged, this, &MainWindow::updateInputPlot);
}

void MainWindow::loadUsf()
{
    QSettings settings;
    const QString path = QFileDialog::getOpenFileName(this, "Open TEMcompany USF sounding",
        settings.value("lastUsfDirectory").toString(), "Universal Sounding Format (*.usf *.USF);;All files (*)");
    if (path.isEmpty()) return;
    try {
        m_sounding = pytem::UsfReader::read(path.toStdString());
    } catch (const std::exception &error) {
        QMessageBox::critical(this, "USF import failed", QString::fromUtf8(error.what()));
        return;
    }
    m_loadedPath = path;
    settings.setValue("lastUsfDirectory", QFileInfo(path).absolutePath());
    m_fileLabel->setText(QString("%1 — %2\n%3, %4; elevation %5 m")
        .arg(QString::fromStdString(m_sounding.soundingName), QFileInfo(path).fileName())
        .arg(m_sounding.longitude, 0, 'f', 6).arg(m_sounding.latitude, 0, 'f', 6)
        .arg(m_sounding.elevation, 0, 'f', 2));
    m_txRadius->setValue(m_sounding.equivalentCircularRadius());
    m_rxOffset->setValue(m_sounding.radialReceiverOffset());
    m_geometry->setCurrentIndex(m_sounding.radialReceiverOffset() > 0.0 ? 1 : 0);
    m_moment->blockSignals(true);
    m_moment->clear();
    for (const auto &moment : m_sounding.moments)
        m_moment->addItem(QString("%1 — channel %2, %3 stacks, %4 Hz")
            .arg(QString::fromStdString(moment.name)).arg(moment.channel).arg(moment.stackCount)
            .arg(moment.meanFrequency, 0, 'f', 2));
    m_moment->blockSignals(false);
    m_moment->setEnabled(true);
    m_moment->setCurrentIndex(0);
    selectMoment(0);
    m_runButton->setEnabled(true);
    m_haveResult = false;
    m_exportButton->setEnabled(false);
}

const pytem::UsfMoment *MainWindow::selectedMoment() const
{
    const int index = m_moment->currentIndex();
    return index >= 0 && index < static_cast<int>(m_sounding.moments.size())
        ? &m_sounding.moments[static_cast<std::size_t>(index)] : nullptr;
}

void MainWindow::selectMoment(int)
{
    const auto *moment = selectedMoment();
    if (!moment) return;
    m_dataTable->blockSignals(true);
    m_dataTable->setRowCount(static_cast<int>(moment->times.size()));
    int selected = 0;
    for (int row = 0; row < static_cast<int>(moment->times.size()); ++row) {
        auto *useItem = new QTableWidgetItem;
        useItem->setFlags(Qt::ItemIsEnabled | Qt::ItemIsUserCheckable | Qt::ItemIsSelectable);
        const double voltage = moment->voltages[static_cast<std::size_t>(row)];
        const double error = moment->standardErrors[static_cast<std::size_t>(row)];
        const bool reliable = voltage > 0.0 && (error <= 0.0 || voltage / error >= 3.0);
        useItem->setCheckState(reliable ? Qt::Checked : Qt::Unchecked);
        selected += reliable ? 1 : 0;
        m_dataTable->setItem(row, 0, useItem);
        m_dataTable->setItem(row, 1, numberItem(moment->times[static_cast<std::size_t>(row)]));
        m_dataTable->setItem(row, 2, numberItem(voltage));
        m_dataTable->setItem(row, 3, numberItem(error));
    }
    m_dataTable->blockSignals(false);
    QStringList stages;
    for (std::size_t i = 0; i < moment->lowPassFrequencies.size(); ++i)
        stages << QString("LP %1 kHz (order %2)").arg(moment->lowPassFrequencies[i] / 1000.0).arg(moment->lowPassOrders[i]);
    for (std::size_t i = 0; i < moment->highPassFrequencies.size(); ++i)
        stages << QString("HP %1 Hz (order %2)").arg(moment->highPassFrequencies[i]).arg(moment->highPassOrders[i]);
    m_systemLabel->setText(QString("%1 waveform points; %2; %3/%4 gates selected")
        .arg(moment->waveformTimes.size()).arg(stages.isEmpty() ? "no filter stages" : stages.join(" × "))
        .arg(selected).arg(moment->times.size()));
    m_status->setText(QString("Loaded %1 moment from %2").arg(QString::fromStdString(moment->name), QFileInfo(m_loadedPath).fileName()));
    updateInputPlot();
}

void MainWindow::addLayer()
{
    const int row = m_layerTable->rowCount();
    if (row > 0) m_layerTable->item(row - 1, 0)->setText("10");
    m_layerTable->insertRow(row);
    m_layerTable->setItem(row, 0, new QTableWidgetItem(""));
    m_layerTable->setItem(row, 1, numberItem(50.0));
}

void MainWindow::removeLayer()
{
    if (m_layerTable->rowCount() <= 1) return;
    m_layerTable->removeRow(m_layerTable->rowCount() - 1);
    m_layerTable->item(m_layerTable->rowCount() - 1, 0)->setText("");
}

bool MainWindow::validateInputs(QString &error) const
{
    if (!selectedMoment()) { error = "Open a USF sounding first."; return false; }
    int used = 0;
    double previous = 0.0;
    for (int row = 0; row < m_dataTable->rowCount(); ++row) {
        if (m_dataTable->item(row, 0)->checkState() != Qt::Checked) continue;
        bool okT = false, okV = false;
        const double time = m_dataTable->item(row, 1)->text().toDouble(&okT);
        const double voltage = m_dataTable->item(row, 2)->text().toDouble(&okV);
        if (!okT || !okV || time <= previous || voltage <= 0.0) {
            error = QString("Selected data row %1 is invalid or out of time order.").arg(row + 1); return false;
        }
        previous = time;
        ++used;
    }
    if (used < 3) { error = "Select at least three positive gates."; return false; }
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

pytem::InversionOptions MainWindow::buildOptions() const
{
    pytem::InversionOptions options;
    options.model.geometry = static_cast<pytem::Geometry>(m_geometry->currentData().toInt());
    options.model.txSize = m_txRadius->value();
    options.model.rxX = m_rxOffset->value();
    const auto *moment = selectedMoment();
    std::vector<double> gateOpen, gateClose;
    const double errorFloor = m_errorFloor->value() / 100.0;
    for (int row = 0; row < m_dataTable->rowCount(); ++row) {
        if (m_dataTable->item(row, 0)->checkState() != Qt::Checked) continue;
        const auto index = static_cast<std::size_t>(row);
        const double value = std::abs(m_dataTable->item(row, 2)->text().toDouble());
        const double sem = std::abs(m_dataTable->item(row, 3)->text().toDouble());
        options.model.times.push_back(m_dataTable->item(row, 1)->text().toDouble());
        options.observed.push_back(value);
        options.noiseStd.push_back(std::max(sem, errorFloor * value));
        gateOpen.push_back(moment->gateOpen[index]);
        gateClose.push_back(moment->gateClose[index]);
    }
    for (int row = 0; row < m_layerTable->rowCount(); ++row) {
        if (row + 1 < m_layerTable->rowCount()) options.model.thicknesses.push_back(m_layerTable->item(row, 0)->text().toDouble());
        options.model.resistivities.push_back(m_layerTable->item(row, 1)->text().toDouble());
    }
    if (m_useWaveform->isChecked()) {
        const auto convolution = pytem::MatrixConvolution::build(
            options.model.times, gateOpen, gateClose,
            moment->waveformTimes, moment->waveformAmplitudes, 300, 81, 1.0);
        options.model.stepTimes = convolution.stepTimes;
        options.model.responseMatrix = convolution.matrix;
    }
    if (m_useFilters->isChecked()) {
        options.model.lowPassFrequencies = moment->lowPassFrequencies;
        options.model.lowPassOrders = moment->lowPassOrders;
        options.model.highPassFrequencies = moment->highPassFrequencies;
        options.model.highPassOrders = moment->highPassOrders;
    }
    options.rhoMin = m_rhoMin->value();
    options.rhoMax = m_rhoMax->value();
    options.maxIterations = m_maxIterations->value();
    options.alphaSteps = m_alphaSteps->value();
    options.calculateSensitivity = m_sensitivity->isChecked();
    return options;
}

void MainWindow::runInversion()
{
    QString error;
    if (!validateInputs(error)) { QMessageBox::warning(this, "Cannot run inversion", error); return; }
    try { m_lastOptions = buildOptions(); }
    catch (const std::exception &exception) { QMessageBox::critical(this, "Matrix setup failed", QString::fromUtf8(exception.what())); return; }
    m_haveResult = false;
    m_log->clear();
    m_log->appendPlainText(QString("Native C++ DLF inversion — matrix %1 × %2")
        .arg(m_lastOptions.model.responseMatrix.size()).arg(m_lastOptions.model.stepTimes.size()));
    m_progress->setRange(0, m_lastOptions.maxIterations);
    m_progress->setValue(0);
    setRunning(true);
    m_workerThread = new QThread(this);
    m_worker = new InversionWorker(m_lastOptions);
    m_worker->moveToThread(m_workerThread);
    connect(m_workerThread, &QThread::started, m_worker, &InversionWorker::run);
    connect(m_worker, &InversionWorker::progress, this, &MainWindow::showProgress);
    connect(m_worker, &InversionWorker::resultReady, this, &MainWindow::inversionComplete);
    connect(m_worker, &InversionWorker::failed, this, &MainWindow::inversionFailed);
    m_workerThread->start();
}

void MainWindow::cancelInversion()
{
    if (m_worker) { m_worker->requestCancel(); m_status->setText("Cancelling..."); }
}

void MainWindow::showProgress(int iteration, double rms, const QString &detail)
{
    m_progress->setValue(iteration);
    m_status->setText(QString("Iteration %1 — RMS %2").arg(iteration).arg(rms, 0, 'g', 4));
    m_log->appendPlainText(QString("Iteration %1  RMS=%2  %3").arg(iteration).arg(rms, 0, 'g', 6).arg(detail));
}

void MainWindow::inversionComplete(const pytem::InversionResult &result)
{
    m_lastResult = result;
    m_haveResult = true;
    displayResult(result);
    m_exportButton->setEnabled(true);
    finishWorker();
}

void MainWindow::inversionFailed(const QString &message)
{
    m_log->appendPlainText("Stopped: " + message);
    m_status->setText(message == "Inversion cancelled" ? "Cancelled" : "Inversion failed");
    if (message != "Inversion cancelled") QMessageBox::critical(this, "Inversion failed", message);
    finishWorker();
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

void MainWindow::displayResult(const pytem::InversionResult &result)
{
    QVector<QPointF> observed, predicted;
    for (std::size_t i = 0; i < m_lastOptions.model.times.size(); ++i) {
        observed.push_back({m_lastOptions.model.times[i], m_lastOptions.observed[i]});
        predicted.push_back({m_lastOptions.model.times[i], std::abs(result.predicted[i])});
    }
    m_soundingPlot->setCurves({{"Observed", observed, QColor("#202124"), true, false},
                               {"Predicted", predicted, QColor("#d95f02"), false, false}});
    double bottom = 0.0;
    for (double h : m_lastOptions.model.thicknesses) bottom += h;
    bottom += std::max(10.0, bottom * 0.35);
    auto curve = [&](const std::vector<double> &rho) {
        QVector<QPointF> points;
        double top = 0.0;
        for (std::size_t i = 0; i < rho.size(); ++i) {
            const double base = i < m_lastOptions.model.thicknesses.size() ? top + m_lastOptions.model.thicknesses[i] : bottom;
            points.push_back({rho[i], top}); points.push_back({rho[i], base});
            if (i + 1 < rho.size()) points.push_back({rho[i + 1], base});
            top = base;
        }
        return points;
    };
    m_modelPlot->setCurves({{"Starting", curve(m_lastOptions.model.resistivities), QColor("#999999"), false, false},
                            {"Recovered", curve(result.resistivities), QColor("#1b75bc"), false, false}});
    const double finalRms = result.rmsHistory.empty() ? 0.0 : result.rmsHistory.back();
    m_status->setText(QString("Finished — RMS %1; %2").arg(finalRms, 0, 'g', 4).arg(QString::fromStdString(result.message)));
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
    if (!m_haveResult) return;
    QSettings settings;
    QString path = QFileDialog::getSaveFileName(this, "Export native inversion",
        settings.value("lastExportDirectory").toString() + "/pytem_cpp_result.json",
        "JSON result (*.json);;Layer model CSV (*.csv)");
    if (path.isEmpty()) return;
    if (!path.endsWith(".json", Qt::CaseInsensitive) && !path.endsWith(".csv", Qt::CaseInsensitive)) path += ".json";
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) { QMessageBox::critical(this, "Export failed", file.errorString()); return; }
    if (path.endsWith(".json", Qt::CaseInsensitive)) {
        QJsonObject object;
        object["source_usf"] = m_loadedPath;
        object["moment"] = QString::fromStdString(selectedMoment()->name);
        object["times"] = jsonArray(m_lastOptions.model.times);
        object["observed"] = jsonArray(m_lastOptions.observed);
        object["predicted"] = jsonArray(m_lastResult.predicted);
        object["thicknesses"] = jsonArray(m_lastOptions.model.thicknesses);
        object["resistivities"] = jsonArray(m_lastResult.resistivities);
        object["sensitivity"] = jsonArray(m_lastResult.sensitivity);
        object["rms_history"] = jsonArray(m_lastResult.rmsHistory);
        file.write(QJsonDocument(object).toJson(QJsonDocument::Indented));
    } else {
        QTextStream out(&file);
        out << "layer,top_depth_m,bottom_depth_m,resistivity_ohm_m,sensitivity\n";
        double top = 0.0;
        for (std::size_t i = 0; i < m_lastResult.resistivities.size(); ++i) {
            out << i + 1 << ',' << QString::number(top, 'g', 12) << ',';
            if (i < m_lastOptions.model.thicknesses.size()) { top += m_lastOptions.model.thicknesses[i]; out << QString::number(top, 'g', 12); }
            else out << "inf";
            out << ',' << QString::number(m_lastResult.resistivities[i], 'g', 12) << ',';
            if (i < m_lastResult.sensitivity.size()) out << QString::number(m_lastResult.sensitivity[i], 'g', 12);
            out << '\n';
        }
    }
    settings.setValue("lastExportDirectory", QFileInfo(path).absolutePath());
}

void MainWindow::setRunning(bool running)
{
    m_runButton->setEnabled(!running && selectedMoment());
    m_cancelButton->setEnabled(running);
}

void MainWindow::updateGeometryControls()
{
    m_rxOffset->setEnabled(m_geometry->currentData().toInt() == static_cast<int>(pytem::Geometry::CircleOffset));
}
