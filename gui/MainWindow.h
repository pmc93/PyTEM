#pragma once

#include "TemSolver.h"
#include "UsfReader.h"

#include <QMainWindow>

class InversionWorker;
class PlotWidget;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QTableWidget;
class QThread;

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void loadUsf();
    void selectMoment(int index);
    void addLayer();
    void removeLayer();
    void runInversion();
    void cancelInversion();
    void showProgress(int iteration, double rms, const QString &detail);
    void inversionComplete(const pytem::InversionResult &result);
    void inversionFailed(const QString &message);
    void exportResult();
    void updateGeometryControls();
    void updateInputPlot();

private:
    void buildInterface();
    void setRunning(bool running);
    bool validateInputs(QString &error) const;
    pytem::InversionOptions buildOptions() const;
    const pytem::UsfMoment *selectedMoment() const;
    void displayResult(const pytem::InversionResult &result);
    void finishWorker();

    QTableWidget *m_dataTable = nullptr;
    QTableWidget *m_layerTable = nullptr;
    QComboBox *m_moment = nullptr;
    QComboBox *m_geometry = nullptr;
    QDoubleSpinBox *m_txRadius = nullptr;
    QDoubleSpinBox *m_rxOffset = nullptr;
    QDoubleSpinBox *m_errorFloor = nullptr;
    QDoubleSpinBox *m_rhoMin = nullptr;
    QDoubleSpinBox *m_rhoMax = nullptr;
    QSpinBox *m_maxIterations = nullptr;
    QSpinBox *m_alphaSteps = nullptr;
    QCheckBox *m_useWaveform = nullptr;
    QCheckBox *m_useFilters = nullptr;
    QCheckBox *m_sensitivity = nullptr;
    QPushButton *m_runButton = nullptr;
    QPushButton *m_cancelButton = nullptr;
    QPushButton *m_exportButton = nullptr;
    QProgressBar *m_progress = nullptr;
    QLabel *m_fileLabel = nullptr;
    QLabel *m_systemLabel = nullptr;
    QLabel *m_status = nullptr;
    QPlainTextEdit *m_log = nullptr;
    PlotWidget *m_soundingPlot = nullptr;
    PlotWidget *m_modelPlot = nullptr;

    pytem::UsfSounding m_sounding;
    pytem::InversionOptions m_lastOptions;
    pytem::InversionResult m_lastResult;
    bool m_haveResult = false;
    QString m_loadedPath;
    QThread *m_workerThread = nullptr;
    InversionWorker *m_worker = nullptr;
};

