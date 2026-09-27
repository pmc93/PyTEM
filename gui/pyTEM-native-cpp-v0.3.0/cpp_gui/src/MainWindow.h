#pragma once

#include "TemSolver.h"
#include "UsfReader.h"

#include <QMainWindow>
#include <QString>

#include <cstddef>
#include <vector>

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
    void selectSounding(int index);
    void previousSounding();
    void nextSounding();
    void selectMoment(int index);
    void addLayer();
    void removeLayer();
    void setLayerCount(int count);
    void runInversion();
    void cancelInversion();
    void showProgress(int jobIndex, int iteration, double rms, const QString &detail);
    void inversionComplete(int jobIndex, const pytem::InversionResult &result);
    void inversionFailed(int jobIndex, const QString &message);
    void batchFinished();
    void exportResult();
    void updateGeometryControls();
    void updateInputPlot();

private:
    void buildInterface();
    void setRunning(bool running);
    struct SoundingState {
        pytem::UsfSounding sounding;
        QString path;
        int momentChoice = -1;
        pytem::Geometry geometry = pytem::Geometry::CircleCentral;
        double txRadius = 1.0;
        double rxOffset = 0.0;
        std::vector<std::vector<bool>> gateEnabled;
        pytem::InversionOptions options;
        pytem::InversionResult result;
        std::vector<std::string> resultMomentNames;
        QString error;
        bool haveResult = false;
    };
    SoundingState *activeSounding();
    const SoundingState *activeSounding() const;
    void saveActiveSoundingEdits();
    bool validateInputs(const SoundingState &state, QString &error) const;
    pytem::InversionOptions buildOptions(const SoundingState &state) const;
    std::vector<const pytem::UsfMoment *> selectedMoments(const SoundingState &state) const;
    void displayResult(const SoundingState &state);
    void displayInput(const SoundingState &state);
    void finishWorker();

    QTableWidget *m_dataTable = nullptr;
    QTableWidget *m_layerTable = nullptr;
    QComboBox *m_soundingSelector = nullptr;
    QComboBox *m_moment = nullptr;
    QComboBox *m_geometry = nullptr;
    QComboBox *m_transform = nullptr;
    QComboBox *m_jacobian = nullptr;
    QDoubleSpinBox *m_txRadius = nullptr;
    QDoubleSpinBox *m_rxOffset = nullptr;
    QDoubleSpinBox *m_errorFloor = nullptr;
    QDoubleSpinBox *m_rhoMin = nullptr;
    QDoubleSpinBox *m_rhoMax = nullptr;
    QSpinBox *m_maxIterations = nullptr;
    QSpinBox *m_alphaSteps = nullptr;
    QSpinBox *m_layerCount = nullptr;
    QSpinBox *m_parallelJobs = nullptr;
    QCheckBox *m_useWaveform = nullptr;
    QCheckBox *m_useFilters = nullptr;
    QCheckBox *m_sensitivity = nullptr;
    QCheckBox *m_cacheJacobian = nullptr;
    QComboBox *m_regularizationNorm = nullptr;
    QPushButton *m_previousButton = nullptr;
    QPushButton *m_nextButton = nullptr;
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

    std::vector<SoundingState> m_soundings;
    int m_activeSoundingIndex = -1;
    std::vector<std::size_t> m_rowMomentIndices;
    std::vector<std::size_t> m_rowGateIndices;
    std::vector<int> m_jobIterations;
    int m_completedJobs = 0;
    QThread *m_workerThread = nullptr;
    InversionWorker *m_worker = nullptr;
};
