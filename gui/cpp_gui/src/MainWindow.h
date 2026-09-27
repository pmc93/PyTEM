#pragma once

#include "TemSolver.h"
#include "UsfReader.h"

#include <QMainWindow>
#include <QElapsedTimer>
#include <QString>

#include <cstddef>
#include <vector>

class InversionWorker;
class MapTileLoader;
class PlotWidget;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QRadioButton;
class QSpinBox;
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
    void setLayerCount(int count);
    void toggleDataPoint(int tableRow);
    void toggleSoundingData(int soundingIndex);
    void navigateToSounding(int soundingIndex);
    void changeMapBackground();
    void runInversion();
    void killInversion();
    void showProgress(int jobIndex, int iteration, double rms, const QString &detail);
    void inversionComplete(int jobIndex, const pytem::InversionResult &result);
    void inversionFailed(int jobIndex, const QString &message);
    void batchFinished();
    void exportResult();
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
        std::vector<std::vector<bool>> gateEnabledBeforeExclusion;
        pytem::InversionOptions options;
        pytem::InversionResult result;
        std::vector<std::string> resultMomentNames;
        std::vector<std::size_t> resultMomentIndices;
        std::vector<std::vector<std::size_t>> resultGateIndices;
        QString error;
        bool haveResult = false;
        bool includeInBatch = true;
    };
    SoundingState *activeSounding();
    const SoundingState *activeSounding() const;
    void saveActiveSoundingEdits();
    bool validateInputs(const SoundingState &state, QString &error) const;
    pytem::InversionOptions buildOptions(const SoundingState &state) const;
    std::vector<const pytem::UsfMoment *> selectedMoments(const SoundingState &state) const;
    void displayResult(const SoundingState &state);
    void displayInput(const SoundingState &state);
    void updateSystemSummary();
    void updateMapPlot();
    void loadMapTiles();
    void updateBatchTiming();
    void writeTimingReport();
    void finishWorker();

    QComboBox *m_soundingSelector = nullptr;
    QComboBox *m_moment = nullptr;
    QComboBox *m_transform = nullptr;
    QComboBox *m_jacobian = nullptr;
    QDoubleSpinBox *m_maxDepth = nullptr;
    QDoubleSpinBox *m_errorFloor = nullptr;
    QSpinBox *m_layerCount = nullptr;
    QSpinBox *m_parallelJobs = nullptr;
    QCheckBox *m_useWaveform = nullptr;
    QCheckBox *m_useFilters = nullptr;
    QCheckBox *m_useGpu = nullptr;
    QCheckBox *m_sensitivity = nullptr;
    QCheckBox *m_cacheJacobian = nullptr;
    QComboBox *m_regularizationNorm = nullptr;
    QPushButton *m_previousButton = nullptr;
    QPushButton *m_nextButton = nullptr;
    QPushButton *m_runButton = nullptr;
    QPushButton *m_killButton = nullptr;
    QPushButton *m_exportButton = nullptr;
    QProgressBar *m_progress = nullptr;
    QProgressBar *m_mapProgress = nullptr;
    QRadioButton *m_noMap = nullptr;
    QRadioButton *m_openStreetMap = nullptr;
    QRadioButton *m_satelliteMap = nullptr;
    QLabel *m_geometryLabel = nullptr;
    QLabel *m_systemLabel = nullptr;
    QLabel *m_status = nullptr;
    QLabel *m_timeEstimate = nullptr;
    QPlainTextEdit *m_log = nullptr;
    PlotWidget *m_soundingPlot = nullptr;
    PlotWidget *m_modelPlot = nullptr;
    PlotWidget *m_mapPlot = nullptr;
    MapTileLoader *m_mapTileLoader = nullptr;

    std::vector<SoundingState> m_soundings;
    int m_activeSoundingIndex = -1;
    bool m_batchKilled = false;
    std::vector<std::size_t> m_rowMomentIndices;
    std::vector<std::size_t> m_rowGateIndices;
    std::vector<int> m_jobIterations;
    int m_completedJobs = 0;
    int m_successfulJobs = 0;
    int m_totalJobs = 0;
    QElapsedTimer m_batchTimer;
    qint64 m_lastEtaUpdateMs = -1;
    QString m_timingReportPath;
    double m_preparationSeconds = 0.0;
    QThread *m_workerThread = nullptr;
    InversionWorker *m_worker = nullptr;
};
