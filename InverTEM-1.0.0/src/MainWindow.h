#pragma once

#include "MatrixConvolution.h"
#include "TemSolver.h"
#include "UsfReader.h"

#include <QJsonObject>
#include <QMainWindow>
#include <QElapsedTimer>
#include <QString>
#include <QVector>

#include <cstddef>
#include <limits>
#include <map>
#include <vector>

class InversionWorker;
class MapTileLoader;
class PlotWidget;
class QCheckBox;
class QCloseEvent;
class QComboBox;
class QDoubleSpinBox;
class QDialog;
class QLabel;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QRadioButton;
class QSpinBox;
class QStackedWidget;
class QThread;
class QTimer;

class MainWindow final : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void newProject();
    void saveProject();
    void loadProject();
    void loadUsf();
    void correctElevations();
    void selectSounding(int index);
    void previousSounding();
    void nextSounding();
    void selectMoment(int index);
    void setLayerCount(int count);
    void toggleDataPoint(int tableRow);
    void editDataPoints(const QVector<int> &pointIds, bool restore);
    void undoGateEdit();
    void selectPlotMode(int index);
    void previousTransectPage();
    void nextTransectPage();
    void toggleSoundingData(int soundingIndex);
    void navigateToSounding(int soundingIndex);
    void changeMapBackground();
    void runInversion();
    void killInversion();
    void showProgress(int jobIndex, int iteration, double rms, const QString &detail);
    void inversionComplete(int jobIndex, const pytem::InversionResult &result);
    void inversionFailed(int jobIndex, const QString &message);
    void batchFinished();
    void exportModelledData();
    void clearSavedResults();
    void moveKeptUsfFiles();
    void updateInputPlot();

private:
    void buildInterface();
    void setRunning(bool running);
    struct SoundingState {
        struct SavedModel {
            QString key;
            QString label;
            int momentChoice = -1;
            pytem::InversionOptions options;
            pytem::InversionResult result;
            std::vector<std::string> momentNames;
            std::vector<std::size_t> momentIndices;
            std::vector<std::vector<std::size_t>> gateIndices;
            double doiDepth = -1.0;
        };
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
        std::vector<std::size_t> resultMomentIndices;
        std::vector<std::vector<std::size_t>> resultGateIndices;
        std::vector<SavedModel> savedModels;
        QString error;
        bool haveResult = false;
        bool resultStale = false; // gates changed since this result was computed
        bool includeInBatch = true;
        int group = 0; // user grouping, 0: none
        double gpsElevation = std::numeric_limits<double>::quiet_NaN(); // kept while a DEM height replaces it
        bool bookmark = false;
        std::vector<std::vector<double>> originalErrors; // standard errors before any STD change, empty if unchanged
        std::vector<int> sources; // averaged soundings: their raw soundings in m_rawSoundings (this session only)
    };
    struct ProjectGeometry {
        bool locked = false;
        double loopX = 0.0;
        double loopY = 0.0;
        double coilX = 0.0;
        double coilY = 0.0;
    };
    SoundingState *activeSounding();
    const SoundingState *activeSounding() const;
    void saveActiveSoundingEdits();
    bool validateInputs(const SoundingState &state, QString &error) const;
    pytem::InversionOptions buildOptions(const SoundingState &state) const;
    bool gateUsed(const SoundingState &state, std::size_t momentIndex, std::size_t gate) const;
    std::vector<const pytem::UsfMoment *> selectedMoments(const SoundingState &state) const;
    std::vector<const pytem::UsfMoment *> fittedMoments(const SoundingState &state) const;
    // Every fitted moment keeps at least three gates (else the batch skips it).
    bool hasUsableGates(const SoundingState &state) const;
    void displayResult(const SoundingState &state);
    void displayInput(const SoundingState &state);
    void activateSavedModel(SoundingState &state,
                            const SoundingState::SavedModel &saved);
    void updateSystemSummary();
    void updateTransectPlot();
    void updateTransectNavigation();
    void updateSoundingNavigation();
    std::vector<int> shownSoundings() const;
    std::vector<int> transectPage() const;
    void alignTransectPage();
    void rebuildGroupFilter();
    void updateMapPlot();
    void loadMapTiles();
    void updateBatchTiming();
    void updateResistivityAxis();
    void writeTimingReport();
    void finishWorker();
    SoundingState readSoundingState(const QString &path, pytem::UsfSounding sounding) const;
    void rebuildSoundingSelector(int selectedIndex = 0);
    void resetProjectState();
    void resetInversionSettings();
    enum class GateEdit { Toggle, Remove, Restore };
    void editGates(const QVector<int> &pointIds, GateEdit edit);
    struct GateReference { int sounding = -1; std::size_t moment = 0, gate = 0; };
    std::vector<GateReference> resolvePoints(const QVector<int> &pointIds) const;
    void editReferences(const std::vector<GateReference> &references, GateEdit edit);
    void refreshAfterEdit(bool activeChanged); // plots, map and project state after gate or STD edits

    // Processing (Processing.cpp), after EEMstudio's tools.
    void averageSoundings();       // windowed averaging along the lines of XYZ data
    void editNoiseModel();         // STDs from a uniform STD and a noise level per moment
    void removeHighStd();          // cull gates with relative STD >= N
    void selectPoints(const QVector<int> &pointIds);
    void selectionCommand(char key, bool wholePage); // Q A N E W, or 27 (Esc) to clear
    void changeSelectionStd(double addRelative); // < 0: restore the original STDs
    void toggleBookmark();
    void nextBookmark();
    // The value plotted for a gate: |dB/dt|, or late-time apparent resistivity [Ohm m].
    double displayValue(const pytem::UsfSounding &sounding, double time, double value) const;
    std::vector<GateReference> m_selection;
    std::vector<SoundingState> m_rawSoundings;   // the soundings before averaging (this session only)
    std::map<std::string, double> m_noiseLevel;  // per moment name, dB/dt [V/m^2] at 1 ms
    double m_noiseUniformStd = 0.03;
    void pushGateUndo(const std::vector<int> &soundingIndices);
    void readSolverSettings();
    bool confirmProjectReplacement();
    bool writeProject(const QString &path);
    bool readProject(const QString &path);
    void markProjectModified();
    void updateProjectTitle();

    QComboBox *m_soundingSelector = nullptr;
    QComboBox *m_moment = nullptr;
    QDoubleSpinBox *m_maxDepth = nullptr;
    QDoubleSpinBox *m_firstDepth = nullptr;
    QDoubleSpinBox *m_errorFloor = nullptr;
    QDoubleSpinBox *m_resistivityAxisMinimum = nullptr;
    QDoubleSpinBox *m_resistivityAxisMaximum = nullptr;
    QSpinBox *m_layerCount = nullptr;
    QSpinBox *m_parallelJobs = nullptr;
    QSpinBox *m_transectSize = nullptr;
    // Read from invertem_solver.txt beside the executable (written with these defaults when missing).
    bool m_pytemJoint = true;
    pytem::SciSettings m_sci;
    std::vector<pytem::SciSettings> m_sciRuns; // every combination of the SCI values in invertem_solver.txt
    int m_sciJobs = 0;                         // soundings per SCI run (0: not SCI)
    // Shared-grid gate matrices keyed by the moments' gates and waveform.
    mutable std::map<std::vector<double>, std::vector<pytem::MatrixConvolution>> m_convolutionCache;
    pytem::TransformMethod m_transform = pytem::TransformMethod::Euler;
    pytem::JacobianMethod m_jacobian = pytem::JacobianMethod::Analytical;
    pytem::JacobianUpdateMethod m_jacobianUpdate = pytem::JacobianUpdateMethod::FullEveryIteration;
    int m_broydenRefresh = 3;
    QCheckBox *m_useGpu = nullptr;
    QCheckBox *m_sensitivity = nullptr;
    QCheckBox *m_cacheJacobian = nullptr;
    QCheckBox *m_adaptiveAlpha = nullptr;
    QCheckBox *m_autoResistivityAxis = nullptr;
    QComboBox *m_regularizationNorm = nullptr;
    QComboBox *m_mapPointSize = nullptr;
    QComboBox *m_mapColour = nullptr; // map colours: RMS or ground elevation
    QComboBox *m_plotMode = nullptr;
    QComboBox *m_yUnit = nullptr;          // dB/dt or apparent resistivity
    QPushButton *m_averageButton = nullptr;
    QPushButton *m_noiseButton = nullptr;
    QPushButton *m_highStdButton = nullptr;
    QSpinBox *m_group = nullptr;
    QComboBox *m_groupFilter = nullptr;
    QPushButton *m_previousButton = nullptr;
    QPushButton *m_nextButton = nullptr;
    QPushButton *m_previousTransectButton = nullptr;
    QPushButton *m_nextTransectButton = nullptr;
    QPushButton *m_runButton = nullptr;
    QPushButton *m_killButton = nullptr;
    QPushButton *m_exportModelledButton = nullptr;
    QPushButton *m_clearResultsButton = nullptr;
    QPushButton *m_moveKeptButton = nullptr;
    QPushButton *m_autoFilterButton = nullptr;
    // Gate and inclusion snapshots taken before each edit, newest last.
    struct GateSnapshot { int sounding; std::vector<std::vector<bool>> gates; bool included; };
    std::vector<std::vector<GateSnapshot>> m_gateUndo;
    QPushButton *m_importButton = nullptr;
    QPushButton *m_elevationButton = nullptr;
    QPushButton *m_newProjectButton = nullptr;
    QPushButton *m_saveProjectButton = nullptr;
    QPushButton *m_loadProjectButton = nullptr;
    QProgressBar *m_progress = nullptr;
    QProgressBar *m_mapProgress = nullptr;
    QRadioButton *m_noMap = nullptr;
    QRadioButton *m_openStreetMap = nullptr;
    QRadioButton *m_satelliteMap = nullptr;
    QLabel *m_geometryLabel = nullptr;
    QLabel *m_projectLabel = nullptr;
    QLabel *m_systemLabel = nullptr;
    QLabel *m_status = nullptr;
    QLabel *m_timeEstimate = nullptr;
    QLabel *m_transectRangeLabel = nullptr;
    QPlainTextEdit *m_log = nullptr;
    PlotWidget *m_soundingPlot = nullptr;
    QLabel *m_fileLabel = nullptr;
    PlotWidget *m_modelPlot = nullptr;
    PlotWidget *m_transectLowPlot = nullptr;
    PlotWidget *m_transectHighPlot = nullptr;
    PlotWidget *m_mapPlot = nullptr;
    QStackedWidget *m_plotModeStack = nullptr;
    MapTileLoader *m_mapTileLoader = nullptr;

    std::vector<SoundingState> m_soundings;
    ProjectGeometry m_projectGeometry;
    QString m_projectPath;
    QString m_projectName = "Untitled";
    bool m_projectModified = false;
    int m_activeSoundingIndex = -1;
    bool m_batchKilled = false;
    std::vector<std::size_t> m_rowMomentIndices;
    std::vector<std::size_t> m_rowGateIndices;
    std::vector<GateReference> m_transectPointReferences;
    int m_transectStart = 0;
    std::vector<int> m_jobIterations;
    std::vector<double> m_liveRms; // RMS after the latest iteration, NaN until the first
    int m_completedJobs = 0;
    int m_successfulJobs = 0;
    int m_totalJobs = 0;
    QElapsedTimer m_batchTimer;
    QTimer *m_elapsedTimer = nullptr;
    QTimer *m_mapRefreshTimer = nullptr;
    QDialog *m_storingDialog = nullptr; // blocks the window while SCI results are stored
    qint64 m_lastEtaUpdateMs = -1;
    qint64 m_estimatedRemainingMs = -1;
    QString m_timingReportPath;
    double m_preparationSeconds = 0.0;
    QThread *m_workerThread = nullptr;
    InversionWorker *m_worker = nullptr;
};
