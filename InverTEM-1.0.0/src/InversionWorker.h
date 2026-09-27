#pragma once

#include "TemSolver.h"

#include <QObject>
#include <QString>
#include <atomic>
#include <vector>

Q_DECLARE_METATYPE(pytem::InversionResult)

struct InversionJob {
    int index = 0;
    QString label;
    pytem::InversionOptions options;
    double easting = 0.0, northing = 0.0, elevation = 0.0; // metres, for SCI neighbours
};

class InversionWorker final : public QObject
{
    Q_OBJECT

public:
    explicit InversionWorker(std::vector<InversionJob> jobs,
                             unsigned maximumParallelJobs = 0);
    void requestCancel();

public slots:
    void run();

signals:
    void progress(int jobIndex, int iteration, double rms, const QString &detail);
    void resultReady(int jobIndex, const pytem::InversionResult &result);
    void failed(int jobIndex, const QString &message);
    void allFinished();

private:
    std::vector<InversionJob> m_jobs;
    unsigned m_maximumParallelJobs = 0;
    std::atomic_bool m_cancelled{false};
};
