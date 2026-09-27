#pragma once

#include "TemSolver.h"

#include <QObject>
#include <atomic>

Q_DECLARE_METATYPE(pytem::InversionResult)

class InversionWorker final : public QObject
{
    Q_OBJECT

public:
    explicit InversionWorker(pytem::InversionOptions options);
    void requestCancel();

public slots:
    void run();

signals:
    void progress(int iteration, double rms, const QString &detail);
    void resultReady(const pytem::InversionResult &result);
    void failed(const QString &message);

private:
    pytem::InversionOptions m_options;
    std::atomic_bool m_cancelled{false};
};

