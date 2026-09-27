#include "InversionWorker.h"

#include <QString>
#include <exception>

InversionWorker::InversionWorker(pytem::InversionOptions options)
    : m_options(std::move(options))
{
}

void InversionWorker::requestCancel()
{
    m_cancelled.store(true, std::memory_order_relaxed);
}

void InversionWorker::run()
{
    try {
        const auto result = pytem::TemSolver::invert(
            m_options,
            [this](int iteration, double rms, const std::string &detail) {
                emit progress(iteration, rms, QString::fromStdString(detail));
            },
            [this]() { return m_cancelled.load(std::memory_order_relaxed); });
        emit resultReady(result);
    } catch (const std::exception &error) {
        emit failed(QString::fromUtf8(error.what()));
    }
}

