#include "InversionWorker.h"

#include <QString>
#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

InversionWorker::InversionWorker(std::vector<InversionJob> jobs,
                                 unsigned maximumParallelJobs)
    : m_jobs(std::move(jobs)),
      m_maximumParallelJobs(maximumParallelJobs)
{
}

void InversionWorker::requestCancel()
{
    m_cancelled.store(true, std::memory_order_relaxed);
}

void InversionWorker::run()
{
    if (m_jobs.empty()) {
        emit allFinished();
        return;
    }
    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    const unsigned requested = m_maximumParallelJobs == 0 ? hardware : m_maximumParallelJobs;
    const unsigned workerCount = std::max(1u, std::min<unsigned>(
        hardware, std::min<unsigned>(requested, static_cast<unsigned>(m_jobs.size()))));
    std::vector<bool> delivered(m_jobs.size(), false);
    std::mutex deliveredMutex;

    // Fast SCI inverts every job together in one constrained system.
    if (m_jobs.front().options.spatialConstraints) {
        std::vector<pytem::InversionOptions> options;
        std::vector<double> eastings, northings, elevations;
        for (const auto &job : m_jobs) {
            options.push_back(job.options);
            eastings.push_back(job.easting);
            northings.push_back(job.northing);
            elevations.push_back(job.elevation);
        }
        try {
            const auto results = pytem::TemSolver::invertSci(
                options, eastings, northings, elevations, std::min(hardware, requested),
                [this](std::size_t s, int iteration, double rms, const std::string &detail) {
                    emit progress(m_jobs[s].index, iteration, rms, QString::fromStdString(detail));
                },
                [this]() { return m_cancelled.load(std::memory_order_relaxed); });
            for (std::size_t s = 0; s < results.size(); ++s)
                emit resultReady(m_jobs[s].index, results[s]);
        } catch (const std::exception &error) {
            if (!m_cancelled.load(std::memory_order_relaxed))
                for (const auto &job : m_jobs)
                    emit failed(job.index, QString::fromUtf8(error.what()));
        }
        emit allFinished();
        return;
    }

    // The pyTEM joint scheme runs one independent inversion per worker, as the
    // notebook does. The normal path keeps every objective function independent
    // but advances all active inversions through shared batch stages.
    const bool joint = m_jobs.front().options.pytemJoint;
    if (!joint) try {
        std::vector<pytem::InversionOptions> options;
        options.reserve(m_jobs.size());
        for (const auto &job : m_jobs)
            options.push_back(job.options);
        const auto results = pytem::TemSolver::invertIndependentBatch(
            options, std::min(hardware, requested),
            [this](std::size_t batchIndex, int iteration, double rms,
                   const std::string &detail) {
                if (batchIndex >= m_jobs.size())
                    return;
                emit progress(m_jobs[batchIndex].index, iteration, rms,
                              QString::fromStdString(detail));
            },
            [this]() {
                return m_cancelled.load(std::memory_order_relaxed);
            },
            [this, &delivered, &deliveredMutex](
                std::size_t batchIndex,
                const pytem::InversionResult &result) {
                if (batchIndex >= m_jobs.size())
                    return;
                {
                    std::lock_guard<std::mutex> lock(deliveredMutex);
                    delivered[batchIndex] = true;
                }
                emit resultReady(m_jobs[batchIndex].index, result);
            });
        (void) results;
        emit allFinished();
        return;
    } catch (const std::exception &error) {
        if (m_cancelled.load(std::memory_order_relaxed)) {
            emit allFinished();
            return;
        }
        // A single unusual dataset must not prevent the remaining independent
        // inversions from completing. Retry through the proven per-job path.
        for (std::size_t index = 0; index < m_jobs.size(); ++index) {
            bool alreadyDelivered = false;
            {
                std::lock_guard<std::mutex> lock(deliveredMutex);
                alreadyDelivered = delivered[index];
            }
            if (!alreadyDelivered)
                emit progress(m_jobs[index].index, 0, 0.0,
                    QString("Batch path unavailable (%1); using independent fallback")
                        .arg(QString::fromUtf8(error.what())));
        }
    }

    const unsigned threadsPerJob = std::max(1u, hardware / workerCount);
    std::atomic_size_t next{0};
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (unsigned worker = 0; worker < workerCount; ++worker) {
        workers.emplace_back([&, threadsPerJob]() {
            while (!m_cancelled.load(std::memory_order_relaxed)) {
                const std::size_t jobPosition = next.fetch_add(1);
                if (jobPosition >= m_jobs.size())
                    break;
                {
                    std::lock_guard<std::mutex> lock(deliveredMutex);
                    if (delivered[jobPosition])
                        continue;
                }
                auto job = m_jobs[jobPosition];
                job.options.model.maxThreads = threadsPerJob;
                for (auto &dataSet : job.options.additionalDataSets)
                    dataSet.model.maxThreads = threadsPerJob;
                try {
                    const auto invert = job.options.pytemJoint
                        ? &pytem::TemSolver::invertJoint : &pytem::TemSolver::invert;
                    const auto result = invert(
                        job.options,
                        [this, index = job.index](int iteration, double rms,
                                                   const std::string &detail) {
                            emit progress(index, iteration, rms,
                                          QString::fromStdString(detail));
                        },
                        [this]() { return m_cancelled.load(std::memory_order_relaxed); });
                    emit resultReady(job.index, result);
                } catch (const std::exception &error) {
                    emit failed(job.index, QString::fromUtf8(error.what()));
                }
            }
        });
    }
    for (auto &worker : workers)
        worker.join();
    emit allFinished();
}
