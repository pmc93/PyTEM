#include "InversionWorker.h"

#include <QString>
#include <algorithm>
#include <atomic>
#include <exception>
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
                auto job = m_jobs[jobPosition];
                job.options.model.maxThreads = threadsPerJob;
                for (auto &dataSet : job.options.additionalDataSets)
                    dataSet.model.maxThreads = threadsPerJob;
                try {
                    const auto result = pytem::TemSolver::invert(
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
