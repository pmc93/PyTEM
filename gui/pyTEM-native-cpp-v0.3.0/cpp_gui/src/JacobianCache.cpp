#include "JacobianCache.h"

#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace pytem {
namespace {

constexpr char cacheMagic[] = "PYTEMJAC2";

struct Entry {
    std::mutex mutex;
    std::condition_variable ready;
    bool computing = false;
    bool available = false;
    DenseMatrix matrix;
};

std::mutex entriesMutex;
std::unordered_map<std::string, std::shared_ptr<Entry>> entries;

std::uint64_t fnv1a(const std::string &value)
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::filesystem::path cachePath(const std::string &directory,
                                const std::string &signature)
{
    std::ostringstream name;
    name << "first_jacobian_" << std::hex << std::setfill('0')
         << std::setw(16) << fnv1a(signature) << ".bin";
    return std::filesystem::path(directory) / name.str();
}

template<typename Value>
bool readValue(std::ifstream &stream, Value &value)
{
    return static_cast<bool>(stream.read(reinterpret_cast<char *>(&value), sizeof(Value)));
}

bool readMatrix(const std::filesystem::path &path,
                const std::string &signature,
                DenseMatrix &matrix)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        return false;
    char magic[sizeof(cacheMagic)]{};
    if (!stream.read(magic, sizeof(magic))
        || std::memcmp(magic, cacheMagic, sizeof(cacheMagic)) != 0)
        return false;
    std::uint64_t signatureSize = 0, rows = 0, columns = 0;
    if (!readValue(stream, signatureSize) || !readValue(stream, rows) || !readValue(stream, columns))
        return false;
    if (signatureSize != signature.size() || rows == 0 || columns == 0
        || rows > 100000 || columns > 10000
        || columns > 100000000ULL / rows)
        return false;
    std::string storedSignature(static_cast<std::size_t>(signatureSize), '\0');
    if (!stream.read(storedSignature.data(), static_cast<std::streamsize>(storedSignature.size()))
        || storedSignature != signature)
        return false;
    DenseMatrix loaded(static_cast<std::size_t>(rows),
                       std::vector<double>(static_cast<std::size_t>(columns)));
    for (auto &row : loaded) {
        if (!stream.read(reinterpret_cast<char *>(row.data()),
                         static_cast<std::streamsize>(row.size() * sizeof(double))))
            return false;
        for (double value : row)
            if (!std::isfinite(value))
                return false;
    }
    char extra = 0;
    if (stream.read(&extra, 1))
        return false;
    matrix = std::move(loaded);
    return true;
}

void writeMatrix(const std::filesystem::path &path,
                 const std::string &signature,
                 const DenseMatrix &matrix)
{
    if (matrix.empty() || matrix.front().empty())
        return;
    const std::size_t columns = matrix.front().size();
    for (const auto &row : matrix)
        if (row.size() != columns)
            throw std::runtime_error("Cannot cache a ragged Jacobian matrix");

    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error)
        return;
    std::ostringstream suffix;
    suffix << ".tmp." << std::this_thread::get_id();
    const auto temporary = path.string() + suffix.str();
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream)
        return;
    const std::uint64_t signatureSize = signature.size();
    const std::uint64_t rows = matrix.size();
    const std::uint64_t columnCount = columns;
    stream.write(cacheMagic, sizeof(cacheMagic));
    stream.write(reinterpret_cast<const char *>(&signatureSize), sizeof(signatureSize));
    stream.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
    stream.write(reinterpret_cast<const char *>(&columnCount), sizeof(columnCount));
    stream.write(signature.data(), static_cast<std::streamsize>(signature.size()));
    for (const auto &row : matrix)
        stream.write(reinterpret_cast<const char *>(row.data()),
                     static_cast<std::streamsize>(row.size() * sizeof(double)));
    stream.close();
    if (!stream) {
        std::filesystem::remove(temporary, error);
        return;
    }
    error.clear();
    std::filesystem::remove(path, error);
    error.clear();
    std::filesystem::rename(temporary, path, error);
    if (error)
        std::filesystem::remove(temporary, error);
}

} // namespace

JacobianCacheResult JacobianCache::loadOrCompute(
    const std::string &signature,
    const std::string &directory,
    const std::function<DenseMatrix()> &compute)
{
    std::shared_ptr<Entry> entry;
    {
        std::lock_guard<std::mutex> lock(entriesMutex);
        auto &slot = entries[signature];
        if (!slot)
            slot = std::make_shared<Entry>();
        entry = slot;
    }

    std::unique_lock<std::mutex> lock(entry->mutex);
    if (entry->available)
        return {entry->matrix, JacobianCacheSource::Memory};
    while (entry->computing) {
        entry->ready.wait(lock);
        if (entry->available)
            return {entry->matrix, JacobianCacheSource::Memory};
    }
    entry->computing = true;
    lock.unlock();

    try {
        DenseMatrix matrix;
        JacobianCacheSource source = JacobianCacheSource::Computed;
        if (!directory.empty() && readMatrix(cachePath(directory, signature), signature, matrix)) {
            source = JacobianCacheSource::Disk;
        } else {
            matrix = compute();
            if (!directory.empty())
                writeMatrix(cachePath(directory, signature), signature, matrix);
        }
        lock.lock();
        entry->matrix = matrix;
        entry->available = true;
        entry->computing = false;
        lock.unlock();
        entry->ready.notify_all();
        return {std::move(matrix), source};
    } catch (...) {
        lock.lock();
        entry->computing = false;
        lock.unlock();
        entry->ready.notify_all();
        throw;
    }
}

void JacobianCache::clearMemory()
{
    std::lock_guard<std::mutex> lock(entriesMutex);
    entries.clear();
}

} // namespace pytem
