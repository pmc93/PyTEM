#pragma once

#include <functional>
#include <string>
#include <vector>

namespace pytem {

using DenseMatrix = std::vector<std::vector<double>>;

enum class JacobianCacheSource {
    Computed,
    Memory,
    Disk,
};

struct JacobianCacheResult {
    DenseMatrix matrix;
    JacobianCacheSource source = JacobianCacheSource::Computed;
};

class JacobianCache final
{
public:
    // Loads an existing memory/disk entry without calculating a miss.
    static bool tryLoad(const std::string &signature,
                        const std::string &directory,
                        JacobianCacheResult &result);

    static JacobianCacheResult loadOrCompute(
        const std::string &signature,
        const std::string &directory,
        const std::function<DenseMatrix()> &compute);

    static void clearMemory();
};

} // namespace pytem
