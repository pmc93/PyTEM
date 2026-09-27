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
    static JacobianCacheResult loadOrCompute(
        const std::string &signature,
        const std::string &directory,
        const std::function<DenseMatrix()> &compute);

    static void clearMemory();
};

} // namespace pytem
