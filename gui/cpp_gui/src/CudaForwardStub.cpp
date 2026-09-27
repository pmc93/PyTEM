#include "CudaForward.h"

namespace pytem {

bool cudaForwardAvailable(std::string *reason)
{
    if (reason)
        *reason = "CUDA support was not compiled into this build";
    return false;
}

bool cudaStepResponse(const ForwardModel &, const std::vector<double> &,
                      std::vector<double> &, std::string *error)
{
    if (error)
        *error = "CUDA support was not compiled into this build";
    return false;
}

} // namespace pytem
