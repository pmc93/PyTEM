#pragma once

#include <string>
#include <vector>

namespace pytem {

struct ForwardModel;

// Runtime availability includes both CUDA build support and a usable device.
bool cudaForwardAvailable(std::string *reason = nullptr);

// Calculates the DLF circular-loop step response on the GPU. Returns false
// when CUDA cannot be used so the caller can transparently fall back to CPU.
bool cudaStepResponse(const ForwardModel &model,
                      const std::vector<double> &evaluationTimes,
                      std::vector<double> &response,
                      std::string *error = nullptr);

} // namespace pytem
