// ============================================================================
// torch_engine.h  —  DA3: libtorch-only TorchScript engine wrapper.
//
// Same shape as Peter's MegaFlow TorchEngine, trimmed for DA3:
//   * namespace da3
//   * disk-load ctor only (single 350MB .pt, in-memory load not needed)
//   * NO fp32 cast on outputs (model returns fp32 already; matches Python)
// ============================================================================
#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <map>
#include <stdexcept>

namespace da3 {

class TorchEngine {
public:
    // Load a traced .pt from `path` onto CUDA (device 0 by default; the caller's
    // CUDAStreamGuard in run() steers actual execution to the right device).
    TorchEngine(const std::string& path, const std::string& name,
                std::vector<std::string> inNames,
                std::vector<std::string> outNames,
                bool autocastFp16 = false);

    // Maps inputs by name -> positional args, forwards, maps outputs back to
    // names. Ordered on the caller's stream.
    std::map<std::string, torch::Tensor>
    run(const std::map<std::string, torch::Tensor>& inputs, cudaStream_t stream);

    const std::vector<std::string>& inputNames()  const { return inNames_; }
    const std::vector<std::string>& outputNames() const { return outNames_; }
    const std::string& name() const { return name_; }

private:
    std::string name_;
    torch::jit::script::Module module_;
    std::vector<std::string> inNames_;
    std::vector<std::string> outNames_;
    bool autocastFp16_;
};

} // namespace da3
