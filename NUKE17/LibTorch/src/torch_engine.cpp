// ============================================================================
// torch_engine.cpp  —  DA3 TorchEngine impl (see header).
// Identical pattern to MegaFlow's TorchEngine, sans the in-memory ctor and the
// "cast outputs to fp32" line (DA3 model returns fp32 already; we don't want
// to second-guess the trace).
// ============================================================================
#include "torch_engine.h"

#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace da3 {

TorchEngine::TorchEngine(const std::string& path, const std::string& name,
                         std::vector<std::string> inNames,
                         std::vector<std::string> outNames,
                         bool autocastFp16)
    : name_(name),
      inNames_(std::move(inNames)),
      outNames_(std::move(outNames)),
      autocastFp16_(autocastFp16)
{
    try {
        module_ = torch::jit::load(path, torch::kCUDA);
    } catch (const std::exception& e) {
        throw std::runtime_error("TorchEngine[" + name_ + "] load failed from '" +
                                 path + "': " + e.what());
    }
    module_.eval();
}

std::map<std::string, torch::Tensor>
TorchEngine::run(const std::map<std::string, torch::Tensor>& inputs, cudaStream_t stream)
{
    torch::NoGradGuard ng;
    auto s = c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device());
    c10::cuda::CUDAStreamGuard guard(s);

    std::vector<torch::jit::IValue> args;
    args.reserve(inNames_.size());
    for (const auto& n : inNames_) {
        auto it = inputs.find(n);
        if (it == inputs.end())
            throw std::runtime_error("TorchEngine[" + name_ + "] missing input '" + n + "'");
        args.emplace_back(it->second);
    }

    torch::jit::IValue out;
    if (autocastFp16_) {
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        out = module_.forward(args);
        at::autocast::set_autocast_enabled(at::kCUDA, false);
        at::autocast::clear_cache();
    } else {
        out = module_.forward(args);
    }

    std::vector<torch::Tensor> outs;
    if (out.isTensor()) {
        outs.push_back(out.toTensor());
    } else if (out.isTuple()) {
        for (const auto& e : out.toTuple()->elements())
            outs.push_back(e.toTensor());
    } else if (out.isTensorList()) {
        for (const auto& e : out.toTensorList())
            outs.push_back(e);
    } else if (out.isList()) {
        for (const auto& e : out.toList())
            outs.push_back(e.get().toTensor());
    } else {
        throw std::runtime_error("TorchEngine[" + name_ + "] unexpected output type: " +
                                 out.tagKind());
    }
    if (outs.size() != outNames_.size())
        throw std::runtime_error("TorchEngine[" + name_ + "] output count " +
            std::to_string(outs.size()) + " != names " + std::to_string(outNames_.size()));

    std::map<std::string, torch::Tensor> result;
    for (size_t i = 0; i < outNames_.size(); ++i)
        result[outNames_[i]] = outs[i];
    return result;
}

} // namespace da3
