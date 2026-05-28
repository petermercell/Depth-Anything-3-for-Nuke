// ============================================================================
// LibTorchDepthAnything3.h  —  per-frame DA3 inference in Nuke via libtorch.
//
// Same UX as TRT_DepthAnything3 (knobs: Depth Only / Invert / Input is Linear)
// but the inference backend is the .pt traced by nuke_da3_v4.py instead of a
// trtexec engine. Advantage: cross-GPU portability + no engine build step;
// the trade-off vs TRT is throughput, not quality.
//
// INTERNAL RESIZE (like LibTorchCorridorKey): the plate can be ANY resolution.
// The plugin bilinear-resizes it to the model's traced resolution, runs the
// forward, then resizes the depth back to plate res. DA3 traces are non-square,
// so resolution is a (width, height) pair, exposed as a preset dropdown
// (2058x1092 / 3080x1624) plus a Custom option with explicit W/H fields.
//
// Architecture:
//   * single input (plate), single output (RGBA)
//   * per-frame inference: first engine() call for a new frame takes a mutex,
//     pulls the entire input as a (1,3,plate_H,plate_W) tensor, resizes to
//     (1,3,model_H,model_W), runs forward, resizes depth back to plate res,
//     caches the (plate_H,plate_W) depth top-down. Subsequent scanlines read it.
//
// Why a Solve button isn't needed (unlike MegaFlow): DA3 is monocular. Every
// frame is independent. Standard Iop semantics fit perfectly.
// ============================================================================
#pragma once

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/Knob.h"
#include "DDImage/Format.h"
#include "DDImage/OutputContext.h"

#include <torch/torch.h>
#include <cuda_runtime.h>

#include <climits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "torch_engine.h"

using namespace DD::Image;

class LibTorchDepthAnything3 : public Iop {
public:
    LibTorchDepthAnything3(Node* node);
    ~LibTorchDepthAnything3() override;

    const char* Class() const override { return description.name; }
    const char* node_help() const override;

    int  minimum_inputs() const override { return 1; }
    int  maximum_inputs() const override { return 1; }
    bool test_input(int n, Op* op) const override;
    const char* input_label(int n, char* buf) const override;

    void knobs(Knob_Callback f) override;
    int  knob_changed(Knob* k) override;

    void _validate(bool for_real) override;
    void _request(int x, int y, int r, int t, ChannelMask m, int count) override;
    void engine(int y, int x, int r, ChannelMask m, Row& row) override;

    static const Iop::Description description;

private:
    enum ResPreset {
        RES_2058x1092 = 0,
        RES_3080x1624 = 1,
        RES_CUSTOM    = 2,
    };

    // ---- knobs ----
    const char* modelPath_;     // path to the .pt traced by nuke_da3_v4.py
    int  resPreset_;            // ResPreset: which trace resolution the .pt is
    int  modelWidth_;           // used only when resPreset_ == RES_CUSTOM
    int  modelHeight_;          // used only when resPreset_ == RES_CUSTOM
    bool depthOnly_;            // true: RGB = depth (BW preview), A = depth
    bool invertDepth_;          // 1 - d (near = 1 -> near = 0)
    bool inputIsLinear_;        // apply linear -> sRGB OETF before feeding
    bool halfPrecision_;        // model was traced with --half; feed fp16
    int  gpuDevice_;            // CUDA device index

    // ---- engine state (lazy-loaded on first compute) ----
    std::unique_ptr<da3::TorchEngine> engine_;
    cudaStream_t stream_ = nullptr;
    std::string loadedFrom_;
    bool loadedHalf_ = false;
    int  loadedDevice_ = -1;

    // ---- plate geometry (from input 0) ----
    int fX_ = 0, fY_ = 0, fW_ = 0, fH_ = 0;

    // ---- per-frame cache ----
    std::mutex cacheMutex_;
    int  cachedFrame_ = INT_MIN;
    int  cachedW_ = 0, cachedH_ = 0;        // plate dims the cache was built at
    int  cachedModelW_ = 0, cachedModelH_ = 0;  // model dims used (cache key)
    bool cachedLinearFlag_ = false;
    std::vector<float> cachedDepth_;        // top-down row-major, size plate W*H

    // ---- helpers ----
    void effectiveModelWH(int& w, int& h) const;  // resolve preset -> (w,h)
    void ensureEngine();
    void computeFrame(int frame);
};
