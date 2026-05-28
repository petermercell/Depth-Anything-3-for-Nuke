// ============================================================================
// LibTorchDepthAnything3.cpp  —  per-frame DA3 inference Iop (libtorch).
//
// Channel routing (unchanged):
//   * Depth Only OFF (default): R,G,B passthrough; A = depth
//   * Depth Only ON:            R = G = B = depth (BW preview); A = depth
//   * Invert Depth:             d -> 1-d
//   * Input is Linear:          linear -> sRGB OETF before feeding the model
//
// Internal resize (like LibTorchCorridorKey): plate is resized to the model's
// traced (W,H) on the GPU, run, and the depth resized back to plate res.
// Resolution comes from a preset dropdown (2058x1092 / 3080x1624 / Custom).
// If the plate already matches the model res, both resizes are skipped.
//
// Cache key: frame + plate dims + model dims + linear flag. Depth Only / Invert
// are read-side (applied at scanline time), so they don't invalidate the cache.
// ============================================================================
#include "LibTorchDepthAnything3.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/nn/functional.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace Fn = torch::nn::functional;

// --- linear -> sRGB OETF ------------------------------------------------------
static inline float lin2srgb(float c) {
    c = std::max(c, 0.0f);
    return (c <= 0.0031308f) ? c * 12.92f
                             : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// ---------------------------------------------------------------------------
// ctor / dtor
// ---------------------------------------------------------------------------
LibTorchDepthAnything3::LibTorchDepthAnything3(Node* node)
    : Iop(node),
      modelPath_(""),
      resPreset_(RES_2058x1092),
      modelWidth_(2058),
      modelHeight_(1092),
      depthOnly_(false),
      invertDepth_(false),
      inputIsLinear_(true),
      halfPrecision_(false),
      gpuDevice_(0)
{}

LibTorchDepthAnything3::~LibTorchDepthAnything3() {
    engine_.reset();
}

const char* LibTorchDepthAnything3::node_help() const {
    return "Depth Anything 3 (monocular depth) via libtorch.\n"
           "Plate can be ANY resolution — the plugin resizes it internally to "
           "the model's traced size, runs inference, and resizes the depth back.\n"
           "Set 'Model Resolution' to match the .pt you traced "
           "(2058x1092 or 3080x1624; pick Custom for anything else).";
}

bool LibTorchDepthAnything3::test_input(int /*n*/, Op* op) const {
    return dynamic_cast<Iop*>(op) != nullptr;
}

const char* LibTorchDepthAnything3::input_label(int /*n*/, char* /*buf*/) const {
    return "plate";
}

// ---------------------------------------------------------------------------
// resolve the preset (or custom fields) into a concrete model (w,h)
// ---------------------------------------------------------------------------
void LibTorchDepthAnything3::effectiveModelWH(int& w, int& h) const {
    switch (resPreset_) {
        case RES_2058x1092: w = 2058; h = 1092; break;
        case RES_3080x1624: w = 3080; h = 1624; break;
        case RES_CUSTOM:    w = modelWidth_; h = modelHeight_; break;
        default:            w = 2058; h = 1092; break;
    }
}

// ---------------------------------------------------------------------------
// knobs
// ---------------------------------------------------------------------------
void LibTorchDepthAnything3::knobs(Knob_Callback f) {
    File_knob(f, &modelPath_, "model_path", "model (.pt)");
    Tooltip(f, "Path to the .pt traced by nuke_da3_v4.py "
               "(e.g. DepthAnything3_mono_large_2058x1092_fp32.pt).");

    static const char* const presets[] = {
        "2058 x 1092", "3080 x 1624", "Custom", nullptr
    };
    Enumeration_knob(f, &resPreset_, presets, "res_preset", "Model Resolution");
    Tooltip(f, "Resolution your .pt was traced at. The plate is resized to this "
               "internally and the depth is resized back. Pick Custom to type "
               "your own W/H.");

    Int_knob(f, &modelWidth_, "model_width", "Model Width");
    SetRange(f, 28, 8192);
    Tooltip(f, "Used only when Model Resolution = Custom. Must be a multiple of 14 "
               "and match the .pt trace.");

    Int_knob(f, &modelHeight_, "model_height", "Model Height");
    SetRange(f, 28, 8192);
    Tooltip(f, "Used only when Model Resolution = Custom. Must be a multiple of 14 "
               "and match the .pt trace.");

    Int_knob(f, &gpuDevice_, "gpu_device", "GPU device");
    SetRange(f, 0, 8);
    Tooltip(f, "CUDA device index. Engine reloads when changed.");

    Bool_knob(f, &halfPrecision_, "half_precision", "half precision (fp16)");
    Tooltip(f, "Enable if your .pt was traced with --half. Input is cast to fp16 "
               "before forward; output is cast back to fp32.");

    Divider(f, "Output");

    Bool_knob(f, &depthOnly_, "depth_only", "Depth Only (BW)");
    Tooltip(f, "Output depth as greyscale RGB (preview). Alpha still gets depth.");

    Bool_knob(f, &invertDepth_, "invert_depth", "Invert Depth");
    Tooltip(f, "Flip near/far. Default: far = 1, near = 0.");

    Bool_knob(f, &inputIsLinear_, "input_is_linear", "Input is Linear");
    Tooltip(f, "Convert linear -> sRGB before inference (Nuke default is linear).");

    // ---- credits ----
    Divider(f, "");
    Text_knob(f, "",
              "LibTorchDepthAnything3 for Nuke by Peter Mercell, 2026\n"
              "www.petermercell.com\n"
              "\n"
              "Depth Anything 3 by Bingyi Kang et al.\n"
              "github.com/DepthAnything/Depth-Anything-3\n"
              "Licensed under Apache-2.0.");
}

int LibTorchDepthAnything3::knob_changed(Knob* k) {
    if (!k) return Iop::knob_changed(k);
    const std::string n = k->name();

    // Grey the Custom W/H fields unless Custom is selected. (Also runs on panel
    // open so the initial state is right.)
    if (n == "res_preset" || n == "showPanel") {
        const bool custom = (resPreset_ == RES_CUSTOM);
        if (Knob* wK = knob("model_width"))  wK->enable(custom);
        if (Knob* hK = knob("model_height")) hK->enable(custom);
    }

    // Inference-affecting knobs -> invalidate cache (and reload engine if needed).
    if (n == "model_path"   || n == "gpu_device"  || n == "half_precision" ||
        n == "input_is_linear" || n == "res_preset" ||
        n == "model_width"  || n == "model_height")
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cachedFrame_ = INT_MIN;
        if (n == "model_path" || n == "gpu_device" || n == "half_precision") {
            engine_.reset();
            loadedFrom_.clear();
            loadedDevice_ = -1;
        }
        return 1;
    }
    return Iop::knob_changed(k);
}

// ---------------------------------------------------------------------------
// validate / request
// ---------------------------------------------------------------------------
void LibTorchDepthAnything3::_validate(bool for_real) {
    if (!input(0)) {
        set_out_channels(Mask_None);
        return;
    }
    copy_info();
    fX_ = info_.x();
    fY_ = info_.y();
    fW_ = info_.w();
    fH_ = info_.h();
    info_.turn_on(Mask_RGBA);
    set_out_channels(Mask_RGBA);
}

void LibTorchDepthAnything3::_request(int /*x*/, int /*y*/, int /*r*/, int /*t*/,
                                      ChannelMask /*m*/, int count) {
    // We always need the WHOLE input frame to run inference, regardless of the
    // ROI the downstream node asked for.
    if (input(0))
        input(0)->request(fX_, fY_, fX_ + fW_, fY_ + fH_, Mask_RGB, count);
}

// ---------------------------------------------------------------------------
// engine load (lazy)
// ---------------------------------------------------------------------------
void LibTorchDepthAnything3::ensureEngine() {
    const std::string mp = modelPath_ ? modelPath_ : "";
    if (mp.empty()) {
        error("LibTorchDepthAnything3: model path not set");
        return;
    }

    const bool sameFile   = (loadedFrom_   == mp);
    const bool sameHalf   = (loadedHalf_   == halfPrecision_);
    const bool sameDevice = (loadedDevice_ == gpuDevice_);
    if (engine_ && sameFile && sameHalf && sameDevice) return;

    engine_.reset();
    try {
        stream_ = c10::cuda::getCurrentCUDAStream(gpuDevice_).stream();
        engine_ = std::make_unique<da3::TorchEngine>(
            mp, "da3",
            std::vector<std::string>{"image"},
            std::vector<std::string>{"depth"},
            /*autocastFp16*/ false
        );
        loadedFrom_   = mp;
        loadedHalf_   = halfPrecision_;
        loadedDevice_ = gpuDevice_;
    } catch (const std::exception& e) {
        engine_.reset();
        loadedFrom_.clear();
        loadedDevice_ = -1;
        error("LibTorchDepthAnything3: model load failed: %s", e.what());
    }
}

// ---------------------------------------------------------------------------
// per-frame inference (mutex-protected; first scanline of a new frame triggers)
// ---------------------------------------------------------------------------
void LibTorchDepthAnything3::computeFrame(int frame) {
    std::lock_guard<std::mutex> lock(cacheMutex_);

    int mw, mh;
    effectiveModelWH(mw, mh);

    if (cachedFrame_ == frame &&
        cachedW_ == fW_ && cachedH_ == fH_ &&
        cachedModelW_ == mw && cachedModelH_ == mh &&
        cachedLinearFlag_ == inputIsLinear_ &&
        !cachedDepth_.empty())
        return;

    ensureEngine();
    if (!engine_) return;

    // ---- pull entire plate as a top-down (1,3,plateH,plateW) fp32 host tensor ----
    auto host = torch::empty({1, 3, fH_, fW_}, torch::kFloat32);
    auto a = host.accessor<float, 4>();

    Row inRow(fX_, fX_ + fW_);
    for (int ty = 0; ty < fH_; ++ty) {
        if (aborted()) return;
        const int ny = fY_ + fH_ - 1 - ty;  // tensor top (ty=0) <- Nuke top (highest y)
        input(0)->get(ny, fX_, fX_ + fW_, Mask_RGB, inRow);
        const float* R = inRow[Chan_Red];
        const float* G = inRow[Chan_Green];
        const float* B = inRow[Chan_Blue];

        for (int tx = 0; tx < fW_; ++tx) {
            float rr = R[fX_ + tx];
            float gg = G[fX_ + tx];
            float bb = B[fX_ + tx];
            if (inputIsLinear_) { rr = lin2srgb(rr); gg = lin2srgb(gg); bb = lin2srgb(bb); }
            a[0][0][ty][tx] = std::clamp(rr, 0.0f, 1.0f);
            a[0][1][ty][tx] = std::clamp(gg, 0.0f, 1.0f);
            a[0][2][ty][tx] = std::clamp(bb, 0.0f, 1.0f);
        }
    }

    // ---- resize to model res, run forward, resize depth back ----
    try {
        torch::Device dev(torch::kCUDA, gpuDevice_);
        torch::Tensor in_cuda = host.to(dev, /*non_blocking*/ true);
        if (halfPrecision_) in_cuda = in_cuda.to(torch::kHalf);

        const bool needResize = (fW_ != mw || fH_ != mh);
        torch::Tensor in_model = needResize
            ? Fn::interpolate(in_cuda,
                  Fn::InterpolateFuncOptions()
                      .size(std::vector<int64_t>{(int64_t)mh, (int64_t)mw})
                      .mode(torch::kBilinear).align_corners(false))
            : in_cuda;

        auto out = engine_->run({{"image", in_model}}, stream_);
        torch::Tensor depth = out["depth"];   // (1,1,mh,mw)
        if (depth.dtype() != torch::kFloat32) depth = depth.to(torch::kFloat32);

        // resize depth back to plate res
        torch::Tensor depth_full = needResize
            ? Fn::interpolate(depth,
                  Fn::InterpolateFuncOptions()
                      .size(std::vector<int64_t>{(int64_t)fH_, (int64_t)fW_})
                      .mode(torch::kBilinear).align_corners(false))
            : depth;

        depth_full = depth_full.to(torch::kCPU).contiguous();   // (1,1,fH,fW)

        if (depth_full.dim() != 4 || depth_full.size(2) != fH_ || depth_full.size(3) != fW_) {
            error("LibTorchDepthAnything3: unexpected output shape after resize "
                  "(%ld,%ld,%ld,%ld), expected (1,1,%d,%d).",
                  (long)depth_full.size(0), (long)depth_full.size(1),
                  (long)depth_full.size(2), (long)depth_full.size(3), fH_, fW_);
            return;
        }

        const size_t n = (size_t)fH_ * (size_t)fW_;
        cachedDepth_.assign(depth_full.data_ptr<float>(), depth_full.data_ptr<float>() + n);
        cachedW_           = fW_;
        cachedH_           = fH_;
        cachedModelW_      = mw;
        cachedModelH_      = mh;
        cachedFrame_       = frame;
        cachedLinearFlag_  = inputIsLinear_;
    } catch (const std::exception& e) {
        error("LibTorchDepthAnything3: inference failed: %s. Most common cause: "
              "'Model Resolution' (%dx%d) doesn't match the .pt trace size, or the "
              ".pt was traced fp16 but 'half precision' is off (or vice versa).",
              e.what(), mw, mh);
    }
}

// ---------------------------------------------------------------------------
// per-scanline output
// ---------------------------------------------------------------------------
void LibTorchDepthAnything3::engine(int y, int x, int r, ChannelMask m, Row& row) {
    const int frame = (int)outputContext().frame();
    computeFrame(frame);

    // Always read input for passthrough channels.
    Row in(x, r);
    input(0)->get(y, x, r, m, in);

    const bool cacheOk = (cachedFrame_ == frame &&
                          cachedW_ == fW_ && cachedH_ == fH_ &&
                          !cachedDepth_.empty());

    const int  ty = (fY_ + fH_ - 1) - y;   // nuke y -> top-down tensor row
    const bool yInside = (ty >= 0 && ty < fH_);
    const float* depthRow = (cacheOk && yInside)
                            ? &cachedDepth_[(size_t)ty * (size_t)fW_]
                            : nullptr;

    foreach(ch, m) {
        float* out = row.writable(ch);
        const float* inp = in[ch];

        const bool isAlpha   = (ch == Chan_Alpha);
        const bool isColorBW = (depthOnly_ &&
                                (ch == Chan_Red || ch == Chan_Green || ch == Chan_Blue));
        const bool writeDepth = (isAlpha || isColorBW) && depthRow;

        if (!writeDepth) {
            std::copy(inp + x, inp + r, out + x);
            continue;
        }

        for (int px = x; px < r; ++px) {
            const int tx = px - fX_;
            if (tx < 0 || tx >= fW_) {
                out[px] = inp[px];
            } else {
                float d = depthRow[tx];
                if (invertDepth_) d = 1.0f - d;
                out[px] = d;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------
static Iop* build(Node* node) { return new LibTorchDepthAnything3(node); }
const Iop::Description LibTorchDepthAnything3::description(
    "LibTorchDepthAnything3",
    "AI/LibTorchDepthAnything3",
    build
);
