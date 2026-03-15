/*
 * TRT_DepthAnything3 - TensorRT Depth Anything 3 Node for Nuke
 *
 * Model details:
 *   Input:  "input"  float32[1,3,H,W]  (NCHW, RGB [0,1])
 *   Output: "depth"  float32[1,1,H,W]  (NCHW, depth [0,1])
 *   ImageNet normalization is baked into the ONNX/engine.
 *   Depth min-max normalization is baked in — output is [0,1].
 *
 * Built for: TensorRT 10.9.0 (enqueueV3 + named tensors)
 *            Nuke NDK (14.1+)
 *            CUDA 12.8
 *            Engine embedded in binary — no external files needed.
 *
 * Author: Peter Mercell
 * Website: petermercell.com
 *
 * Depth Anything 3 by Bingyi Kang et al.
 * https://github.com/DepthAnything/Depth-Anything-3
 * Licensed under Apache-2.0
 */

#include "DDImage/Iop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/Thread.h"
#include "DDImage/Format.h"
#include "DDImage/Interest.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <iostream>
#include <mutex>

using namespace DD::Image;

// ---------------------------------------------------------------------------
// Embedded engine data (linked via objcopy from engine.bin)
// ---------------------------------------------------------------------------
extern "C" {
    extern const char _binary_engine_bin_start[];
    extern const char _binary_engine_bin_end[];
}

// ---------------------------------------------------------------------------
// TensorRT logger
// ---------------------------------------------------------------------------
class TRTLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT_DepthAnything3] " << msg << std::endl;
    }
};

static TRTLogger gLogger;

// ---------------------------------------------------------------------------
// Tensor names (from ONNX export)
// ---------------------------------------------------------------------------
static const char* kInputTensorName  = "input";
static const char* kOutputTensorName = "depth";

// ---------------------------------------------------------------------------
// Fixed model resolution — must match the ONNX/engine build
// Change these if you build a different resolution engine.
// ---------------------------------------------------------------------------
static const int kModelW = 2058;
static const int kModelH = 1092;

// ---------------------------------------------------------------------------
// CUDA error check
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            error("CUDA error: %s at %s:%d", cudaGetErrorString(err),           \
                  __FILE__, __LINE__);                                           \
            return;                                                             \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Bilinear resize (CPU, planar)
// ---------------------------------------------------------------------------
static void bilinearResize(const float* src, int srcW, int srcH,
                           float* dst, int dstW, int dstH,
                           int channels)
{
    for (int c = 0; c < channels; ++c)
    {
        const float* srcC = src + c * srcW * srcH;
        float* dstC       = dst + c * dstW * dstH;

        for (int dy = 0; dy < dstH; ++dy)
        {
            float sy = (dy + 0.5f) * srcH / (float)dstH - 0.5f;
            int y0   = (int)std::floor(sy);
            int y1   = y0 + 1;
            float fy = sy - y0;

            y0 = std::max(0, std::min(y0, srcH - 1));
            y1 = std::max(0, std::min(y1, srcH - 1));

            for (int dx = 0; dx < dstW; ++dx)
            {
                float sx = (dx + 0.5f) * srcW / (float)dstW - 0.5f;
                int x0   = (int)std::floor(sx);
                int x1   = x0 + 1;
                float fx = sx - x0;

                x0 = std::max(0, std::min(x0, srcW - 1));
                x1 = std::max(0, std::min(x1, srcW - 1));

                float v00 = srcC[y0 * srcW + x0];
                float v10 = srcC[y0 * srcW + x1];
                float v01 = srcC[y1 * srcW + x0];
                float v11 = srcC[y1 * srcW + x1];

                dstC[dy * dstW + dx] =
                    (1 - fy) * ((1 - fx) * v00 + fx * v10)
                  +      fy  * ((1 - fx) * v01 + fx * v11);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// sRGB <-> linear conversion
// ---------------------------------------------------------------------------
static inline float srgbToLinear(float v)
{
    return (v <= 0.04045f) ? v / 12.92f
                           : std::pow((v + 0.055f) / 1.055f, 2.4f);
}

static inline float linearToSrgb(float v)
{
    return (v <= 0.0031308f) ? v * 12.92f
                             : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

// ---------------------------------------------------------------------------
// TRT_DepthAnything3 node
// ---------------------------------------------------------------------------
class TRT_DepthAnything3 : public Iop
{

public:

    TRT_DepthAnything3(Node* node);
    ~TRT_DepthAnything3() override;

    const char* Class() const override { return description.name; }
    const char* node_help() const override
    {
        return "Depth Anything 3 — monocular depth estimation.\n\n"
               "Runs a TensorRT engine of Depth Anything 3 Mono Large\n"
               "to produce a depth map in the alpha channel.\n\n"
               "Input: RGB image (any resolution, linear or sRGB)\n"
               "Output: RGBA (alpha = depth map [0,1])\n\n"
               "Model resolution: 2058x1092 (147x78 patches)\n"
               "Engine is embedded — no external files needed.\n\n"
               "TRT_DepthAnything3 for Nuke by Peter Mercell, 2026\n"
               "www.petermercell.com\n\n"
               "Depth Anything 3 by Bingyi Kang et al.\n"
               "https://github.com/DepthAnything/Depth-Anything-3\n"
               "Licensed under Apache-2.0.";
    }

    void knobs(Knob_Callback f) override;

    void _validate(bool for_real) override;
    void _request(int x, int y, int r, int t,
                  ChannelMask channels, int count) override;
    void _open() override;
    void _close() override;
    void engine(int y, int x, int r,
                ChannelMask channels, Row& row) override;

    static const Iop::Description description;

private:

    // --- knobs ---
    bool        outputDepthOnly_;   // Show depth as greyscale RGB
    bool        invertDepth_;       // Invert depth (near=white, far=black)
    bool        linearInput_;       // Input is linear (needs sRGB conversion for model)
    int         gpuDevice_;

    // --- frame geometry ---
    int frameW_;
    int frameH_;
    int frameX_;
    int frameY_;

    // --- CPU buffers ---
    std::vector<float> cpuFrameIn_;   // 3 * frameW * frameH (planar CHW)
    std::vector<float> cpuDepthOut_;  // frameW * frameH
    std::vector<float> modelInput_;   // 3 * kModelW * kModelH
    std::vector<float> modelOutput_;  // kModelW * kModelH

    // --- CUDA ---
    float*       d_input_;
    float*       d_output_;
    cudaStream_t stream_;

    // --- TensorRT 10.x ---
    nvinfer1::IRuntime*          runtime_;
    nvinfer1::ICudaEngine*       engineTRT_;
    nvinfer1::IExecutionContext*  context_;

    // --- frame-level inference lock ---
    std::mutex inferenceMutex_;
    bool       inferenceRan_;

    // --- engine state ---
    bool engineLoaded_;

    // --- methods ---
    void loadEngine();
    void freeEngine();
    void allocateGPU();
    void freeGPU();
    void fetchAllRows();
    void preprocessFrame();
    void runInference();
    void postprocessDepth();
    void doFullInference();
};

// ---------------------------------------------------------------------------
// Ctor / Dtor
// ---------------------------------------------------------------------------
TRT_DepthAnything3::TRT_DepthAnything3(Node* node)
    : Iop(node)
    , outputDepthOnly_(false)
    , invertDepth_(false)
    , linearInput_(true)
    , gpuDevice_(0)
    , frameW_(0), frameH_(0), frameX_(0), frameY_(0)
    , d_input_(nullptr), d_output_(nullptr), stream_(nullptr)
    , runtime_(nullptr), engineTRT_(nullptr), context_(nullptr)
    , inferenceRan_(false)
    , engineLoaded_(false)
{
}

TRT_DepthAnything3::~TRT_DepthAnything3()
{
    freeGPU();
    freeEngine();
}

// ---------------------------------------------------------------------------
// Knobs
// ---------------------------------------------------------------------------
void TRT_DepthAnything3::knobs(Knob_Callback f)
{
    Bool_knob(f, &outputDepthOnly_, "depth_only", "Depth Only (BW)");
    Tooltip(f, "Output depth map as greyscale RGB instead of passthrough + alpha.");

    Bool_knob(f, &invertDepth_, "invert", "Invert Depth");
    Tooltip(f, "Invert depth values. Default: far=1, near=0. Inverted: near=1, far=0.");

    Bool_knob(f, &linearInput_, "linear_input", "Input is Linear");
    Tooltip(f, "Enable if your Nuke project is in linear colorspace.\n"
               "The model expects sRGB input — this converts linear to sRGB\n"
               "before feeding the model.\n\n"
               "Default: ON (Nuke default is linear).");
    SetFlags(f, Knob::STARTLINE);

    // Hidden — GPU device selection
    Int_knob(f, &gpuDevice_, "gpu", "GPU Device");
    SetFlags(f, Knob::HIDDEN);

    Divider(f, "");
    Text_knob(f, "Depth Anything 3 Mono Large | 2058x1092 | FP32\n"
                 "\n"
                 "TRT_DepthAnything3 for Nuke by Peter Mercell, 2026\n"
                 "www.petermercell.com\n"
                 "\n"
                 "Depth Anything 3 by Bingyi Kang et al.\n"
                 "github.com/DepthAnything/Depth-Anything-3\n"
                 "Licensed under Apache-2.0.");
}

// ---------------------------------------------------------------------------
// Load TensorRT engine from embedded data (TRT 10.x)
// ---------------------------------------------------------------------------
void TRT_DepthAnything3::loadEngine()
{
    if (engineLoaded_) return;

    const char* engineData = _binary_engine_bin_start;
    size_t engineSize = _binary_engine_bin_end - _binary_engine_bin_start;

    if (engineSize == 0)
    {
        error("Embedded engine data is empty.");
        return;
    }

    std::cerr << "[TRT_DepthAnything3] Loading embedded engine ("
              << (engineSize / (1024 * 1024)) << " MB)..." << std::endl;

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_)
    {
        error("Failed to create TensorRT runtime.");
        return;
    }

    engineTRT_ = runtime_->deserializeCudaEngine(engineData, engineSize);
    if (!engineTRT_)
    {
        error("Failed to deserialize embedded TensorRT engine.");
        return;
    }

    // Verify tensor names (TRT 10.x API)
    int nbIO = engineTRT_->getNbIOTensors();
    bool foundInput = false, foundOutput = false;

    for (int i = 0; i < nbIO; ++i)
    {
        const char* name = engineTRT_->getIOTensorName(i);
        std::cerr << "[TRT_DepthAnything3] Tensor: " << name << std::endl;
        if (strcmp(name, kInputTensorName) == 0)  foundInput = true;
        if (strcmp(name, kOutputTensorName) == 0) foundOutput = true;
    }

    if (!foundInput)
    {
        error("Engine missing input tensor '%s'", kInputTensorName);
        freeEngine();
        return;
    }
    if (!foundOutput)
    {
        error("Engine missing output tensor '%s'", kOutputTensorName);
        freeEngine();
        return;
    }

    context_ = engineTRT_->createExecutionContext();
    if (!context_)
    {
        error("Failed to create TensorRT execution context.");
        freeEngine();
        return;
    }

    engineLoaded_ = true;
    std::cerr << "[TRT_DepthAnything3] Embedded engine loaded successfully."
              << std::endl;
}

void TRT_DepthAnything3::freeEngine()
{
    if (context_)   { delete context_;   context_   = nullptr; }
    if (engineTRT_) { delete engineTRT_; engineTRT_ = nullptr; }
    if (runtime_)   { delete runtime_;   runtime_   = nullptr; }
    engineLoaded_ = false;
}

// ---------------------------------------------------------------------------
// GPU buffers
// ---------------------------------------------------------------------------
void TRT_DepthAnything3::allocateGPU()
{
    freeGPU();

    size_t inBytes  = 3 * kModelW * kModelH * sizeof(float);
    size_t outBytes = 1 * kModelW * kModelH * sizeof(float);

    cudaSetDevice(gpuDevice_);
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaMalloc(&d_input_,  inBytes));
    CUDA_CHECK(cudaMalloc(&d_output_, outBytes));

    // TRT 10.x: set tensor addresses by name
    if (!context_->setTensorAddress(kInputTensorName, d_input_))
    {
        error("Failed to set input tensor address");
        return;
    }
    if (!context_->setTensorAddress(kOutputTensorName, d_output_))
    {
        error("Failed to set output tensor address");
        return;
    }
}

void TRT_DepthAnything3::freeGPU()
{
    if (stream_)   { cudaStreamDestroy(stream_);  stream_   = nullptr; }
    if (d_input_)  { cudaFree(d_input_);          d_input_  = nullptr; }
    if (d_output_) { cudaFree(d_output_);         d_output_ = nullptr; }
}

// ---------------------------------------------------------------------------
// _open / _close
// ---------------------------------------------------------------------------
void TRT_DepthAnything3::_open()
{
    loadEngine();
    if (engineLoaded_)
        allocateGPU();
}

void TRT_DepthAnything3::_close()
{
    freeGPU();
    freeEngine();
}

// ---------------------------------------------------------------------------
// _validate
// ---------------------------------------------------------------------------
void TRT_DepthAnything3::_validate(bool for_real)
{
    copy_info();

    ChannelSet out = info().channels();
    out += Chan_Alpha;
    set_out_channels(out);
    info_.turn_on(Chan_Alpha);

    const Format& fmt = info().format();
    frameX_ = fmt.x();
    frameY_ = fmt.y();
    frameW_ = fmt.w();
    frameH_ = fmt.h();

    if (for_real)
    {
        cpuFrameIn_.resize(3 * frameW_ * frameH_, 0.0f);
        cpuDepthOut_.resize(frameW_ * frameH_, 0.0f);
        modelInput_.resize(3 * kModelW * kModelH, 0.0f);
        modelOutput_.resize(kModelW * kModelH, 0.0f);

        inferenceRan_ = false;
    }
}

// ---------------------------------------------------------------------------
// _request — request full frame from input
// ---------------------------------------------------------------------------
void TRT_DepthAnything3::_request(int x, int y, int r, int t,
                                  ChannelMask channels, int count)
{
    ChannelSet need = channels;
    need += Chan_Red;
    need += Chan_Green;
    need += Chan_Blue;

    input0().request(frameX_, frameY_,
                     frameX_ + frameW_, frameY_ + frameH_,
                     need, count);
}

// ---------------------------------------------------------------------------
// Fetch all input rows into the planar buffer
// ---------------------------------------------------------------------------
void TRT_DepthAnything3::fetchAllRows()
{
    ChannelSet need(Chan_Red);
    need += Chan_Green;
    need += Chan_Blue;

    float* rPlane = cpuFrameIn_.data();
    float* gPlane = cpuFrameIn_.data() + frameW_ * frameH_;
    float* bPlane = cpuFrameIn_.data() + 2 * frameW_ * frameH_;

    for (int y = frameY_; y < frameY_ + frameH_; ++y)
    {
        Row row(frameX_, frameX_ + frameW_);
        input0().get(y, frameX_, frameX_ + frameW_, need, row);

        if (aborted())
            return;

        const float* rIn = row[Chan_Red]  + frameX_;
        const float* gIn = row[Chan_Green] + frameX_;
        const float* bIn = row[Chan_Blue]  + frameX_;

        int rowIdx = (y - frameY_) * frameW_;

        for (int i = 0; i < frameW_; ++i)
        {
            rPlane[rowIdx + i] = rIn[i];
            gPlane[rowIdx + i] = gIn[i];
            bPlane[rowIdx + i] = bIn[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Preprocess: resize to model resolution + linear->sRGB (if needed)
// ImageNet normalization is baked into the ONNX — NOT applied here.
// ---------------------------------------------------------------------------
void TRT_DepthAnything3::preprocessFrame()
{
    // Resize to model resolution first (keeps cpuFrameIn_ untouched for passthrough)
    bilinearResize(cpuFrameIn_.data(), frameW_, frameH_,
                   modelInput_.data(), kModelW, kModelH, 3);

    // If input is linear, convert resized model input to sRGB
    // (model expects sRGB [0,1], ImageNet norm is baked into ONNX)
    if (linearInput_)
    {
        int pixels = kModelW * kModelH;
        for (int c = 0; c < 3; ++c)
        {
            float* plane = modelInput_.data() + c * pixels;
            for (int i = 0; i < pixels; ++i)
            {
                float v = plane[i];
                v = std::max(0.0f, std::min(1.0f, v));
                plane[i] = linearToSrgb(v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Postprocess: resize depth back to frame resolution
// No sigmoid needed — depth is already [0,1] from the ONNX model.
// ---------------------------------------------------------------------------
void TRT_DepthAnything3::postprocessDepth()
{
    // Resize depth map from model resolution to frame resolution
    bilinearResize(modelOutput_.data(), kModelW, kModelH,
                   cpuDepthOut_.data(), frameW_, frameH_, 1);

    if (invertDepth_)
    {
        int pixels = frameW_ * frameH_;
        for (int i = 0; i < pixels; ++i)
            cpuDepthOut_[i] = 1.0f - cpuDepthOut_[i];
    }
}

// ---------------------------------------------------------------------------
// TensorRT inference (TRT 10.x: enqueueV3)
// ---------------------------------------------------------------------------
void TRT_DepthAnything3::runInference()
{
    if (!engineLoaded_ || !context_) return;

    size_t inBytes  = 3 * kModelW * kModelH * sizeof(float);
    size_t outBytes = 1 * kModelW * kModelH * sizeof(float);

    cudaMemcpyAsync(d_input_, modelInput_.data(), inBytes,
                    cudaMemcpyHostToDevice, stream_);

    bool ok = context_->enqueueV3(stream_);
    if (!ok)
        std::cerr << "[TRT_DepthAnything3] enqueueV3 FAILED!" << std::endl;

    cudaMemcpyAsync(modelOutput_.data(), d_output_, outBytes,
                    cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);
}

// ---------------------------------------------------------------------------
// Full inference pipeline (called once per frame under lock)
// ---------------------------------------------------------------------------
void TRT_DepthAnything3::doFullInference()
{
    fetchAllRows();

    if (aborted()) return;

    preprocessFrame();
    runInference();
    postprocessDepth();
}

// ---------------------------------------------------------------------------
// engine() — called per scanline by Nuke (multi-threaded)
// ---------------------------------------------------------------------------
void TRT_DepthAnything3::engine(int y, int x, int r,
                                ChannelMask channels, Row& row)
{
    if (!engineLoaded_)
    {
        input0().get(y, x, r, channels, row);
        return;
    }

    // First thread does the full fetch + inference
    {
        std::lock_guard<std::mutex> lock(inferenceMutex_);
        if (!inferenceRan_)
        {
            doFullInference();
            inferenceRan_ = true;
        }
    }

    // Read from pre-computed buffers
    int localY = y - frameY_;
    int rowOffset = localY * frameW_;

    float* rOut = row.writable(Chan_Red);
    float* gOut = row.writable(Chan_Green);
    float* bOut = row.writable(Chan_Blue);

    if (outputDepthOnly_)
    {
        // Depth as greyscale
        for (int i = x; i < r; ++i)
        {
            int localX = i - frameX_;
            float d = cpuDepthOut_[rowOffset + localX];
            rOut[i] = d;
            gOut[i] = d;
            bOut[i] = d;
        }
    }
    else
    {
        // Passthrough original RGB
        const float* rPlane = cpuFrameIn_.data();
        const float* gPlane = cpuFrameIn_.data() + frameW_ * frameH_;
        const float* bPlane = cpuFrameIn_.data() + 2 * frameW_ * frameH_;

        for (int i = x; i < r; ++i)
        {
            int localX = i - frameX_;
            rOut[i] = rPlane[rowOffset + localX];
            gOut[i] = gPlane[rowOffset + localX];
            bOut[i] = bPlane[rowOffset + localX];
        }
    }

    // Depth always goes to alpha
    if (channels & Mask_Alpha)
    {
        float* aOut = row.writable(Chan_Alpha);
        for (int i = x; i < r; ++i)
        {
            int localX = i - frameX_;
            aOut[i] = cpuDepthOut_[rowOffset + localX];
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
static Iop* build(Node* node)
{
    return new TRT_DepthAnything3(node);
}

const Iop::Description TRT_DepthAnything3::description(
    "TRT_DepthAnything3",
    "AI/TRT_DepthAnything3",
    build
);
