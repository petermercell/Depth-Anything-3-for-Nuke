#!/usr/bin/env python3
"""
Depth Anything 3 Mono -> ONNX Export for TensorRT

Exports DA3Mono to ONNX format for building TensorRT engines.
Uses decomposed backbone+head approach to avoid sky estimation
(which uses quantile/boolean indexing — not ONNX-compatible).

Pipeline: DA3 safetensors -> ONNX -> trtexec -> TRT engine -> C++ Nuke plugin

Usage:
    python nuke_da3_onnx_export.py \
        --model-path /path/to/DA3MONO-LARGE.safetensors \
        --config-path /path/to/config.json \
        --width 2058 --height 1092 \
        --output-dir ./output

Then build TRT engine:
    # FP16 (recommended for RTX A5000):
    trtexec --onnx=DepthAnything3_mono_large_2058x1092.onnx \
            --saveEngine=DepthAnything3_mono_large_2058x1092_fp16.engine \
            --fp16 --workspace=4096

    # FP32:
    trtexec --onnx=DepthAnything3_mono_large_2058x1092.onnx \
            --saveEngine=DepthAnything3_mono_large_2058x1092_fp32.engine \
            --workspace=4096

Resolution must be multiples of 14.
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DA3OnnxWrapper(nn.Module):
    """
    ONNX-friendly wrapper for DA3Mono.
    
    Uses decomposed backbone+head to avoid sky estimation
    (quantile, boolean indexing, torch.where with masks are not
    well supported in ONNX/TensorRT).
    
    Input:  [B, 3, H, W] RGB tensor in range [0, 1]
    Output: [B, 1, H, W] normalized depth in range [0, 1]
    
    Preprocessing (ImageNet normalization) is baked into the model
    so the C++ plugin only needs to handle:
    - sRGB-to-linear conversion (if needed)
    - Feed [0,1] RGB
    - Read [0,1] depth output
    """
    
    def __init__(self, da3_model):
        super().__init__()
        self.backbone = da3_model.backbone
        self.head = da3_model.head
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # ImageNet normalization using scalars (ONNX-safe)
        x = torch.cat([
            (x[:, 0:1, :, :] - 0.485) / 0.229,
            (x[:, 1:2, :, :] - 0.456) / 0.224,
            (x[:, 2:3, :, :] - 0.406) / 0.225,
        ], dim=1)
        
        # DA3 expects [B, S, C, H, W] where S=views (1 for mono)
        x = x.unsqueeze(1)
        
        # Backbone: extract features
        features, _ = self.backbone(x)
        
        # Head: decode depth (no sky estimation)
        head_output = self.head(features, H, W, patch_start_idx=0)
        depth = head_output['depth']
        
        # Ensure [B, 1, H, W]
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 4 and depth.shape[1] != 1:
            depth = depth[:, 0:1, :, :]
        
        # Normalize to [0, 1]
        depth_min = depth.amin(dim=(2, 3), keepdim=True)
        depth_max = depth.amax(dim=(2, 3), keepdim=True)
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        
        return depth


def load_da3_model(model_path: str, config_path: str, device: str):
    """Load DA3Mono model from local files."""
    from depth_anything_3.cfg import create_object
    
    LOGGER.info(f"Loading config: {config_path}")
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    model_config = config_data.get("config", config_data)
    
    LOGGER.info("Creating model architecture...")
    model = create_object(model_config)
    
    LOGGER.info(f"Loading weights: {model_path}")
    state_dict = load_file(model_path)
    
    # DA3 safetensors keys start with "model." - strip it
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[6:] if key.startswith("model.") else key
        cleaned_state_dict[new_key] = value
    
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        LOGGER.warning(f"Missing keys: {len(missing)}")
    if unexpected:
        LOGGER.warning(f"Unexpected keys: {len(unexpected)}")
    
    model = model.to(device).eval()
    LOGGER.info(f"Model loaded on {device}")
    
    return model


def export_onnx(
    model_path: str,
    config_path: str,
    width: int,
    height: int,
    output_dir: str,
    opset: int = 17,
):
    """Export DA3Mono to ONNX."""
    
    W, H = width, height
    
    # Validate dimensions
    if W % 14 != 0:
        suggested = (W // 14) * 14
        LOGGER.error(f"Width {W} is not a multiple of 14! Try {suggested} or {suggested + 14}.")
        sys.exit(1)
    if H % 14 != 0:
        suggested = (H // 14) * 14
        LOGGER.error(f"Height {H} is not a multiple of 14! Try {suggested} or {suggested + 14}.")
        sys.exit(1)
    
    patches_w = W // 14
    patches_h = H // 14
    LOGGER.info(f"Resolution: {W}×{H} = {patches_w}×{patches_h} patches")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    da3_model = load_da3_model(model_path, config_path, DEVICE)
    
    # Create ONNX wrapper (decomposed, no sky estimation)
    LOGGER.info("Creating ONNX wrapper (decomposed backbone+head, no sky estimation)...")
    wrapper = DA3OnnxWrapper(da3_model)
    wrapper = wrapper.to(DEVICE).eval()
    
    # Test forward pass
    LOGGER.info(f"Testing forward pass at {W}×{H}...")
    test_input = torch.randn(1, 3, H, W, device=DEVICE, dtype=torch.float32)
    
    with torch.no_grad():
        test_output = wrapper(test_input)
        LOGGER.info(f"  Input:  {test_input.shape} {test_input.dtype}")
        LOGGER.info(f"  Output: {test_output.shape} {test_output.dtype}")
        LOGGER.info(f"  Range:  [{test_output.min():.4f}, {test_output.max():.4f}]")
    
    # Export to ONNX
    filename = f"DepthAnything3_mono_large_{W}x{H}.onnx"
    output_path = os.path.join(output_dir, filename)
    
    LOGGER.info(f"Exporting ONNX (opset {opset})...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            test_input,
            output_path,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["depth"],
            # Fixed shape — no dynamic axes for TRT
        )
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    LOGGER.info(f"ONNX saved: {output_path} ({size_mb:.1f} MB)")
    
    # Verify with ONNX Runtime if available
    try:
        import onnxruntime as ort
        LOGGER.info("Verifying with ONNX Runtime...")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(output_path, providers=providers)
        
        test_np = test_input.cpu().numpy()
        ort_output = session.run(None, {"input": test_np})[0]
        ort_tensor = torch.from_numpy(ort_output).to(DEVICE)
        
        diff = (test_output - ort_tensor).abs().max()
        LOGGER.info(f"  PyTorch vs ORT max diff: {diff:.6f}")
        
        if diff < 0.01:
            LOGGER.info("  ONNX verification PASSED")
        else:
            LOGGER.warning(f"  ONNX verification: diff={diff:.6f} (may be acceptable)")
    except ImportError:
        LOGGER.info("onnxruntime not installed, skipping verification")
        LOGGER.info("  pip install onnxruntime-gpu")
    
    # Simplify with onnx-simplifier if available
    simplified_path = None
    try:
        import onnxsim
        import onnx
        LOGGER.info("Simplifying ONNX model...")
        model_onnx = onnx.load(output_path)
        model_simplified, check = onnxsim.simplify(model_onnx)
        if check:
            simplified_filename = f"DepthAnything3_mono_large_{W}x{H}_simplified.onnx"
            simplified_path = os.path.join(output_dir, simplified_filename)
            onnx.save(model_simplified, simplified_path)
            simp_size = os.path.getsize(simplified_path) / (1024 * 1024)
            LOGGER.info(f"Simplified ONNX saved: {simplified_path} ({simp_size:.1f} MB)")
        else:
            LOGGER.warning("Simplification check failed, using original ONNX")
    except ImportError:
        LOGGER.info("onnx-simplifier not installed, skipping simplification")
        LOGGER.info("  pip install onnxsim")
    
    # Print TensorRT build instructions
    onnx_for_trt = simplified_path if simplified_path else output_path
    onnx_for_trt_basename = os.path.basename(onnx_for_trt)
    
    print("\n" + "=" * 70)
    print("ONNX EXPORT SUCCESS!")
    print("=" * 70)
    print(f"\nONNX model: {output_path}")
    if simplified_path:
        print(f"Simplified: {simplified_path}")
    print(f"\nInput:  [1, 3, {H}, {W}]  float32  RGB [0, 1]")
    print(f"Output: [1, 1, {H}, {W}]  float32  depth [0, 1]")
    
    print(f"\n--- TensorRT Engine Build ---")
    print(f"# FP16 (recommended for RTX A5000):")
    print(f"trtexec --onnx={onnx_for_trt_basename} \\")
    print(f"        --saveEngine=DA3_mono_large_{W}x{H}_fp16.engine \\")
    print(f"        --fp16 --workspace=4096")
    print(f"\n# FP32 (if FP16 has NaN issues):")
    print(f"trtexec --onnx={onnx_for_trt_basename} \\")
    print(f"        --saveEngine=DA3_mono_large_{W}x{H}_fp32.engine \\")
    print(f"        --workspace=4096")
    
    print(f"\n--- C++ Plugin Notes ---")
    print(f"Same pattern as TRTBiRefNet / TRTCorridorKey:")
    print(f"  - Input binding:  'input'  [1, 3, {H}, {W}]")
    print(f"  - Output binding: 'depth'  [1, 1, {H}, {W}]")
    print(f"  - Preprocessing:  sRGB→linear (if Nuke is in linear),")
    print(f"                    ImageNet norm is BAKED IN to the ONNX")
    print(f"  - Postprocessing: depth is already [0,1] normalized")
    print(f"  - No sky estimation (skipped for TRT compatibility)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Export DA3Mono to ONNX for TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2K plate (2058x1092, matching your current setup)
  python nuke_da3_onnx_export.py \\
      --model-path model.safetensors --config-path config.json \\
      --width 2058 --height 1092

  # Standard square (518x518)
  python nuke_da3_onnx_export.py \\
      --model-path model.safetensors --config-path config.json \\
      --width 518 --height 518

Resolution must be multiples of 14.
        """
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to DA3MONO-LARGE.safetensors"
    )
    parser.add_argument(
        "--config-path", type=str, required=True,
        help="Path to config.json"
    )
    parser.add_argument(
        "--width", type=int, required=True,
        help="Input width (must be multiple of 14)"
    )
    parser.add_argument(
        "--height", type=int, required=True,
        help="Input height (must be multiple of 14)"
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    LOGGER.info(f"Device: {DEVICE}")
    LOGGER.info(f"Resolution: {args.width}×{args.height}")
    
    export_onnx(
        model_path=args.model_path,
        config_path=args.config_path,
        width=args.width,
        height=args.height,
        output_dir=args.output_dir,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
