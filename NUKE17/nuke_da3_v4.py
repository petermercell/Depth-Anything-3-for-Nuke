#!/usr/bin/env python3
"""
Depth Anything 3 Mono -> Nuke Cattery Export (v3)

This version follows Rafael's DepthAnythingV2 pattern exactly:
- Fixed resolution tracing
- Explicit patch dimension handling
- Clean TorchScript output without debug prints

Usage:
    python nuke_da3_v3.py \
        --model-path /path/to/DA3MONO-LARGE.safetensors \
        --config-path /path/to/config.json \
        --resolution 518 \
        --output-dir ./output

The resolution MUST match your Nuke Reformat node exactly!
Common options:
    518 = 37×14 (37×37 patches) - standard DA size
    532 = 38×14 (38×38 patches) 
    560 = 40×14 (40×40 patches)
    
After tracing:
1. Use Nuke's CatFileCreator to create .cat from .pt
2. Set Reformat to EXACT same resolution (e.g., 518×518)
3. Input channels: 3, Output channels: 1
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DA3MonoNukeWrapper(nn.Module):
    """
    Wrapper for DA3Mono that handles both FP16 and FP32.
    
    For FP16: Uses decomposed approach with custom sky estimation (FP32 internally)
    For FP32: Uses full model forward with all post-processing
    
    CRITICAL: Input resolution MUST match the traced resolution exactly!
    
    Input:  [B, 3, H, W] RGB tensor in range [0, 1]
    Output: [B, 1, H, W] normalized depth in range [0, 1]
    """
    
    def __init__(self, da3_model, use_fp16: bool = False):
        super().__init__()
        self.use_fp16 = use_fp16
        
        if use_fp16:
            # For FP16, use decomposed approach with custom sky estimation
            self.backbone = da3_model.backbone
            self.head = da3_model.head
        else:
            # For FP32, use full model with all post-processing
            self.model = da3_model
        
        # Sky threshold
        self.sky_threshold = 0.3
    
    def _process_sky_estimation_fp16(self, depth: torch.Tensor, sky: torch.Tensor) -> torch.Tensor:
        """
        Custom sky estimation that works with FP16.
        Temporarily converts to FP32 for quantile calculation.
        """
        original_dtype = depth.dtype
        
        # Convert to FP32 for quantile operations
        depth_f32 = depth.float()
        sky_f32 = sky.float()
        
        # Compute sky mask (same as original)
        non_sky_mask = sky_f32 < self.sky_threshold
        
        # Check if we have enough non-sky pixels
        if non_sky_mask.sum() <= 10:
            return depth
        if (~non_sky_mask).sum() <= 10:
            return depth
        
        # Get non-sky depth values
        non_sky_depth = depth_f32[non_sky_mask]
        
        # Sample if too many points (same as original)
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        
        # Compute 99th percentile (quantile in FP32)
        non_sky_max = torch.quantile(sampled_depth, 0.99)
        
        # Set sky regions to max depth
        depth_f32 = torch.where(~non_sky_mask, non_sky_max, depth_f32)
        
        # Convert back to original dtype
        return depth_f32.to(original_dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # ImageNet normalization using scalars (no device-bound tensors!)
        # This avoids cuda:0 vs cpu mismatch in Nuke 16.1+ / 17.0+
        x = torch.cat([
            (x[:, 0:1, :, :] - 0.485) / 0.229,
            (x[:, 1:2, :, :] - 0.456) / 0.224,
            (x[:, 2:3, :, :] - 0.406) / 0.225,
        ], dim=1)
        
        # DA3 expects [B, S, C, H, W] where S=views (1 for mono)
        x = x.unsqueeze(1)  # [B, C, H, W] -> [B, 1, C, H, W]
        
        if self.use_fp16:
            # FP16 path: decomposed approach with custom sky estimation
            features, _ = self.backbone(x)
            head_output = self.head(features, H, W, patch_start_idx=0)
            depth = head_output['depth']
            
            # Apply sky estimation if sky mask is available
            if 'sky' in head_output:
                sky = head_output['sky']
                depth = self._process_sky_estimation_fp16(depth, sky)
        else:
            # FP32 path: use full forward() with all post-processing
            output = self.model(x)
            depth = output['depth']
        
        # Ensure output is [B, 1, H, W]
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 4 and depth.shape[1] != 1:
            depth = depth[:, 0:1, :, :]
        
        # Normalize depth to [0, 1] using scalar eps (no torch.tensor!)
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
    
    # Load weights
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        LOGGER.warning(f"Missing keys: {len(missing)}")
    if unexpected:
        LOGGER.warning(f"Unexpected keys: {len(unexpected)}")
    
    model = model.to(device).eval()
    LOGGER.info(f"Model loaded on {device}")
    
    return model


def trace_model(
    model_path: str,
    config_path: str,
    resolution: int,
    width: int,
    height: int,
    use_half: bool,
    output_dir: str,
):
    """Trace DA3Mono for Nuke's Cattery."""
    
    # Use width/height if provided, otherwise use square resolution
    if width > 0 and height > 0:
        W, H = width, height
    else:
        W, H = resolution, resolution
    
    # Validate dimensions are multiples of 14
    if W % 14 != 0:
        suggested = (W // 14) * 14
        LOGGER.error(f"Width {W} is not a multiple of 14!")
        LOGGER.error(f"Try {suggested} or {suggested + 14} instead.")
        sys.exit(1)
    
    if H % 14 != 0:
        suggested = (H // 14) * 14
        LOGGER.error(f"Height {H} is not a multiple of 14!")
        LOGGER.error(f"Try {suggested} or {suggested + 14} instead.")
        sys.exit(1)
    
    patches_w = W // 14
    patches_h = H // 14
    LOGGER.info(f"Resolution: {W}×{H} = {patches_w}×{patches_h} patches")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model on GPU
    da3_model = load_da3_model(model_path, config_path, DEVICE)
    
    # Create wrapper
    LOGGER.info("Creating Nuke wrapper...")
    wrapper = DA3MonoNukeWrapper(da3_model, use_fp16=use_half)
    wrapper = wrapper.to(DEVICE).eval()
    
    # Set precision
    dtype = torch.float32
    if use_half:
        LOGGER.info("Converting to FP16...")
        wrapper = wrapper.half()
        dtype = torch.float16
    
    # Test forward pass
    LOGGER.info(f"Testing forward pass on GPU at {W}×{H}...")
    test_input = torch.randn(1, 3, H, W, device=DEVICE, dtype=dtype)
    
    with torch.no_grad():
        test_output = wrapper(test_input)
        LOGGER.info(f"  Input:  {test_input.shape} {test_input.dtype}")
        LOGGER.info(f"  Output: {test_output.shape} {test_output.dtype}")
        LOGGER.info(f"  Range:  [{test_output.min():.4f}, {test_output.max():.4f}]")
    
    # Trace on CUDA - vanilla, no post-processing optimizations
    # NOTE: Do NOT use torch.jit.optimize_for_inference — it introduces
    #   CUDA-specific fused ops that cause device mismatches in Nuke 16.1+/17.0
    # NOTE: Do NOT use torch.jit.freeze — it breaks DINOv2's
    #   _get_intermediate_layers_not_chunked
    LOGGER.info("Tracing with torch.jit.trace on CUDA...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, test_input)
    
    # Verify traced model
    LOGGER.info("Verifying traced model...")
    with torch.no_grad():
        traced_output = traced(test_input)
        diff = (test_output - traced_output).abs().max()
        LOGGER.info(f"  Max difference: {diff:.6f}")
    
    # Save directly — vanilla trace graph is device-agnostic in IR,
    # only parameter tensors carry device info. Nuke's Cattery handles
    # device placement when loading.
    precision_str = "fp16" if use_half else "fp32"
    filename = f"DepthAnything3_mono_large_{W}x{H}_{precision_str}.pt"
    output_path = os.path.join(output_dir, filename)
    
    traced.save(output_path)
    
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    LOGGER.info(f"Saved: {output_path} ({size_mb:.1f} MB)")
    
    # Print instructions
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nModel saved to: {output_path}")
    print(f"\nNuke Setup Instructions:")
    print("-" * 60)
    print(f"1. Create .cat file:")
    print(f"   - TorchScript file: {filename}")
    print(f"   - Channels in: 3")
    print(f"   - Channels out: 1")
    print(f"   - Model ID: DepthAnything3")
    print(f"\n2. Reformat settings:")
    print(f"   - Type: to box")
    print(f"   - Width: {W}")
    print(f"   - Height: {H}")
    print(f"   - Resize type: fit")
    print(f"   - Filter: Cubic (Keys)")
    print(f"   - Check 'force shape'")
    print(f"\n3. Node graph:")
    print(f"   Read -> Reformat({W}×{H}) -> Inference(.cat)")
    print(f"        -> Reformat(original) -> Output")
    print(f"\nIMPORTANT: Resolution MUST be exactly {W}×{H}!")
    print("=" * 60)
    
    return traced


def main():
    parser = argparse.ArgumentParser(
        description="Export DA3Mono for Nuke's Cattery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 518 resolution (matches DA2)
  python nuke_da3_v3.py --model-path model.safetensors --config-path config.json --resolution 518
  
  # FP32 for better compatibility
  python nuke_da3_v3.py --model-path model.safetensors --config-path config.json --resolution 518 --no-half
  
  # Higher resolution
  python nuke_da3_v3.py --model-path model.safetensors --config-path config.json --resolution 560

Common resolutions (must be multiple of 14):
  518 = 37×14 (37×37 patches) - standard
  532 = 38×14 (38×38 patches)
  560 = 40×14 (40×40 patches)
  588 = 42×14 (42×42 patches)
        """
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Path to DA3MONO-LARGE.safetensors"
    )
    parser.add_argument(
        "--config-path", 
        type=str, 
        required=True,
        help="Path to config.json"
    )
    parser.add_argument(
        "--resolution", 
        type=int, 
        default=518,
        help="Square resolution to trace at (must be multiple of 14)"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=0,
        help="Width for non-square (must be multiple of 14, use with --height)"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=0,
        help="Height for non-square (must be multiple of 14, use with --width)"
    )
    parser.add_argument(
        "--half", 
        action="store_true",
        help="Use FP16 (smaller file, may have dtype issues)"
    )
    parser.add_argument(
        "--no-half", 
        action="store_true",
        help="Use FP32 (recommended for compatibility)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./",
        help="Output directory for .pt file"
    )
    
    args = parser.parse_args()
    
    # Default to FP32 unless --half is explicitly set
    use_half = args.half and not args.no_half
    
    # Determine resolution
    if args.width > 0 and args.height > 0:
        res_str = f"{args.width}×{args.height}"
    else:
        res_str = f"{args.resolution}×{args.resolution}"
    
    LOGGER.info(f"Device: {DEVICE}")
    LOGGER.info(f"Precision: {'FP16' if use_half else 'FP32'}")
    LOGGER.info(f"Resolution: {res_str}")
    
    trace_model(
        model_path=args.model_path,
        config_path=args.config_path,
        resolution=args.resolution,
        width=args.width,
        height=args.height,
        use_half=use_half,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
