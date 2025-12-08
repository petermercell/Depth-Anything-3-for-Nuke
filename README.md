# Depth Anything 3 for Nuke

**Monocular Depth Estimation for Foundry Nuke**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)
[![Nuke](https://img.shields.io/badge/Nuke-16.0-yellow.svg)](https://www.foundry.com/products/nuke)

## Description

This plugin brings **Depth Anything 3** monocular depth estimation to Foundry's Nuke compositing software. It generates high-quality depth maps from single RGB images using the DA3Mono-Large model (350M parameters).

Depth Anything 3 represents the state-of-the-art in monocular depth estimation, significantly outperforming previous versions (DA1, DA2) in geometric accuracy.

## Pre-traced Models

Pre-traced models for various resolutions (720p, 2K, 3K, 4K) are available on my [Patreon](https://www.patreon.com/posts/depth-anything-3-145386188).

## Usage

> ⚠️ **IMPORTANT:** The input resolution MUST match the traced model resolution exactly!

### Node Graph Setup

```
Read → Reformat → Inference → Reformat → Output
       (to model)   (.cat)    (to original)
```

### Reformat Settings (before Inference)

- Type: **to box**
- Width/Height: exact model resolution
- Resize type: **fit**
- Filter: **Cubic (Keys)**
- ☑️ **force shape** ← critical!

### Creating .CAT File

If you need to create a `.cat` file from the `.pt` model:

1. Open Nuke's **CatFileCreator**
2. Set TorchScript file: `DepthAnything3_mono_large_WxH_fp32.pt`
3. Cat file: `DepthAnything3_mono_large_WxH_fp32.cat`
4. Channels in: `rgba.red, rgba.green, rgba.blue`
5. Channels out: `rgba.alpha`
6. Model ID: `DepthAnything3`
7. Create the `.cat` file

## For Developers

### Requirements for Tracing

- Python 3.10
- PyTorch 2.1.1 (must match Nuke 16.0)
- `depth_anything_3` package
- `safetensors`

### Step by Step Installation

```bash
conda create -n da3_nuke16 python=3.10 -y
conda activate da3_nuke16
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install safetensors
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install -e .

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "from depth_anything_3.api import DepthAnything3; print('DA3 imported OK')"
```

### Download Model

```bash
huggingface-cli download depth-anything/DA3MONO-LARGE --local-dir ./DA3MONO-LARGE
```

### Tracing New Resolutions

The included `nuke_da3_v3.py` script can trace models at any resolution (width and height must be multiples of 14):

```bash
# Square resolution
python nuke_da3_v3.py \
    --model-path /path/to/model.safetensors \
    --config-path /path/to/config.json \
    --resolution 2520 \
    --no-half \
    --output-dir ./output

# Non-square (16:9) resolution - saves VRAM!
python nuke_da3_v3.py \
    --model-path /path/to/model.safetensors \
    --config-path /path/to/config.json \
    --width 2058 \
    --height 1092 \
    --no-half \
    --output-dir ./output
```

### Common Resolutions (must be ×14)

| Format | Resolution | Patches | VRAM (FP32) |
|--------|------------|---------|-------------|
| 720p | 1288×728 | 92×52 | ~4 GB |
| 2K DCI | 2058×1092 | 147×78 | ~6-8 GB |
| 3K DCI | 3080×1624 | 220×116 | ~12-16 GB |
| 4K DCI | 4102×2170 | 293×155 | ~20-24 GB |

## Compatibility

| Component | Version |
|-----------|---------|
| Nuke | 15.1, 15.2, 16.0 |
| PyTorch | 2.1.1 |
| CUDA | 11.8+ |
| OS | Rocky Linux 8/9+, Windows 10/11 |

## Credits

### Depth Anything 3 Model

- **Paper:** [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647)
- **Authors:** Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, Bingyi Kang
- **Organization:** ByteDance Seed Team
- **Repository:** https://github.com/ByteDance-Seed/Depth-Anything-3
- **Project Page:** https://depth-anything-3.github.io/

### Nuke Integration

Tracing and wrapper by **Peter Mercell**

### Acknowledgments

Special thanks to:
- **Rafael Silva** for his excellent [Depth-Anything-for-Nuke](https://github.com/rafaelperez/Depth-Anything-for-Nuke) project, which served as the foundation and inspiration for this work.
- **PozzettiAndrea** for [ComfyUI-DepthAnythingV3](https://github.com/PozzettiAndrea/ComfyUI-DepthAnythingV3) reference implementation.

## License

This Nuke integration is released under the **Apache License 2.0**.

The DA3Mono-Large model is licensed under Apache License 2.0 by ByteDance.

See [LICENSE.txt](LICENSE.txt) for full license text.

## Citation

If you use this in your work, please cite the original paper:

```bibtex
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and 
          Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```
