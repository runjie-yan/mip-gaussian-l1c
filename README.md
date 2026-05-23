# Mip-Splatting with L1 Contribution Rasterization

This repository extends [Mip-Splatting](https://niujinshuchong.github.io/mip-splatting/) with an efficient **per-Gaussian L1 contribution metric** computed at zero additional memory cost. The metric measures exactly how much each Gaussian contributes to the L1 reconstruction loss, enabling principled pruning, importance scoring, and policy-gradient-style optimization over Gaussian existence.

The original Mip-Splatting documentation is preserved in [README_MIP_GS.md](README_MIP_GS.md). For a detailed explanation of the L1 contribution algorithm, see [RERASTER_L1_IMPLEMENTATION.md](RERASTER_L1_IMPLEMENTATION.md).

---

## What's New

Standard 3DGS pruning relies on opacity or gradient-magnitude proxies. This repo adds a direct, differentiable measure:

> **L1 contribution of Gaussian $g_i$**: how much the L1 loss would increase if $g_i$ were removed.

The computation is fused into the existing backward pass — no second rasterization, no extra pixel-level cache. The result is exposed as a gradient on a user-supplied `l1_token` tensor, making it composable with any PyTorch training loop.

---

## Installation

```bash
git clone https://github.com/runjie-yan/mip-gaussian-l1c.git
cd mip-gaussian-l1c-public

conda create -y -n mip-gs-l1c python=3.8
conda activate mip-gs-l1c

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
conda install cudatoolkit-dev=11.3 -c conda-forge

pip install -r requirements.txt

pip install submodules/diff-gaussian-rasterization # core rasterization module
pip install submodules/simple-knn/
```

---

## L1 Contribution API

The key addition is two new optional arguments to `GaussianRasterizer.forward` and the low-level `rasterize_gaussians` function.

### `GaussianRasterizer.forward`

```python
# Basic mip-gaussian rasterizer
rendered_image, radii = rasterizer(
    means3D, means2D, opacities,
    shs=shs,
    scales=scales,
    rotations=rotations,
)

# With L1 contribution enabled:
rendered_image, radii, l1_map = rasterizer(
    means3D, means2D, opacities,
    shs=shs,
    scales=scales,
    rotations=rotations,
    gt_image=gt_image,   # required when l1_token is used
    l1_token=l1_token,   # triggers contribution computation
)
```

**New arguments:**

| Argument | Type | Description |
|---|---|---|
| `gt_image` | `torch.Tensor [3, H, W]` | Ground-truth image (RGB, float32). Required when `l1_token` is provided. |
| `l1_token` | `torch.Tensor [N]`, `requires_grad=True` | Dummy token. After `loss.backward()`, `.grad` holds per-Gaussian L1 contributions. |

**Return values (when `l1_token` is provided):**

| Value | Shape | Description |
|---|---|---|
| `rendered_image` | `[3, H, W]` | Rendered color image (unchanged). |
| `radii` | `[N]` int32 | Screen-space radii; 0 for culled Gaussians (unchanged). |
| `l1_map` | `[H, W]` | Per-pixel summed absolute residual $\sum_c \|C_{render,c} - C_{GT,c}\|$. |

### Minimal usage example

```python
import torch
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

# --- setup rasterizer (standard Mip-Splatting settings) ---
raster_settings = GaussianRasterizationSettings(...)
rasterizer = GaussianRasterizer(raster_settings=raster_settings)

# --- create token (one scalar per Gaussian) ---
l1_token = torch.zeros(num_gaussians, device="cuda", requires_grad=True)

# --- forward ---
rendered_image, radii, l1_map = rasterizer(
    means3D=means3D,
    means2D=means2D,
    opacities=opacities,
    shs=shs,
    scales=scales,
    rotations=rotations,
    gt_image=gt_image,
    l1_token=l1_token,
)

# --- loss ---
loss = (rendered_image - gt_image).abs().mean()
# Including l1_map in the graph activates contribution accumulation in backward.
# Using .mean() makes grad_l1_map a uniform weight map (equivalent to per-pixel weight = 1/HW).
loss += l1_map.mean()

# NOTE: l1_map must be connected to the scalar loss. If l1_map is not used here,
# grad_l1_map will be None, the kernel skips accumulation entirely, and
# l1_token.grad stays all zeros — consistent with standard PyTorch autograd behavior.

# --- backward ---
loss.backward()

# --- read contributions ---
contributions = l1_token.grad  # shape [N], Gaussian caused loss change, can be positive or negative
```

### Custom per-pixel weighting

`l1_map` is a standard tensor in the autograd graph. Any differentiable operation on it will produce the corresponding `grad_l1_map`, which is used as a per-pixel weight $w_p$ when accumulating contributions:

```python
# Uniform weights (same as .mean() up to a scale factor)
loss += l1_map.sum()

# Mask-based: only count contributions inside a region of interest
loss += (l1_map * roi_mask).sum()
```

### Policy-gradient interpretation

`l1_token` can be interpreted as log-probabilities $\log p(g_i)$ of Gaussian existence. The gradient accumulated into `l1_token.grad` is then a REINFORCE-style policy gradient:

$$\nabla_{\log p(g_i)} J = \sum_p w_p \,\Delta L_{p,i}$$

where $\Delta L_{p,i} = |R_p| - |R_p - \Delta C_{p,i}|$ is the per-pixel change in L1 loss when Gaussian $i$ is removed, and $w_p$ comes from `grad_l1_map`. This makes the feature directly applicable to learned Gaussian pruning or sparsity regularization.

## Citation

If you use the L1 contribution feature, please cite the Mip-Splatting paper:

```bibtex
@InProceedings{Yu2024MipSplatting,
    author    = {Yu, Zehao and Chen, Anpei and Huang, Binbin and Sattler, Torsten and Geiger, Andreas},
    title     = {Mip-Splatting: Alias-free 3D Gaussian Splatting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {19447-19456}
}
```
and our paper:
```bibtex
@misc{yan2026generative3dgaussianslearned,
      title={Generative 3D Gaussians with Learned Density Control}, 
      author={Runjie Yan and Yan-Pei Cao and Peng Wang and Ding Liang and Yuan-Chen Guo},
      year={2026},
      eprint={2605.16355},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2605.16355}, 
}
```