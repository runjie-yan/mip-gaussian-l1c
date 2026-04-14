# L1 Contribution Rasterization: Implementation Notes

## 1. Background

In 3D Gaussian Splatting, pruning heuristics typically rely on opacity or gradient-magnitude proxies. A more principled metric is the **per-Gaussian L1 contribution**:

$$\Delta L_i = \mathcal{L}_1(I_{\text{render}},\, I_{\text{GT}}) - \mathcal{L}_1(I_{\text{render} \setminus \{g_i\}},\, I_{\text{GT}})$$

A positive $\Delta L_i$ means Gaussian $g_i$ reduces the L1 loss — it is useful. A negative value means it increases the loss — it is harmful or redundant.

Computing this naively requires $N$ extra forward passes (one per Gaussian). This implementation computes it for all Gaussians simultaneously, fused into the existing backward pass, with **zero additional memory overhead**.

---

## 2. Algorithm

### 2.1 Forward pass

The forward pass is unchanged from standard Mip-Splatting. When `l1_token` is provided, one extra step runs in Python after the CUDA kernel returns:

```python
residual = (color - gt_image).detach()          # [3, H, W]  (C=3, NUM_CHANNELS hardcoded)
l1_map   = residual.abs().sum(dim=0)  # [H, W]
```

`l1_map` is the per-pixel summed absolute residual $\sum_c |R_c|$. It is returned as a regular tensor in the autograd graph. The `residual` tensor is saved in the context for use in the backward pass.

### 2.2 Backward pass — contribution accumulation

The backward CUDA kernel (`renderCUDA` in `backward.cu`) already re-traverses all Gaussians in reverse depth order to compute standard gradients. The L1 contribution is accumulated in the same loop at negligible extra cost.

For each (Gaussian $i$, pixel $p$) pair encountered during the backward traversal:

1. **Reconstruct the background color behind Gaussian $i$ at pixel $p$.**  
   The backward kernel maintains `accum_rec[ch]`, the accumulated color from all Gaussians *behind* the current one. This is the color pixel $p$ would show if Gaussian $i$ and everything in front of it were removed.

2. **Compute the color change $\Delta C$ from removing Gaussian $i$.**  
   $$\Delta C_{p,i,c} = T_{p,i} \cdot \alpha_{p,i} \cdot (c_i - \text{back}_{p,c})$$
   where $T_{p,i}$ is the transmittance at pixel $p$ just before Gaussian $i$, $\alpha_{p,i}$ is its blending weight, $c_i$ is its color, and $\text{back}_{p,c}$ is `accum_rec[ch]`.

3. **Compute the per-pixel L1 change.**  
   $$\Delta L_{p,i} = |R_{p,c}| - |R_{p,c} - \Delta C_{p,i,c}|$$
   summed over channels $c$, where $R_{p,c}$ is the pre-computed residual from the forward pass.

4. **Weight and accumulate.**  
   $$\text{contribution}_i \mathrel{+}= \sum_p w_p \cdot \Delta L_{p,i}$$
   where $w_p = \texttt{l1\_weight}[p]$ (the gradient flowing back through `l1_map`).

The CUDA code for steps 2–4 inside the per-channel loop:

```cpp
if (l1_output && residual_image) {
    float back   = accum_rec[ch];
    float delta_c = T * alpha * (c - back);
    float resid  = residual_image[ch * H * W + pix_id];
    float weight = l1_weight ? l1_weight[pix_id] : 1.0f;
    l1_diff += weight * (abs(resid) - abs(resid - delta_c));
}
```

After the channel loop:

```cpp
if (l1_output)
    atomicAdd(&l1_output[global_id], l1_diff);
```

`l1_output` is the buffer that becomes `l1_token.grad` after `loss.backward()`.

### 2.3 How `l1_token.grad` is populated

`l1_token` is a dummy tensor with `requires_grad=True`. It is passed through the Python API but is not used in the forward computation. In the backward pass, `grad_l1_token` (initialized to `torch.zeros_like(radii, dtype=torch.float32)` — explicit float32 because `radii` is int32) is filled by the CUDA kernel and returned as the gradient for `l1_token`. PyTorch's autograd engine then writes it into `l1_token.grad`.

The per-pixel weight $w_p$ is `grad_l1_map` — the gradient of the scalar loss with respect to `l1_map`. This is determined entirely by how the user connects `l1_map` to the loss:

| User code | `grad_l1_map` | Effect |
|---|---|---|
| `loss += l1_map.sum()` | all-ones `[H, W]` | raw L1 contributions |
| `loss += l1_map.mean()` | `1/(H*W)` everywhere | contributions normalized by image size |
| `loss += (l1_map * mask).sum()` | `mask` | contributions only inside masked region |

---

## 3. Code Changes

### 3.1 `cuda_rasterizer/backward.cu` — `renderCUDA` kernel

**New kernel parameters:**

```cpp
const float* __restrict__ residual_image,  // flat [C * H * W], layout (ch, y, x)
float*       __restrict__ l1_output,        // flat [N], one entry per Gaussian
const float* __restrict__ l1_weight         // flat [H * W], layout (y, x)
```

Both `residual_image` and `l1_output` are nullable. The contribution block is guarded by `if (l1_output && residual_image)`, so there is zero overhead when the feature is not used.

The kernel indexes these as:
- `residual_image[ch * H * W + pix_id]` — matches a contiguous `[3, H, W]` tensor (C=3)
- `l1_weight[pix_id]` — matches a contiguous `[H, W]` tensor (same flat memory)

### 3.2 `rasterize_points.cu` — C++ binding

`RasterizeGaussiansBackwardCUDA` accepts three new tensors:

```cpp
const torch::Tensor& residual_image,
const torch::Tensor& l1_output,
const torch::Tensor& l1_weight
```

Empty tensors (`numel() == 0`) are passed as `nullptr` to the CUDA kernel:

```cpp
residual_image.numel() > 0 ? residual_image.contiguous().data_ptr<float>() : nullptr,
l1_output.numel()      > 0 ? l1_output.contiguous().data_ptr<float>()      : nullptr,
l1_weight.numel()      > 0 ? l1_weight.contiguous().data_ptr<float>()      : nullptr
```

### 3.3 `diff_gaussian_rasterization/__init__.py` — Python autograd function

**`_RasterizeGaussians.forward`**

- Accepts `gt_image` and `l1_token` as optional trailing arguments.
- When `l1_token is not None`, computes `residual` and `l1_map` in Python, saves `residual` in the context alongside the standard buffers, sets `ctx.l1_enabled = True`, and returns `(color, radii, l1_map)` instead of `(color, radii)`.
- Raises `ValueError` if `l1_token` is provided without `gt_image`.
- Validates that `l1_token.shape[0] == means3D.shape[0]`.

**`_RasterizeGaussians.backward`**

- Detects `ctx.l1_enabled` to decide which saved-tensor layout to unpack.
- Allocates `grad_l1_token = torch.zeros_like(radii, dtype=torch.float32)` when enabled (explicit float32 because `radii` is int32); passes an empty tensor otherwise.
- Extracts `grad_l1_map` from the trailing gradient arguments when `l1_enabled`.
- Returns `grad_l1_token` as the gradient for `l1_token` (index 10 in the return tuple), `None` for `gt_image` (index 9).

**`GaussianRasterizer.forward`**

- Passes `gt_image` and `l1_token` through to `rasterize_gaussians` unchanged.
- Validates `gt_image` shape against `raster_settings` dimensions.
- Validates `l1_token` length against `means3D`.

---

## 4. API Reference

### `GaussianRasterizer.forward`

```python
def forward(
    self,
    means3D,          # [N, 3]
    means2D,          # [N, 3]  (screenspace, requires_grad=True)
    opacities,        # [N, 1]
    shs=None,         # [N, K, 3]  K=(max_sh_degree+1)^2; mutually exclusive with colors_precomp
    colors_precomp=None,  # [N, 3]  mutually exclusive with shs
    scales=None,      # [N, 3]  mutually exclusive with cov3D_precomp
    rotations=None,   # [N, 4]  quaternion (w, x, y, z)
    cov3D_precomp=None,   # [N, 6]  upper-triangular 3D covariance; mutually exclusive with scales/rotations
    gt_image=None,    # [3, H, W]  required if l1_token is used
    l1_token=None,    # [N]  requires_grad=True
) -> tuple
```

Returns `(color, radii)` normally, or `(color, radii, l1_map)` when `l1_token` is provided.

| Return | Shape | Description |
|---|---|---|
| `color` | `[3, H, W]` | Rendered image (NUM_CHANNELS=3 hardcoded) |
| `radii` | `[N]` int32 | Screen-space radii; 0 for culled Gaussians |
| `l1_map` | `[H, W]` | Per-pixel $\sum_c \|R_c\|$; only when `l1_token` is given |

After `loss.backward()`:

| Tensor | Shape | Description |
|---|---|---|
| `l1_token.grad` | `[N]` | Per-Gaussian L1 contribution (weighted sum over pixels) |

---

## 5. Policy Gradient Interpretation

Setting `l1_token` to the log-probability of each Gaussian's existence $\log p_\theta(g_i)$ connects this to REINFORCE. The gradient of the expected loss with respect to $\theta$ is:

$$
\nabla_\theta \mathbb{E}\bigl[\mathcal{L}\bigr]
= \mathbb{E}\!\left[\sum_i \bigl(\mathcal{L}(\{g_j\}) - \mathcal{L}(\{g_j\}_{j \neq i})\bigr)\,\nabla_\theta \log p_\theta(g_i)\right]
$$

The term $\mathcal{L}(\{g_j\}) - \mathcal{L}(\{g_j\}_{j \neq i})$ is exactly $-\Delta L_i$ (the negative of the L1 contribution). The kernel computes this difference per pixel and accumulates it, so `l1_token.grad` directly provides the policy gradient signal for each Gaussian.

This is useful for learned pruning: if `l1_token` is the output of a small network that predicts Gaussian keep/drop probabilities, the gradient flows end-to-end without any separate evaluation pass.

---

## 6. Limitations

- The contribution estimate uses a **leave-one-out linearization**: it assumes removing Gaussian $i$ does not change the transmittance of Gaussians in front of it. This is exact for non-overlapping Gaussians and a good approximation otherwise.
- `residual` is computed from the rendered color **before** any tone-mapping or clamping applied outside the rasterizer. If the user clamps the output before computing the loss, the residual used internally will differ slightly.
- `l1_token.grad` accumulates contributions from a **single view**. For a scene-level importance score, average over multiple views.
