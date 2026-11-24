#!/usr/bin/env python3
"""Precision comparison test: CUDA build vs ROCm (HIP) build of
mip-gaussian-l1c's diff-gaussian-rasterization.

What it does
------------
Builds a small deterministic synthetic Gaussian scene (fixed seed), runs
one forward + backward pass through the rasterizer (including the L1
contribution path), and either dumps the results (CUDA reference) or
compares them against a dumped reference (ROCm run).

Modes
-----
  --dump-reference DIR   Run on the CUDA build, save inputs + outputs to DIR.
  --compare DIR          Run on the ROCm build with identical inputs, compare
                         against the reference in DIR, print PASS/FAIL.
  --smoke                Run once, no reference: just check all outputs are
                         finite / sane (first run on a new ROCm build).

Why the tolerances are what they are
-------------------------------------
fp32 atomicAdd reduction order is non-deterministic even between two runs
on the SAME GPU, so bitwise equality is NOT expected. The thresholds below
allow fp32 reduction noise (~1e-6) while catching real bugs: wrong math
shows up as errors orders of magnitude larger (>= 1e-2).

  rendered color / l1_map : PSNR >= 50 dB, max abs diff <= 1e-4,
                            mean abs diff <= 1e-6, SSIM >= 0.999 (if available)
  gradients (each tensor): rel. L2 error <= 1e-4, cosine sim >= 0.9999,
                            max abs diff <= 1e-4
  radii                  : exact equality (int tensor)
  num_rendered           : exact equality (derived from radii: count > 0)

Camera / scene construction follows the upstream repo exactly:
  getWorld2View2 / getProjectionMatrix from mip-gaussian-l1c/utils/graphics_utils.py
  kernel_size = 0.1 (upstream default, arguments/__init__.py)
  subpixel_offset = zeros (H, W, 2) (gaussian_renderer/__init__.py)
  means2D = zeros (P, 3) with requires_grad=True (screenspace_points)

Usage
-----
  # 1. On an NVIDIA machine with the CUDA build installed:
  python test_precision.py --dump-reference ./ref_cuda --seed 0
  # 2. Copy ref_cuda/ to the ROCm machine, build + install the HIP port, then:
  python test_precision.py --compare ./ref_cuda --seed 0
  # (Smoke test on a fresh ROCm build, no reference needed:)
  python test_precision.py --smoke

Exit code: 0 when all checks pass, 1 otherwise.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Camera math, copied from upstream mip-gaussian-l1c/utils/graphics_utils.py
# (getWorld2View2, getProjectionMatrix) and scene/cameras.py (Camera).
# ---------------------------------------------------------------------------

def get_world2view(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def get_projection_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


# ---------------------------------------------------------------------------
# Scene + rasterizer
# ---------------------------------------------------------------------------

def build_scene(seed, n_gaussians, image_h, image_w, device):
    """Deterministic synthetic scene. Camera at origin looking down +Z
    (COLMAP convention); Gaussians in front of the camera, z in [2, 4]."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    P = n_gaussians
    means3D = torch.empty(P, 3)
    means3D[:, 0] = torch.rand(P, generator=g) * 2.0 - 1.0   # x in [-1, 1]
    means3D[:, 1] = torch.rand(P, generator=g) * 2.0 - 1.0   # y in [-1, 1]
    means3D[:, 2] = torch.rand(P, generator=g) * 2.0 + 2.0   # z in [2, 4]
    means3D = means3D.to(device).detach().requires_grad_(True)

    scales = torch.exp(torch.rand(P, 3, generator=g) * 2.0 - 3.0)
    scales = scales.to(device).detach().requires_grad_(True)

    rots = torch.randn(P, 4, generator=g)
    rots = rots / rots.norm(dim=1, keepdim=True)
    rots = rots.to(device).detach().requires_grad_(True)

    opacities = torch.rand(P, 1, generator=g) * 0.5 + 0.5  # 2-D (P,1): C++ backward returns [P,1]; 1-D fails on torch 2.11
    opacities = opacities.to(device).detach().requires_grad_(True)

    # SH degree 3 -> 16 coefficients, 3 channels.
    shs = torch.randn(P, 16, 3, generator=g) * 0.3
    shs = shs.to(device).detach().requires_grad_(True)

    means2D = torch.zeros(P, 3, device=device, requires_grad=True)

    # Camera: R = I, T = 0 (camera at origin, +Z forward), 60 deg FoV.
    fov = math.radians(60.0)
    R = np.eye(3, dtype=np.float32)
    T = np.zeros(3, dtype=np.float32)
    w2v = torch.tensor(get_world2view(R, T)).transpose(0, 1).to(device)
    proj = get_projection_matrix(0.01, 100.0, fov, fov).transpose(0, 1).to(device)
    full_proj = w2v.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    campos = torch.tensor([0.0, 0.0, 0.0], device=device)
    tanfovx = math.tan(fov / 2.0)
    tanfovy = math.tan(fov / 2.0)

    gt_image = torch.rand(3, image_h, image_w, generator=g).to(device)
    l1_token = torch.ones(P, device=device, requires_grad=True)

    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    settings = GaussianRasterizationSettings(
        image_height=image_h,
        image_width=image_w,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=0.1,
        subpixel_offset=torch.zeros(image_h, image_w, 2, device=device),
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=w2v,
        projmatrix=full_proj,
        sh_degree=3,
        campos=campos,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    inputs = dict(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        shs=shs,
        scales=scales,
        rotations=rots,
        gt_image=gt_image,
        l1_token=l1_token,
    )
    return rasterizer, inputs


def run_once(rasterizer, inputs):
    """One forward + backward. Returns dict of tensors (on CPU)."""
    color, radii, l1_map = rasterizer(
        means3D=inputs["means3D"],
        means2D=inputs["means2D"],
        opacities=inputs["opacities"],
        shs=inputs["shs"],
        scales=inputs["scales"],
        rotations=inputs["rotations"],
        gt_image=inputs["gt_image"],
        l1_token=inputs["l1_token"],
    )
    # Scalar loss that exercises both the color path and the L1 path.
    loss = color.sum() + l1_map.sum()
    loss.backward()
    out = {
        "color": color.detach().cpu(),
        "radii": radii.detach().cpu(),
        "l1_map": l1_map.detach().cpu(),
        "grad_means3D": inputs["means3D"].grad.cpu(),
        "grad_means2D": inputs["means2D"].grad.cpu(),
        "grad_shs": inputs["shs"].grad.cpu(),
        "grad_opacities": inputs["opacities"].grad.cpu(),
        "grad_scales": inputs["scales"].grad.cpu(),
        "grad_rotations": inputs["rotations"].grad.cpu(),
        "grad_l1_token": inputs["l1_token"].grad.cpu(),
        "num_rendered": int((radii > 0).sum().item()),
    }
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def rel_l2(a, b):
    num = float(((a - b) ** 2).sum())
    den = float((b ** 2).sum()) + 1e-12
    return math.sqrt(num / den)


def cosine_sim(a, b):
    af = a.flatten().double()
    bf = b.flatten().double()
    denom = float(af.norm() * bf.norm())
    if denom == 0:
        return 1.0 if float((af - bf).abs().max()) == 0 else 0.0
    return float(torch.dot(af, bf) / denom)


def ssim_or_none(a, b):
    try:
        from skimage.metrics import structural_similarity
    except (ImportError, ValueError):
        return None
    # skimage expects (H, W, C) float in [0, 1]; our tensors are (C, H, W).
    a_np = a.permute(1, 2, 0).numpy()
    b_np = b.permute(1, 2, 0).numpy()
    return float(structural_similarity(a_np, b_np, data_range=1.0, channel_axis=2))


def check_image(name, got, ref):
    m = {
        "psnr_db": psnr(got, ref),
        "max_abs_diff": float((got - ref).abs().max()),
        "mean_abs_diff": float((got - ref).abs().mean()),
    }
    s = ssim_or_none(got, ref)
    if s is not None:
        m["ssim"] = s
    ok = (
        m["psnr_db"] >= 50.0
        and m["max_abs_diff"] <= 1e-4
        and m["mean_abs_diff"] <= 1e-6
        and (s is None or s >= 0.999)
    )
    return ok, m


def check_grad(name, got, ref):
    m = {
        "rel_l2": rel_l2(got, ref),
        "cosine_sim": cosine_sim(got, ref),
        "max_abs_diff": float((got - ref).abs().max()),
    }
    ok = (
        m["rel_l2"] <= 1e-4
        and m["cosine_sim"] >= 0.9999
        and m["max_abs_diff"] <= 1e-4
    )
    return ok, m


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def env_info():
    info = {
        "torch_version": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
    return info


def cmd_smoke(args):
    device = torch.device("cuda")
    rasterizer, inputs = build_scene(
        args.seed, args.n_gaussians, args.image_h, args.image_w, device
    )
    out = run_once(rasterizer, inputs)
    checks = {
        "color finite": bool(torch.isfinite(out["color"]).all()),
        "radii non-negative": bool((out["radii"] >= 0).all()),
        "l1_map finite and >= 0": bool(
            torch.isfinite(out["l1_map"]).all() and (out["l1_map"] >= 0).all()
        ),
        "num_rendered > 0": out["num_rendered"] > 0,
    }
    for k in [
        "grad_means3D",
        "grad_means2D",
        "grad_shs",
        "grad_opacities",
        "grad_scales",
        "grad_rotations",
        "grad_l1_token",
    ]:
        checks[f"{k} finite"] = bool(torch.isfinite(out[k]).all())
    print(json.dumps({"env": env_info(), "smoke": checks}, indent=2))
    ok = all(checks.values())
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_dump_reference(args):
    device = torch.device("cuda")
    rasterizer, inputs = build_scene(
        args.seed, args.n_gaussians, args.image_h, args.image_w, device
    )
    out = run_once(rasterizer, inputs)
    os.makedirs(args.dump_reference, exist_ok=True)
    # Save inputs too, so --compare uses bit-identical inputs.
    payload = {f"in_{k}": v.detach().cpu() for k, v in inputs.items()}
    payload.update({f"out_{k}": v for k, v in out.items() if k != "num_rendered"})
    torch.save(payload, os.path.join(args.dump_reference, "reference.pt"))
    manifest = {
        "seed": args.seed,
        "n_gaussians": args.n_gaussians,
        "image_h": args.image_h,
        "image_w": args.image_w,
        "num_rendered": out["num_rendered"],
        "env": env_info(),
        "note": "Reference generated on the CUDA build. Compare on the ROCm build with --compare.",
    }
    with open(os.path.join(args.dump_reference, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print("Reference saved to", args.dump_reference)
    return 0


def cmd_compare(args):
    ref_path = os.path.join(args.compare, "reference.pt")
    payload = torch.load(ref_path, map_location="cpu", weights_only=False)
    with open(os.path.join(args.compare, "manifest.json")) as f:
        manifest = json.load(f)
    if (
        manifest["seed"] != args.seed
        or manifest["n_gaussians"] != args.n_gaussians
        or manifest["image_h"] != args.image_h
        or manifest["image_w"] != args.image_w
    ):
        print("ERROR: --compare manifest does not match --seed/--n-gaussians/--image size.")
        return 1

    device = torch.device("cuda")
    # Rebuild the scene from the SAVED inputs (bit-identical, not re-randomized).
    rasterizer, _ = build_scene(
        args.seed, args.n_gaussians, args.image_h, args.image_w, device
    )
    inputs = {}
    for k, v in payload.items():
        if k.startswith("in_"):
            vv = v.to(device).detach()
            if vv.is_floating_point():
                vv.requires_grad_(True)
            inputs[k[3:]] = vv
    out = run_once(rasterizer, inputs)

    results = {}
    all_ok = True
    for name in ["color", "l1_map"]:
        ok, m = check_image(name, out[name], payload[f"out_{name}"])
        results[name] = {"pass": ok, **m}
        all_ok &= ok
    for name in [
        "grad_means3D",
        "grad_means2D",
        "grad_shs",
        "grad_opacities",
        "grad_scales",
        "grad_rotations",
        "grad_l1_token",
    ]:
        ok, m = check_grad(name, out[name], payload[f"out_{name}"])
        results[name] = {"pass": ok, **m}
        all_ok &= ok
    radii_ok = bool(torch.equal(out["radii"], payload["out_radii"]))
    results["radii_exact"] = {"pass": radii_ok}
    all_ok &= radii_ok
    n_ok = out["num_rendered"] == manifest["num_rendered"]
    results["num_rendered"] = {
        "pass": n_ok,
        "got": out["num_rendered"],
        "ref": manifest["num_rendered"],
    }
    all_ok &= n_ok

    print(json.dumps({"env": env_info(), "ref_env": manifest["env"], "results": results}, indent=2))
    print("COMPARE:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-gaussians", type=int, default=2000)
    p.add_argument("--image-h", type=int, default=256)
    p.add_argument("--image-w", type=int, default=256)
    p.add_argument("--dump-reference", default=None)
    p.add_argument("--compare", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA/HIP device visible (torch.cuda.is_available() is False).")
        return 1
    if args.smoke:
        return cmd_smoke(args)
    if args.dump_reference:
        return cmd_dump_reference(args)
    if args.compare:
        return cmd_compare(args)
    p.error("one of --smoke, --dump-reference, --compare is required")


if __name__ == "__main__":
    sys.exit(main())
