"""
megaflow_cache.py - Pre-compute MegaFlow dense bidirectional trajectory
maps for a shot and save to .npz. Lets you pick points interactively
later (see megaflow_pick.py) without re-running the model.

A cache is bound to (input clip, fix_width, ref_frame). To pick from a
different reference frame, rebuild the cache.

Usage (video):
    python megaflow_cache.py --input assets/longboard.mp4 \\
        --nuke_first_frame 1 --ref_frame 25 \\
        --output cache/longboard.npz

Usage (EXR sequence in a folder):
    python megaflow_cache.py \\
        --input /home/pm/Documents/MEGAFLOW/megaflow/assets/exr/longboard \\
        --nuke_first_frame 1 --ref_frame 25 \\
        --exr_colorspace linear \\
        --output cache/longboard_ref25.npz

Cache format (.npz, no pickle):
    traj_maps : (T, 2, H_model, W_model)  float16  (absolute pixel coords in
                                                    MODEL space)
    meta      : 0-d str array containing a JSON dict with:
                {version, input_path, nuke_first_frame, last_frame,
                 ref_frame, ref_idx, native_h, native_w, model_h, model_w,
                 fix_width, iters, dtype, n_frames}
"""

import argparse
import glob
import json
import os

# Must be set BEFORE importing cv2 so OpenCV enables its OpenEXR codec.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch

from megaflow.model import MegaFlow
from megaflow.utils.basic import gridcloud2d
import megaflow.utils.frame_utils as frame_utils


CACHE_VERSION = 1

_FLOAT_EXTS = (".exr", ".hdr")   # treated as scene-linear float by default


# ---------------------------------------------------------------------------
# Float -> 8-bit encoding (for EXR / HDR / linear float input)
# ---------------------------------------------------------------------------

def _linear_to_srgb(c):
    """Linear scene values -> sRGB display values. Expects c >= 0."""
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308,
                    c * 12.92,
                    1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def _float_rgb_to_uint8(rgb_float, colorspace="linear", exposure=0.0):
    """rgb_float: (H, W, 3) float, channel order RGB.
       colorspace: 'linear' applies an sRGB OETF; 'srgb' assumes already
       display-encoded (just clip + quantize).
    """
    img = np.asarray(rgb_float, dtype=np.float32)
    if exposure:
        img = img * (2.0 ** float(exposure))
    img = np.clip(img, 0.0, None)
    if colorspace == "linear":
        img = _linear_to_srgb(img)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def _read_exr_rgb_float(path):
    """Read an EXR/HDR file -> (H, W, 3) float32 in RGB order. Tries OpenCV
    first (needs the OpenEXR codec), then imageio as a fallback."""
    img = None
    try:
        img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    except Exception:
        img = None

    if img is not None:
        img = np.asarray(img)
        if img.ndim == 2:                      # mono -> 3ch
            img = np.repeat(img[..., None], 3, axis=2)
        img = img[..., :3].astype(np.float32)
        img = img[..., ::-1]                   # OpenCV BGR -> RGB
        return np.ascontiguousarray(img)

    # Fallback: imageio (already RGB)
    try:
        import imageio.v2 as imageio
        arr = np.asarray(imageio.imread(path)).astype(np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        return np.ascontiguousarray(arr[..., :3])
    except Exception as e:
        raise RuntimeError(
            f"Could not read EXR/HDR '{path}'. Enable OpenCV's OpenEXR codec "
            f"(OPENCV_IO_ENABLE_OPENEXR=1 + a build with EXR support) or "
            f"install imageio with the EXR plugin. Underlying error: {e}"
        )


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def calculate_dynamic_size(orig_h, orig_w, fix_width, patch_size=14):
    new_w = fix_width
    new_h = round(orig_h * (new_w / orig_w) / patch_size) * patch_size
    return int(new_h), int(new_w)


def get_native_frames(input_path, exr_colorspace="linear", exr_exposure=0.0):
    """Yield (RGB uint8 frame at NATIVE resolution, (H, W)).

    Directories may mix uint8 (png/jpg/...) and float (exr/hdr) images;
    float images are exposure-adjusted and encoded to 8-bit via
    `exr_colorspace`.
    """
    if os.path.isdir(input_path):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.BMP", "*.exr",
                "*.EXR", "*.hdr", "*.tif", "*.tiff")
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(input_path, ext)))
        image_paths = sorted(set(image_paths))
        if not image_paths:
            raise FileNotFoundError(f"No images found in {input_path}")
        for path in image_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext in _FLOAT_EXTS:
                rgb = _read_exr_rgb_float(path)
                img = _float_rgb_to_uint8(rgb, exr_colorspace, exr_exposure)
            else:
                raw = np.asarray(frame_utils.read_gen(path))
                if raw.dtype.kind == "f":      # float tiff / linear png, etc.
                    img = _float_rgb_to_uint8(
                        raw[..., :3], exr_colorspace, exr_exposure
                    )
                else:
                    img = raw.astype(np.uint8)[..., :3]
            yield img, img.shape[:2]
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame, frame.shape[:2]
        cap.release()


# ---------------------------------------------------------------------------
# MegaFlow inference (bidirectional, full-res trajectory)
# ---------------------------------------------------------------------------

def _run_megaflow(frames_btchw, model, num_reg_refine, device):
    """(1, T, 3, H, W) at MODEL res -> (T, 2, H, W) absolute coords."""
    _, _, _, H, W = frames_btchw.shape
    compute_dtype = (torch.bfloat16
                     if device == "cuda" and torch.cuda.is_bf16_supported()
                     else torch.float16)
    with torch.inference_mode(), torch.autocast(
        device_type=device, dtype=compute_dtype, enabled=(device == "cuda")
    ):
        grid_xy = gridcloud2d(1, H, W, norm=False, device=device).float()
        grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H, W)
        flow_final = model.forward_track(
            frames_btchw, num_reg_refine=num_reg_refine
        )["flow_final"]
        traj = flow_final.to(device) + grid_xy
    return traj[0].float()


def _identity_traj(H, W, device):
    grid_xy = gridcloud2d(1, H, W, norm=False, device=device).float()
    return grid_xy.permute(0, 2, 1).reshape(1, 2, H, W)


def _to_model_tensor(native_frames, idx_list, new_H, new_W, device):
    frames = []
    for i in idx_list:
        img = cv2.resize(native_frames[i], (new_W, new_H),
                         interpolation=cv2.INTER_LINEAR)
        frames.append(torch.from_numpy(img).permute(2, 0, 1).float())
    return torch.stack(frames, 0)[None].to(device)


def build_full_trajectory(native_frames, ref_idx, model, num_reg_refine,
                          fix_width, patch_size, device):
    """Returns (T_total, 2, H_model, W_model) numpy float32 plus model size."""
    T_total = len(native_frames)
    orig_H, orig_W = native_frames[0].shape[:2]
    new_H, new_W = calculate_dynamic_size(orig_H, orig_W, fix_width, patch_size)

    # Forward
    fwd_idx = list(range(ref_idx, T_total))
    if len(fwd_idx) >= 2:
        if device == "cuda":
            torch.cuda.empty_cache()
        fwd = _run_megaflow(
            _to_model_tensor(native_frames, fwd_idx, new_H, new_W, device),
            model, num_reg_refine, device,
        )
    else:
        fwd = _identity_traj(new_H, new_W, device)

    # Backward
    bwd_idx = list(range(ref_idx, -1, -1))
    if len(bwd_idx) >= 2:
        if device == "cuda":
            torch.cuda.empty_cache()
        bwd = _run_megaflow(
            _to_model_tensor(native_frames, bwd_idx, new_H, new_W, device),
            model, num_reg_refine, device,
        )
    else:
        bwd = _identity_traj(new_H, new_W, device)

    full = torch.cat([torch.flip(bwd, dims=[0]), fwd[1:]], dim=0)
    full[ref_idx] = _identity_traj(new_H, new_W, device)[0]
    return full.cpu().numpy(), (new_H, new_W)


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def save_cache(path, traj_maps_f32, meta, dtype="float16"):
    if dtype == "float16":
        arr = traj_maps_f32.astype(np.float16)
    else:
        arr = traj_maps_f32.astype(np.float32)
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(path,
             traj_maps=arr,
             meta=np.array(json.dumps(meta)))


def load_cache(path):
    data = np.load(path, allow_pickle=False)
    traj = data["traj_maps"].astype(np.float32)
    meta = json.loads(str(data["meta"]))
    return traj, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Video file, or a folder of frames (png/jpg/exr/...).")
    p.add_argument("--output", required=True, help="Output .npz path")
    p.add_argument("--nuke_first_frame", type=int, default=1)
    p.add_argument("--ref_frame", type=int, default=None,
                   help="Tracker reference. Default = nuke_first_frame.")
    p.add_argument("--fix_width", type=int, default=518)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--exr_colorspace", choices=["linear", "srgb"],
                   default="linear",
                   help="Color space of EXR/HDR/float input. 'linear' applies "
                        "an sRGB OETF before quantizing; 'srgb' assumes the "
                        "data is already display-encoded.")
    p.add_argument("--exr_exposure", type=float, default=0.0,
                   help="Exposure (stops) applied to EXR/float input before "
                        "encoding to 8-bit.")
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading megaflow-track on {device}...")
    model = MegaFlow.from_pretrained("megaflow-track", device=device).eval()

    print(f"Reading frames from {args.input}...")
    native_frames, native_size = [], None
    for frame, shape in get_native_frames(
        args.input,
        exr_colorspace=args.exr_colorspace,
        exr_exposure=args.exr_exposure,
    ):
        if native_size is None:
            native_size = shape
        native_frames.append(frame)
    T_total = len(native_frames)
    if T_total < 2:
        raise RuntimeError("Need at least 2 frames.")
    orig_H, orig_W = native_size

    nuke_first = args.nuke_first_frame
    ref_frame = args.ref_frame if args.ref_frame is not None else nuke_first
    ref_idx = ref_frame - nuke_first
    if not (0 <= ref_idx < T_total):
        raise ValueError(
            f"ref_frame {ref_frame} outside "
            f"[{nuke_first}, {nuke_first + T_total - 1}]."
        )

    print(f"  {T_total} frames at {orig_W}x{orig_H}, "
          f"ref={ref_frame} (idx {ref_idx})")
    print("Running MegaFlow forward+backward...")
    traj_maps, (model_h, model_w) = build_full_trajectory(
        native_frames, ref_idx, model, args.iters,
        args.fix_width, patch_size=14, device=device,
    )

    meta = {
        "version": CACHE_VERSION,
        "input_path": os.path.abspath(args.input),
        "nuke_first_frame": nuke_first,
        "last_frame": nuke_first + T_total - 1,
        "ref_frame": ref_frame,
        "ref_idx": ref_idx,
        "native_h": orig_H,
        "native_w": orig_W,
        "model_h": model_h,
        "model_w": model_w,
        "fix_width": args.fix_width,
        "iters": args.iters,
        "dtype": args.dtype,
        "n_frames": T_total,
    }

    print(f"Saving cache -> {args.output} ({args.dtype})...")
    save_cache(args.output, traj_maps, meta, dtype=args.dtype)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Cache size: {size_mb:.1f} MB, shape {traj_maps.shape} {args.dtype}")
    print("Pick points with megaflow_pick.py (or the Nuke gizmo).")


if __name__ == "__main__":
    main()
