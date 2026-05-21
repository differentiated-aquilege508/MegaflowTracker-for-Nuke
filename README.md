# MegaflowTracker-for-Nuke

Turn [MegaFlow](https://github.com/cvg/megaflow) dense point trajectories into native Nuke **Tracker4** / **CornerPin2D** nodes.

Solve the shot **once** on the GPU, cache the result to a `.npz`, then pick as many points as you like inside Nuke and bake them into trackers — instantly, on the CPU, as many times as you want, without ever re-running the model.

The cache is solved relative to a single **reference frame** (the frame your picks are anchored to). Re-picking, adding points, and exporting are all free — but **changing the reference frame means rebuilding the `.npz`** (re-running the GPU solve), since the trajectories are computed forward and backward from that frame.

---

## How it works

The workflow is split into a slow offline step and a fast interactive one:

1. **Solve once (GPU).** `megaflow_cache.py` runs MegaFlow forward **and** backward from a chosen reference frame and bakes the full dense trajectory field to a `.npz` cache. This is the only heavy/slow part. Video files and image sequences are supported, **including EXR/HDR** (correctly encoded to the model's input space rather than truncated).
2. **Pick (Nuke).** Drop the `MEGAFlowTracker_PM` gizmo, point it at the `.npz`, and place points directly in the Viewer.
3. **Track It (Nuke).** The gizmo samples the cache at your picked points and pastes a `Tracker4` and/or `CornerPin2D` straight into your script. Pure `numpy`, no GPU, no PyTorch, nothing written to disk.

Because picking only samples a pre-baked cache, you can re-pick different features, add/remove points, and export again in seconds — no re-solve.

---

## Repository contents

| File | Purpose |
| --- | --- |
| `megaflow_cache.py` | Offline solver. Runs MegaFlow and writes the trajectory cache `.npz`. |
| `MegaFlowTracker.tcl` | The Nuke gizmo (a self-contained Group). Picks points and exports trackers. |

The gizmo embeds all of its pick/sample logic, so these two files are everything you need.

---

## Requirements

- **MegaFlow** installed and working — see https://github.com/cvg/megaflow. A CUDA GPU is strongly recommended for the solve step.
- **Nuke** for the pick/export step. The gizmo uses only Nuke's bundled `numpy` and standard knob APIs (developed on Nuke 17).

---

## Installation

### 1. MegaFlow + this repo

Follow the MegaFlow install instructions, then drop `megaflow_cache.py` next to the MegaFlow package (so the `import megaflow...` calls resolve):

```bash
git clone https://github.com/cvg/megaflow
cd megaflow
# ... follow MegaFlow's setup / model download ...
# then copy megaflow_cache.py into this folder
```

### 2. The Nuke gizmo

Pick whichever you prefer:

**Quick (no install):** open `MegaFlowTracker.tcl` in a text editor, copy everything, and paste it into the Nuke Node Graph. The `MEGAFlowTracker_PM` node appears.

**Permanent (menu entry):** copy `MegaFlowTracker.tcl` into a folder on your `NUKE_PATH` (e.g. `~/.nuke`) and add to your `~/.nuke/menu.py`:

```python
import nuke
m = nuke.menu("Nodes").addMenu("MegaFlow")
m.addCommand("MegaFlowTracker",
             "nuke.nodePaste('/path/to/MegaFlowTracker.tcl')")
```

---

## Usage

### Step 1 — Solve once (build the cache)

```bash
cd /home/pm/Documents/MEGAFLOW/megaflow
conda activate megaflow

python megaflow_cache.py \
    --input /home/pm/Documents/MEGAFLOW/megaflow/assets/exr/longboard \
    --ref_frame 25 --nuke_first_frame 1 \
    --exr_colorspace linear \
    --output cache/longboard_ref25.npz
```

Key arguments:

- `--input` — a video file **or** a folder of frames (`png`, `jpg`, `tif`, `exr`, `hdr`, …).
- `--ref_frame` — the frame your picks will be valid at. The cache is bound to this reference; to pick from a different frame, rebuild the cache.
- `--nuke_first_frame` — the frame number the first input frame maps to in Nuke (so the exported keyframes line up with your timeline).
- `--exr_colorspace` — `linear` (default; applies an sRGB encode to scene-linear EXR) or `srgb` (input already display-encoded). Also `--exr_exposure` (in stops) if a plate is very dark/hot.
- `--output` — the `.npz` cache path.

### Step 2 — Pick points in Nuke

1. Connect the `MEGAFlowTracker_PM` gizmo to your plate.
2. Set **NPZ FILE** to the `.npz` from Step 1.
3. Use **Add Point** / **Remove Point** to create picks, drag their handles in the Viewer to position them, and tick **enable** on the ones you want exported. Disabled points are skipped.

### Step 3 — Track It

1. Set **Export** to `tracker4`, `cornerpin`, or `both`.
2. Click **TRACK IT**. A `Tracker4` (and/or `CornerPin2D`) is created at the top level of your script, next to the gizmo, with animated tracks sampled from the cache.

---

## Notes

- **CornerPin** needs exactly **4** enabled points. With a different count, `both` falls back to `tracker4`.
- For image sequences, frames are read in **sorted filename order** — use zero-padded names (`shot.0001.exr`, `shot.0002.exr`, …) so the order is correct.
- The cache stores trajectories in the model's working resolution; the gizmo handles the model↔native↔Nuke coordinate conversions for you (including the y-up flip).
- The gizmo needs **`numpy`** available in Nuke's Python. Nuke ships with it bundled, so this works out of the box in a standard install — but if you run a custom/standalone Python or a stripped environment, make sure `numpy` is importable (`pip install numpy` into Nuke's Python) or **TRACK IT** will fail to import.
- If a cache won't read EXRs, your OpenCV build may lack the OpenEXR codec — installing `imageio` (with an EXR backend) covers the fallback path.

---

## Credits & License

- Built on **MegaFlow** (Apache-2.0) — https://github.com/cvg/megaflow.
- Tracker4 / CornerPin2D serialisation adapted from `lprestini/ml-runner` (Apache-2.0).
- Licensed under the **Apache License 2.0** — free for commercial use, modification, and distribution (see `LICENSE`). © Peter Mercell, 2026.
