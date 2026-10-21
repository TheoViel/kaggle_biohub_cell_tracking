"""Render a short GIF of the pipeline's output on one stack: pick a cell, follow it over time.

    .venv/bin/python scripts/make_pipeline_gif.py --sample 6bba_32db13fc --t0 0 --cell 182,168 --out assets/pipeline_demo.gif

Styled like webapp/trajectory_viewer.py: one z slice, detections within a z
tolerance drawn as dots faded by their distance to the slice, and the track
drawn backwards from the current frame. Needs only the stack and a predicted geff.
"""
import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tracksdata as td
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from params import SCALE  # noqa: E402

SC = np.array(SCALE)
PRED_DIR = ROOT / "logs/edge_run/2026-09-28/0/geff_0.45_pp"
SIZE = 560           # output pixels
Z_TOL = 3.0          # slices
CELL_PX = 17         # ring diameter in stack pixels
SKY, DODGER, ORANGE = "#87ceeb", "#1e90ff", "#ff8c00"
BG = "#0b0f14"


def load_graph(path):
    r = td.graph.IndexedRXGraph.from_geff(path)
    g = r[0] if isinstance(r, tuple) else r
    n = g.node_attrs()
    pos = {int(r["node_id"]): (int(r["t"]), np.array([r["z"], r["y"], r["x"]], float)) for r in n.iter_rows(named=True)}
    e = g.edge_attrs()
    edges = list(zip(e["source_id"].to_list(), e["target_id"].to_list())) if g.num_edges() else []
    return pos, edges


def follow(pos, edges, start, n):
    nxt = {}
    for a, b in edges:
        nxt.setdefault(a, []).append(b)
    chain = [start]
    while len(chain) < n and len(nxt.get(chain[-1], [])) >= 1:
        chain.append(nxt[chain[-1]][0])
    return chain


class Scene:
    def __init__(self, sample, t0, n_steps, cell):
        grp = zarr.open_group(ROOT / "input/train" / f"{sample}.zarr", mode="r")
        self.stack = np.asarray(grp["0"][t0:t0 + n_steps + 1])
        q = grp.attrs.get("image_statistics", {}).get("quantiles", {})
        self.lo = next(float(v) for k, v in q.items() if abs(float(k) - 0.01) < 1e-6)
        self.hi = next(float(v) for k, v in q.items() if abs(float(k) - 0.999) < 1e-6)
        pos, edges = load_graph(PRED_DIR / f"{sample}.geff")
        self.t0, self.n = t0, n_steps
        self.by_t = {}
        for i, (t, x) in pos.items():
            if t0 <= t <= t0 + n_steps:
                self.by_t.setdefault(t, []).append(x)
        self.by_t = {t: np.array(v) for t, v in self.by_t.items()}
        cands = [i for i, (t, x) in pos.items() if t == t0 and np.hypot(x[1] - cell[0], x[2] - cell[1]) < 4]
        assert cands, "no detection at that position"
        start = cands[0]
        chain = follow(pos, edges, start, n_steps + 1)
        assert len(chain) == n_steps + 1, f"track only {len(chain)} long"
        self.track = np.array([pos[i][1] for i in chain])  # (n+1, 3) z,y,x
        self.fig = plt.figure(figsize=(SIZE / 100, SIZE / 100), dpi=100, facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])

    def image(self, k, z):
        z = int(np.clip(round(z), 0, self.stack.shape[1] - 1))
        sl = self.stack[k, z].astype(float)
        lo, hi = np.percentile(sl, [1, 99.7])  # per-slice stretch: deeper slices are much dimmer
        im = np.clip((sl - lo) / (hi - lo), 0, 1) ** 0.8
        return np.round(im * 95) / 95  # 96 gray levels leave room in a GIF's 256-colour palette for the overlay colours

    def frame(self, k, center, half, z=None, rings=True, trail_to=None, selected=True, caption="", label=True, full_track=False):
        """Render timepoint index k, cropped to center (y, x) +- half pixels."""
        ax = self.ax
        ax.clear(); ax.set_facecolor(BG); ax.axis("off")
        z = self.track[k, 0] if z is None else z
        ax.imshow(self.image(k, z), cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
        ax.set_xlim(center[1] - half, center[1] + half); ax.set_ylim(center[0] + half, center[0] - half)
        if rings and (self.t0 + k) in self.by_t:
            d = self.by_t[self.t0 + k]
            dz = np.abs(d[:, 0] - z)
            keep = dz <= Z_TOL
            fade = 1 - 0.7 * (dz[keep] / Z_TOL)
            ax.scatter(d[keep, 2], d[keep, 1], s=34, color=SKY, edgecolors=DODGER, linewidths=0.9, alpha=fade.clip(0.3, 1), zorder=4)
        end = k if trail_to is None else trail_to
        if selected:
            tr = self.track[:end + 1]
            if len(tr) > 1:
                ax.plot(tr[:, 2], tr[:, 1], color=ORANGE, lw=2.2, alpha=0.45, solid_capstyle="round", zorder=5)
            ax.scatter(*tr[-1, [2, 1]], s=120, color=ORANGE, edgecolors="white", linewidths=1.4, zorder=7)
        # overlays
        ax.text(0.03, 0.965, f"t = {self.t0 + k}" if label else "", transform=ax.transAxes, color="white", fontsize=15, fontweight="bold", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="#000000aa", ec="none"))
        if caption:
            ax.text(0.5, 0.035, caption, transform=ax.transAxes, color="white", fontsize=14, fontweight="bold", ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.45", fc="#000000b8", ec="none"))
        bar_px = 20 / SCALE[1]
        x1 = 1 - 0.04; x0 = x1 - bar_px / (2 * half)
        ax.plot([x0, x1], [0.145, 0.145], transform=ax.transAxes, color="white", lw=3)
        ax.text((x0 + x1) / 2, 0.16, "20 µm", transform=ax.transAxes, color="white", fontsize=10, ha="center", va="bottom")
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()


def _palette():
    """One 256-colour palette for every frame: the gray ramp, the overlay colours, and the trail blended over each gray.

    A per-frame quantiser drops the small orange marker on bright frames, so the palette is fixed instead.
    """
    from matplotlib.colors import to_rgb
    gray = np.round(np.linspace(0, 255, 96))[:, None] * np.ones((1, 3))
    orange = np.array(to_rgb(ORANGE)) * 255
    trail = 0.45 * orange + 0.55 * gray            # the trail is drawn at alpha 0.45
    extra = np.array([orange, np.array(to_rgb(SKY)) * 255, np.array(to_rgb(DODGER)) * 255, to_rgb(BG)], dtype=float)
    extra[3] = np.array(to_rgb(BG)) * 255
    blend = np.concatenate([0.5 * np.array(to_rgb(SKY)) * 255 + 0.5 * gray[::4], 0.5 * np.array(to_rgb(DODGER)) * 255 + 0.5 * gray[::4]])
    pal = np.concatenate([gray, trail, extra, blend])[:256]
    return np.vstack([pal, np.zeros((256 - len(pal), 3))])


def indexed(frame, _cache={}):
    from PIL import Image
    from scipy.spatial import cKDTree
    if not _cache:
        pal = _palette()
        _cache["tree"], _cache["pal"] = cKDTree(pal), pal
    idx = _cache["tree"].query(frame.reshape(-1, 3).astype(float))[1].astype(np.uint8).reshape(frame.shape[:2])
    img = Image.fromarray(idx, mode="P")
    img.putpalette(_cache["pal"].astype(np.uint8).ravel().tolist())
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--cell", required=True, help="y,x of the cell at t0")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--hold", type=float, default=6.0, help="seconds to stay on the final frame, whatever --slow is")
    ap.add_argument("--slow", type=float, default=4.0, help="multiplier on every frame duration")
    ap.add_argument("--half", type=float, default=48)
    ap.add_argument("--out", default="assets/pipeline_demo.gif")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    cell = [float(v) for v in a.cell.split(",")]
    sc = Scene(a.sample, a.t0, a.steps, cell)
    H, W = sc.stack.shape[2:]
    full_c, full_half = (H / 2, W / 2), H / 2
    c0 = sc.track[0, 1:]
    frames, dur = [], []

    def add(img, d):
        frames.append(img); dur.append(d)

    z0 = sc.track[0, 0]
    # 1. the raw stack, 2. detections on it
    add(sc.frame(0, full_c, full_half, z=z0, rings=False, selected=False, caption="Raw stack (one z-slice)"), 1.0)
    add(sc.frame(0, full_c, full_half, z=z0, rings=True, selected=False, caption="Cell centers detected"), 1.4)
    # 3. pick a cell, zoom in
    n_zoom = 8
    for i in range(n_zoom + 1):
        u = (1 - np.cos(np.pi * i / n_zoom)) / 2
        ctr = (1 - u) * np.array(full_c) + u * c0
        half = (1 - u) * full_half + u * a.half
        add(sc.frame(0, ctr, half, z=z0, rings=True, selected=True, caption="Pick a cell" if i < n_zoom else ""), 1.0 if i == 0 else (0.07 if i < n_zoom else 0.5))
    # 4. follow it, centered on the cell
    for k in range(1, a.steps + 1):
        add(sc.frame(k, sc.track[k, 1:], a.half, caption="Follow it over time"), 0.4)
    # 5. zoom out to the whole trajectory
    kk = a.steps
    for i in range(n_zoom + 1):
        u = (1 - np.cos(np.pi * i / n_zoom)) / 2
        ctr = (1 - u) * sc.track[kk, 1:] + u * np.array(full_c)
        half = (1 - u) * a.half + u * full_half
        add(sc.frame(kk, ctr, half, rings=True, caption="Its trajectory" if i == n_zoom else ""), 0.07 if i < n_zoom else a.hold / a.slow)
    if a.preview:
        for i in (1, 2, 11, 25, 40, len(frames) - 1):
            imageio.imwrite(f"/tmp/claude-0/-data-kaggle-biohub-cell-tracking/d8f3b20c-37c1-4368-bd7e-8f5ecc353d5b/scratchpad/prev_{i}.png", frames[i])
        print(len(frames), "frames"); return
    pal = [indexed(f) for f in frames]
    pal[0].save(a.out, save_all=True, append_images=pal[1:], duration=[int(d * a.slow * 1000) for d in dur], loop=0, optimize=False)
    print(a.out, len(frames), "frames", sum(dur) * a.slow, "s")


if __name__ == "__main__":
    main()
