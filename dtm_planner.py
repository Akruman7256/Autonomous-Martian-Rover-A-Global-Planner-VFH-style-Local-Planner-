#!/usr/bin/env python3
"""
dtm_planner.py -- DTM elevation map -> 3-class traversability map -> cost grid -> A* path.

Classes:  0 FREE | 1 DIFFICULT (passable, costly) | 2 BLOCKED (slope / nodata / inflated)

Inputs (--dtm):  .tif (GeoTIFF DTM, needs rasterio) | .npz (xs, ys, H) | .obj (+ --world, uses terrain_to_grid.py)
Usage:
  python3 dtm_planner.py --dtm mars.tif --start 0 0 --goal 15 10 --plot
  python3 dtm_planner.py --dtm /workspace/rover_project/worlds/nasa_jpl/mars_terrain/meshes/mars_terrain.obj --world /workspace/mars_1km_robhar.world --start 0 0 --goal 200 400 --plot
Library:
  cm = build_costmap(*load_dtm("mars.tif"), res=0.2);  cm.save("cm.npz")   # once
  cm = CostMap.load("cm.npz");  path = plan(cm, (0, 0), (15, 10))          # many times, instant
"""
import argparse, heapq, math, os
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator

FREE, DIFFICULT, BLOCKED = 0, 1, 2
SQ2 = math.sqrt(2)
CACHE_VERSION = 2


# ============================ 1. load DTM ============================
def load_dtm(path, world=None):
    """-> xs, ys (ascending, metres), H[row=y, col=x]"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        d = np.load(path); return d["xs"], d["ys"], d["H"]
    if ext in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(path) as s:
            H = s.read(1).astype(float)
            if s.nodata is not None:
                H[H == s.nodata] = np.nan
            t = s.transform
            xs = t.c + (np.arange(s.width) + .5) * t.a
            ys = t.f + (np.arange(s.height) + .5) * t.e
        if ys[0] > ys[-1]:
            ys, H = ys[::-1], H[::-1]
        return xs, ys, np.ascontiguousarray(H)
    if ext == ".obj":                       # reuse your Gazebo-correct transform (scale + roll)
        import terrain_to_grid as t
        try:
            xs, ys, H, _, _ = t.load_world_heightmap(world_file=world, obj_path=path)
        except NotImplementedError:             # mesh is rotated (e.g. yaw) in the world file -> rasterise instead
            xs, ys, H = _rasterise_rotated_obj(path, world, t)
        return xs, ys, H
    raise ValueError(f"unsupported DTM type: {ext}")


def _rasterise_rotated_obj(path, world, t):
    """Fallback: apply scale+pose from the world file to every vertex, then interpolate onto a straight grid."""
    from scipy.interpolate import griddata
    info = t.parse_world(world)
    print(f"[dtm] mesh is rotated in the world -> rasterising. scale={info['scale']} pose={info['pose']} ({info['pose_source']})")
    V = np.array([[float(v) for v in l.split()[1:4]] for l in open(path) if l.startswith("v ")])
    W = t.to_world(V, info["scale"], info["pose"])
    step = max(math.sqrt(np.ptp(W[:, 0]) * np.ptp(W[:, 1]) / len(W)), 0.05)
    while (np.ptp(W[:, 0]) / step) * (np.ptp(W[:, 1]) / step) > 4e6: step *= 1.5
    xs = np.arange(W[:, 0].min(), W[:, 0].max(), step); ys = np.arange(W[:, 1].min(), W[:, 1].max(), step)
    X, Y = np.meshgrid(xs, ys)
    H = griddata(W[:, :2], W[:, 2], (X, Y), method="linear")        # NaN outside the mesh -> becomes BLOCKED
    print(f"[dtm] raster {H.shape} @ {step:.3f} m | world x[{xs[0]:.1f},{xs[-1]:.1f}] y[{ys[0]:.1f},{ys[-1]:.1f}] z[{np.nanmin(H):.2f},{np.nanmax(H):.2f}]")
    return xs, ys, H


# ============================ 2. costmap ============================
class CostMap:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def w2g(self, x, y):
        return int(round((y - self.y0) / self.res)), int(round((x - self.x0) / self.res))

    def g2w(self, r, c):
        return self.x0 + c * self.res, self.y0 + r * self.res

    @property
    def grid(self):                        # binary grid, drop-in for the old PP.py consumers
        return (self.cls == BLOCKED).astype(np.uint8)

    def save(self, p):
        np.savez_compressed(p, **self.__dict__)

    @staticmethod
    def load(p):
        d = np.load(p); return CostMap(**{k: d[k][()] if d[k].ndim == 0 else d[k] for k in d.files})


def build_costmap(xs, ys, H, res=0.1, footprint=0.6, rover_r=0.33,
                  slope_diff=8.0, slope_max=15.0,       # deg: > diff = difficult, > max = blocked      
                  safety=0.5, min_blob=0.2,             # m soft-cost halo around obstacles / m^2 specks to drop
                  diff_mult=3.0, slope_w=4.0):
    dx = xs[1] - xs[0]
    while ((xs[-1] - xs[0]) / res) * ((ys[-1] - ys[0]) / res) > 4e6:      # protect RAM
        res *= 1.5; print(f"[costmap] grid too big -> coarsening to {res:.2f} m")
    if footprint < 2 * res:
        footprint = 2 * res; print(f"[costmap] footprint raised to {footprint:.2f} m (can't see detail finer than the grid)")

    bad = np.isnan(H)
    if bad.any():                                       # fill holes with nearest value (flagged blocked below)
        H = H[tuple(ndi.distance_transform_edt(bad, return_distances=False, return_indices=True))]
    if res > dx * 1.01:                                 # anti-alias before coarsening
        H = ndi.gaussian_filter(H, res / dx / 2)
    gx = np.arange(xs[0], xs[-1] + 1e-9, res); gy = np.arange(ys[0], ys[-1] + 1e-9, res)
    if abs(res - dx) < 1e-9 and abs(res - (ys[1] - ys[0])) < 1e-9:
        Z, nodata = H.copy(), bad
    else:
        X, Y = np.meshgrid(gx, gy)
        pts = np.stack([Y.ravel(), X.ravel()], -1)
        Z = RegularGridInterpolator((ys, xs), H, bounds_error=False, fill_value=None)(pts).reshape(X.shape)
        nodata = RegularGridInterpolator((ys, xs), bad.astype(float), bounds_error=False,
                                         fill_value=1.0)(pts).reshape(X.shape) > 0.5

    # ---- terrain features (all vectorised, footprint-scale) ----
    Zs = ndi.gaussian_filter(Z, footprint / res / 2)            # what the rover "feels" as its local plane
    gyy, gxx = np.gradient(Zs, res)
    slope = np.degrees(np.arctan(np.hypot(gxx, gyy)))
    

    # ---- classify: worst feature wins ----
    cls = np.zeros(Z.shape, dtype=np.uint8)

    cls[slope > slope_diff] = DIFFICULT
    cls[(slope > slope_max) | nodata] = BLOCKED
    
    cls = ndi.median_filter(cls, size=3)

    # Never allow unknown terrain to become traversable.
    cls[nodata] = BLOCKED

    # Keep the map boundary blocked.
    cls[0, :] = cls[-1, :] = BLOCKED
    cls[:, 0] = cls[:, -1] = BLOCKED

    # ---- inflate by rover radius + soft halo ----
    dist = ndi.distance_transform_edt(cls != BLOCKED) * res      # metres to nearest obstacle
    cls[dist <= rover_r] = BLOCKED
    cost = 1.0 + slope_w * np.clip(slope / slope_max, 0, 1) ** 2
    cost[cls == DIFFICULT] *= diff_mult
    halo = (dist > rover_r) & (dist < rover_r + safety)
    cost[halo] += 5.0 * np.exp(-(dist[halo] - rover_r) / (safety / 3))
    cost[cls == BLOCKED] = np.inf

    region, _ = ndi.label(np.isfinite(cost), structure=np.ones((3, 3)))   # connected traversable regions
    return CostMap(cls=cls, cost=cost, Z=Z, slope=slope, region=region,
                   x0=float(gx[0]), y0=float(gy[0]), res=float(res))


# ============================ 3. A* ============================
def astar(cost, s, g):
    """8-connected A* on a float cost array (inf = blocked). No corner cutting. Returns [(r,c)], g-values."""
    P = np.pad(cost, 1, constant_values=np.inf)                  # inf border -> no bounds checks in the loop
    R, C = P.shape
    flat = P.ravel()
    S, Gl = (s[0] + 1) * C + s[1] + 1, (g[0] + 1) * C + g[1] + 1
    gr, gc = g[0] + 1, g[1] + 1
    mv = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2)]

    def h(i):                                                    # octile distance, admissible (min cost = 1)
        dr, dc = abs(i // C - gr), abs(i % C - gc)
        return (dr + dc) + (SQ2 - 2) * min(dr, dc)

    G = {S: 0.0}; par = {}; closed = set(); pq = [(h(S), S)]
    while pq:
        _, u = heapq.heappop(pq)
        if u in closed: continue
        closed.add(u)
        if u == Gl: break
        gu = G[u]
        for dr, dc, ln in mv:
            v = u + dr * C + dc
            if flat[v] == math.inf or v in closed: continue
            if dr and dc and (flat[u + dr * C] == math.inf or flat[u + dc] == math.inf): continue
            gv = gu + ln * 0.5 * (flat[u] + flat[v])
            if gv < G.get(v, math.inf):
                G[v] = gv; par[v] = u; heapq.heappush(pq, (gv + h(v), v))
    if Gl not in G: return None, None
    idx = [Gl]
    while idx[-1] != S: idx.append(par[idx[-1]])
    idx = idx[::-1]
    return [(i // C - 1, i % C - 1) for i in idx], [G[i] for i in idx]


def smooth(cost, path, gv, tol=1.05):
    """Line-of-sight shortcutting: skip waypoints if the straight line is not costlier than the A* detour."""
    def line(a, b):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        rr = np.linspace(a[0], b[0], n).round().astype(int); cc = np.linspace(a[1], b[1], n).round().astype(int)
        return cost[rr, cc].mean() * math.dist(a, b)
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line(path[i], path[j]) <= tol * (gv[j] - gv[i]): j -= 1
        out.append(path[j]); i = j
    return out


# ============================ 4. plan ============================
def plan(cm, start_w, goal_w, do_smooth=True):
    """-> list of (x, y) world waypoints, or None. Prints WHY if it can't plan."""
    R, C = cm.cls.shape
    clamp = lambda c: (min(max(c[0], 0), R - 1), min(max(c[1], 0), C - 1))
    sizes = np.bincount(cm.region.ravel()); sizes[0] = 0
    if sizes.sum() == 0:
        print("No traversable cells at all -> thresholds too strict for this terrain (raise --slope-max)."); return None
    cells = {"start": cm.w2g(*start_w), "goal": cm.w2g(*goal_w)}
    reg = {}
    for n, cell in cells.items():
        r, c = clamp(cell); inside = (r, c) == tuple(cell)
        reg[n] = int(cm.region[r, c])
        print(f"  {n}: world {start_w if n == 'start' else goal_w} -> cell {cell}{'' if inside else ' (OUTSIDE map!)'} | "
              f"{['FREE', 'DIFFICULT', 'BLOCKED'][cm.cls[r, c]]} | slope {cm.slope[r, c]:.1f} deg | "
              f"region {reg[n]} ({100 * sizes[reg[n]] / sizes.sum():.1f}% of traversable cells)")
    cands = [v for v in reg.values() if v > 0]
    comp = max(cands, key=lambda v: sizes[v]) if cands else int(sizes.argmax())
    if reg["start"] != reg["goal"] or reg["start"] == 0:
        print(f"  NOTE: start/goal are not in the same traversable region -> planning inside the larger one (region {comp}).")
    snapped = {}
    for n, cell in cells.items():
        r, c = clamp(cell)
        if cm.region[r, c] != comp:
            idx = ndi.distance_transform_edt(cm.region != comp, return_distances=False, return_indices=True)
            nr, nc = int(idx[0][r, c]), int(idx[1][r, c])
            print(f"  WARNING: {n} moved {math.dist((r, c), (nr, nc)) * cm.res:.1f} m to world "
                  f"({cm.g2w(nr, nc)[0]:.1f}, {cm.g2w(nr, nc)[1]:.1f}) because it was not on the main traversable region.")
            snapped[n] = (nr, nc)
        else:
            snapped[n] = (r, c)
    path, gv = astar(cm.cost, snapped["start"], snapped["goal"])
    if path is None: return None
    if do_smooth: path = smooth(cm.cost, path, gv)
    return [cm.g2w(r, c) for r, c in path]


def plot(cm, wp, f="path.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    ext = [cm.x0, cm.x0 + cm.cls.shape[1] * cm.res, cm.y0, cm.y0 + cm.cls.shape[0] * cm.res]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im = ax[0].imshow(cm.Z, origin="lower", extent=ext, cmap="terrain"); plt.colorbar(im, ax=ax[0], shrink=.8)
    ax[0].set_title("DTM height (m)")
    ax[1].imshow(cm.cls, origin="lower", extent=ext, cmap=ListedColormap(["#4caf50", "#ffd54f", "#d32f2f"]), vmin=0, vmax=2)
    ax[1].set_title("green=free  yellow=difficult  red=blocked")
    if wp:
        px, py = zip(*wp)
        for a in ax: a.plot(px, py, "c-", lw=2); a.plot(px[0], py[0], "ko"); a.plot(px[-1], py[-1], "kx", ms=10)
    plt.tight_layout(); plt.savefig(f, dpi=110); print("saved", f)


if __name__ == "__main__":
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtm", required=True); ap.add_argument("--world")
    ap.add_argument("--start", nargs=2, type=float, required=True); ap.add_argument("--goal", nargs=2, type=float, required=True)
    ap.add_argument("--res", type=float, default=None, help="cell size in m (default: the DTM's own spacing)"); ap.add_argument("--footprint", type=float, default=0.6)
    ap.add_argument("--rover-r", type=float, default=0.33)
    ap.add_argument("--slope-diff", type=float, default=8); ap.add_argument("--slope-max", type=float, default=15)
    ap.add_argument("--cache", help="costmap .npz: loaded if it exists, else built and saved")
    ap.add_argument("--out"); ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    t = time.time()
    mt = lambda f: os.path.getmtime(f) if f and os.path.exists(f) else 0
    key = repr((
        CACHE_VERSION,
        os.path.abspath(a.dtm),
        mt(a.dtm),
        a.world,
        mt(a.world),
        a.res,
        a.footprint,
        a.rover_r,
        a.slope_diff,
        a.slope_max,
    ))
    cm = None
    if a.cache and os.path.exists(a.cache):
        c = CostMap.load(a.cache)
        if str(getattr(c, "key", "")) == key: cm = c; print("[cache] using", a.cache)
        else: print("[cache] settings or DTM changed -> rebuilding")
    if cm is None:
        xs, ys, H = load_dtm(a.dtm, a.world)
        res = a.res or max(xs[1] - xs[0], ys[1] - ys[0])
        print(f"[dtm] DTM spacing {xs[1]-xs[0]:.3f} m -> using grid res {res:.3f} m")
        cm = build_costmap(xs, ys, H, res=res, footprint=a.footprint, rover_r=a.rover_r,
                           slope_diff=a.slope_diff, slope_max=a.slope_max)
        cm.key = key
        if a.cache: cm.save(a.cache)
    n = cm.cls.size
    print(f"costmap {cm.cls.shape} in {time.time()-t:.2f}s | free {100*(cm.cls==0).sum()/n:.0f}% "
          f"difficult {100*(cm.cls==1).sum()/n:.0f}% blocked {100*(cm.cls==2).sum()/n:.0f}%")
    t = time.time(); wp = plan(cm, tuple(a.start), tuple(a.goal))
    print("NO PATH" if wp is None else f"path: {len(wp)} waypoints, {sum(math.dist(wp[i], wp[i+1]) for i in range(len(wp)-1)):.1f} m, planned in {time.time()-t:.2f}s")
    if wp and a.out:
        import csv
        with open(a.out, "w", newline="") as f: w = csv.writer(f); w.writerow(["x", "y"]); w.writerows(wp)
    if a.plot: plot(cm, wp)
