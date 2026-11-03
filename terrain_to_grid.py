"""
terrain_to_grid.py -- builds the HEIGHT MAP + occupancy grid from exactly what Gazebo loads.

The raw OBJ is NOT what Gazebo shows. mars_1km.world applies, to the mesh:
    <scale>0.05 0.25 0.05</scale>          (per OBJ axis x, y, z)
    model pose roll = 1.57                 (turns the Y-up OBJ into Z-up)
so   world_x = 0.05*OBJ_x ,  world_y = -0.05*OBJ_z ,  world_z(height) = 0.25*OBJ_y
This module reads the .world file itself (mesh uri, scale, pose) and applies that transform, so the
height map is the one the lidar actually hits. Nothing is hard-coded except the world-file path.

Set the world file with  MARS_WORLD=/path/to/mars_1km.world   (default: mars_1km.world next to this file)
Optional OBJ override:   MARS_OBJ=/path/to/mars_terrain.obj
"""
import math, os, sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, uniform_filter, label

HERE = os.path.dirname(os.path.abspath(__file__))
WORLD_FILE = os.environ.get("MARS_WORLD", os.path.join(HERE, "/workspace/mars_1km.world"))
MODEL_NAME = "mars_terrain"
# used ONLY if the world file can't be found (values copied from your mars_1km.world)
FALLBACK = dict(scale=(0.05, 0.25, 0.05), pose=(0, 0, 0, 1.57, 0, 0), uri="mars_terrain.obj", pose_source="FALLBACK")

# ---------- grid knobs ----------
RESOLUTION    = 0.1      # m per cell
SLOPE_MAX_DEG = 12.0     # steeper than this = non-traversable
ROVER_RADIUS  = 0.2      # obstacle inflation
FOOTPRINT     = 0.4      # m: slope is measured over the rover's footprint (ignores 0.25 m mesh ripples the wheels roll over)
CROP          = None     # (x_min, x_max, y_min, y_max) in world metres, None = whole map


# ======================= world file =======================
def _floats(t):
    return [float(v) for v in t.split()]


def parse_world(world_file=None, model=MODEL_NAME):
    """mesh uri / scale / pose of the terrain model, as Gazebo will use them."""
    world_file = world_file or WORLD_FILE
    if not os.path.exists(world_file):
        print(f"WARNING: world file '{world_file}' not found -> using the scale/pose from your mars_1km.world "
              f"({FALLBACK['scale']}, roll {FALLBACK['pose'][3]}). Set MARS_WORLD to be sure.")
        return dict(FALLBACK, obj=None)
    world = ET.parse(world_file).getroot().find("world")
    mdl = next(m for m in world.findall("model") if m.get("name") == model)
    mesh = mdl.find(".//collision//mesh")                    # the collision mesh is what the lidar hits
    if mesh is None:
        mesh = mdl.find(".//visual//mesh")
    uri = mesh.find("uri").text.strip()
    sc = mesh.find("scale")
    scale = _floats(sc.text) if sc is not None else [1, 1, 1]
    pose, src = [0.0] * 6, "default"
    if mdl.find("pose") is not None:
        pose, src = _floats(mdl.find("pose").text), "model <pose>"
    st = world.find("state")                                 # a saved <state> OVERRIDES the model pose in Gazebo
    if st is not None:
        for sm in st.findall("model"):
            if sm.get("name") == model and sm.find("pose") is not None:
                pose, src = _floats(sm.find("pose").text), "world <state> pose"
    return dict(uri=uri, scale=tuple(scale), pose=tuple(pose), pose_source=src, obj=None)


def _resolve_obj(info, world_file):
    cands = []
    if os.environ.get("MARS_OBJ"):
        cands.append(os.environ["MARS_OBJ"])
    uri = info["uri"]
    if uri.startswith("file://"):
        cands.append(uri[len("file://"):])
    base = os.path.basename(uri)
    cands += [os.path.join(os.path.dirname(os.path.abspath(world_file or WORLD_FILE)), base),
              os.path.join(HERE, base), os.path.join(os.getcwd(), base)]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Terrain OBJ not found. Tried:\n  " + "\n  ".join(cands) + "\nSet MARS_OBJ=/path/to/mars_terrain.obj")


def rpy_matrix(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def to_world(V, scale, pose):
    """OBJ vertices -> Gazebo world frame: scale (in mesh axes) -> rotate (roll,pitch,yaw) -> translate."""
    R = rpy_matrix(*pose[3:6])
    return (V * np.asarray(scale)) @ R.T + np.asarray(pose[:3])


# ======================= height map =======================
def load_world_heightmap(world_file=None, obj_path=None, verbose=True, scale=None):
    """
    Returns (xs, ys, H, diag, info):  H[row=y, col=x] is the terrain height in the Gazebo WORLD frame,
    xs/ys ascending (metres). diag = 'anti'|'main' = which diagonal the OBJ splits each grid quad on
    (needed for exact triangle interpolation).
    """
    world_file = world_file or WORLD_FILE
    info = parse_world(world_file)
    if scale is not None:                       # what-if: pretend the world file had this <scale>
        info['scale'] = tuple(scale)
    obj = obj_path or _resolve_obj(info, world_file)
    V, F = [], []
    with open(obj) as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f ") and len(F) < 2000:
                F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    V = np.array(V)
    # the height axis is the OBJ axis with by far the most distinct values; the other two form the grid
    nuniq = [len(np.unique(V[:, i].round(4))) for i in range(3)]
    up = int(np.argmax(nuniq))
    a, b = [i for i in range(3) if i != up]
    ua, ub = np.unique(V[:, a].round(4)), np.unique(V[:, b].round(4))
    col, row = np.searchsorted(ua, V[:, a].round(4)), np.searchsorted(ub, V[:, b].round(4))
    W = to_world(V, info["scale"], info["pose"])
    nc, nr = len(ua), len(ub)
    cnt_c, cnt_r = np.bincount(col, minlength=nc), np.bincount(row, minlength=nr)
    xs = np.bincount(col, W[:, 0], nc) / cnt_c
    ys = np.bincount(row, W[:, 1], nr) / cnt_r
    # grid-alignment check: world x must follow the OBJ column axis, world y the row axis
    if np.abs(W[:, 0] - xs[col]).max() > 0.02 or np.abs(W[:, 1] - ys[row]).max() > 0.02:
        raise NotImplementedError("Mesh pose rotates the grid by something other than a flip/tilt; extend load_world_heightmap().")
    H = np.full((nr, nc), np.nan)
    H[row, col] = W[:, 2]
    # which diagonal do the faces use? (index space: row=b axis, col=a axis)
    sgn = []
    for tri in F[:200]:
        p = [(row[i], col[i]) for i in tri]
        for i in range(3):
            for j in range(i + 1, 3):
                if p[i][0] != p[j][0] and p[i][1] != p[j][1]:
                    sgn.append(np.sign((p[j][0] - p[i][0]) * (p[j][1] - p[i][1])))
    anti = np.median(sgn) < 0
    if xs[0] > xs[-1]:
        xs, H, anti = xs[::-1], H[:, ::-1], not anti
    if ys[0] > ys[-1]:
        ys, H, anti = ys[::-1], H[::-1, :], not anti
    diag = "anti" if anti else "main"
    info.update(obj=obj, up_axis_in_obj=up)
    if verbose:
        print(f"[terrain] OBJ   : {obj}")
        print(f"[terrain] scale : {info['scale']}   pose: {info['pose']}  (from {info['pose_source']})")
        print(f"[terrain] world : x[{xs[0]:.2f},{xs[-1]:.2f}] y[{ys[0]:.2f},{ys[-1]:.2f}] z[{np.nanmin(H):.2f},{np.nanmax(H):.2f}] m, "
              f"{nc}x{nr} vertices, spacing {xs[1]-xs[0]:.3f} m")
    return np.ascontiguousarray(xs), np.ascontiguousarray(ys), np.ascontiguousarray(H), diag, info


# ======================= occupancy grid =======================
def build_grid_from_obj(hm=None, crop=CROP, res=RESOLUTION, slope_max=SLOPE_MAX_DEG, rover_r=ROVER_RADIUS, footprint=FOOTPRINT):
    """hm = (xs, ys, H) from load_world_heightmap (loaded for you if None).
    Returns grid[row=y, col=x] (1 = blocked), meta, Z (height), slope (deg)."""
    xs, ys, H = hm if hm is not None else load_world_heightmap()[:3]
    interp = RegularGridInterpolator((ys, xs), H, bounds_error=False, fill_value=np.nan)
    x0, x1, y0, y1 = crop if crop else (xs[0], xs[-1], ys[0], ys[-1])
    cols, rows = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    cx = x0 + (np.arange(cols) + 0.5) * res                  # cell CENTRES, same convention as grid2world()
    cy = y0 + (np.arange(rows) + 0.5) * res
    X, Y = np.meshgrid(cx, cy)
    Z = interp(np.stack([Y.ravel(), X.ravel()], -1)).reshape(rows, cols)
    Zs = np.where(np.isnan(Z), np.nanmean(Z), Z)
    k = int(round(footprint / res))
    if k > 1:                                                # average height over the footprint, then take the slope
        Zs = uniform_filter(Zs, size=k, mode="nearest")
    dzdy, dzdx = np.gradient(Zs, res, res)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    blocked = (slope > slope_max) | np.isnan(Z)
    if rover_r > 0:
        blocked = distance_transform_edt(~blocked) * res <= rover_r
    grid = blocked.astype(np.uint8)
    grid[0, :] = grid[-1, :] = 1
    grid[:, 0] = grid[:, -1] = 1
    return grid, dict(origin_x=x0, origin_y=y0, resolution=res), Z, slope


# ======================= self-check =======================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, ys, H, diag, info = load_world_heightmap()
    grid, meta, Z, slope = build_grid_from_obj(hm=(xs, ys, H))
    ox = H[np.argmin(abs(ys)), np.argmin(abs(xs))]
    print(f"[terrain] height at world origin (0,0): {ox:+.2f} m   | quad diagonal: {diag}")
    for t in (5, 10, 15, 20, 25, 30):
        print(f"[terrain] cells steeper than {t:2d} deg: {100*(slope > t).mean():5.1f} %")
    print("[terrain] grid", grid.shape, "blocked", round(100 * grid.mean(), 1), "%", meta)
    np.save("grid.npy", grid)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ext = [xs[0], xs[-1], ys[0], ys[-1]]
    for a, d, t, cm in zip(ax, (H, slope, grid), ("height (m) - Gazebo world frame", "slope (deg)", "occupancy (black=blocked)"),
                           ("terrain", "magma", "gray_r")):
        im = a.imshow(d, origin="lower", extent=ext, cmap=cm); a.set_title(t); plt.colorbar(im, ax=a, shrink=.8)
        a.plot(0, 0, "c*", ms=12); a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
    plt.tight_layout(); plt.savefig("heightmap_check.png", dpi=110)
    print("saved heightmap_check.png (cyan star = world origin / rover spawn)")
    if "--verify" in sys.argv:       # independent check against a ray-cast of the transformed mesh (needs trimesh)
        import trimesh
        V, Fc = [], []
        for line in open(info["obj"]):
            if line.startswith("v "): V.append([float(t) for t in line.split()[1:4]])
            elif line.startswith("f "): Fc.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
        mesh = trimesh.Trimesh(to_world(np.array(V), info["scale"], info["pose"]), np.array(Fc), process=False)
        import PP_terrain as PT
        rng = np.random.default_rng(0)
        x, y = rng.uniform(xs[0] + 1, xs[-1] - 1, 300), rng.uniform(ys[0] + 1, ys[-1] - 1, 300)
        loc, ir, _ = mesh.ray.intersects_location(np.stack([x, y, np.full(300, 50.)], -1), np.tile([0, 0, -1.], (300, 1)), multiple_hits=False)
        err = np.abs(PT.height_at(x[ir], y[ir]) - loc[:, 2])
        print(f"[verify] height_at vs ray-cast mesh: max error {err.max()*1000:.2f} mm over {len(ir)} points")
