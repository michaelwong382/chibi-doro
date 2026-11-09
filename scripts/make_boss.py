#!/usr/bin/env python3
"""Replace the square socket under the Doro head with a male cylindrical boss.

Implements instructions_prompt.md:
  * boss axis    = centroid (symmetry) axis of the square socket
  * datum plane  = plane through 3 of the socket's 4 opening corners
  * diameter     = --diameter-pct of the bounding-box width (X)
  * height       = --height-pct of the head's height measured along the boss axis, and enough
                   that the head rests on the boss alone
  * nothing else on the head changes; the output is a closed, manifold solid

Method: delete only the socket's wall and floor triangles, close the opening with a cap that
continues the head's curvature (a thin-plate spline through the rim and the surrounding surface),
then boolean-union a cylinder (manifold3d) whose base is buried inside the head, so only the
projecting part shows. The result is reloaded from disk and checked.

Usage: python scripts/make_boss.py [--input headGK.stl] [--output headGK_boss.stl]
                                   [--diameter-pct 25] [--height-pct 12] [--report report.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import manifold3d as m3d
import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

# A point near the socket floor and the rough direction out of the socket; only used to find it.
SOCKET_FLOOR_HINT = np.array([-0.14, 0.93, -25.53])
SOCKET_AXIS_HINT = np.array([0.0, 0.32, -0.95])
FLAT_DEG = 2.0  # max dihedral (deg) between neighbouring triangles of one planar socket face
BURY = 2.0  # the boss starts this far inside the datum plane, hidden in the head
SEGMENTS = 128  # facets around the boss (chord error ~0.0014 at this size)
EDIT_MARGIN = 0.5  # original triangles this close to the boss volume may be re-cut by it
CAP_STEP = 0.25  # spacing (mm) between the cap's rings, about the rim's own vertex spacing
CAP_FIT_BAND = 3.0  # surrounding surface within this distance (mm) of the rim shapes the cap
CAP_INNER = 0.3  # innermost ring, as a fraction of the rim; well inside the boss, so never seen


# ---------------------------------------------------------------------------- helpers
def planar_components(mesh: trimesh.Trimesh, flat_deg: float) -> np.ndarray:
    """Label faces by connected region, joining neighbours only across near-flat edges."""
    adj = mesh.face_adjacency[mesh.face_adjacency_angles < np.radians(flat_deg)]
    n = len(mesh.faces)
    g = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n))
    return sp.csgraph.connected_components(g, directed=False)[1]


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    c = points.mean(0)
    _, _, vt = np.linalg.svd(points - c)
    n = vt[2]
    return c, n, float(np.abs((points - c) @ n).max())


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    c = u @ v / np.linalg.norm(u) / np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def boundary_loops(faces: np.ndarray, n_vertices: int) -> list[list[int]]:
    """Ordered boundary loops, following the faces' half-edge direction."""
    he = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = he[:, 0].astype(np.int64) * n_vertices + he[:, 1]
    rkey = he[:, 1].astype(np.int64) * n_vertices + he[:, 0]
    nxt = {}
    for u, v in he[~np.isin(rkey, key)]:
        if int(u) in nxt:
            raise RuntimeError("boundary is not a set of simple loops")
        nxt[int(u)] = int(v)
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, u = [], start
        while u not in seen:
            seen.add(u)
            loop.append(u)
            u = nxt[u]
        loops.append(loop)
    return loops


def tri_keys(ids: np.ndarray) -> np.ndarray:
    """One int per triangle from its 3 vertex ids; rotation-invariant, winding-sensitive."""
    k = np.argmin(ids, axis=1)
    rows = np.arange(len(ids))
    t = np.column_stack([ids[rows, k], ids[rows, (k + 1) % 3], ids[rows, (k + 2) % 3]])
    t = t.astype(np.int64)
    return (t[:, 0] << 42) | (t[:, 1] << 21) | t[:, 2]


def shared_ids(v1: np.ndarray, v2: np.ndarray):
    """Common integer ids for two vertex arrays (exact float32 positions)."""
    uniq, inv = np.unique(np.vstack([v1, v2]).astype(np.float32), axis=0, return_inverse=True)
    inv = inv.ravel()
    return uniq, inv[: len(v1)], inv[len(v1):]


def frame(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.array([1.0, 0, 0]) - a[0] * a
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def axial(points: np.ndarray, p0: np.ndarray, a: np.ndarray):
    """Distance along the axis (outward positive) and radial distance from it."""
    s = (points - p0) @ a
    return s, np.linalg.norm(points - p0 - s[..., None] * a, axis=-1)


def to_manifold(vertices, faces) -> m3d.Manifold:
    m = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(vertices, np.float32),
                              tri_verts=np.asarray(faces, np.uint32)))
    if m.status() != m3d.Error.NoError:
        raise RuntimeError(f"manifold3d rejected the mesh: {m.status()}")
    return m


# ---------------------------------------------------------------------------- socket
def find_socket(mesh: trimesh.Trimesh) -> dict:
    labels = planar_components(mesh, FLAT_DEG)
    hint = SOCKET_AXIS_HINT / np.linalg.norm(SOCKET_AXIS_HINT)
    cand = np.where(mesh.face_normals @ hint > np.cos(np.radians(10)))[0]
    dist = np.linalg.norm(mesh.triangles_center[cand] - SOCKET_FLOOR_HINT, axis=1)
    floor = np.where(labels == labels[cand[np.argmin(dist)]])[0]

    in_floor = np.zeros(len(mesh.faces), bool)
    in_floor[floor] = True
    adj = mesh.face_adjacency
    across = adj[in_floor[adj[:, 0]] ^ in_floor[adj[:, 1]]]
    outside = np.where(in_floor[across[:, 0]], across[:, 1], across[:, 0])
    wall_labels = np.unique(labels[outside])
    if len(wall_labels) != 4:
        raise RuntimeError(f"expected 4 socket walls around the floor, found {len(wall_labels)}")
    walls = [np.where(labels == lab)[0] for lab in wall_labels]

    _, a, floor_resid = fit_plane(mesh.vertices[np.unique(mesh.faces[floor])])
    a = a if a @ hint > 0 else -a  # outward: from the socket floor toward its opening
    wall_planes = [fit_plane(mesh.vertices[np.unique(mesh.faces[w])]) for w in walls]
    resid = max([floor_resid] + [p[2] for p in wall_planes])
    if resid > 0.05:
        raise RuntimeError(f"socket faces are not planar (max residual {resid:.3f})")
    drafts = [float(np.degrees(np.arcsin(abs(p[1] @ a)))) for p in wall_planes]
    if max(drafts) - min(drafts) > 0.5:
        raise RuntimeError(f"socket walls are not symmetric about the floor normal: {drafts}")

    socket = np.concatenate([floor] + walls)
    keep = np.ones(len(mesh.faces), bool)
    keep[socket] = False
    loops = boundary_loops(mesh.faces[keep], len(mesh.vertices))
    if len(loops) != 1:
        raise RuntimeError(f"removing the socket should leave one opening, found {len(loops)}")
    rim = loops[0]

    floor_v = set(np.unique(mesh.faces[floor]).tolist())
    wall_v = [set(np.unique(mesh.faces[w]).tolist()) for w in walls]
    corner_v = set()
    for i, j in itertools.combinations(range(4), 2):
        corner_v |= wall_v[i] & wall_v[j]
    floor_corners = sorted(corner_v & floor_v)
    open_corners = sorted(corner_v & set(rim))
    if len(floor_corners) != 4 or len(open_corners) != 4:
        raise RuntimeError("could not identify the socket's 4 floor and 4 opening corners")
    return dict(faces=socket, axis=a, drafts=drafts, rim=rim,
                floor_corners=mesh.vertices[floor_corners],
                open_corners=mesh.vertices[open_corners])


def smooth_cap(head: trimesh.Trimesh, socket_faces: np.ndarray, rim: list[int],
               p0: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Close the socket opening with a surface that carries on the head's curvature.

    Around the socket the head is a height field over the plane square to the axis (every face
    there faces out along it), so the cap's heights come from a thin-plate spline through the rim
    and the surrounding surface. The opening is star-shaped around the axis, so the cap is meshed
    as scaled copies of the rim loop, fanned to the axis at the middle. Its outer loop is the rim
    itself, so it joins the head exactly.

    Returns the new vertices, the cap's faces (rim ids as-is, new vertices numbered from
    len(head.vertices)), and how far the cap rises above a flat fan from p0 at most.
    """
    V = head.vertices
    u, v = frame(a)

    def local(P):
        d = P - p0
        return np.column_stack([d @ u, d @ v]), d @ a

    rim = np.asarray(rim)
    rim_uv, rim_s = local(V[rim])
    others = np.setdiff1d(np.arange(len(V)), np.unique(head.faces[socket_faces]))
    uv, s = local(V[others])
    band = (cKDTree(rim_uv).query(uv)[0] < CAP_FIT_BAND) & (np.abs(s) < 4)
    surface = RBFInterpolator(np.vstack([rim_uv, uv[band]]), np.concatenate([rim_s, s[band]]),
                              kernel="thin_plate_spline")

    n = len(rim)
    reach = float(np.linalg.norm(rim_uv, axis=1).max())
    scales = np.linspace(1, CAP_INNER, int(np.ceil(reach * (1 - CAP_INNER) / CAP_STEP)) + 1)[1:]
    ring_uv = np.concatenate([rim_uv * k for k in scales] + [np.zeros((1, 2))])
    ring_s = surface(ring_uv)
    new_v = p0 + ring_uv[:, :1] * u + ring_uv[:, 1:] * v + ring_s[:, None] * a

    base = len(V)
    rings = [rim] + [base + i * n + np.arange(n) for i in range(len(scales))]
    centre = base + len(scales) * n
    j, j1 = np.arange(n), (np.arange(n) + 1) % n
    faces = []
    for outer, inner in zip(rings[:-1], rings[1:]):  # same winding as a fan (centre, next, this)
        faces += [np.column_stack([inner[j], outer[j1], outer[j]]),
                  np.column_stack([inner[j], inner[j1], outer[j1]])]
    faces.append(np.column_stack([np.full(n, centre), rings[-1][j1], rings[-1][j]]))

    # how much higher than the old flat fan from p0 the cap now sits, over the exposed corners
    flat_s = np.concatenate([rim_s * k for k in scales] + [np.zeros(1)])
    return new_v, np.vstack(faces), float((ring_s - flat_s).max())


def sides(corners: np.ndarray) -> list[float]:
    """Side lengths of a quad given its 4 corners in any order (the 4 shortest of 6 distances)."""
    d = [float(np.linalg.norm(corners[i] - corners[j])) for i, j in itertools.combinations(range(4), 2)]
    return [round(x, 3) for x in sorted(d)[:4]]


def corner_name(p: np.ndarray, centre: np.ndarray) -> str:
    # the model faces -Y with +Z up, so +X is the character's left
    return ("front" if p[1] < centre[1] else "back") + "-" + ("left" if p[0] > centre[0] else "right")


# ---------------------------------------------------------------------------- build
def build(head: trimesh.Trimesh, diameter_pct: float, height_pct: float) -> tuple:
    V = head.vertices
    sock = find_socket(head)
    a = sock["axis"]
    f0 = sock["floor_corners"].mean(0)  # the socket's centroid axis is f0 + t*a

    # Datum plane: of the 4 ways to pick 3 opening corners, use the one most square to the axis.
    oc = sock["open_corners"]
    options = []
    for tri in itertools.combinations(range(4), 3):
        p = oc[list(tri)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        n /= np.linalg.norm(n)
        n = n if n @ a > 0 else -n
        other = (set(range(4)) - set(tri)).pop()
        options.append((angle_deg(n, a), tri, n, float((oc[other] - p[0]) @ n), other))
    tilt, tri, n_d, fourth_off, fourth = min(options, key=lambda o: o[0])
    p0 = f0 + ((oc[tri[0]] - f0) @ n_d) / (a @ n_d) * a  # boss axis meets the datum plane

    width = float(head.extents[0])
    r = diameter_pct / 100 * width / 2
    s_all = V @ a
    head_height = float(s_all.max() - s_all.min())
    height = height_pct / 100 * head_height

    socket_only = np.zeros(len(V), bool)
    socket_only[np.unique(head.faces[sock["faces"]])] = True
    socket_only[sock["rim"]] = False
    s, rad = axial(V, p0, a)
    need = float(s[~socket_only & (rad > r)].max())  # lowest other point when resting on the boss
    if need >= height:
        raise RuntimeError(f"a {height:.3f} boss would not clear the head (needs > {need:.3f})")

    # 1) Remove the socket and close the opening with a cap that follows the head's curvature,
    #    so the corners of the old square outside the round boss end up flush with the head.
    keep = np.ones(len(head.faces), bool)
    keep[sock["faces"]] = False
    cap_v, cap, cap_rise = smooth_cap(head, sock["faces"], sock["rim"], p0, a)
    verts = np.vstack([V, cap_v])
    faces = np.vstack([head.faces[keep], cap])
    cap_n = np.cross(verts[cap[:, 1]] - verts[cap[:, 0]], verts[cap[:, 2]] - verts[cap[:, 0]])
    if np.any(cap_n @ a <= 0):
        raise RuntimeError("socket opening is not star-shaped around the boss axis")
    used = np.unique(faces)
    remap = np.full(len(verts), -1)
    remap[used] = np.arange(len(used))
    capped = trimesh.Trimesh(verts[used], remap[faces], process=False)
    if not capped.is_watertight or capped.euler_number != head.euler_number:
        raise RuntimeError("capping the socket did not give a closed surface of the same topology")

    # 2) Union the boss: from BURY inside the datum plane out to `height` along the axis.
    u, v = frame(a)
    t = np.linspace(0, 2 * np.pi, SEGMENTS, endpoint=False)
    ring = p0 + r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    k = np.arange(SEGMENTS)
    k1 = (k + 1) % SEGMENTS
    n = SEGMENTS
    cyl_v = np.vstack([ring - BURY * a, ring + height * a, p0 - BURY * a, p0 + height * a])
    cyl_f = np.vstack([np.column_stack([k, k1, k1 + n]), np.column_stack([k, k1 + n, k + n]),
                       np.column_stack([np.full(n, 2 * n + 1), k + n, k1 + n]),
                       np.column_stack([np.full(n, 2 * n), k1, k])])
    result = to_manifold(capped.vertices, capped.faces) + to_manifold(cyl_v, cyl_f)
    if result.status() != m3d.Error.NoError or len(result.decompose()) != 1:
        raise RuntimeError("boolean union failed")
    out = result.to_mesh()
    out_v = np.asarray(out.vert_properties)[:, :3].astype(np.float64)
    out_f = np.asarray(out.tri_verts).astype(np.int64)

    # 3) manifold3d tidies sliver triangles when it builds a solid (here it flips the edge of one
    #    near-zero-width sliver, ~1e-5 across, on the side of the head). Put back every original
    #    triangle outside the socket and the boss's reach so the rest stays bit-for-bit original.
    fs, fr = axial(head.triangles, p0, a)
    near = np.any((fr < r + EDIT_MARGIN) & (fs > -BURY - EDIT_MARGIN)
                  & (fs < height + EDIT_MARGIN), axis=1)
    protect = ~near
    protect[sock["faces"]] = False
    uniq, o_ids, n_ids = shared_ids(head.vertices, out_v)
    ko, kn = tri_keys(o_ids[head.faces]), tri_keys(n_ids[out_f])
    lost = np.where(protect & ~np.isin(ko, kn))[0]
    if len(lost):
        lost_uid = np.unique(o_ids[head.faces[lost]])
        drop = ~np.isin(kn, ko) & np.all(np.isin(n_ids[out_f], lost_uid), axis=1)
        to_out = np.full(len(uniq), -1)
        to_out[n_ids] = np.arange(len(out_v))
        gone = lost_uid[to_out[lost_uid] < 0]  # vertices the engine collapsed away
        to_out[gone] = len(out_v) + np.arange(len(gone))
        out_v = np.vstack([out_v, uniq[gone]])
        out_f = np.vstack([out_f[~drop], to_out[o_ids[head.faces[lost]]]])

    info = {
        "socket": {
            "triangles": int(len(sock["faces"])),
            "axis": a.round(5).tolist(),
            "axis_tilt_from_-Z_toward_+Y_deg": round(angle_deg(a, np.array([0, 0, -1.0])), 3),
            "floor_centre": f0.round(4).tolist(),
            "floor_sides": sides(sock["floor_corners"]),
            "opening_sides": sides(oc),
            "depth_along_axis": round(float((oc.mean(0) - f0) @ a), 3),
            "wall_draft_deg": round(float(np.mean(sock["drafts"])), 2),
        },
        "datum": {
            "corners": [corner_name(oc[i], oc.mean(0)) for i in tri],
            "unused_corner": corner_name(oc[fourth], oc.mean(0)),
            "unused_corner_offset": round(fourth_off, 4),
            "angle_to_axis_deg": round(tilt, 3),
            "axis_point": p0.round(4).tolist(),
        },
        "boss": {
            "bbox_width_x": round(width, 4),
            "diameter": round(2 * r, 4),
            "head_height_along_axis": round(head_height, 4),
            "height": round(height, 4),
            "min_height_to_clear": round(need, 4),
        },
        "cap": {
            "vertices": int(len(cap_v)),
            "max_rise_over_flat_fan": round(cap_rise, 4),
        },
        "original_triangles_restored": int(len(lost)),
    }
    geom = dict(a=a, p0=p0, r=r, height=height, protect=protect, socket=sock["faces"])
    return out_v, out_f, info, geom


# ---------------------------------------------------------------------------- verify
def verify(path: str, head: trimesh.Trimesh, g: dict) -> dict:
    """Reload the written STL and check it against the spec's acceptance criteria."""
    m = trimesh.load(path, force="mesh")
    F = m.faces
    checks = {}
    checks["watertight: every edge shared by exactly 2 triangles"] = m.is_watertight
    checks["consistent winding, outward normals"] = m.is_winding_consistent and m.is_volume
    checks["one body"] = len(m.split(only_watertight=False)) == 1
    checks[f"topology unchanged (Euler {head.euler_number} -> {m.euler_number})"] = (
        m.euler_number == head.euler_number)
    checks["no degenerate triangles"] = bool(m.area_faces.min() > 1e-10)
    s_f = np.sort(F, axis=1)
    checks["no duplicate triangles"] = len(np.unique(s_f, axis=0)) == len(F)
    used = np.zeros(len(m.vertices), bool)
    used[F.ravel()] = True
    checks["no unreferenced (hanging) vertices"] = bool(used.all())
    # each vertex's triangles must form a single fan (no pinched / bow-tie vertices)
    rows, cols = [], []
    for col in range(2):
        vv = m.face_adjacency_edges[:, col]
        for side, out in ((0, rows), (1, cols)):
            f = m.face_adjacency[:, side]
            out.append(f * 3 + np.argmax(F[f] == vv[:, None], axis=1))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(3 * len(F),) * 2)
    lab = sp.csgraph.connected_components(graph, directed=False)[1]
    fans = len(np.unique(np.column_stack([F.ravel(), lab]), axis=0))
    checks["no non-manifold vertices"] = fans == len(m.vertices)

    _, o_ids, n_ids = shared_ids(head.vertices, m.vertices)
    ko, kn = tri_keys(o_ids[head.faces]), tri_keys(n_ids[F])
    kept = np.isin(ko, kn)
    checks[f"rest of head bit-for-bit original ({int(kept.sum())}/{len(kept)} triangles kept)"] = (
        bool(kept[g["protect"]].all()))

    a, p0, r, h = g["a"], g["p0"], g["r"], g["height"]
    s, rad = axial(m.vertices, p0, a)
    end = np.abs(s - s.max()) < 1e-3
    clear = float(s.max() - s[rad > r + 1e-3].max())
    checks[f"boss end face: dia {2 * rad[end].max():.3f}, height {s.max():.3f}"] = bool(
        abs(rad[end].max() - r) < 2e-3 and abs(s.max() - h) < 2e-3)
    checks[f"rests on the boss alone ({clear:.3f} clearance)"] = clear > 0.1
    off = float(axial(m.center_mass[None], p0, a)[1][0])
    checks[f"centre of mass over the boss ({off:.3f} off-axis < radius {r:.3f})"] = off < r
    return {"checks": {k: bool(v) for k, v in checks.items()}, "triangles": int(len(F)),
            "volume": round(float(m.volume), 3), "volume_original": round(float(head.volume), 3),
            "clearance": round(clear, 4), "com_offset": round(off, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default="headGK.stl")
    ap.add_argument("--output", default="headGK_boss.stl")
    ap.add_argument("--diameter-pct", type=float, default=25.0)
    ap.add_argument("--height-pct", type=float, default=12.0)
    ap.add_argument("--report", help="also write the measurements and checks to this JSON file")
    args = ap.parse_args()

    head = trimesh.load(args.input, force="mesh")  # merges the STL's duplicate vertices
    if not (head.is_watertight and head.is_winding_consistent):
        raise RuntimeError("input head is not a closed, consistently wound mesh")
    out_v, out_f, info, geom = build(head, args.diameter_pct, args.height_pct)
    trimesh.Trimesh(out_v, out_f, process=False).export(args.output)
    info["result"] = verify(args.output, head, geom)
    info["spec"] = {"diameter_pct_of_width": args.diameter_pct,
                    "height_pct_of_head_height": args.height_pct}

    print(json.dumps({k: info[k] for k in ("socket", "datum", "boss", "cap")}, indent=2))
    print(f"\nwrote {args.output}: {info['result']['triangles']} triangles, "
          f"volume {info['result']['volume_original']} -> {info['result']['volume']}")
    for name, ok in info["result"]["checks"].items():
        print(("PASS  " if ok else "FAIL  ") + name)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(info, fh, indent=2)
    return 0 if all(info["result"]["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
