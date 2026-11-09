#!/usr/bin/env python3
"""Render README images of the original head and the boss version (pure numpy, no GPU).

The socket faces removed from the original are tinted orange; the faces added in the
boss version (cap + boss) are tinted blue. Images are RGBA PNGs with transparent backgrounds.

Usage: python scripts/render.py [--original headGK.stl] [--modified headGK_boss.stl] [--out renders]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import trimesh
from PIL import Image

SS = 2  # supersampling factor
BASE = (226, 222, 214)  # clay
REMOVED = (240, 135, 70)  # socket faces that get replaced (orange)
ADDED = (80, 150, 235)  # new cap + boss faces (blue)


def srgb_to_lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def corner_normals(V, F, smooth_deg=35.0):
    """Per-corner normals averaged only over neighbouring faces within smooth_deg (keeps hard edges)."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fa = np.linalg.norm(fn, axis=1)
    fn = fn / np.maximum(fa, 1e-30)[:, None]
    cv = F.ravel()
    cf = np.repeat(np.arange(len(F)), 3)
    order = np.argsort(cv, kind="stable")
    cv, cf = cv[order], cf[order]
    starts = np.flatnonzero(np.r_[True, cv[1:] != cv[:-1]])
    counts = np.diff(np.r_[starts, len(cv)])
    rep = np.repeat(counts, counts)
    gstart = np.repeat(starts, counts)
    i_idx = np.repeat(np.arange(len(cv)), rep)
    offs = np.arange(rep.sum()) - np.repeat(np.cumsum(rep) - rep, rep)
    j_idx = np.repeat(gstart, rep) + offs
    fi, fj = cf[i_idx], cf[j_idx]
    w = (np.einsum("ij,ij->i", fn[fi], fn[fj]) > np.cos(np.radians(smooth_deg))) * fa[fj]
    acc = np.column_stack([np.bincount(i_idx, weights=w * fn[fj, k], minlength=len(cv)) for k in range(3)])
    out = np.empty_like(acc)
    out[order] = acc
    out /= np.maximum(np.linalg.norm(out, axis=1), 1e-30)[:, None]
    return out.reshape(len(F), 3, 3), fn


def look(forward, up):
    f = np.asarray(forward, float)
    f /= np.linalg.norm(f)
    u = np.asarray(up, float) - (np.dot(up, f)) * f
    u /= np.linalg.norm(u)
    return f, u, np.cross(f, u)


def rasterize(S, F, W, H):
    """Exact z-buffer: returns per-pixel triangle id (-1 = empty), barycentrics and depth."""
    zbuf = np.full(W * H, np.inf)
    tbuf = np.full(W * H, -1, np.int64)
    bbuf = np.zeros((W * H, 2))
    T = S[F]
    xmin = np.floor(T[:, :, 0].min(1) - 0.5).astype(int)
    xmax = np.ceil(T[:, :, 0].max(1) - 0.5).astype(int)
    ymin = np.floor(T[:, :, 1].min(1) - 0.5).astype(int)
    ymax = np.ceil(T[:, :, 1].max(1) - 0.5).astype(int)
    onscreen = (xmax >= 0) & (ymax >= 0) & (xmin < W) & (ymin < H)
    xmin, ymin = np.maximum(xmin, 0), np.maximum(ymin, 0)
    xmax, ymax = np.minimum(xmax, W - 1), np.minimum(ymax, H - 1)
    size = np.maximum(xmax - xmin, ymax - ymin) + 1
    prev = 0
    for K in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        ids = np.where(onscreen & (size > prev) & (size <= K))[0]
        prev = K
        if len(ids) == 0:
            continue
        oy, ox = np.divmod(np.arange(K * K), K)
        step = max(1, int(6e6 // (K * K)))
        for c0 in range(0, len(ids), step):
            t = ids[c0:c0 + step]
            px = xmin[t, None] + ox[None, :]
            py = ymin[t, None] + oy[None, :]
            valid = (px <= xmax[t, None]) & (py <= ymax[t, None])
            x0, y0, z0 = T[t, 0, 0, None], T[t, 0, 1, None], T[t, 0, 2, None]
            x1, y1, z1 = T[t, 1, 0, None], T[t, 1, 1, None], T[t, 1, 2, None]
            x2, y2, z2 = T[t, 2, 0, None], T[t, 2, 1, None], T[t, 2, 2, None]
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            ok_den = np.abs(den) > 1e-12
            den = np.where(ok_den, den, 1.0)
            cx, cy = px + 0.5, py + 0.5
            l0 = ((y1 - y2) * (cx - x2) + (x2 - x1) * (cy - y2)) / den
            l1 = ((y2 - y0) * (cx - x2) + (x0 - x2) * (cy - y2)) / den
            l2 = 1 - l0 - l1
            eps = -1e-7
            inside = valid & ok_den & (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
            if not inside.any():
                continue
            r, c = np.nonzero(inside)
            z = (l0 * z0 + l1 * z1 + l2 * z2)[r, c]
            pix = py[r, c] * W + px[r, c]
            o = np.lexsort((z, pix))
            pix_o = pix[o]
            first = np.r_[True, pix_o[1:] != pix_o[:-1]]
            sel = o[first]
            pix_s, z_s = pix[sel], z[sel]
            better = z_s < zbuf[pix_s]
            sel, pix_s = sel[better], pix_s[better]
            zbuf[pix_s] = z_s[better]
            tbuf[pix_s] = t[r[sel]]
            bbuf[pix_s, 0] = l1[r[sel], c[sel]]
            bbuf[pix_s, 1] = l2[r[sel], c[sel]]
    return tbuf.reshape(H, W), bbuf.reshape(H, W, 2), zbuf.reshape(H, W)


def shift(img, dy, dx, fill):
    out = np.full_like(img, fill)
    h, w = img.shape[:2]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = img[ys, xs]
    return out


def render(V, F, CN, albedo, forward, up, center, span, size, table=None):
    """Orthographic render; span = model units across the image; table = a point on a
    horizontal table surface to draw under the model (edge-on views only). Returns RGBA."""
    f, u, rgt = look(forward, up)
    W = H = size * SS
    scale = W / span
    P = V - center
    S = np.column_stack([P @ rgt * scale + W / 2, H / 2 - P @ u * scale, P @ f])
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    front = np.where(fn @ f < 0)[0]
    tb, bb, zb = rasterize(S, F[front], W, H)
    hit = tb >= 0
    tri = front[tb[hit]]
    b1, b2 = bb[hit, 0], bb[hit, 1]
    n = (1 - b1 - b2)[:, None] * CN[tri, 0] + b1[:, None] * CN[tri, 1] + b2[:, None] * CN[tri, 2]
    n /= np.linalg.norm(n, axis=1)[:, None]
    view = -f
    key = 1.0 * view + 0.85 * u - 0.55 * rgt
    fill = 1.0 * view + 0.7 * rgt - 0.25 * u
    rim = -0.9 * view + 0.6 * u + 0.2 * rgt
    key, fill, rim = (x / np.linalg.norm(x) for x in (key, fill, rim))
    nd = lambda L: np.clip(n @ L, 0, None)
    hemi = 0.5 + 0.5 * (n @ u)
    ambient = 0.16 + 0.10 * hemi
    hv = key + view
    hv /= np.linalg.norm(hv)
    spec = 0.05 * np.clip(n @ hv, 0, None) ** 24
    rim_t = 0.25 * nd(rim) * (1 - np.clip(n @ view, 0, 1)) ** 1.5
    light = ambient + 0.78 * nd(key) + 0.22 * nd(fill) + rim_t

    # screen-space ambient occlusion: a neighbour occludes only if it sits in front of this
    # pixel's tangent plane (so smooth slopes are not darkened), within a limited range
    depth = np.where(hit, zb, np.inf)
    gx = np.zeros(hit.shape)
    gy = np.zeros(hit.shape)
    nf = np.minimum(n @ f, -0.05)  # front-facing: n.f < 0
    gx[hit] = -(n @ rgt) / nf / scale  # d(depth)/d(pixel x)
    gy[hit] = (n @ u) / nf / scale  # d(depth)/d(pixel y), screen y points down
    occ = np.zeros(hit.shape)
    nsamp = 0
    rng = np.random.default_rng(1)
    with np.errstate(invalid="ignore"):
        for rad in (2, 5, 10, 18, 30):
            for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False) + rng.uniform(0, 2 * np.pi):
                dy, dx = int(round(rad * SS * np.sin(ang))), int(round(rad * SS * np.cos(ang)))
                nb = shift(depth, dy, dx, np.inf)
                expect = depth + gx * dx + gy * dy
                dz = expect - nb  # >0: neighbour is in front of the tangent plane
                occ += np.clip((dz - 0.02) / 0.3, 0, 1) * (dz < 2.0)
                nsamp += 1
        ao_raw = np.where(hit, occ / nsamp, 0)
        k = 2 * SS + 1  # small box blur to hide sampling noise
        acc = np.zeros(hit.shape)
        cnt = np.zeros(hit.shape)
        for dy in range(-(k // 2), k // 2 + 1):
            for dx in range(-(k // 2), k // 2 + 1):
                acc += shift(ao_raw, dy, dx, 0.0)
                cnt += shift(hit.astype(float), dy, dx, 0.0)
        ao = 1 - 0.85 * np.clip(acc / np.maximum(cnt, 1) * 1.6, 0, 0.85)

    col = np.zeros((H, W, 3))
    col[hit] = albedo[tri] * (light * ao[hit])[:, None] + spec[:, None]

    # thin outline on depth discontinuities and silhouettes
    edge = np.zeros(hit.shape, bool)
    with np.errstate(invalid="ignore"):
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nb = shift(depth, dy, dx, np.inf)
            edge |= hit & ((nb - depth > 0.9) | ~np.isfinite(nb))
    col[edge] *= 0.45

    rgba = np.zeros((H, W, 4))
    rgba[..., :3] = lin_to_srgb(col) * hit[..., None]
    rgba[..., 3] = hit
    if table is not None:  # table edge as a line, with a faint band for the tabletop below it
        ty = int(round(H / 2 - (np.asarray(table) - center) @ u * scale))
        rows = np.arange(H)[:, None]
        below = ~hit & (rows >= ty)
        rgba[below, :3] = (0.55, 0.57, 0.60)
        rgba[below, 3] = 0.10
        rgba[below & (rows < ty + 2 * SS), 3] = 0.9
    # downsample (premultiplied) for anti-aliasing
    pm = rgba.copy()
    pm[..., :3] *= pm[..., 3:4]
    pm = pm.reshape(size, SS, size, SS, 4).mean(axis=(1, 3))
    a = pm[..., 3:4]
    rgb = np.where(a > 0, pm[..., :3] / np.maximum(a, 1e-9), 0)
    out = np.concatenate([rgb, a], axis=2)
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGBA")


def tri_keys(ids):
    k = np.argmin(ids, axis=1)
    r = np.arange(len(ids))
    t = np.column_stack([ids[r, k], ids[r, (k + 1) % 3], ids[r, (k + 2) % 3]]).astype(np.int64)
    return (t[:, 0] << 42) | (t[:, 1] << 21) | t[:, 2]


def points_on_triangles(P, T, tol=1e-5):
    """(n, m) bool: point n lies on triangle m (within tol of its plane and inside its edges)."""
    v0, e1, e2 = T[:, 0], T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
    nrm = np.cross(e1, e2)
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    d = P[:, None, :] - v0[None]
    dist = np.abs(np.einsum("nmk,mk->nm", d, nrm))
    d00, d01, d11 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    d20, d21 = np.einsum("nmk,mk->nm", d, e1), np.einsum("nmk,mk->nm", d, e2)
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return (dist < tol) & (v >= -1e-6) & (w >= -1e-6) & (1 - v - w >= -1e-6)


def rot_to(a, target):
    a, target = a / np.linalg.norm(a), target / np.linalg.norm(target)
    v = np.cross(a, target)
    c = a @ target
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--original", default="headGK.stl")
    ap.add_argument("--modified", default="headGK_boss.stl")
    ap.add_argument("--out", default="renders")
    ap.add_argument("--size", type=int, default=800)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    o = trimesh.load(args.original, force="mesh")
    m = trimesh.load(args.modified, force="mesh")
    allv = np.vstack([o.vertices, m.vertices]).astype(np.float32)
    _, inv = np.unique(allv, axis=0, return_inverse=True)
    inv = inv.ravel()
    ko = tri_keys(inv[: len(o.vertices)][o.faces])
    km = tri_keys(inv[len(o.vertices):][m.faces])
    removed = ~np.isin(ko, km)
    added = ~np.isin(km, ko)
    # Triangles where the boss merely re-cut the surrounding surface are not new geometry:
    # tint only the socket (orange) and the new cap + boss (blue).
    on = points_on_triangles(m.triangles_center[added], o.triangles[removed])
    retri = np.where(added)[0][on.any(axis=1)]
    recut = np.where(removed)[0][on.any(axis=0)]
    added[retri] = False
    removed[recut] = False

    # boss axis from the new geometry: the end face is the largest flat group among added faces
    fn_m = m.face_normals[added]
    area_m = m.area_faces[added]
    q = np.round(fn_m, 3)
    uq, lab = np.unique(q, axis=0, return_inverse=True)
    best = np.argmax(np.bincount(lab.ravel(), weights=area_m))
    a = fn_m[lab.ravel() == best].mean(0)
    a /= np.linalg.norm(a)
    end_c = m.triangles_center[added][lab.ravel() == best].mean(0)

    meshes = {}
    for name, mesh, mask, tint in (("original", o, removed, REMOVED), ("boss", m, added, ADDED)):
        alb = np.tile(srgb_to_lin(BASE), (len(mesh.faces), 1))
        alb[mask] = srgb_to_lin(tint)
        CN, _ = corner_normals(mesh.vertices, mesh.faces)
        meshes[name] = (mesh, alb, CN)

    center = o.bounds.mean(0)
    span = float(np.max(o.extents)) * 1.18
    views = {  # name: (forward, up)  -- model faces -Y, +Z up, rose on +X (character's left)
        "front": ((0, 1, 0), (0, 0, 1)),
        "back": ((0, -1, 0), (0, 0, 1)),
        "left": ((-1, 0, 0), (0, 0, 1)),
        "right": ((1, 0, 0), (0, 0, 1)),
        "top": ((0, 0, -1), (0, 1, 0)),
        "bottom": ((0, 0, 1), (0, -1, 0)),
        "three_quarter": ((0.55, 0.8, -0.32), (0, 0, 1)),
        "three_quarter_below": ((0.55, -0.45, 0.70), (0, 0, 1)),
    }
    for name, (mesh, alb, CN) in meshes.items():
        for vname, (fwd, up) in views.items():
            img = render(mesh.vertices, mesh.faces, CN, alb, fwd, up, center, span, args.size)
            path = os.path.join(args.out, f"{name}_{vname}.png")
            img.save(path, optimize=True)
            print("wrote", path)

    # close-ups of the socket / boss, from 50 deg off the socket axis so the boss height reads
    side = np.array([1.0, -0.5, 0.0])
    side -= (side @ a) * a
    side /= np.linalg.norm(side)
    th = np.radians(50)
    fwd = -a * np.cos(th) + side * np.sin(th)
    for name, (mesh, alb, CN) in meshes.items():
        img = render(mesh.vertices, mesh.faces, CN, alb, fwd, (0, -1, 0), end_c - 1.0 * a, 24.0, args.size)
        path = os.path.join(args.out, f"{name}_closeup.png")
        img.save(path, optimize=True)
        print("wrote", path)

    # resting on the boss: rotate so the boss axis points straight down, table at the end face
    R = rot_to(a, np.array([0, 0, -1.0]))
    mesh, alb, _ = meshes["boss"]
    Vr = (mesh.vertices - end_c) @ R.T
    CNr = corner_normals(Vr, mesh.faces)[0]
    c_r = (Vr.min(0) + Vr.max(0)) / 2
    c_r[2] = Vr[:, 2].min() + 0.42 * (Vr[:, 2].max() - Vr[:, 2].min())
    for vname, fwd in (("resting_side", (-1, 0, 0)), ("resting_front", (0, 1, 0))):
        img = render(Vr, mesh.faces, CNr, alb, fwd, (0, 0, 1), c_r, span, args.size,
                     table=(0, 0, Vr[:, 2].min()))
        path = os.path.join(args.out, f"boss_{vname}.png")
        img.save(path, optimize=True)
        print("wrote", path)


if __name__ == "__main__":
    main()
