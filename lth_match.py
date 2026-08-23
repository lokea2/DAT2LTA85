"""
lth_match.py — Physics-to-render triangle matcher.

Extracts the render geometry from the binary DAT, builds per-texture numpy
matrices for fast batch queries, then for each physics polygon finds the best
matching render triangle (the "donor") that will supply UV and texture data
for the LTA exporter.

An optional C accelerator (``lth_accel.so`` / ``lth_accel.dll``) can be
compiled with ``build_accel.py`` and provides a ~10× speed-up on the hot
search loop.  The module falls back to a pure-NumPy implementation
automatically when the library is absent.
"""

import ctypes
import logging
import os
import platform
import struct
from collections import defaultdict
from functools import lru_cache

import numpy as np

from lth_types import Vec3, Vec2, read_lithtech_string

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# C accelerator
# ---------------------------------------------------------------------------

def _load_accel() -> ctypes.CDLL | None:
    """Try to load lth_accel.dll / lth_accel.so from the same directory.

    Returns the loaded CDLL on success, or ``None`` when the library is
    absent or fails to load.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    lib_name = "lth_accel.dll" if platform.system() == "Windows" else "lth_accel.so"
    lib_path = os.path.join(here, lib_name)

    if not os.path.isfile(lib_path):
        return None

    try:
        lib = ctypes.CDLL(lib_path)

        _dp = ctypes.POINTER(ctypes.c_double)

        lib.collect_candidates.restype  = ctypes.c_int
        lib.collect_candidates.argtypes = [
            _dp,              # normals  [K*3]
            _dp,              # centers  [K*3]
            _dp,              # verts    [K*9]
            _dp,              # n_arr    [3]
            ctypes.c_double,  # raw_dist
            _dp,              # phys_arr [9]
            _dp,              # phys_center [3]
            ctypes.c_int,     # K
            ctypes.c_double,  # epsilon
            ctypes.c_void_p,  # out CandResult[K]
        ]

        lib.is_point_inside_triangle.restype  = ctypes.c_int
        lib.is_point_inside_triangle.argtypes = [_dp, _dp, _dp, _dp, ctypes.c_double]

        lib.is_coplanar_tri.restype  = ctypes.c_int
        lib.is_coplanar_tri.argtypes = [_dp, ctypes.c_double, _dp, ctypes.c_double]

        return lib

    except Exception as exc:
        log.warning("Failed to load %s: %s", lib_path, exc)
        return None


_LIB: ctypes.CDLL | None = _load_accel()

if _LIB is not None:
    log.info("C accelerator loaded — geometry operations are hardware-accelerated.")
else:
    log.info(
        "C accelerator not found. Run build_accel.py to compile it. "
        "Falling back to NumPy implementation."
    )


class _CandResult(ctypes.Structure):
    """Mirror of the ``CandResult`` struct defined in lth_accel.c."""

    _fields_ = [
        ("orig_idx", ctypes.c_int32),
        ("matches",  ctypes.c_int32),
        ("winding",  ctypes.c_int32),
        ("area",     ctypes.c_float),
        ("dist",     ctypes.c_float),
        ("v_err",    ctypes.c_float),
    ]


# ---------------------------------------------------------------------------
# Texture-key normalisation
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def clean_tex_key(t: str) -> str:
    """Normalise a texture path to a canonical lookup key.

    Strips leading slashes, lower-cases the path, converts backslashes,
    and removes ``.dtx`` / ``.tga`` extensions.
    """
    t = t.replace("\\", "/").lstrip("/").lower()
    if t.endswith(".dtx") or t.endswith(".tga"):
        return t[:-4]
    return t


# ---------------------------------------------------------------------------
# Coplanarity / point-in-triangle tests
# ---------------------------------------------------------------------------

_DP = ctypes.POINTER(ctypes.c_double)

# Default coplanarity tolerance (world units).
_COPLANAR_EPSILON = 25.0


def is_coplanar(n: Vec3, dist: float, tri: dict, epsilon: float = _COPLANAR_EPSILON) -> bool:
    """Return True if all three vertices of *tri* lie within *epsilon* of the plane."""
    if _LIB is not None:
        verts = np.array(
            [[tri["v0"].x, tri["v0"].y, tri["v0"].z],
             [tri["v1"].x, tri["v1"].y, tri["v1"].z],
             [tri["v2"].x, tri["v2"].y, tri["v2"].z]],
            dtype=np.float64,
        )
        n_arr = np.array([n.x, n.y, n.z], dtype=np.float64)
        return bool(_LIB.is_coplanar_tri(
            n_arr.ctypes.data_as(_DP),
            ctypes.c_double(dist),
            verts.ctypes.data_as(_DP),
            ctypes.c_double(epsilon),
        ))

    for v in (tri["v0"], tri["v1"], tri["v2"]):
        if abs(n.x * v.x + n.y * v.y + n.z * v.z - dist) > epsilon:
            return False
    return True


def is_point_inside_triangle_np(
    center_arr, v0a, v1a, v2a, eps: float = 0.01
) -> bool:
    """Test whether *center_arr* lies inside the triangle (v0a, v1a, v2a).

    Uses barycentric coordinates.  Delegates to the C accelerator when
    available, otherwise falls back to a NumPy implementation.
    """
    if _LIB is not None:
        c  = np.ascontiguousarray(center_arr, dtype=np.float64)
        a0 = np.ascontiguousarray(v0a, dtype=np.float64)
        a1 = np.ascontiguousarray(v1a, dtype=np.float64)
        a2 = np.ascontiguousarray(v2a, dtype=np.float64)
        return bool(_LIB.is_point_inside_triangle(
            c.ctypes.data_as(_DP),
            a0.ctypes.data_as(_DP),
            a1.ctypes.data_as(_DP),
            a2.ctypes.data_as(_DP),
            ctypes.c_double(eps),
        ))

    n = np.cross(v1a - v0a, v2a - v0a)
    area_sqr = np.dot(n, n)
    if area_sqr < 1e-12:
        return False
    p = center_arr
    u = np.dot(n, np.cross(v1a - p, v2a - p)) / area_sqr
    v = np.dot(n, np.cross(v2a - p, v0a - p)) / area_sqr
    w = np.dot(n, np.cross(v0a - p, v1a - p)) / area_sqr
    return u >= -eps and v >= -eps and w >= -eps


# ---------------------------------------------------------------------------
# Render-geometry extraction
# ---------------------------------------------------------------------------

def extract_render_geometry_by_polyindex(f, header, version_flag: str):
    """Parse the render BSP and return per-texture triangle pools.

    Returns:
        main_pool_by_tex: dict[tex_key → list[tri]] for the main render nodes.
        wm_pool_by_tex:   dict[wm_name → dict[tex_key → list[tri]]] for
                          WorldModel render nodes.
    """
    f.seek(header.render_data_pos)
    (render_node_count,) = struct.unpack("<I", f.read(4))

    main_pool_by_tex: dict = defaultdict(list)
    wm_pool_by_tex:   dict = defaultdict(lambda: defaultdict(list))

    def _process_render_node(target_by_tex: dict) -> None:
        Vec3.read(f)  # center
        Vec3.read(f)  # half_dims
        (section_count,) = struct.unpack("<I", f.read(4))

        sections = []
        for _ in range(section_count):
            tex0           = read_lithtech_string(f).replace("\\", "/").lower()
            tex1           = read_lithtech_string(f).replace("\\", "/").lower()
            shader_code    = struct.unpack("<B", f.read(1))[0]
            tri_count      = struct.unpack("<I", f.read(4))[0]
            texture_effect = read_lithtech_string(f)
            w, h, size     = struct.unpack("<3I", f.read(12))
            f.read(size)

            is_lm = (
                (shader_code in (4, 9) and tex0.startswith("lightanim"))
                or tex0 == "lightanim_base"
            )
            sections.append({
                "count":          tri_count,
                "is_lm":          is_lm,
                "tex":            tex0,
                "tex_key":        clean_tex_key(tex0),
                "shader_code":    shader_code,
                "tex1":           tex1,
                "texture_effect": texture_effect,
            })

        (vert_count,) = struct.unpack("<I", f.read(4))
        raw_verts = np.empty((vert_count, 3), dtype=np.float64)
        vec3_obj, uv_obj, uv1_obj, color_obj = [], [], [], []

        for i in range(vert_count):
            v   = Vec3.read(f)
            uv0 = Vec2.read(f)
            uv1 = Vec2.read(f)
            raw_verts[i] = (v.x, v.y, v.z)
            vec3_obj.append(v)
            uv_obj.append(uv0)
            uv1_obj.append(uv1)
            color_obj.append(f.read(4))
            f.read(12)  # vertex normal
            if version_flag == "-v1":
                f.read(24)  # tangent + binormal

        (tri_count_total,) = struct.unpack("<I", f.read(4))
        all_indices = [
            (*struct.unpack("<3I", f.read(12)), struct.unpack("<I", f.read(4))[0])
            for _ in range(tri_count_total)
        ]

        cursor = 0
        for sec in sections:
            count    = sec["count"]
            sec_slice = all_indices[cursor : cursor + count]
            cursor  += count

            if sec["is_lm"]:
                continue

            tex_key = sec["tex_key"]
            for i0, i1, i2, poly_idx in sec_slice:
                v0a = raw_verts[i0]
                v1a = raw_verts[i1]
                v2a = raw_verts[i2]
                normal   = np.cross(v1a - v0a, v2a - v0a)
                nlen     = np.linalg.norm(normal)
                normal_unit = normal / nlen if nlen > 1e-8 else normal
                verts_arr   = np.stack([v0a, v1a, v2a])
                target_by_tex[tex_key].append({
                    "tex":            sec["tex"],
                    "tex1":           sec["tex1"],
                    "texture_effect": sec["texture_effect"],
                    "tex_key":        tex_key,
                    "shader_code":    sec["shader_code"],
                    "v0":  vec3_obj[i0], "v1":  vec3_obj[i1], "v2":  vec3_obj[i2],
                    "uv0": uv_obj[i0],   "uv1": uv_obj[i1],   "uv2": uv_obj[i2],
                    "uv1_0": uv1_obj[i0], "uv1_1": uv1_obj[i1], "uv1_2": uv1_obj[i2],
                    "c0":  color_obj[i0], "c1": color_obj[i1], "c2": color_obj[i2],
                    "v0a": v0a, "v1a": v1a, "v2a": v2a,
                    "verts_arr":       verts_arr,
                    "center":          verts_arr.mean(axis=0),
                    "normal_unit":     normal_unit,
                    "render_poly_idx": poly_idx,
                })

        _skip_render_node_tail(f)

    def _skip_render_node_tail(f) -> None:
        """Skip sky portals, occluders, lightgroups, and child flags."""
        # Sky portals
        n = struct.unpack("<I", f.read(4))[0]
        for _ in range(n):
            f.read(12 * struct.unpack("<B", f.read(1))[0] + 16)
        # Occluders
        n = struct.unpack("<I", f.read(4))[0]
        for _ in range(n):
            f.read(12 * struct.unpack("<B", f.read(1))[0] + 20)
        # Lightgroups
        n = struct.unpack("<I", f.read(4))[0]
        for _ in range(n):
            f.read(struct.unpack("<H", f.read(2))[0] + 12)
            f.read(struct.unpack("<I", f.read(4))[0])
            for _ in range(struct.unpack("<I", f.read(4))[0]):
                for _ in range(struct.unpack("<I", f.read(4))[0]):
                    f.read(16)
                    f.read(struct.unpack("<I", f.read(4))[0])
        f.read(9)  # child_flags (1) + child_indices (8)

    for _ in range(render_node_count):
        _process_render_node(main_pool_by_tex)

    (wm_node_count,) = struct.unpack("<I", f.read(4))
    for _ in range(wm_node_count):
        wm_name = read_lithtech_string(f)
        (subnode_count,) = struct.unpack("<I", f.read(4))
        for _ in range(subnode_count):
            _process_render_node(wm_pool_by_tex[wm_name])
        f.read(4)  # no_child_flag

    return main_pool_by_tex, wm_pool_by_tex


# ---------------------------------------------------------------------------
# Search-matrix builder
# ---------------------------------------------------------------------------

def build_pool_matrices(pool_by_tex: dict) -> dict:
    """Pre-compute contiguous NumPy arrays from a texture pool for fast search.

    Returns a dict mapping tex_key → ``{"normals", "centers", "verts"}``.
    """
    matrices = {}
    for tex_key, tris in pool_by_tex.items():
        if not tris:
            continue
        matrices[tex_key] = {
            "normals": np.ascontiguousarray(
                np.stack([t["normal_unit"] for t in tris]), dtype=np.float64),
            "centers": np.ascontiguousarray(
                np.stack([t["center"]      for t in tris]), dtype=np.float64),
            "verts":   np.ascontiguousarray(
                np.stack([t["verts_arr"]   for t in tris]), dtype=np.float64),
        }
    return matrices


# ---------------------------------------------------------------------------
# Candidate collection (C path + NumPy fallback)
# ---------------------------------------------------------------------------

def _collect_candidates_c(
    tex_tris, mat, n_arr, raw_dist, n_obj, phys_arr, phys_center_arr
) -> list[dict]:
    """Collect candidates using the C accelerator."""
    K = len(tex_tris)
    if K == 0 or mat is None:
        return []

    normals    = mat["normals"]
    centers    = mat["centers"]
    verts_flat = mat["verts"].reshape(K * 9)

    phys_flat   = np.ascontiguousarray(phys_arr,        dtype=np.float64)
    phys_c_flat = np.ascontiguousarray(phys_center_arr, dtype=np.float64)
    n_c         = np.ascontiguousarray(n_arr,           dtype=np.float64)
    out_buf     = (_CandResult * K)()

    count = _LIB.collect_candidates(
        normals.ctypes.data_as(_DP),
        centers.ctypes.data_as(_DP),
        verts_flat.ctypes.data_as(_DP),
        n_c.ctypes.data_as(_DP),
        ctypes.c_double(raw_dist),
        phys_flat.ctypes.data_as(_DP),
        phys_c_flat.ctypes.data_as(_DP),
        ctypes.c_int(K),
        ctypes.c_double(_COPLANAR_EPSILON),
        ctypes.cast(out_buf, ctypes.c_void_p),
    )

    return [
        {
            "tri":     tex_tris[out_buf[i].orig_idx],
            "matches": out_buf[i].matches,
            "winding": out_buf[i].winding,
            "area":    float(out_buf[i].area),
            "dist":    float(out_buf[i].dist),
            "v_err":   float(out_buf[i].v_err),
        }
        for i in range(count)
    ]


def _collect_candidates_numpy(
    tex_tris, mat, n_arr, raw_dist, n_obj, phys_arr, phys_center_arr
) -> list[dict]:
    """Collect candidates using a vectorised NumPy implementation (fallback)."""
    K = len(tex_tris)
    if K == 0 or mat is None:
        return []

    dots     = mat["normals"] @ n_arr
    surv_idx = np.where(np.abs(dots) >= 0.5)[0]
    if len(surv_idx) == 0:
        return []

    winding_scores = (dots[surv_idx] > 0).astype(np.int32)
    verts_surv     = mat["verts"][surv_idx]

    diff    = verts_surv[:, :, np.newaxis, :] - phys_arr[np.newaxis, np.newaxis, :, :]
    matches = np.any(np.all(np.abs(diff) < 20.0, axis=3), axis=2).sum(axis=1).astype(np.int32)
    v_errs  = np.sqrt(np.sum(diff ** 2, axis=3)).min(axis=2).sum(axis=1)
    dists   = np.linalg.norm(mat["centers"][surv_idx] - phys_center_arr, axis=1)

    e1    = verts_surv[:, 1, :] - verts_surv[:, 0, :]
    e2    = verts_surv[:, 2, :] - verts_surv[:, 0, :]
    areas = np.linalg.norm(np.cross(e1, e2), axis=1) * 0.5

    candidates = []
    for i, orig_idx in enumerate(surv_idx):
        m = int(matches[i])
        if m < 3 and not is_coplanar(n_obj, raw_dist, tex_tris[orig_idx]):
            continue
        candidates.append({
            "tri":     tex_tris[orig_idx],
            "matches": m,
            "winding": int(winding_scores[i]),
            "area":    float(areas[i]),
            "dist":    float(dists[i]),
            "v_err":   float(v_errs[i]),
        })
    return candidates


def _collect_candidates(
    tex_tris, mat, n_arr, raw_dist, n_obj, phys_arr, phys_center_arr
) -> list[dict]:
    if _LIB is not None:
        return _collect_candidates_c(tex_tris, mat, n_arr, raw_dist, n_obj,
                                     phys_arr, phys_center_arr)
    return _collect_candidates_numpy(tex_tris, mat, n_arr, raw_dist, n_obj,
                                     phys_arr, phys_center_arr)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def find_best_donor(
    n_obj, raw_dist, tex_key, phys_arr, phys_center_arr, n_arr,
    primary_pool, primary_mat, fallback_pool=None, fallback_mat=None,
):
    """Find the render triangle that best matches a physics polygon.

    Searches *primary_pool* first; if no candidates are found and a
    *fallback_pool* is provided, it is searched next.

    Returns the best-matching triangle dict, or ``None`` when nothing is found.
    """
    tex_tris   = primary_pool.get(tex_key, [])
    mat        = primary_mat.get(tex_key)
    candidates = _collect_candidates(tex_tris, mat, n_arr, raw_dist, n_obj,
                                     phys_arr, phys_center_arr)

    if not candidates and fallback_pool is not None and fallback_pool is not primary_pool:
        tex_tris_fb = fallback_pool.get(tex_key, [])
        mat_fb      = fallback_mat.get(tex_key) if fallback_mat else None
        candidates  = _collect_candidates(tex_tris_fb, mat_fb, n_arr, raw_dist, n_obj,
                                          phys_arr, phys_center_arr)

    if not candidates:
        return None

    healthy = [c for c in candidates if c["area"] >= 1.0] or candidates

    inside = [
        c for c in healthy
        if is_point_inside_triangle_np(
            phys_center_arr,
            c["tri"]["v0a"], c["tri"]["v1a"], c["tri"]["v2a"],
        )
    ]
    pool = inside if inside else healthy
    pool.sort(
        key=lambda c: (c["matches"], -c["v_err"], c["winding"], -c["dist"], c["area"]),
        reverse=True,
    )
    return pool[0]["tri"]
