"""
lth_render.py — Render BSP parser and vertex-colour extractor.

``read_render_data``         — Parses every render node and writes a detailed
                               human-readable log (triangle counts, lightmap
                               grid-size estimates, lightgroups, …).

``write_physics_vertex_colors`` — Matches physics BSP polygons to render
                                  triangles and writes per-vertex BGR colours,
                                  which are later consumed by lth_ambient.py.
"""

import math
import statistics
import struct
from collections import defaultdict

from lth_types import Vec3, Vec3u, Vec2, read_lithtech_string


# ---------------------------------------------------------------------------
# Shader-code lookup tables
# ---------------------------------------------------------------------------

#: Human-readable shader names used in the parse log.
SHADER_TYPE_NAMES: dict[int, str] = {
    0:  "None",
    1:  "Gouraud",
    2:  "Lightmap",
    4:  "Lightmap_Texture",
    5:  "Skypan",
    6:  "SkyPortal",
    7:  "Occluder",
    8:  "DualTexture",
    9:  "Lightmap_DualTexture",
    10: "Splitter",
    11: "Unknown",
}

#: Simplified shader names used by the exporter.
EPCShaderType: dict[int, str] = {
    0:  "None",
    1:  "Gouraud",
    2:  "Lightmap",
    4:  "Lightmap",
    5:  "Skypan",
    6:  "SkyPortal",
    7:  "Occluder",
    8:  "Gouraud",
    9:  "Lightmap",
    10: "Splitter",
    11: "Unknown",
}

# Texture name prefix that identifies a section as a lightmap atlas.
_LIGHTANIM_PREFIX = "lightanim"

# Minimum UV-space edge length (pixels) to include in grid-size estimation.
_LM_MIN_EDGE_UV_PX = 0.5


# ---------------------------------------------------------------------------
# Edge-length helpers (moved out of inner loop for clarity)
# ---------------------------------------------------------------------------

def _edge_lengths(va: tuple, vb: tuple, ua: tuple, ub: tuple,
                  lm_w: int, lm_h: int) -> tuple[float, float]:
    """Return the 3-D world length and UV-pixel length of one triangle edge."""
    len_3d = math.sqrt(
        (vb[0] - va[0]) ** 2 +
        (vb[1] - va[1]) ** 2 +
        (vb[2] - va[2]) ** 2
    )
    len_uv = math.sqrt(
        ((ub[0] - ua[0]) * lm_w) ** 2 +
        ((ub[1] - ua[1]) * lm_h) ** 2
    )
    return len_3d, len_uv


def _estimate_lm_grid_size(
    sec_tris: list[tuple],
    pos_arr: list[tuple],
    uv0_arr: list[tuple],
    lm_w: int,
    lm_h: int,
    out,
    section_label: str,
) -> None:
    """Estimate the LMGridSize for one lightmap section and print the result."""
    raw_vals: list[float] = []

    for tri_idx, (i0, i1, i2, pidx) in enumerate(sec_tris):
        v0, v1, v2 = pos_arr[i0], pos_arr[i1], pos_arr[i2]
        u0, u1, u2 = uv0_arr[i0], uv0_arr[i1], uv0_arr[i2]

        edges = [
            _edge_lengths(v0, v1, u0, u1, lm_w, lm_h),
            _edge_lengths(v1, v2, u1, u2, lm_w, lm_h),
            _edge_lengths(v2, v0, u2, u0, lm_w, lm_h),
        ]

        tri_gs_vals: list[float] = []
        log_parts: list[str] = []
        for e_3d, e_uv in edges:
            if e_uv > _LM_MIN_EDGE_UV_PX:
                gs = e_3d / e_uv
                tri_gs_vals.append(gs)
                log_parts.append(f"{gs:.1f} ({e_3d:.1f}u/{e_uv:.1f}px)")
            else:
                log_parts.append("SKIP")

        if tri_gs_vals:
            tri_avg = sum(tri_gs_vals) / len(tri_gs_vals)
            raw_vals.append(tri_avg)
            if tri_idx < 5:  # Log the first few triangles to keep output manageable.
                print(
                    f"    Poly {pidx:4d} | Edges: [{', '.join(log_parts)}] "
                    f"-> Avg: {tri_avg:.2f}",
                    file=out,
                )

    if not raw_vals:
        print(f"    -> WARNING: No valid edges for {section_label}.", file=out)
        return

    median_gs = statistics.median(raw_vals)
    snapped   = max(1, int(round(median_gs)))
    print(
        f"    -> LMGridSize: median={median_gs:.2f}  "
        f"(min={min(raw_vals):.2f}, max={max(raw_vals):.2f})  "
        f"-> rounded to {snapped}",
        file=out,
    )


# ---------------------------------------------------------------------------
# Parse-log writer
# ---------------------------------------------------------------------------

def read_render_data(f, header, out, version_flag: str) -> None:
    """Parse the render BSP and write a detailed log to *out*."""
    f.seek(header.render_data_pos)
    (render_node_count,) = struct.unpack("<I", f.read(4))
    print(f"\nRenderData — RenderNode count: {render_node_count}", file=out)

    for i in range(render_node_count):
        center    = Vec3.read(f)
        half_dims = Vec3.read(f)
        (section_count,) = struct.unpack("<I", f.read(4))

        print(f"\n[RenderNode #{i}]", file=out)
        print(f"  Center:   {center}", file=out)
        print(f"  HalfDims: {half_dims}", file=out)
        print(f"  Sections: {section_count}", file=out)

        sections = []
        for s in range(section_count):
            tex0        = read_lithtech_string(f)
            _tex1       = read_lithtech_string(f)
            shader_code = struct.unpack("<B", f.read(1))[0]
            tri_count   = struct.unpack("<I", f.read(4))[0]
            _tex_effect = read_lithtech_string(f)
            w, h, size  = struct.unpack("<3I", f.read(12))
            f.read(size)

            is_lm       = tex0.lower().startswith(_LIGHTANIM_PREFIX)
            shader_name = SHADER_TYPE_NAMES.get(shader_code, f"Unknown({shader_code})")
            sections.append({
                "tex0": tex0, "shader_code": shader_code,
                "tri_count": tri_count, "is_lm": is_lm,
                "lm_w": w, "lm_h": h,
            })
            print(
                f"  [Section {s}] tex0={tex0!r}, tris={tri_count}, "
                f"shader={shader_name}",
                file=out,
            )

        (vertex_count,) = struct.unpack("<I", f.read(4))
        print(f"  Vertices: {vertex_count}", file=out)

        pos_arr, uv0_arr = [], []
        for _ in range(vertex_count):
            pos = Vec3.read(f)
            uv0 = Vec2.read(f)
            Vec2.read(f)          # uv1 — not needed for grid-size estimation
            f.seek(4 + 12, 1)    # colour + normal
            if version_flag == "-v1":
                f.seek(24, 1)    # tangent + binormal
            pos_arr.append((pos.x, pos.y, pos.z))
            uv0_arr.append((uv0.x, uv0.y))

        (tri_count_total,) = struct.unpack("<I", f.read(4))
        print(f"  Triangles: {tri_count_total}", file=out)

        all_tris = [
            (*struct.unpack("<3I", f.read(12)), struct.unpack("<I", f.read(4))[0])
            for _ in range(tri_count_total)
        ]

        # Lightmap grid-size estimation per lightmap section.
        cursor = 0
        for s_idx, sec in enumerate(sections):
            cnt       = sec["tri_count"]
            sec_tris  = all_tris[cursor : cursor + cnt]
            cursor   += cnt

            if not sec["is_lm"] or sec["lm_w"] == 0 or sec["lm_h"] == 0:
                continue

            label = f"Section {s_idx} ({sec['tex0']}, {sec['lm_w']}×{sec['lm_h']})"
            print(f"\n  [LMGridSize] {label} — {cnt} tris", file=out)
            _estimate_lm_grid_size(
                sec_tris, pos_arr, uv0_arr,
                sec["lm_w"], sec["lm_h"], out, label,
            )

        _skip_render_node_tail(f, out)

    # WorldModel render nodes
    (wm_node_count,) = struct.unpack("<I", f.read(4))
    print(f"\nWorldModelNodes: {wm_node_count}", file=out)

    for wm_index in range(wm_node_count):
        name = read_lithtech_string(f)
        print(f"\n[WMRenderNode #{wm_index}] Name: {name}", file=out)
        (subnode_count,) = struct.unpack("<I", f.read(4))

        for subnode_index in range(subnode_count):
            print(f"\n  [SubRenderNode #{subnode_index}]", file=out)
            center    = Vec3.read(f)
            half_dims = Vec3.read(f)
            (section_count,) = struct.unpack("<I", f.read(4))
            print(f"    Center:   {center}", file=out)
            print(f"    HalfDims: {half_dims}", file=out)
            print(f"    Sections: {section_count}", file=out)

            for s in range(section_count):
                tex0        = read_lithtech_string(f)
                tex1        = read_lithtech_string(f)
                shader_code = struct.unpack("<B", f.read(1))[0]
                tri_count   = struct.unpack("<I", f.read(4))[0]
                tex_effect  = read_lithtech_string(f)
                _w, _h, size = struct.unpack("<3I", f.read(12))
                f.read(size)
                shader_name = SHADER_TYPE_NAMES.get(shader_code, f"Unknown({shader_code})")
                print(
                    f"    [Section {s}] tex0={tex0!r}, tris={tri_count}, "
                    f"shader={shader_name}",
                    file=out,
                )
                if tex1:       print(f"      tex1={tex1!r}", file=out)
                if tex_effect: print(f"      effect={tex_effect!r}", file=out)

            (vertex_count,) = struct.unpack("<I", f.read(4))
            print(f"    Vertices: {vertex_count}", file=out)
            for _ in range(vertex_count):
                Vec3.read(f)
                f.seek(8 + 8, 1)  # uv0 + uv1
                f.read(4)          # colour
                f.seek(12, 1)      # normal
                if version_flag == "-v1":
                    f.seek(24, 1)  # tangent + binormal

            (tri_count,) = struct.unpack("<I", f.read(4))
            print(f"    Triangles: {tri_count}", file=out)
            for _ in range(tri_count):
                f.read(12 + 4)  # indices + poly_idx

            _skip_render_node_tail(f, out)

        (no_child_flag,) = struct.unpack("<I", f.read(4))
        print(f"  NoChildFlag: {no_child_flag}", file=out)

    # World lightgroups
    (lightgroups_len,) = struct.unpack("<I", f.read(4))
    print(f"\n[WorldLightGroups] Count: {lightgroups_len}", file=out)
    for i in range(lightgroups_len):
        name      = read_lithtech_string(f)
        color     = Vec3.read(f)
        offset    = Vec3u.read(f)
        size      = Vec3u.read(f)
        total_len = size.x * size.y * size.z
        f.read(total_len)
        print(f"  [{i}] {name}", file=out)
        print(f"      Color:  {color}", file=out)
        print(f"      Offset: {offset}", file=out)
        print(f"      Size:   {size}", file=out)
        print(f"      Data:   {total_len} bytes", file=out)


def _skip_render_node_tail(f, out=None) -> None:
    """Advance past sky portals, occluders, lightgroups, and child flags."""
    # Sky portals
    n = struct.unpack("<I", f.read(4))[0]
    if n and out:
        print(f"  SkyPortals: {n}", file=out)
    for _ in range(n):
        f.read(12 * struct.unpack("<B", f.read(1))[0] + 16)

    # Occluders
    n = struct.unpack("<I", f.read(4))[0]
    if n and out:
        print(f"  Occluders: {n}", file=out)
    for _ in range(n):
        vert_count = struct.unpack("<B", f.read(1))[0]
        f.read(12 * vert_count + 12 + 4)
        m_id = struct.unpack("<I", f.read(4))[0]
        if out:
            print(f"    Occluder hashcode: {m_id}", file=out)

    # Lightgroups
    n = struct.unpack("<I", f.read(4))[0]
    if n and out:
        print(f"  LightGroups: {n}", file=out)
    for lg_index in range(n):
        lg_name = read_lithtech_string(f)
        if out:
            print(f"  LightGroup {lg_index}: {lg_name}", file=out)
        f.seek(12, 1)
        (intensity_len,) = struct.unpack("<I", f.read(4))
        f.seek(intensity_len, 1)
        (section_lm_len,) = struct.unpack("<I", f.read(4))
        for _ in range(section_lm_len):
            (sublm_len,) = struct.unpack("<I", f.read(4))
            for _ in range(sublm_len):
                f.seek(16, 1)
                (datalen,) = struct.unpack("<I", f.read(4))
                f.seek(datalen, 1)

    child_flags   = struct.unpack("<B", f.read(1))[0]
    child_indices = struct.unpack("<2I", f.read(8))
    if out:
        print(f"  ChildFlags: {child_flags}", file=out)
        print(f"  ChildIndices: {child_indices[0]}, {child_indices[1]}", file=out)


# ---------------------------------------------------------------------------
# Physics vertex-colour extractor
# ---------------------------------------------------------------------------

def _clean_tex_key(t: str) -> str:
    t = t.replace("\\", "/").lower()
    if t.endswith(".dtx") or t.endswith(".tga"):
        return t[:-4]
    return t


def _build_render_tri_pool(f, header, version_flag: str):
    """Read the render BSP and build per-texture pools of coloured triangles.

    Returns:
        main_pool: dict[tex_key → list[tri]]
        wm_pool:   dict[wm_name → dict[tex_key → list[tri]]]

    Each tri dict: ``{"k0", "k1", "k2": (x,y,z), "c0", "c1", "c2": (b,g,r)}``.
    """
    f.seek(header.render_data_pos)
    (render_node_count,) = struct.unpack("<I", f.read(4))

    main_pool: dict = defaultdict(list)
    wm_pool:   dict = defaultdict(lambda: defaultdict(list))

    def _process_node(target_pool: dict) -> None:
        Vec3.read(f)  # center
        Vec3.read(f)  # half_dims
        (section_count,) = struct.unpack("<I", f.read(4))

        sections = []
        for _ in range(section_count):
            tex0        = read_lithtech_string(f).replace("\\", "/").lower()
            read_lithtech_string(f)              # tex1
            shader_code = struct.unpack("<B", f.read(1))[0]
            tri_count   = struct.unpack("<I", f.read(4))[0]
            read_lithtech_string(f)              # texture_effect
            _w, _h, size = struct.unpack("<3I", f.read(12))
            f.read(size)

            is_lm = (
                (shader_code in (4, 9) and tex0.startswith(_LIGHTANIM_PREFIX))
                or tex0 == "lightanim_base"
            )
            sections.append({
                "count":   tri_count,
                "is_lm":   is_lm,
                "tex_key": _clean_tex_key(tex0),
            })

        (vert_count,) = struct.unpack("<I", f.read(4))
        pos_arr, color_arr = [], []
        for _ in range(vert_count):
            v = Vec3.read(f)
            f.read(8 + 8)      # uv0 + uv1
            rgba = f.read(4)   # B G R A
            f.read(12)         # normal
            if version_flag == "-v1":
                f.read(24)     # tangent + binormal
            pos_arr.append((v.x, v.y, v.z))
            color_arr.append((rgba[0], rgba[1], rgba[2]))

        (tri_total,) = struct.unpack("<I", f.read(4))
        all_idx = []
        for _ in range(tri_total):
            i0, i1, i2 = struct.unpack("<3I", f.read(12))
            f.read(4)  # poly_index (not needed for colour extraction)
            all_idx.append((i0, i1, i2))

        cursor = 0
        for sec in sections:
            count   = sec["count"]
            sl      = all_idx[cursor : cursor + count]
            cursor += count
            if sec["is_lm"]:
                continue
            tex_key = sec["tex_key"]
            for i0, i1, i2 in sl:
                target_pool[tex_key].append({
                    "k0": pos_arr[i0],    "k1": pos_arr[i1],    "k2": pos_arr[i2],
                    "c0": color_arr[i0],  "c1": color_arr[i1],  "c2": color_arr[i2],
                })

        _skip_render_node_tail(f)

    for _ in range(render_node_count):
        _process_node(main_pool)

    (wm_node_count,) = struct.unpack("<I", f.read(4))
    for _ in range(wm_node_count):
        wm_name = read_lithtech_string(f)
        (subnode_count,) = struct.unpack("<I", f.read(4))
        for _ in range(subnode_count):
            _process_node(wm_pool[wm_name])
        f.read(4)  # no_child_flag

    return main_pool, wm_pool


def write_physics_vertex_colors(f, header, world_tree, out, version_flag: str) -> None:
    """Match physics polygons to render triangles and write per-vertex BGR colours.

    For each physics polygon:
      1. Build a frozenset of the polygon's vertex positions.
      2. Find render triangles whose three vertices all belong to that set.
      3. Map each physics vertex to the BGR value from the matched render data.
      4. Vertices without a direct match (BSP splits) are written as BGR(0,0,0).

    The output format is compatible with lth_ambient.py.
    """
    main_pool, wm_pool = _build_render_tri_pool(f, header, version_flag)

    print(f"\nOffset: (0.0, 0.0, 0.0)", file=out)

    for wm in world_tree.world_models:
        wm_name     = wm.world_name
        render_pool = wm_pool.get(wm_name, main_pool)

        print(f"\n[WorldModel: {wm_name}]", file=out)

        surface_to_polies: dict = defaultdict(list)
        for poly_idx, (surface_idx, plane_idx, indices) in enumerate(wm.polies):
            surface_to_polies[surface_idx].append((poly_idx, plane_idx, indices))

        for surface_idx in sorted(surface_to_polies):
            surface  = wm.surfaces[surface_idx]
            flags    = surface.get("Flags", 0)
            tex_idx  = surface.get("TextureIndex", 0)
            tex_name = (
                wm.texture_names[tex_idx]
                if 0 <= tex_idx < len(wm.texture_names)
                else "Default"
            )
            tex_key      = _clean_tex_key(tex_name)
            render_tris  = render_pool.get(tex_key) or main_pool.get(tex_key, [])

            print(
                f"\n  [Surface {surface_idx}] tex={tex_name}  "
                f"flags=0x{flags:08X}",
                file=out,
            )

            for poly_idx, plane_idx, indices in surface_to_polies[surface_idx]:
                n, dist = wm.planes[plane_idx]
                print(
                    f"    [Poly {poly_idx}] "
                    f"n=({n.x:.4f},{n.y:.4f},{n.z:.4f}) dist={dist:.4f}",
                    file=out,
                )

                phys_keys = frozenset(
                    (wm.points[i].x, wm.points[i].y, wm.points[i].z)
                    for i in indices
                )

                # Collect vertex colours from render triangles whose vertices
                # exactly match the physics polygon's vertex set.
                color_map: dict[tuple, tuple] = {}
                for tri in render_tris:
                    if tri["k0"] in phys_keys and tri["k1"] in phys_keys and tri["k2"] in phys_keys:
                        color_map.setdefault(tri["k0"], tri["c0"])
                        color_map.setdefault(tri["k1"], tri["c1"])
                        color_map.setdefault(tri["k2"], tri["c2"])

                for vi, pt_idx in enumerate(indices):
                    pt  = wm.points[pt_idx]
                    key = (pt.x, pt.y, pt.z)
                    bgr = color_map.get(key, (0, 0, 0))
                    print(
                        f"      v{vi} pos=({pt.x:.6f},{pt.y:.6f},{pt.z:.6f}) "
                        f"BGR({bgr[0]}, {bgr[1]}, {bgr[2]})",
                        file=out,
                    )
