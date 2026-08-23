"""
lth_ambient.py — Ambient-light extraction from paired render dumps.

Given two parse-log text files — one produced with normal lighting and one
with every surface's ``AmbientLight`` set to zero — this module computes the
per-surface ambient colour contribution by subtracting the zero-ambient vertex
colours from the originals.

Usage (via dat2lta85.py):
    python dat2lta85.py -comp <original.txt> <ambient_zero.txt>
"""

import os
import re
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_offset(filepath: str) -> tuple[float, float, float]:
    """Extract the world-space offset from the first ``Offset:`` line."""
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*Offset:\s*\(([^,]+),([^,]+),([^)]+)\)", line)
            if m:
                return float(m.group(1)), float(m.group(2)), float(m.group(3))
    return (0.0, 0.0, 0.0)


def _normalize_pos(
    pos: tuple[float, float, float],
    offset: tuple[float, float, float],
    precision: int = 1,
) -> tuple[float, float, float]:
    """Translate and round a position to reduce floating-point noise."""
    return (
        round(pos[0] + offset[0], precision),
        round(pos[1] + offset[1], precision),
        round(pos[2] + offset[2], precision),
    )


def _parse_render_txt(
    filepath: str,
    offset: tuple[float, float, float],
    precision: int = 1,
) -> dict:
    """Parse a render dump and return a dict keyed by (wm, surface, tex, pos).

    Values are ``(B, G, R)`` integer tuples.
    """
    result: dict = {}
    cur_wm   = "0"
    cur_surf = "0"
    cur_tex  = "unknown"

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            s = line.strip()

            m = re.search(r"\[WorldModel:?\s*(.*?)\]", s)
            if m:
                wm_name = m.group(1).strip()
                cur_wm  = wm_name or "0"
                continue

            m_surf = re.search(r"\[Surface (\d+)\]", s)
            m_tex  = re.search(r"tex=([^\s]+)", s)

            if m_surf:
                cur_surf = m_surf.group(1)
            if m_tex:
                cur_tex = m_tex.group(1).replace("\\", "/").lower()
            if m_surf or m_tex:
                continue

            m_vert = re.search(
                r"v\d+\s+pos=\(([^,]+),([^,]+),([^)]+)\)\s+"
                r"BGR\((\d+),\s*(\d+),\s*(\d+)\)",
                s,
            )
            if m_vert:
                pos = _normalize_pos(
                    (float(m_vert.group(1)), float(m_vert.group(2)), float(m_vert.group(3))),
                    offset,
                    precision,
                )
                key = (cur_wm, cur_surf, cur_tex, pos)
                result.setdefault(
                    key,
                    (int(m_vert.group(4)), int(m_vert.group(5)), int(m_vert.group(6))),
                )

    return result


# ---------------------------------------------------------------------------
# Colour selection heuristic
# ---------------------------------------------------------------------------

def _get_recommended_color(
    colors: list[tuple[int, int, int]],
    total_verts: int,
) -> tuple[int, int, int]:
    """Choose the most representative ambient colour, filtering noise.

    Rules applied in order:
      - Skip pure black (shadow / invisible zones).
      - Skip singleton values when the sample is large (noise filter).
      - Skip very dark colours that represent less than 5 % of the surface.
    """
    for (b, g, r), count in Counter(colors).most_common():
        if b == 0 and g == 0 and r == 0:
            continue
        if count <= 2 and total_verts > 10:
            continue
        if (b + g + r) < 8 and count < total_verts * 0.05:
            continue
        return (b, g, r)
    return (0, 0, 0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compare_ambient(orig_path: str, zero_path: str) -> None:
    """Compute ambient colours from two render dumps and write a report.

    Args:
        orig_path:  Path to the render dump with normal AmbientLight values.
        zero_path:  Path to the render dump with AmbientLight = 0 everywhere.

    Output:
        ``<orig_basename>_ambient.txt`` in the current directory.
    """
    orig_offset = _parse_offset(orig_path)
    zero_offset = _parse_offset(zero_path)

    orig_map = _parse_render_txt(orig_path, orig_offset)
    zero_map = _parse_render_txt(zero_path, zero_offset)

    # Index the zero-ambient map by (WorldModel, position) only, so that
    # surface-ID or texture mismatches between the two files don't block lookups.
    zero_by_pos: dict = {}
    for (wm, _surf, _tex, pos), color in zero_map.items():
        zero_by_pos[(wm, pos)] = color

    matched   = 0
    unmatched = 0
    by_group: dict = defaultdict(lambda: defaultdict(list))

    for (wm, surf_id, tex, pos), (ob, og, or_) in orig_map.items():
        zero = zero_by_pos.get((wm, pos))
        if zero is None:
            unmatched += 1
            continue

        zb, zg, zr = zero
        diff = (
            max(0, ob - zb),
            max(0, og - zg),
            max(0, or_ - zr),
        )
        by_group[wm][(surf_id, tex)].append(diff)
        matched += 1

    out_name = os.path.splitext(os.path.basename(orig_path))[0] + "_ambient.txt"

    with open(out_name, "w", encoding="utf-8") as out:
        out.write(f"Original:    {orig_path}\n")
        out.write(f"Ambient=0:   {zero_path}\n")
        out.write(f"Orig offset: {orig_offset}\n")
        out.write(f"Zero offset: {zero_offset}\n\n")
        out.write(f"Matched vertices:   {matched}\n")
        out.write(f"Unmatched vertices: {unmatched}\n")

        for wm in sorted(by_group, key=lambda k: "" if k is None else k):
            label = "Main (RenderNodes)" if wm == "0" else f"WM: {wm}"
            out.write(f"\n{'=' * 80}\n  {label}\n{'=' * 80}\n")

            sorted_surfaces = sorted(
                by_group[wm], key=lambda k: (int(k[0]), k[1])
            )
            for surf_id, tex in sorted_surfaces:
                colors      = by_group[wm][(surf_id, tex)]
                total_verts = len(colors)
                top_5       = Counter(colors).most_common(5)
                rec_b, rec_g, rec_r = _get_recommended_color(colors, total_verts)

                surf_label = f"Surface {surf_id}" if surf_id != "0" else "All Geometry"
                out.write(f"{surf_label:<15} {tex:<45} [{total_verts} verts]\n")

                if rec_b == 0 and rec_g == 0 and rec_r == 0:
                    out.write("  => Recommended Ambient: ZERO\n")
                else:
                    out.write(
                        f"  => Recommended Ambient: "
                        f"RGB({rec_r/255:.4f}, {rec_g/255:.4f}, {rec_b/255:.4f})"
                        f"  [B:{rec_b} G:{rec_g} R:{rec_r}]\n"
                    )

                for rank, ((b, g, r), count) in enumerate(top_5, 1):
                    marker = ""
                    if (b, g, r) not in {(0, 0, 0), (rec_b, rec_g, rec_r)}:
                        marker = "   <-- rejected (noise)"
                    if b == 0 and g == 0 and r == 0:
                        out.write(
                            f"     {rank}. B:{b:>3}  G:{g:>3}  R:{r:>3}"
                            f"   ({count:>4} verts) - ZERO\n"
                        )
                    else:
                        out.write(
                            f"     {rank}. B:{b:>3}  G:{g:>3}  R:{r:>3}"
                            f"   ({count:>4} verts){marker}\n"
                        )

    print(f"Matched: {matched}, Unmatched: {unmatched}")
    print(f"Report written to: {out_name}")
