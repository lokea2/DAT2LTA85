"""
dat2lta85.py — Convert a LithTech Jupiter v85 DAT world file to LTA format.

Usage:
    python dat2lta85.py <mapname.dat> -v1 | -v2
    python dat2lta85.py -comp <original.txt> <ambient_zero.txt>

Version flags:
    -v1   Vertex layout includes tangent + binormal (24 extra bytes per vertex).
    -v2   Vertex layout omits tangent + binormal.

Outputs (normal mode):
    <mapname>.txt             — Human-readable parse log.
    <mapname>.lta             — Render geometry in LTA format.
    <mapname>_PhysicsDATA.lta — Physics BSP surfaces in LTA format.

Output (-comp mode):
    <original>_ambient.txt    — Per-surface recommended ambient RGB values.
"""

import os
import sys

from lth_header import LithtechHeader, LithtechWorldInfo, WorldTree, read_world_info
from lth_objects import read_world_objects, read_blind_objects
from lth_render import read_render_data, write_physics_vertex_colors
from lth_export import export_rendernodes_to_lta, write_surfaces_to_lta
from lth_ambient import compare_ambient
from lth_match import extract_render_geometry_by_polyindex, build_pool_matrices


def convert(dat_path: str, version_flag: str) -> None:
    """Run the full DAT → LTA conversion pipeline.

    Args:
        dat_path:     Path to the source ``.dat`` file.
        version_flag: ``"-v1"`` or ``"-v2"`` (vertex layout selector).
    """
    map_name        = os.path.splitext(os.path.basename(dat_path))[0]
    info_txt        = f"{map_name}.txt"
    output_lta      = f"{map_name}.lta"
    output_phys_lta = f"{map_name}_PhysicsDATA.lta"

    # --- Pass 1: parse log ---------------------------------------------------
    with open(info_txt, "w", encoding="utf-8") as out:
        with open(dat_path, "rb") as f:
            header = LithtechHeader()
            header.read(f)
            header.print_info(out)

            world_info = LithtechWorldInfo()
            world_info.read(f)
            world_info.print_info(out)

            world_tree = WorldTree()
            world_tree.read(f)
            world_tree.print_info(out)

            keyframer_basekeynames, scattervolume_names, occluder_names = \
                read_world_objects(f, header, out)
            keyframer_keys_map = read_blind_objects(
                f, header, keyframer_basekeynames, scattervolume_names, out
            )
            read_render_data(f, header, out, version_flag)
            write_physics_vertex_colors(f, header, world_tree, out, version_flag)

    # --- Pass 2: build render pools (shared by both LTA outputs) ------------
    print("Extracting render geometry …")
    with open(dat_path, "rb") as f:
        header = LithtechHeader()
        header.read(f)
        read_world_info(f)
        WorldTree().read(f)  # advance past the tree to render_data_pos
        main_pool_by_tex, wm_pool_by_tex = \
            extract_render_geometry_by_polyindex(f, header, version_flag)

    print("Building search matrices …")
    main_pool_matrices = build_pool_matrices(main_pool_by_tex)
    wm_pool_matrices   = {
        wm: build_pool_matrices(p) for wm, p in wm_pool_by_tex.items()
    }

    # --- Pass 3: render LTA export ------------------------------------------
    with open(dat_path, "rb") as f, open(output_lta, "w", encoding="utf-8") as out_lta:
        header = LithtechHeader()
        header.read(f)
        read_world_info(f)

        export_rendernodes_to_lta(
            f, header, out_lta, map_name, world_tree,
            keyframer_keys_map, occluder_names, version_flag,
        )

    # --- Pass 4: physics LTA export -----------------------------------------
    with open(output_phys_lta, "w", encoding="utf-8") as out_phys:
        with open(dat_path, "rb") as f:
            write_surfaces_to_lta(
                f, header, world_tree, out_phys, map_name,
                main_pool_by_tex,  main_pool_matrices,
                wm_pool_by_tex,    wm_pool_matrices,
                occluder_names=occluder_names,
                keyframer_keys_map=keyframer_keys_map,
            )

    print(f"{map_name}: conversion complete.")
    print(f"  Parse log:    {info_txt}")
    print(f"  Render LTA:   {output_lta}")
    print(f"  Physics LTA:  {output_phys_lta}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _usage() -> None:
    print(__doc__)
    sys.exit(1)


def main() -> None:
    argv = sys.argv[1:]

    # Ambient comparison mode
    if len(argv) == 3 and argv[0] == "-comp":
        orig_path, zero_path = argv[1], argv[2]
        for path in (orig_path, zero_path):
            if not os.path.isfile(path):
                print(f"Error: file not found: {path}")
                sys.exit(1)
        compare_ambient(orig_path, zero_path)
        sys.exit(0)

    # Normal conversion mode
    if len(argv) != 2:
        _usage()

    dat_path, version_flag = argv

    if not os.path.isfile(dat_path):
        print(f"Error: file not found: {dat_path}")
        sys.exit(1)

    if version_flag not in ("-v1", "-v2"):
        print(f"Error: unknown version flag {version_flag!r}. Use -v1 or -v2.")
        _usage()

    convert(dat_path, version_flag)


if __name__ == "__main__":
    main()
