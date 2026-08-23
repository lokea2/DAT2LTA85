"""
lth_header.py — LithTech Jupiter world-file header and tree parsers.

Reads the fixed-size file header, world-info block, and the WorldTree
(which contains all WorldModel BSP data).  Also exposes surface-flag
decoding and the global world-offset helpers used throughout the exporter.
"""

import struct
from lth_types import Vec3, Vec3u, read_lithtech_string

# ---------------------------------------------------------------------------
# Module-level globals (set once per conversion run)
# ---------------------------------------------------------------------------

info_string: str = ""
WORLD_OFFSET: Vec3 | None = None


# ---------------------------------------------------------------------------
# Surface flags
# ---------------------------------------------------------------------------

_SURFACE_FLAGS: dict[int, str] = {
    1 << 0:  "SOLID",
    1 << 1:  "NONEXISTENT",
    1 << 2:  "INVISIBLE",
    1 << 4:  "SKY",
    1 << 6:  "FLATSHADE",
    1 << 7:  "LIGHTMAP",
    1 << 8:  "NOSUBDIV",
    1 << 10: "PARTICLEBLOCKER",
    1 << 12: "GOURAUDSHADE",
    1 << 17: "PHYSICSBLOCKER",
    1 << 19: "RBSPLITTER",
    1 << 21: "VISBLOCKER",
    1 << 22: "NOTASTEP",
    1 << 24: "RECEIVELIGHT",
    1 << 25: "RECEIVESHADOWS",
    1 << 26: "RECEIVESUNLIGHT",
    1 << 28: "SHADOWMESH",
    1 << 29: "CASTSHADOWMESH",
    1 << 30: "CLIPLIGHT",
}


def decode_surface_flags(flags: int) -> list[str]:
    """Return a list of human-readable flag names set in *flags*."""
    return [name for bit, name in _SURFACE_FLAGS.items() if flags & bit]


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def apply_world_offset(vec: Vec3) -> Vec3:
    """Return *vec* translated by the global world offset (if set)."""
    if WORLD_OFFSET is None:
        return vec
    return Vec3(vec.x + WORLD_OFFSET.x,
                vec.y + WORLD_OFFSET.y,
                vec.z + WORLD_OFFSET.z)


def write_vec3(vec: Vec3) -> str:
    """Format a Vec3 as ``'x y z'`` for LTA output."""
    return f"{vec.x:.6f} {vec.y:.6f} {vec.z:.6f}"


# ---------------------------------------------------------------------------
# File header (60-byte fixed block)
# ---------------------------------------------------------------------------

class LithtechHeader:
    """The 60-byte fixed header at the start of every LithTech Jupiter DAT."""

    def read(self, f) -> None:
        (
            self.version,
            self.object_data_pos,
            self.blind_object_data_pos,
            self.lightgrid_pos,
            self.collision_data_pos,
            self.particle_blocker_data_pos,
            self.render_data_pos,
            self.packer_type,
            self.packer_version,
            *self.future,
        ) = struct.unpack("<15I", f.read(60))

    def print_info(self, out) -> None:
        print("Header:", file=out)
        print(f"  Version:                {self.version}", file=out)
        print(f"  ObjectDataPos:          0x{self.object_data_pos:08X}", file=out)
        print(f"  BlindObjectDataPos:     0x{self.blind_object_data_pos:08X}", file=out)
        print(f"  LightgridPos:           0x{self.lightgrid_pos:08X}", file=out)
        print(f"  CollisionDataPos:       0x{self.collision_data_pos:08X}", file=out)
        print(f"  ParticleBlockerDataPos: 0x{self.particle_blocker_data_pos:08X}", file=out)
        print(f"  RenderDataPos:          0x{self.render_data_pos:08X}", file=out)


# ---------------------------------------------------------------------------
# World info block
# ---------------------------------------------------------------------------

class LithtechWorldInfo:
    """World-info block: info string, extents, and world-space offset."""

    def read(self, f) -> None:
        global info_string
        (strlen,) = struct.unpack("<I", f.read(4))
        info_string = f.read(strlen).decode("utf-8", errors="ignore")
        self.extents_min = Vec3.read(f)
        self.extents_max = Vec3.read(f)
        self.offset      = Vec3.read(f)

    def print_info(self, out) -> None:
        print("\nWorld Info:", file=out)
        print(f"  Info String:  {info_string}", file=out)
        print(f"  Extents Min:  {self.extents_min}", file=out)
        print(f"  Extents Max:  {self.extents_max}", file=out)
        print(f"  Offset:       {self.offset}", file=out)


def read_world_info(f) -> LithtechWorldInfo:
    """Read the world-info block and cache its offset as the global ``WORLD_OFFSET``."""
    global WORLD_OFFSET
    world_info = LithtechWorldInfo()
    world_info.read(f)
    WORLD_OFFSET = world_info.offset
    return world_info


# ---------------------------------------------------------------------------
# WorldTree bit-stream parser
# ---------------------------------------------------------------------------

def _parse_worldtree_bits(data: bytes) -> list[str]:
    """Walk the packed bit-stream and return a text representation of the tree."""
    out: list[str] = ["Parsed WorldLayout Tree:\n"]

    def read_node(reader: dict, prefix: str = "", is_last: bool = True) -> None:
        bit = (reader["data"][reader["byte"]] >> reader["bit"]) & 1
        reader["bit"] += 1
        if reader["bit"] == 8:
            reader["bit"] = 0
            reader["byte"] += 1
        connector = "└── " if is_last else "├── "
        out.append(prefix + connector + ("[Node]\n" if bit else "[Leaf]\n"))
        new_prefix = prefix + ("    " if is_last else "│   ")
        if bit:
            for i in range(4):
                read_node(reader, new_prefix, i == 3)

    read_node({"data": data, "byte": 0, "bit": 0})
    return out


# ---------------------------------------------------------------------------
# WorldModel
# ---------------------------------------------------------------------------

class WorldModel:
    """One BSP world model (physics geometry + surface/texture metadata)."""

    def __init__(self) -> None:
        self.dummy               = 0
        self.world_info_flags    = 0
        self.world_name          = ""
        self.texture_names: list[str] = []
        self.points_len          = 0
        self.planes_len          = 0
        self.surfaces_len        = 0
        self.portals             = 0
        self.polies_len          = 0
        self.leaves_len          = 0
        self.polies_vertices_len = 0
        self.visible_list_len    = 0
        self.leaf_list           = 0
        self.nodes_len           = 0
        self.world_bbox_min      = Vec3()
        self.world_bbox_max      = Vec3()
        self.world_translation   = Vec3()

    def read(self, f) -> None:
        self.dummy,            = struct.unpack("<I", f.read(4))
        self.world_info_flags, = struct.unpack("<I", f.read(4))
        name_len,              = struct.unpack("<H", f.read(2))
        self.world_name = f.read(name_len).decode("utf-8", errors="ignore")

        (
            self.points_len,
            self.planes_len,
            self.surfaces_len,
            self.portals,
            self.polies_len,
            self.leaves_len,
            self.polies_vertices_len,
            self.visible_list_len,
            self.leaf_list,
            self.nodes_len,
        ) = struct.unpack("<10I", f.read(40))

        self.world_bbox_min    = Vec3.read(f)
        self.world_bbox_max    = Vec3.read(f)
        self.world_translation = Vec3.read(f)

        texture_names_size, = struct.unpack("<I", f.read(4))
        _texture_names_len, = struct.unpack("<I", f.read(4))
        self.texture_names = self._parse_texture_names(f.read(texture_names_size))

        self.vertices_len = list(f.read(self.polies_len))
        self.planes = [
            (Vec3.read(f), struct.unpack("<f", f.read(4))[0])
            for _ in range(self.planes_len)
        ]

        self.surfaces = []
        for _ in range(self.surfaces_len):
            flags, tex_idx, tex_flags = struct.unpack("<IHH", f.read(8))
            self.surfaces.append(
                {"Flags": flags, "TextureIndex": tex_idx, "TextureFlags": tex_flags}
            )

        self.polies = []
        for vert_count in self.vertices_len:
            surface_idx = struct.unpack("<I", f.read(4))[0]
            plane_idx   = struct.unpack("<I", f.read(4))[0]
            indices     = list(struct.unpack(f"<{vert_count}I", f.read(4 * vert_count)))
            self.polies.append((surface_idx, plane_idx, indices))

        self.nodes = []
        for _ in range(self.nodes_len):
            poly_idx = struct.unpack("<I", f.read(4))[0]
            zero     = struct.unpack("<H", f.read(2))[0]
            children = struct.unpack("<2i", f.read(8))
            self.nodes.append((poly_idx, zero, children))

        self.points       = [Vec3.read(f) for _ in range(self.points_len)]
        self.root_node_index = struct.unpack("<i", f.read(4))[0]
        self.sections        = struct.unpack("<I", f.read(4))[0]

    @staticmethod
    def _parse_texture_names(raw: bytes) -> list[str]:
        return [
            chunk.decode("utf-8", errors="ignore").replace("\\", "/")
            for chunk in raw.split(b"\x00")
            if chunk
        ]

    def print_info(self, out) -> None:
        print("\nWorldModel Info:", file=out)
        print(f"  Name:              {self.world_name}", file=out)
        print(f"  Points:            {self.points_len}", file=out)
        print(f"  Planes:            {self.planes_len}", file=out)
        print(f"  Surfaces:          {self.surfaces_len}", file=out)
        print(f"  Polies:            {self.polies_len}", file=out)
        print(f"  Nodes:             {self.nodes_len}", file=out)
        print(f"  World BBox Min:    {self.world_bbox_min}", file=out)
        print(f"  World BBox Max:    {self.world_bbox_max}", file=out)
        print(f"  World Translation: {self.world_translation}", file=out)
        print(f"  RootNodeIndex:     {self.root_node_index}", file=out)
        print("\n--- Surfaces ---", file=out)
        for i, s in enumerate(self.surfaces):
            flags      = s["Flags"]
            tex_idx    = s["TextureIndex"]
            flag_str   = ", ".join(decode_surface_flags(flags)) or "None"
            tex_name   = (
                self.texture_names[tex_idx]
                if 0 <= tex_idx < len(self.texture_names)
                else "<INVALID INDEX>"
            )
            print(
                f"  [{i}] Flags: 0x{flags:08X} ({flag_str}), "
                f"TextureIndex: {tex_idx} ({tex_name}), "
                f"TextureFlags: 0x{s['TextureFlags']:04X}",
                file=out,
            )


# ---------------------------------------------------------------------------
# WorldTree
# ---------------------------------------------------------------------------

class WorldTree:
    """Top-level container: bounding box, layout bit-stream, and all WorldModels."""

    def __init__(self) -> None:
        self.root_bbox_min    = Vec3()
        self.root_bbox_max    = Vec3()
        self.sub_nodes_len    = 0
        self.terrain_depth    = 0
        self.world_layout     = b""
        self.layout_tree_strs: list[str] = []
        self.world_models_len = 0
        self.world_models: list[WorldModel] = []

    def read(self, f) -> None:
        self.root_bbox_min  = Vec3.read(f)
        self.root_bbox_max  = Vec3.read(f)
        self.sub_nodes_len, = struct.unpack("<I", f.read(4))
        self.terrain_depth, = struct.unpack("<I", f.read(4))

        layout_bytes      = (self.sub_nodes_len + 7) // 8
        self.world_layout = f.read(layout_bytes)
        self.layout_tree_strs = _parse_worldtree_bits(self.world_layout)

        self.world_models_len, = struct.unpack("<I", f.read(4))
        self.world_models = []
        for _ in range(self.world_models_len):
            wm = WorldModel()
            wm.read(f)
            self.world_models.append(wm)

    def print_info(self, out) -> None:
        print("\nWorldTree Info:", file=out)
        print(f"  Root BBox Min:  {self.root_bbox_min}", file=out)
        print(f"  Root BBox Max:  {self.root_bbox_max}", file=out)
        print(f"  Sub Nodes:      {self.sub_nodes_len}", file=out)
        print(f"  Terrain Depth:  {self.terrain_depth}", file=out)
        print(f"  WorldModels:    {self.world_models_len}", file=out)
        for i, wm in enumerate(self.world_models):
            print(f"\n  [WorldModel {i}]", file=out)
            wm.print_info(out)
