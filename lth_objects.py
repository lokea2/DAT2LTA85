"""
lth_objects.py — WorldObject and BlindObject readers.

Parses the two object sections of a LithTech Jupiter DAT file:

* ``read_world_objects``  — typed property list for every world object.
* ``read_blind_objects``  — engine-specific binary payloads (KeyFramer keys,
                            ScatterVolume particles, …).
* ``read_particle_blockers`` — convex-hull blockers for the particle system.
* ``read_world_objects_lta``  — variant used by the LTA exporter that returns
                                structured dicts instead of printing.
"""

import struct
from lth_types import Vec3, Vec2, Color, Quaternion, read_lithtech_string, st_gethash_ic


# ---------------------------------------------------------------------------
# Property-type constants (matches the engine enum)
# ---------------------------------------------------------------------------

_PROP_STRING   = 0
_PROP_VECTOR   = 1
_PROP_COLOR    = 2
_PROP_REAL     = 3
_PROP_BOOL     = 5
_PROP_LONGINT  = 6
_PROP_ROTATION = 7


def _read_property(f, name: str, data_type: int, data_size: int):
    """Decode one property value according to *data_type*.

    Unknown types are skipped (the cursor is advanced by *data_size* bytes)
    and a descriptive placeholder string is returned.
    """
    # Special case: RenderGroup is stored as float even when typed as longint.
    if name == "RenderGroup" and data_type == _PROP_LONGINT:
        return struct.unpack("<f", f.read(4))[0]

    if data_type == _PROP_STRING:   return read_lithtech_string(f)
    if data_type == _PROP_VECTOR:   return Vec3.read(f)
    if data_type == _PROP_COLOR:    return Color.read(f)
    if data_type == _PROP_REAL:     return struct.unpack("<f", f.read(4))[0]
    if data_type == _PROP_BOOL:     return struct.unpack("<B", f.read(1))[0] != 0
    if data_type == _PROP_LONGINT:  return struct.unpack("<f", f.read(4))[0]
    if data_type == _PROP_ROTATION: return Quaternion.read(f)

    f.seek(data_size, 1)
    return f"<unknown type {data_type}, skipped {data_size} bytes>"


# ---------------------------------------------------------------------------
# Public readers
# ---------------------------------------------------------------------------

def read_world_objects(f, header, out):
    """Read the WorldObject section and return three lookup structures.

    Returns:
        keyframer_basekeynames: list of BaseKeyName strings for KeyFramer objects.
        scattervolume_names:    list of Name strings for ScatterVolume objects.
        occluder_names:         dict mapping occluder name → LithTech hash.
    """
    f.seek(header.object_data_pos)
    (count,) = struct.unpack("<I", f.read(4))
    print(f"\nWorldObjects: {count}", file=out)

    keyframer_basekeynames: list[str] = []
    scattervolume_names:    list[str] = []
    occluder_names:         dict[str, int] = {}

    for i in range(count):
        _object_size  = struct.unpack("<H", f.read(2))[0]
        obj_type      = read_lithtech_string(f)
        (prop_count,) = struct.unpack("<I", f.read(4))
        print(f"\n[Object #{i}] Type: {obj_type}, Properties: {prop_count}", file=out)

        for _ in range(prop_count):
            name      = read_lithtech_string(f)
            data_type = struct.unpack("<B", f.read(1))[0]
            _flags    = struct.unpack("<I", f.read(4))[0]
            data_size = struct.unpack("<H", f.read(2))[0]

            value = _read_property(f, name, data_type, data_size)
            print(f"  - {name} = {value}", file=out)

            if obj_type == "KeyFramer" and name == "BaseKeyName" and isinstance(value, str):
                keyframer_basekeynames.append(value)
            if obj_type == "ScatterVolume" and name == "Name" and isinstance(value, str):
                scattervolume_names.append(value)
            if (obj_type == "DynamicOccluderVolume"
                    and name.startswith("OccluderName")
                    and isinstance(value, str)
                    and value):
                occluder_names.setdefault(value, st_gethash_ic(value))

    return keyframer_basekeynames, scattervolume_names, occluder_names


def read_blind_objects(f, header, keyframer_basekeynames, scattervolume_names, out):
    """Read the BlindObject section (engine-opaque binary payloads).

    Returns:
        keyframer_data_map: dict mapping BaseKeyName → list of key dicts.
    """
    f.seek(header.blind_object_data_pos)
    (object_count,) = struct.unpack("<I", f.read(4))
    print(f"Blind Objects: {object_count}\n", file=out)

    # Engine-defined object IDs
    _ID_KEYFRAMER     = 1_789_855_876
    _ID_SCATTERVOLUME = 1_945_451_140

    keyframer_idx      = 0
    scatter_idx        = 0
    keyframer_data_map: dict[str, list[dict]] = {}

    for obj_idx in range(object_count):
        (engine_size,) = struct.unpack("<I", f.read(4))
        (obj_id,)      = struct.unpack("<I", f.read(4))

        print(f"Blind Object {obj_idx + 1}:", file=out)
        print(f"  - ID = {obj_id} (0x{obj_id:08X})", file=out)

        if obj_id == _ID_KEYFRAMER:
            key_name = (
                keyframer_basekeynames[keyframer_idx]
                if keyframer_idx < len(keyframer_basekeynames)
                else "<unknown KeyFramer>"
            )
            keyframer_idx += 1
            keyframer_data_map[key_name] = []

            (key_count,) = struct.unpack("<I", f.read(4))
            print(f"  - KF Name:  {key_name}", file=out)
            print(f"  - NumKeys:  {key_count}", file=out)

            for key_idx in range(key_count):
                # Key name follows the engine convention: base + "" / "01" / "02" …
                full_key_name = (
                    f"{key_name}{key_idx}"
                    if key_idx == 0
                    else f"{key_name}{key_idx:02d}"
                )

                (key_type,)     = struct.unpack("<H", f.read(2))
                (name_len,)     = struct.unpack("<B", f.read(1))
                (cmd_len,)      = struct.unpack("<B", f.read(1))
                pos             = Vec3.read(f)
                rot_deg         = Vec3.read(f)
                (timestamp,)    = struct.unpack("<f", f.read(4))
                (sound_radius,) = struct.unpack("<f", f.read(4))
                sound_name      = f.read(name_len).decode("ascii", errors="ignore")
                command         = f.read(cmd_len).decode("ascii", errors="ignore")

                # Bézier handles (present only for certain key types)
                bez_prev = bez_next = None
                if   key_type == 0x0001: bez_prev = Vec3.read(f)
                elif key_type == 0x0002: bez_next = Vec3.read(f)
                elif key_type == 0x0003: bez_prev = Vec3.read(f); bez_next = Vec3.read(f)
                elif key_type != 0x0000:
                    raise ValueError(
                        f"Unknown key_type 0x{key_type:04X} at 0x{f.tell():08X}"
                    )

                print(f"    [Key #{key_idx}] {full_key_name}:", file=out)
                print(f"    - Pos         = {pos}", file=out)
                print(f"    - Rotation    = {rot_deg}", file=out)
                print(f"    - TimeStamp   = {timestamp:.6f}", file=out)
                print(f"    - SoundRadius = {sound_radius:.6f}", file=out)
                print(f"    - SoundName   = '{sound_name}'", file=out)
                print(f"    - Command     = '{command}'", file=out)
                if bez_prev: print(f"    - BezierPrev  = {bez_prev}", file=out)
                if bez_next: print(f"    - BezierNext  = {bez_next}", file=out)

                keyframer_data_map[key_name].append({
                    "full_name":    full_key_name,
                    "pos":          pos,
                    "rot_deg":      rot_deg,
                    "timestamp":    timestamp,
                    "sound_radius": sound_radius,
                    "sound_name":   sound_name,
                    "command":      command,
                    "bez_prev":     bez_prev,
                    "bez_next":     bez_next,
                })
            print("", file=out)

        elif obj_id == _ID_SCATTERVOLUME:
            scatter_name = (
                scattervolume_names[scatter_idx]
                if scatter_idx < len(scattervolume_names)
                else "<unknown ScatterVolume>"
            )
            scatter_idx += 1

            (num_volumes,) = struct.unpack("<I", f.read(4))
            print(f"  - ScatterVolume: {scatter_name}", file=out)
            print(f"  - NumVolumes:    {num_volumes}", file=out)

            for vi in range(num_volumes):
                pos  = Vec3.read(f)
                dims = Vec3.read(f)
                (num_particles,) = struct.unpack("<I", f.read(4))
                print(
                    f"    [Volume #{vi}]: pos={pos}, dims={dims}, "
                    f"particles={num_particles}",
                    file=out,
                )
                for pi in range(num_particles):
                    part_pos    = Vec3.read(f)
                    (color,)    = struct.unpack("<I", f.read(4))
                    (scale,)    = struct.unpack("<f", f.read(4))
                    (wave_rot,) = struct.unpack("<B", f.read(1))
                    (wave_start,) = struct.unpack("<B", f.read(1))
                    print(
                        f"      Particle {pi}: pos={part_pos}, "
                        f"color=0x{color:08X}, scale={scale:.6f}, "
                        f"waveRot={wave_rot}, waveStart={wave_start}",
                        file=out,
                    )
                print("", file=out)

        else:
            # Unknown object type — skip its payload.
            f.seek(engine_size, 1)

        print("", file=out)

    return keyframer_data_map


def read_particle_blockers(f, header, out) -> None:
    """Read and print the particle-blocker convex hulls."""
    f.seek(header.particle_blocker_data_pos)
    (count,) = struct.unpack("<I", f.read(4))
    print(f"\nParticle Blockers: {count}", file=out)

    for i in range(count):
        (vcount,) = struct.unpack("<B", f.read(1))
        verts      = [Vec3.read(f) for _ in range(vcount)]
        normal     = Vec3.read(f)
        (radius,)  = struct.unpack("<f", f.read(4))
        print(
            f"  [Blocker #{i}] Verts: {vcount}, "
            f"Radius: {radius:.2f}, Normal: {normal}",
            file=out,
        )
        for v in verts:
            print(f"    - {v}", file=out)


def read_world_objects_lta(f, header):
    """Read WorldObjects and return structured data for the LTA exporter.

    Returns:
        objects: list of property dicts (including ``__Type``).
        metas:   list of per-property metadata dicts (type names for LTA).
    """
    f.seek(header.object_data_pos)
    (count,) = struct.unpack("<I", f.read(4))

    objects: list[dict] = []
    metas:   list[dict] = []

    _META_TYPE = {
        _PROP_STRING:   "string",
        _PROP_VECTOR:   "vector",
        _PROP_COLOR:    "color",
        _PROP_REAL:     "real",
        _PROP_BOOL:     "bool",
        _PROP_LONGINT:  "longint",
        _PROP_ROTATION: "rotation",
    }

    for _ in range(count):
        _object_size  = struct.unpack("<H", f.read(2))[0]
        obj_type      = read_lithtech_string(f)
        (prop_count,) = struct.unpack("<I", f.read(4))

        props: dict = {"__Type": obj_type}
        meta:  dict = {}

        for _ in range(prop_count):
            name      = read_lithtech_string(f)
            data_type = struct.unpack("<B", f.read(1))[0]
            _flags    = struct.unpack("<I", f.read(4))[0]
            data_size = struct.unpack("<H", f.read(2))[0]

            value     = _read_property(f, name, data_type, data_size)
            meta_type = _META_TYPE.get(data_type, "string")

            props[name]     = value
            meta[name]      = {"type": meta_type}

        objects.append(props)
        metas.append(meta)

    return objects, metas
