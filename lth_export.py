"""
lth_export.py — LTA format writer.

Exports render nodes, physics BSP surfaces, KeyFramer animations,
occluders, particle blockers, and world-model brushes to the
LithTech ASCII (LTA) scene format consumed by DEdit.
"""

import math
import os
import struct
from collections import defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

import lth_header
from lth_header import apply_world_offset, decode_surface_flags, write_vec3, LithtechWorldInfo
from lth_match import clean_tex_key, find_best_donor
from lth_objects import read_world_objects_lta
from lth_render import EPCShaderType
from lth_types import Vec3, Vec2, Vec3u, Vertex, Quaternion, read_lithtech_string, current_dir


def get_dtx_texture_size(tex_path, search_dirs=None):
    """Return the (width, height) of a DTX or SPR texture file.

    Args:
        tex_path:    Relative texture path as stored in the DAT file.
        search_dirs: List of root directories to search.  Defaults to the
                     parent directory of this script.

    Returns:
        ``(width, height)`` as integers, or ``(0, 0)`` when the file cannot
        be located or read.
    """
    if search_dirs is None:
        search_dirs = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

    if tex_path.lower().startswith("lightanim") or tex_path.lower().startswith("default"):
        return 0, 0

    tex_rel = tex_path.replace("\\", "/").lstrip("/").lower()
    found = False

    for root in search_dirs:
        full_path = os.path.join(root, tex_rel)

        if not os.path.exists(full_path):
            continue

        found = True
        try:
            if full_path.lower().endswith(".spr"):
                with open(full_path, "rb") as f:
                    f.seek(20)
                    dtx_rel_path = read_lithtech_string(f).replace("\\", "/").lower()
                    full_path = os.path.join(root, dtx_rel_path)
                    if not os.path.exists(full_path):
                        print(f"[MISSING] DTX inside SPR not found: {full_path}")
                        continue

            with open(full_path, "rb") as f:
                f.seek(8)
                w = struct.unpack("<H", f.read(2))[0]
                h = struct.unpack("<H", f.read(2))[0]
                return w, h

        except Exception as e:
            print(f"[WARN] Can't read texture: {full_path} ({e})")

    if not found:
        print(f"[MISSING] Texture not found: {tex_rel}")

    return 0, 0


def generate_opq_exact(v0, v1, v2, uv0, uv1, uv2, width, height):
    if width == 0 or height == 0:
        return Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 1)
    
    def barycentric(p0, p1, p2, p):
        area = lambda a, b, c: (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])
        n = area(p0, p1, p2)
        if abs(n) < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        u = area(p1, p2, p) / n
        v = area(p2, p0, p) / n
        w = 1.0 - u - v
        return np.array([u, v, w])

    t0 = np.array([uv0.x, -uv0.y])
    t1 = np.array([uv1.x, -uv1.y])
    t2 = np.array([uv2.x, -uv2.y])

    bc_o = barycentric(t0, t1, t2, np.array([0.0, 0.0]))
    bc_p = barycentric(t0, t1, t2, np.array([1.0, 0.0]))
    bc_q = barycentric(t0, t1, t2, np.array([0.0, 1.0]))

    p0 = np.array([v0.x, v0.y, v0.z])
    p1 = np.array([v1.x, v1.y, v1.z])
    p2 = np.array([v2.x, v2.y, v2.z])

    O = bc_o[0]*p0 + bc_o[1]*p1 + bc_o[2]*p2
    P = bc_p[0]*p0 + bc_p[1]*p1 + bc_p[2]*p2 - O
    Q = bc_q[0]*p0 + bc_q[1]*p1 + bc_q[2]*p2 - O

    tp = np.linalg.norm(P)
    tq = np.linalg.norm(Q)
    tp = 1.0 / (tp / width) if tp > 1e-8 else 1.0
    tq = 1.0 / (tq / height) if tq > 1e-8 else 1.0

    P = P / np.linalg.norm(P) if np.linalg.norm(P) > 1e-8 else np.array([1.0, 0.0, 0.0])
    Q = Q / np.linalg.norm(Q) if np.linalg.norm(Q) > 1e-8 else np.array([0.0, 1.0, 0.0])

    R = np.cross(Q, P)
    Pn = np.cross(R, Q)
    Qn = np.cross(P, R)

    Pn = Pn / np.linalg.norm(Pn)
    Qn = Qn / np.linalg.norm(Qn)

    pscale = 1.0 / np.dot(P, Pn)
    qscale = 1.0 / np.dot(Q, Qn)

    P_final = Pn * tp * pscale
    Q_final = Qn * tq * qscale * -1

    return Vec3(*O), Vec3(*P_final), Vec3(*Q_final)


def read_vertex_data(f, count, version_flag):
    vertices = []
    for _ in range(count):
        pos = Vec3.read(f)
        uv0 = Vec2.read(f)
        uv1 = Vec2.read(f)
        color = f.read(4)
        normal = Vec3.read(f)
        if version_flag == "-v1":
            tangent = Vec3.read(f)
            binormal = Vec3.read(f)
        vertices.append(Vertex(pos, normal, uv0, uv1, color))
    return vertices


def split_triangles_by_sections(triangles, section_infos):
    result = []
    cursor = 0
    for tex0, tex_w, tex_h, tex1, tex_w1, tex_h1, tri_count, shader_str, texture_effect in section_infos:
        tris = triangles[cursor:cursor + tri_count]
        result.append((
            tex0, tex_w, tex_h,
            tex1, tex_w1, tex_h1,
            tris,
            shader_str,
            texture_effect
        ))
        cursor += tri_count
    return result


def write_polyhedron(out_file, vertices, section_triangles):

    out_file.write("\t\t( polyhedron (\n")
    out_file.write("\t\t\t( color 255 255 255 )\n")
    out_file.write("\t\t\t( pointlist \n")
    for v in vertices:
        adjusted_v = apply_world_offset(v.pos)
        alpha = v.color[3] if v.color else 255
        out_file.write(f"\t\t\t\t( {write_vec3(adjusted_v)} {alpha} {alpha} {alpha} {alpha} )\n")
    out_file.write("\t\t\t)\n\t\t\t( polylist (\n")

    for tex_name, tex_w, tex_h, tex1_name, tex1_w, tex1_h, tris, shader_str, texture_effect in section_triangles:
        for tri in tris:
            v0, v1, v2 = [vertices[i] for i in tri]
            n = v0.normal

            p0 = apply_world_offset(v0.pos)
            p1 = apply_world_offset(v1.pos)
            p2 = apply_world_offset(v2.pos)

            dist = p0.x * n.x + p0.y * n.y + p0.z * n.z

            origin, U, V = generate_opq_exact(p0, p1, p2, v0.uv, v1.uv, v2.uv, tex_w, tex_h)
            origin1, U1, V1 = generate_opq_exact(p0, p1, p2, v0.uv1, v1.uv1, v2.uv1, tex1_w, tex1_h)
            out_file.write("\t\t\t\t( editpoly \n")
            out_file.write(f"\t\t\t\t\t( f {tri[0]} {tri[1]} {tri[2]} )\n")
            out_file.write(f"\t\t\t\t\t( n {write_vec3(n)} )\n")
            out_file.write(f"\t\t\t\t\t( dist {dist:.6f} )\n")
            out_file.write("\t\t\t\t\t( textureinfo \n")
            out_file.write(f"\t\t\t\t\t\t( {write_vec3(origin)} )\n")
            out_file.write(f"\t\t\t\t\t\t( {write_vec3(U)} )\n")
            out_file.write(f"\t\t\t\t\t\t( {write_vec3(V)} )\n")
            out_file.write("\t\t\t\t\t\t( sticktopoly 1 )\n")
            out_file.write(f'\t\t\t\t\t\t( name "{tex_name}" )\n')
            out_file.write("\t\t\t\t\t)\n")
            out_file.write("\t\t\t\t\t( flags )\n")
            out_file.write("\t\t\t\t\t( shade 0 0 0 )\n")
            out_file.write('\t\t\t\t\t( physicsmaterial "Default" )\n')
            out_file.write('\t\t\t\t\t( surfacekey "" )\n')
            out_file.write("\t\t\t\t\t( textures ( \n")
            out_file.write("\t\t\t\t\t\t( 1 ( textureinfo \n")
            out_file.write(f"\t\t\t\t\t\t\t( {write_vec3(origin1)} )\n")
            out_file.write(f"\t\t\t\t\t\t\t( {write_vec3(U1)} )\n")
            out_file.write(f"\t\t\t\t\t\t\t( {write_vec3(V1)} )\n")
            out_file.write(f'\t\t\t\t\t\t\t( sticktopoly 1 )\n')
            out_file.write(f'\t\t\t\t\t\t\t( name "{tex1_name}" )\n')
            out_file.write("\t\t\t\t\t\t) )\n")
            out_file.write("\t\t\t\t\t) )\n")
            out_file.write("\t\t\t\t)\n")

    out_file.write("\t\t\t) )\n\t\t) )\n")


def write_physics_polygons(f, header, out_file):
    tex_name = "Tex\\Blocker.dtx"
    f.seek(header.collision_data_pos)
    (poly_count,) = struct.unpack("<I", f.read(4))

    from lth_types import Vec3
    for i in range(poly_count):
        normal = Vec3.read(f)
        (dist,) = struct.unpack("<f", f.read(4))
        (vcount,) = struct.unpack("<I", f.read(4))

        verts = [Vec3.read(f) for _ in range(vcount)]
        adjusted_verts = [apply_world_offset(v) for v in verts]
        
        P0, Q0 = get_box_mapping_axes(normal)
        O0 = Vec3(0.0, 0.0, 0.0)

        out_file.write("\t\t( polyhedron (\n")
        out_file.write("\t\t\t( color 255 255 255 )\n")
        out_file.write("\t\t\t( pointlist \n")
        for adjusted_v in adjusted_verts:
            out_file.write(f"\t\t\t\t( {format_clean_vec3(adjusted_v)} 255 255 255 255 )\n")
        out_file.write("\t\t\t)\n\t\t\t( polylist (\n")
        out_file.write("\t\t\t\t( editpoly \n")
        out_file.write(f"\t\t\t\t\t( f {' '.join(str(j) for j in range(vcount))} )\n")
        out_file.write(f"\t\t\t\t\t( n {format_clean_vec3(normal)} )\n")
        out_file.write(f"\t\t\t\t\t( dist {dist:.6f} )\n")
        out_file.write("\t\t\t\t\t( textureinfo \n")
        out_file.write(f"\t\t\t\t\t\t( {format_clean_vec3(O0)} )\n")
        out_file.write(f"\t\t\t\t\t\t( {format_clean_vec3(P0)} )\n")
        out_file.write(f"\t\t\t\t\t\t( {format_clean_vec3(Q0)} )\n")
        out_file.write("\t\t\t\t\t\t( sticktopoly 1 )\n")
        out_file.write(f'\t\t\t\t\t\t( name "{tex_name}" )\n')
        out_file.write("\t\t\t\t\t)\n")
        out_file.write("\t\t\t\t\t( flags )\n")
        out_file.write("\t\t\t\t\t( shade 0 0 0 )\n")
        out_file.write("\t\t\t\t\t( physicsmaterial \"Default\" )\n")
        out_file.write("\t\t\t\t\t( surfacekey \"\" )\n")
        out_file.write("\t\t\t\t\t( textures ( \n")
        out_file.write("\t\t\t\t\t\t( 1 ( textureinfo \n")
        out_file.write("\t\t\t\t\t\t\t( 0.000000 0.000000 0.000000 )\n")
        out_file.write("\t\t\t\t\t\t\t( 1.000000 0.000000 0.000000 )\n")
        out_file.write("\t\t\t\t\t\t\t( 0.000000 0.000000 1.000000 )\n")
        out_file.write("\t\t\t\t\t\t\t( sticktopoly 1 )\n")
        out_file.write('\t\t\t\t\t\t\t( name "Default" )\n')
        out_file.write("\t\t\t\t\t\t) )\n")
        out_file.write("\t\t\t\t\t) )\n")
        out_file.write("\t\t\t\t)\n")
        out_file.write("\t\t\t) )\n\t\t) )\n")
    return poly_count

def write_particleblockers_polygons(f, header, out_file):
    tex_name = "Textures\\ParticleBlocker.dtx"
    f.seek(header.particle_blocker_data_pos)
    (poly_count,) = struct.unpack("<I", f.read(4))

    from lth_types import Vec3
    for i in range(poly_count):
        normal = Vec3.read(f)
        (dist,) = struct.unpack("<f", f.read(4))
        (vcount,) = struct.unpack("<I", f.read(4))

        verts = [Vec3.read(f) for _ in range(vcount)]
        adjusted_verts = [apply_world_offset(v) for v in verts]
        
        P0, Q0 = get_box_mapping_axes(normal)
        O0 = Vec3(0.0, 0.0, 0.0)

        out_file.write("\t\t( polyhedron (\n")
        out_file.write("\t\t\t( color 255 255 255 )\n")
        out_file.write("\t\t\t( pointlist \n")
        for adjusted_v in adjusted_verts:
            out_file.write(f"\t\t\t\t( {format_clean_vec3(adjusted_v)} 255 255 255 255 )\n")
        out_file.write("\t\t\t)\n\t\t\t( polylist (\n")
        out_file.write("\t\t\t\t( editpoly \n")
        out_file.write(f"\t\t\t\t\t( f {' '.join(str(j) for j in range(vcount))} )\n")
        out_file.write(f"\t\t\t\t\t( n {format_clean_vec3(normal)} )\n")
        out_file.write(f"\t\t\t\t\t( dist {dist:.6f} )\n")
        out_file.write("\t\t\t\t\t( textureinfo \n")
        out_file.write(f"\t\t\t\t\t\t( {format_clean_vec3(O0)} )\n")
        out_file.write(f"\t\t\t\t\t\t( {format_clean_vec3(P0)} )\n")
        out_file.write(f"\t\t\t\t\t\t( {format_clean_vec3(Q0)} )\n")
        out_file.write("\t\t\t\t\t\t( sticktopoly 1 )\n")
        out_file.write(f'\t\t\t\t\t\t( name "{tex_name}" )\n')
        out_file.write("\t\t\t\t\t)\n")
        out_file.write("\t\t\t\t\t( flags )\n")
        out_file.write("\t\t\t\t\t( shade 0 0 0 )\n")
        out_file.write("\t\t\t\t\t( physicsmaterial \"Default\" )\n")
        out_file.write("\t\t\t\t\t( surfacekey \"\" )\n")
        out_file.write("\t\t\t\t\t( textures ( \n")
        out_file.write("\t\t\t\t\t\t( 1 ( textureinfo \n")
        out_file.write("\t\t\t\t\t\t\t( 0.000000 0.000000 0.000000 )\n")
        out_file.write("\t\t\t\t\t\t\t( 1.000000 0.000000 0.000000 )\n")
        out_file.write("\t\t\t\t\t\t\t( 0.000000 0.000000 1.000000 )\n")
        out_file.write("\t\t\t\t\t\t\t( sticktopoly 1 )\n")
        out_file.write('\t\t\t\t\t\t\t( name "Default" )\n')
        out_file.write("\t\t\t\t\t\t) )\n")
        out_file.write("\t\t\t\t\t) )\n")
        out_file.write("\t\t\t\t)\n")
        out_file.write("\t\t\t) )\n\t\t) )\n")
    return poly_count
    
def write_occ(out_file, skyportals_all):

    for verts, normal, dist, occ_hashcode in skyportals_all:
        out_file.write("\t\t( polyhedron (\n")
        out_file.write("\t\t\t( color 255 255 255 )\n")
        out_file.write("\t\t\t( pointlist \n")

        for v in verts:
            adjusted_v = apply_world_offset(v)
            out_file.write(f"\t\t\t\t( {write_vec3(adjusted_v)} 255 255 255 255 )\n")
        out_file.write("\t\t\t)\n")
        out_file.write("\t\t\t( polylist (\n")
        out_file.write("\t\t\t\t( editpoly \n")
        vcount = len(verts)
        out_file.write(f"\t\t\t\t\t( f {' '.join(str(j) for j in range(vcount))} )\n")
        out_file.write(f"\t\t\t\t\t( n {write_vec3(normal)} )\n")
        out_file.write(f"\t\t\t\t\t( dist {dist:.6f} )\n")
        out_file.write("\t\t\t\t\t( textureinfo \n")
        out_file.write(f"\t\t\t\t\t\t( 0.000000 0.000000 0.000000 )\n")
        out_file.write(f"\t\t\t\t\t\t( 1.000000 0.000000 0.000000 )\n")
        out_file.write(f"\t\t\t\t\t\t( 0.000000 0.000000 1.000000 )\n")
        out_file.write("\t\t\t\t\t\t( sticktopoly 1 )\n")
        out_file.write('\t\t\t\t\t\t( name "Default" )\n')
        out_file.write("\t\t\t\t\t) )\n")
        out_file.write("\t\t\t\t) )\n")
        out_file.write("\t\t\t\t\t( flags )\n")
        out_file.write("\t\t\t\t\t( shade 0 0 0 )\n")
        out_file.write("\t\t\t\t\t( physicsmaterial \"Default\" )\n")
        out_file.write("\t\t\t\t\t( surfacekey \"\" )\n")
        out_file.write("\t\t\t\t\t( textures ( \n")
        out_file.write("\t\t\t\t\t\t( 1 ( textureinfo \n")
        out_file.write("\t\t\t\t\t\t\t( 0.000000 0.000000 0.000000 )\n")
        out_file.write("\t\t\t\t\t\t\t( 1.000000 0.000000 0.000000 )\n")
        out_file.write("\t\t\t\t\t\t\t( 0.000000 0.000000 1.000000 )\n")
        out_file.write("\t\t\t\t\t\t\t( sticktopoly 1 )\n")
        out_file.write("\t\t\t\t\t\t\t( name \"Default\" )\n")
        out_file.write("\t\t\t\t\t\t) )\n")
        out_file.write("\t\t\t\t\t) )\n")
        out_file.write("\t\t) )\n")
        
def write_skypo(out_file, skyportals_all):

    for verts, normal, dist in skyportals_all:
        out_file.write("\t\t( polyhedron (\n")
        out_file.write("\t\t\t( color 255 255 255 )\n")
        out_file.write("\t\t\t( pointlist \n")

        for v in verts:
            adjusted_v = apply_world_offset(v)
            out_file.write(f"\t\t\t\t( {write_vec3(adjusted_v)} 255 255 255 255 )\n")
        out_file.write("\t\t\t)\n")
        out_file.write("\t\t\t( polylist (\n")
        out_file.write("\t\t\t\t( editpoly \n")
        vcount = len(verts)
        out_file.write(f"\t\t\t\t\t( f {' '.join(str(j) for j in range(vcount))} )\n")
        out_file.write(f"\t\t\t\t\t( n {write_vec3(normal)} )\n")
        out_file.write(f"\t\t\t\t\t( dist {dist:.6f} )\n")
        out_file.write("\t\t\t\t\t( textureinfo \n")
        out_file.write(f"\t\t\t\t\t\t( 0.000000 0.000000 0.000000 )\n")
        out_file.write(f"\t\t\t\t\t\t( 1.000000 0.000000 0.000000 )\n")
        out_file.write(f"\t\t\t\t\t\t( 0.000000 0.000000 1.000000 )\n")
        out_file.write("\t\t\t\t\t\t( sticktopoly 1 )\n")
        out_file.write('\t\t\t\t\t\t( name "Default" )\n')
        out_file.write("\t\t\t\t\t) )\n")
        out_file.write("\t\t\t\t) )\n")
        out_file.write("\t\t\t\t\t( flags )\n")
        out_file.write("\t\t\t\t\t( shade 0 0 0 )\n")
        out_file.write("\t\t\t\t\t( physicsmaterial \"Default\" )\n")
        out_file.write("\t\t\t\t\t( surfacekey \"\" )\n")
        out_file.write("\t\t\t\t\t( textures ( \n")
        out_file.write("\t\t\t\t\t\t( 1 ( textureinfo \n")
        out_file.write("\t\t\t\t\t\t\t( 0.000000 0.000000 0.000000 )\n")
        out_file.write("\t\t\t\t\t\t\t( 1.000000 0.000000 0.000000 )\n")
        out_file.write("\t\t\t\t\t\t\t( 0.000000 0.000000 1.000000 )\n")
        out_file.write("\t\t\t\t\t\t\t( sticktopoly 1 )\n")
        out_file.write("\t\t\t\t\t\t\t( name \"Default\" )\n")
        out_file.write("\t\t\t\t\t\t) )\n")
        out_file.write("\t\t\t\t\t) )\n")
        out_file.write("\t\t) )\n")

skyportals_all = []
occluders_all = []

def export_rendernodes_to_lta(f, header, out_file, map_name, worldtree, keyframer_keys_map, occluder_names, version_flag):

    current_pos = f.tell()
    f.seek(header.object_data_pos)
    world_objects, world_objects_meta = read_world_objects_lta(f, header)

    f.seek(header.render_data_pos)
    (render_node_count,) = struct.unpack("<I", f.read(4))

    section_brushes = []
    section_brushes_by_node = defaultdict(list)

    out_file.write(f"( world\n\t( header (\n\t\t( versioncode 2 )\n\t\t( infostring \"{lth_header.info_string}\" )\n\t) )\n\t( polyhedronlist (\n")

    for node_index in range(render_node_count):
        center = Vec3.read(f)
        half_dims = Vec3.read(f)
        (section_count,) = struct.unpack("<I", f.read(4))

        sections = []
        for _ in range(section_count):
            tex0 = read_lithtech_string(f)
            tex1 = read_lithtech_string(f)

            shader_type_byte = f.read(1)[0]
            tri_count = struct.unpack("<I", f.read(4))[0]
            texture_effect = read_lithtech_string(f)
            w, h, size = struct.unpack("<3I", f.read(12))
            f.read(size)

            t0 = tex0.strip()
            t1 = tex1.strip()
            if t0 == "":
                tex0 = "Default"
            if t1 == "":
                tex1 = "Default"

            is_lightanim = (t0.lower() == "lightanim_base")
            shader_str = EPCShaderType.get(shader_type_byte, "Unknown")
            tex_w,  tex_h  = get_dtx_texture_size(tex0, search_dirs=[current_dir])
            tex_w1, tex_h1 = get_dtx_texture_size(tex1, search_dirs=[current_dir])

            sections.append({
                "is_lightanim": is_lightanim,
                "tex0": tex0, "tex1": tex1,
                "tex_w": tex_w, "tex_h": tex_h,
                "tex_w1": tex_w1, "tex_h1": tex_h1,
                "tri_count": tri_count,
                "shader_str": shader_str,
                "texture_effect": texture_effect,
            })

        (vertex_count,) = struct.unpack("<I", f.read(4))
        vertices = read_vertex_data(f, vertex_count, version_flag)

        (tri_count_total,) = struct.unpack("<I", f.read(4))
        all_tris = []
        for _ in range(tri_count_total):
            i0, i1, i2 = struct.unpack("<3I", f.read(12))
            f.read(4)
            all_tris.append((i0, i1, i2))

        if sum(s["tri_count"] for s in sections) != len(all_tris):
            print(f"[WARN] RenderNode {node_index}: sum(section.tri_count) != total tris")

        cursor = 0
        kept_sections = [] 
        for s in sections:
            tris = all_tris[cursor:cursor + s["tri_count"]]
            cursor += s["tri_count"]
            if s["is_lightanim"]:
                continue

            kept_sections.append((
                s["tex0"], s["tex_w"], s["tex_h"],
                s["tex1"], s["tex_w1"], s["tex_h1"],
                tris, s["shader_str"], s["texture_effect"]
            ))

        for section_idx, (tex0, tex_w, tex_h, tex1, tex_w1, tex_h1, tris, shader_str, texture_effect) in enumerate(kept_sections):
            local_verts = []
            index_map = {}
            for i0, i1, i2 in tris:
                for i in (i0, i1, i2):
                    if i not in index_map:
                        index_map[i] = len(local_verts)
                        local_verts.append(vertices[i])

            local_tris = [(index_map[i0], index_map[i1], index_map[i2]) for (i0, i1, i2) in tris]

            write_polyhedron(
                out_file,
                local_verts,
                [(tex0, tex_w, tex_h, tex1, tex_w1, tex_h1, local_tris, shader_str, texture_effect)]
            )

            label = f"Brush_RN{node_index}_S{section_idx}"
            section_brushes_by_node[node_index].append((label, len(section_brushes), shader_str, texture_effect))
            section_brushes.append((label, len(section_brushes)))

        (skip_blocks,) = struct.unpack("<I", f.read(4))
        for _ in range(skip_blocks):
            vert_count = struct.unpack("<B", f.read(1))[0]
            verts = [Vec3.read(f) for _ in range(vert_count)]
            plane_normal = Vec3.read(f)
            plane_dist = struct.unpack("<f", f.read(4))[0]
            skyportals_all.append((verts, plane_normal, plane_dist))

        (skip_occluders,) = struct.unpack("<I", f.read(4))
        for _ in range(skip_occluders):
            vert_count = struct.unpack("<B", f.read(1))[0]
            verts = [Vec3.read(f) for _ in range(vert_count)]
            plane_normal = Vec3.read(f)
            plane_dist = struct.unpack("<f", f.read(4))[0]
            (occ_hashcode,) = struct.unpack("<I", f.read(4))
            occluders_all.append((verts, plane_normal, plane_dist, occ_hashcode))

        (lg_count,) = struct.unpack("<I", f.read(4))
        for lg_index in range(lg_count):
            name_len = struct.unpack("<H", f.read(2))[0]
            group_name = f.read(name_len).decode("utf-8", errors="ignore")
            center = Vec3.read(f)
            intensity_data_len = struct.unpack("<I", f.read(4))[0]
            zero_compressed_data = f.read(intensity_data_len)
            (section_lm_len,) = struct.unpack("<I", f.read(4))
            for s in range(section_lm_len):
                sub_lm_len = struct.unpack("<I", f.read(4))[0]
                for i in range(sub_lm_len):
                    left, top, width, height, datalen = struct.unpack("<5I", f.read(20))
                    compressed = f.read(datalen)

        f.read(1)
        f.read(8)

    # === WMRenderNodes ===
    (wm_node_count,) = struct.unpack("<I", f.read(4))
    
    all_wm_data = {}
    wm_section_props = defaultdict(list)

    for wm_index in range(wm_node_count):
        wm_name = read_lithtech_string(f)

        (subnode_count,) = struct.unpack("<I", f.read(4))
        wm_data = []

        for subnode_index in range(subnode_count):
            center = Vec3.read(f)
            half_dims = Vec3.read(f)
            (section_count,) = struct.unpack("<I", f.read(4))

            sections = []
            for _ in range(section_count):
                tex0 = read_lithtech_string(f)
                tex1 = read_lithtech_string(f)
                shader_type_byte = f.read(1)[0]
                tri_count = struct.unpack("<I", f.read(4))[0]
                texture_effect = read_lithtech_string(f)
                w, h, size = struct.unpack("<3I", f.read(12))
                f.read(size)

                t0 = tex0.strip()
                t1 = tex1.strip()
                if t0 == "":
                    tex0 = "Default"
                if t1 == "":
                    tex1 = "Default"

                is_lightanim = (t0.lower() == "lightanim_base")
                shader_str = EPCShaderType.get(shader_type_byte, "Unknown")
                tex_w,  tex_h  = get_dtx_texture_size(tex0, search_dirs=[current_dir])
                tex_w1, tex_h1 = get_dtx_texture_size(tex1, search_dirs=[current_dir])

                sections.append({
                    "is_lightanim": is_lightanim,
                    "tex0": tex0, "tex1": tex1,
                    "tex_w": tex_w, "tex_h": tex_h,
                    "tex_w1": tex_w1, "tex_h1": tex_h1,
                    "tri_count": tri_count,
                    "shader_str": shader_str,
                    "texture_effect": texture_effect,
                })

            (vertex_count,) = struct.unpack("<I", f.read(4))
            vertices = read_vertex_data(f, vertex_count, version_flag)

            (tri_count_total,) = struct.unpack("<I", f.read(4))
            all_tris = []
            for _ in range(tri_count_total):
                i0, i1, i2 = struct.unpack("<3I", f.read(12))
                f.read(4) 
                all_tris.append((i0, i1, i2))

            expected = sum(s["tri_count"] for s in sections)
            if expected != len(all_tris):
                print(f"[WARN] WMRenderNode {wm_index}/{subnode_index}: "
                      f"sum(section.tri_count)={expected} != total_tris={len(all_tris)}")
                tri_count_total = min(expected, len(all_tris))
            else:
                tri_count_total = len(all_tris)

            cursor = 0
            kept_sections = []
            for s in sections:
                count = s["tri_count"]
                tris = all_tris[cursor:cursor + count]
                cursor += count
                if s["is_lightanim"]:
                    continue
                kept_sections.append((
                    s["tex0"], s["tex_w"], s["tex_h"],
                    s["tex1"], s["tex_w1"], s["tex_h1"],
                    tris, s["shader_str"], s["texture_effect"]
                ))

            section_triangles = []
            for section_idx, (tex0, tex0_w, tex0_h,
                              tex1, tex_w1, tex_h1,
                              tris, shader_str, texture_effect) in enumerate(kept_sections):

                local_verts = []
                index_map = {}
                for i0, i1, i2 in tris:
                    for i in (i0, i1, i2):
                        if i not in index_map:
                            index_map[i] = len(local_verts)
                            local_verts.append(vertices[i])

                local_tris = [(index_map[i0], index_map[i1], index_map[i2]) for (i0, i1, i2) in tris]

                section_triangles.append((tex0, tex0_w, tex0_h,
                                          tex1, tex_w1, tex_h1,
                                          local_tris, shader_str, texture_effect))

                label = f"{wm_name}_{subnode_index}_S{section_idx}"
                wm_section_props[wm_name].append({
                    "label": label,
                    "shader_str": shader_str,
                    "texture_effect": texture_effect,
                    "is_bsp": False
                })

                wm_data.append((local_verts, [(tex0, tex0_w, tex0_h,
                                               tex1, tex_w1, tex_h1,
                                               local_tris, shader_str, texture_effect)]))

            (skip_blocks,) = struct.unpack("<I", f.read(4))
            for _ in range(skip_blocks):
                vert_count = struct.unpack("<B", f.read(1))[0]
                verts = [Vec3.read(f) for _ in range(vert_count)]
                plane_normal = Vec3.read(f)
                plane_dist = struct.unpack("<f", f.read(4))[0]
                skyportals_all.append((verts, plane_normal, plane_dist))

            (skip_occluders,) = struct.unpack("<I", f.read(4))
            for _ in range(skip_occluders):
                vert_count = struct.unpack("<B", f.read(1))[0]
                verts = [Vec3.read(f) for _ in range(vert_count)]
                plane_normal = Vec3.read(f)
                plane_dist = struct.unpack("<f", f.read(4))[0]
                (occ_hashcode,) = struct.unpack("<I", f.read(4))
                occluders_all.append((verts, plane_normal, plane_dist, occ_hashcode))

            (lg_count,) = struct.unpack("<I", f.read(4))
            for lg_index in range(lg_count):
                name_len = struct.unpack("<H", f.read(2))[0]
                group_name = f.read(name_len).decode("utf-8", errors="ignore")
                center = Vec3.read(f)
                (intensity_data_len,) = struct.unpack("<I", f.read(4))
                f.read(intensity_data_len)
                (section_lm_len,) = struct.unpack("<I", f.read(4))
                for s in range(section_lm_len):
                    sub_lm_len = struct.unpack("<I", f.read(4))[0]
                    for i in range(sub_lm_len):
                        left, top, width, height, datalen = struct.unpack("<5I", f.read(20))
                        compressed = f.read(datalen)

            f.read(1)
            f.read(8)

        if wm_data:
            all_wm_data[wm_name] = wm_data
            
        (no_child_flag,) = struct.unpack("<I", f.read(4))

    # --- Strict Fallback: Process WorldModels strictly from PhysicsBSP if absent in RenderNodes ---
    for wm in worldtree.world_models:
        wm_name = wm.world_name
        if wm_name not in all_wm_data:
            wm_data = []
            surface_to_polies = defaultdict(list)
            for poly_idx, (surface_idx, plane_idx, indices) in enumerate(wm.polies):
                surface_to_polies[surface_idx].append((poly_idx, plane_idx, indices))

            for surface_idx in sorted(surface_to_polies.keys()):
                point_map   = {}
                local_points = []
                for _, _, indices in surface_to_polies[surface_idx]:
                    for idx in indices:
                        if idx not in point_map:
                            point_map[idx] = len(local_points)
                            local_points.append(wm.points[idx])

                tex_idx  = wm.surfaces[surface_idx]['TextureIndex']
                tex_name = wm.texture_names[tex_idx] if 0 <= tex_idx < len(wm.texture_names) else "Default"
                tex_w, tex_h = get_dtx_texture_size(tex_name)

                from lth_types import Vec2
                verts = [Vertex(
                    pos    = apply_world_offset(p),
                    normal = Vec3(0.0, 1.0, 0.0),
                    uv     = Vec2(0.0, 0.0),
                    uv1    = Vec2(0.0, 0.0),
                    color  = b'\xff\xff\xff\xff'
                ) for p in local_points]

                tris = []
                for _, plane_idx, indices in surface_to_polies[surface_idx]:
                    local_idx = [point_map[i] for i in indices]
                    for k in range(1, len(local_idx) - 1):
                        tris.append((local_idx[0], local_idx[k], local_idx[k + 1]))

                if tris:
                    flags_raw  = wm.surfaces[surface_idx].get("Flags", 0)
                    flags_set  = set(decode_surface_flags(flags_raw))
                    shader_str = "Lightmap" if "LIGHTMAP" in flags_set else "Gouraud"

                    wm_data.append((
                        verts,
                        [(tex_name, tex_w, tex_h, "Default", 0, 0, tris, shader_str, "")]
                    ))

                    section_idx = len(wm_section_props[wm_name])
                    label = f"{wm_name}_0_S{section_idx}"
                    wm_section_props[wm_name].append({
                        "label":         label,
                        "shader_str":    shader_str,
                        "texture_effect": "",
                        "is_bsp":        True,
                        "flags_raw":     flags_raw,
                    })
                    
            if wm_data:
                all_wm_data[wm_name] = wm_data

    # --- 1. СНАЧАЛА СОЗДАЕМ СЛОВАРЬ СОРТИРОВКИ ---
    objects_by_type = defaultdict(list)
    for i, obj in enumerate(world_objects):
        obj_type = obj.get("__Type", "Unknown")
        objects_by_type[obj_type].append((i, obj))

    # --- 2. Write order ---
    wm_ordered = []
    for obj_type in sorted(objects_by_type.keys()):
        for i, obj in objects_by_type[obj_type]:
            obj_name = obj.get("Name")
            if obj_name and obj_name in all_wm_data:
                wm_ordered.append(obj_name)

    for wm_name in wm_ordered:
        for verts, section_tris in all_wm_data[wm_name]:
            write_polyhedron(out_file, verts, section_tris)

    write_skypo(out_file, skyportals_all)
    write_occ(out_file, occluders_all)

    polyBlocks_count = write_physics_polygons(f, header, out_file)
    polypParticleblock_count = write_particleblockers_polygons(f, header, out_file)


    # === NODE HIERARCHY ===
    brushindex = 0
    base_nodeid = 1
    base_propid = 1
    
    out_file.write(") )\n")
    out_file.write("( nodehierarchy\n")
    out_file.write("\t( worldnode\n")
    out_file.write("\t\t( type null )\n")
    out_file.write(f'\t\t( label \"{map_name}\" )\n')
    out_file.write(f'\t\t( nodeid {base_nodeid} )\n')
    out_file.write("\t\t( flags ( worldroot expanded ) )\n")
    out_file.write("\t\t( properties\n")
    out_file.write(f'\t\t\t( propid 0 )\n')
    out_file.write("\t\t)\n")
    out_file.write("\t\t( childlist (\n")
    base_nodeid += 1
    
    out_file.write("\t\t\t( worldnode\n")
    out_file.write("\t\t\t\t( type null )\n")
    out_file.write(f'\t\t\t\t( label "RenderNodes" )\n')
    out_file.write(f'\t\t\t\t( nodeid {base_nodeid} )\n')
    out_file.write("\t\t\t\t( flags ( ) )\n")
    out_file.write("\t\t\t\t( properties\n")
    out_file.write(f'\t\t\t\t\t( propid 0 )\n')
    out_file.write("\t\t\t\t)\n")
    out_file.write("\t\t\t\t( childlist (\n")
    base_nodeid += 1
    for node_index, section_brushes in section_brushes_by_node.items():
        out_file.write("\t\t\t\t\t( worldnode\n")
        out_file.write("\t\t\t\t\t\t( type null )\n")
        out_file.write(f'\t\t\t\t\t\t( label \"RenderNode{node_index}\" )\n')
        out_file.write(f'\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
        out_file.write("\t\t\t\t\t\t( flags ( ) )\n")
        out_file.write("\t\t\t\t\t\t( properties\n")
        out_file.write(f'\t\t\t\t\t\t\t( propid 0 )\n')
        out_file.write("\t\t\t\t\t\t)\n")
        out_file.write("\t\t\t\t\t\t( childlist (\n")
        base_nodeid += 1

        for label, _, _, _ in section_brushes:
            out_file.write("\t\t\t\t\t\t\t( worldnode\n")
            out_file.write("\t\t\t\t\t\t\t\t( type brush )\n")
            out_file.write(f'\t\t\t\t\t\t\t\t( brushindex {brushindex} )\n')
            out_file.write(f'\t\t\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
            out_file.write("\t\t\t\t\t\t\t\t( flags ( ) )\n")
            out_file.write("\t\t\t\t\t\t\t\t( properties\n")
            out_file.write(f'\t\t\t\t\t\t\t\t\t( name "Brush" )\n')
            out_file.write(f'\t\t\t\t\t\t\t\t\t( propid {base_propid} )\n')
            out_file.write("\t\t\t\t\t\t\t\t)\n")
            out_file.write("\t\t\t\t\t\t\t)\n")
            base_propid += 1
            base_nodeid += 1
            brushindex += 1

        out_file.write("\t\t\t\t\t\t) )\n") 
        out_file.write("\t\t\t\t\t)\n") 
    out_file.write("\t\t\t\t) )\n") 
    out_file.write("\t\t\t)\n") 
    
    # === OBJECT NODES ===

    if len(world_objects) > 0:
        out_file.write("\t\t\t( worldnode\n")
        out_file.write("\t\t\t\t( type null )\n")
        out_file.write(f'\t\t\t\t( label "Objects" )\n')
        out_file.write(f'\t\t\t\t( nodeid {base_nodeid} )\n')
        out_file.write('\t\t\t\t( flags ( ) )\n')
        out_file.write('\t\t\t\t( properties\n')
        out_file.write(f'\t\t\t\t\t( propid 0 )\n')
        out_file.write('\t\t\t\t)\n')
        out_file.write('\t\t\t\t( childlist (\n')
        base_nodeid += 1

        for obj_type in sorted(objects_by_type.keys()):
            out_file.write('\t\t\t\t\t( worldnode\n')
            out_file.write('\t\t\t\t\t\t( type null )\n')
            out_file.write(f'\t\t\t\t\t\t( label "{obj_type}" )\n') 
            out_file.write(f'\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
            out_file.write('\t\t\t\t\t\t( flags ( ) )\n')
            out_file.write('\t\t\t\t\t\t( properties\n')
            out_file.write(f'\t\t\t\t\t\t\t( propid 0 )\n')
            out_file.write('\t\t\t\t\t\t)\n')
            out_file.write('\t\t\t\t\t\t( childlist (\n')
            base_nodeid += 1

            for i, obj in objects_by_type[obj_type]:
                name = obj.get("Name", f"{obj_type}_{i}")
                label = obj.get("__Type", "Object")

                out_file.write('\t\t\t\t\t\t\t( worldnode\n')
                out_file.write('\t\t\t\t\t\t\t\t( type object )\n')
                out_file.write(f'\t\t\t\t\t\t\t\t( label "{name}" )\n') 
                out_file.write(f'\t\t\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
                out_file.write('\t\t\t\t\t\t\t\t( flags ( ) )\n')
                out_file.write('\t\t\t\t\t\t\t\t( properties\n')
                out_file.write(f'\t\t\t\t\t\t\t\t\t( name "{label}" )\n') 
                out_file.write(f'\t\t\t\t\t\t\t\t\t( propid {base_propid} )\n')
                out_file.write('\t\t\t\t\t\t\t\t)\n')
                base_propid += 1
                object_nodeid = base_nodeid
                base_nodeid += 1

                obj_name = obj.get("Name")
                if obj_name in all_wm_data:
                    out_file.write('\t\t\t\t\t\t\t\t( childlist (\n')

                    for section in wm_section_props[obj_name]:
                        out_file.write('\t\t\t\t\t\t\t\t\t( worldnode\n')
                        out_file.write('\t\t\t\t\t\t\t\t\t\t( type brush )\n')
                        out_file.write(f'\t\t\t\t\t\t\t\t\t\t( brushindex {brushindex} )\n')
                        out_file.write(f'\t\t\t\t\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
                        out_file.write('\t\t\t\t\t\t\t\t\t\t( flags ( ) )\n')
                        out_file.write('\t\t\t\t\t\t\t\t\t\t( properties\n')
                        out_file.write(f'\t\t\t\t\t\t\t\t\t\t\t( name "Brush" )\n')
                        out_file.write(f'\t\t\t\t\t\t\t\t\t\t\t( propid {base_propid} )\n')
                        out_file.write('\t\t\t\t\t\t\t\t\t\t)\n')
                        out_file.write('\t\t\t\t\t\t\t\t\t)\n')
                        base_nodeid += 1
                        base_propid += 1
                        brushindex += 1

                    out_file.write('\t\t\t\t\t\t\t\t) )\n') 

                out_file.write('\t\t\t\t\t\t\t)\n') 

            out_file.write('\t\t\t\t\t\t) )\n') 
            out_file.write('\t\t\t\t\t)\n') 

        out_file.write('\t\t\t\t) )\n') 
        out_file.write('\t\t\t)\n') 

        # 4. KeyFramers
        if len(keyframer_keys_map) > 0:
            out_file.write("\t\t\t( worldnode\n")
            out_file.write("\t\t\t\t( type null )\n")
            out_file.write(f'\t\t\t\t( label "KeyFramers" )\n')
            out_file.write(f'\t\t\t\t( nodeid {base_nodeid} )\n')
            out_file.write('\t\t\t\t( flags ( ) ) \n')
            out_file.write('\t\t\t\t( properties\n')
            out_file.write(f'\t\t\t\t\t( propid 0 )\n')
            out_file.write('\t\t\t\t)\n')
            out_file.write('\t\t\t\t( childlist (\n')
            base_nodeid += 1
            
            for key_name in keyframer_keys_map:
                out_file.write("\t\t\t\t\t( worldnode\n")
                out_file.write("\t\t\t\t\t\t( type null )\n")
                out_file.write(f'\t\t\t\t\t\t( label "{key_name}" )\n')
                out_file.write(f'\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
                out_file.write('\t\t\t\t\t\t( flags ( path ) ) \n')
                out_file.write('\t\t\t\t\t\t( properties\n')
                out_file.write(f'\t\t\t\t\t\t\t( propid 0 )\n')
                out_file.write('\t\t\t\t\t\t)\n')
                out_file.write('\t\t\t\t\t\t( childlist (\n')
                base_nodeid += 1
                for entry in keyframer_keys_map[key_name]:
                    out_file.write('\t\t\t\t\t\t\t( worldnode\n')
                    out_file.write('\t\t\t\t\t\t\t\t( type object )\n')
                    out_file.write(f'\t\t\t\t\t\t\t\t( label "Key" )\n')
                    out_file.write(f'\t\t\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
                    out_file.write('\t\t\t\t\t\t\t\t( flags ( ) )\n')
                    out_file.write('\t\t\t\t\t\t\t\t( properties\n')
                    out_file.write(f'\t\t\t\t\t\t\t\t\t( name "Key" )\n')
                    out_file.write(f'\t\t\t\t\t\t\t\t\t( propid {base_propid} )\n')
                    out_file.write('\t\t\t\t\t\t\t\t)\n')
                    out_file.write('\t\t\t\t\t\t\t)\n')
                    base_nodeid += 1
                    base_propid += 1

                out_file.write('\t\t\t\t\t\t) )\n') 
                out_file.write('\t\t\t\t\t)\n') 
                
            out_file.write('\t\t\t\t) )\n') 
            out_file.write('\t\t\t)\n') 

    if len(skyportals_all) > 0:
        out_file.write("\t\t\t( worldnode\n")
        out_file.write("\t\t\t\t( type null )\n")
        out_file.write(f'\t\t\t\t( label "SkyPortals" )\n')
        out_file.write(f'\t\t\t\t( nodeid {base_nodeid} )\n')
        out_file.write('\t\t\t\t( flags ( ) )\n')
        out_file.write('\t\t\t\t( properties\n')
        out_file.write(f'\t\t\t\t\t( propid 0 )\n')
        out_file.write('\t\t\t\t)\n')
        out_file.write('\t\t\t\t( childlist (\n')
        base_nodeid += 1
        for _ in range(len(skyportals_all)):
            out_file.write('\t\t\t\t\t( worldnode\n')
            out_file.write('\t\t\t\t\t\t( type brush )\n')
            out_file.write(f'\t\t\t\t\t\t( brushindex {brushindex} )\n')
            out_file.write(f'\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
            out_file.write('\t\t\t\t\t\t( flags ( ) )\n')
            out_file.write('\t\t\t\t\t\t( properties\n')
            out_file.write('\t\t\t\t\t\t\t( name "Brush" )\n')
            out_file.write(f'\t\t\t\t\t\t\t( propid {base_propid} )\n')
            out_file.write('\t\t\t\t\t\t)\n')
            out_file.write('\t\t\t\t\t)\n')
            brushindex += 1
            base_nodeid += 1
        base_propid += 1
        out_file.write('\t\t\t\t) )\n') 
        out_file.write('\t\t\t)\n') 
        
    if len(occluders_all) > 0:
        out_file.write("\t\t\t( worldnode\n")
        out_file.write("\t\t\t\t( type null )\n")
        out_file.write(f'\t\t\t\t( label "Occluders" )\n')
        out_file.write(f'\t\t\t\t( nodeid {base_nodeid} )\n')
        out_file.write('\t\t\t\t( flags ( ) )\n')
        out_file.write('\t\t\t\t( properties\n')
        out_file.write(f'\t\t\t\t\t( propid 0 )\n')
        out_file.write('\t\t\t\t)\n')
        out_file.write('\t\t\t\t( childlist (\n')
        base_nodeid += 1
        for _ in range(len(occluders_all)):
            out_file.write('\t\t\t\t\t( worldnode\n')
            out_file.write('\t\t\t\t\t\t( type brush )\n')
            out_file.write(f'\t\t\t\t\t\t( brushindex {brushindex} )\n')
            out_file.write(f'\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
            out_file.write('\t\t\t\t\t\t( flags ( ) )\n')
            out_file.write('\t\t\t\t\t\t( properties\n')
            out_file.write('\t\t\t\t\t\t\t( name "Brush" )\n')
            out_file.write(f'\t\t\t\t\t\t\t( propid {base_propid} )\n')
            out_file.write('\t\t\t\t\t\t)\n')
            out_file.write('\t\t\t\t\t)\n')
            brushindex += 1
            base_nodeid += 1
            base_propid += 1
        out_file.write('\t\t\t\t) )\n') 
        out_file.write('\t\t\t)\n') 
        
    if polyBlocks_count > 0:
        out_file.write("\t\t\t( worldnode\n")
        out_file.write("\t\t\t\t( type null )\n")
        out_file.write(f'\t\t\t\t( label "Blockers" )\n')
        out_file.write(f'\t\t\t\t( nodeid {base_nodeid} )\n')
        out_file.write('\t\t\t\t( flags ( ) )\n')
        out_file.write('\t\t\t\t( properties\n')
        out_file.write(f'\t\t\t\t\t( propid 0 )\n')
        out_file.write('\t\t\t\t)\n')
        out_file.write('\t\t\t\t( childlist (\n')
        base_nodeid += 1
        for _ in range(polyBlocks_count):
            out_file.write('\t\t\t\t\t( worldnode\n')
            out_file.write('\t\t\t\t\t\t( type brush )\n')
            out_file.write(f'\t\t\t\t\t\t( brushindex {brushindex} )\n')
            out_file.write(f'\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
            out_file.write('\t\t\t\t\t\t( flags ( ) )\n')
            out_file.write('\t\t\t\t\t\t( properties\n')
            out_file.write('\t\t\t\t\t\t\t( name "Brush" )\n')
            out_file.write(f'\t\t\t\t\t\t\t( propid {base_propid} )\n')
            out_file.write('\t\t\t\t\t\t)\n')
            out_file.write('\t\t\t\t\t)\n')
            brushindex += 1
            base_nodeid += 1
        base_propid += 1
        out_file.write('\t\t\t\t) )\n') 
        out_file.write('\t\t\t)\n') 
        
    if polypParticleblock_count > 0:
        out_file.write("\t\t\t( worldnode\n")
        out_file.write("\t\t\t\t( type null )\n")
        out_file.write(f'\t\t\t\t( label "ParticleBlockers" )\n')
        out_file.write(f'\t\t\t\t( nodeid {base_nodeid} )\n')
        out_file.write('\t\t\t\t( flags ( ) )\n')
        out_file.write('\t\t\t\t( properties\n')
        out_file.write(f'\t\t\t\t\t( propid 0 )\n')
        out_file.write('\t\t\t\t)\n')
        out_file.write('\t\t\t\t( childlist (\n')
        base_nodeid += 1
        for _ in range(polypParticleblock_count):
            out_file.write('\t\t\t\t\t( worldnode\n')
            out_file.write('\t\t\t\t\t\t( type brush )\n')
            out_file.write(f'\t\t\t\t\t\t( brushindex {brushindex} )\n')
            out_file.write(f'\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
            out_file.write('\t\t\t\t\t\t( flags ( ) )\n')
            out_file.write('\t\t\t\t\t\t( properties\n')
            out_file.write('\t\t\t\t\t\t\t( name "Brush" )\n')
            out_file.write(f'\t\t\t\t\t\t\t( propid {base_propid} )\n')
            out_file.write('\t\t\t\t\t\t)\n')
            out_file.write('\t\t\t\t\t)\n')
            brushindex += 1
            base_nodeid += 1
        base_propid += 1
        out_file.write('\t\t\t\t) )\n') 
        out_file.write('\t\t\t)\n') 

    # ЗАКРЫВАЕМ КОРЕНЬ КАРТЫ (Root Map Node)
    out_file.write('\t\t) )\n') # Закрывает childlist корня
    out_file.write('\t)\n')     # Закрывает worldnode корня
        
    out_file.write("( globalproplist (\n")
    out_file.write("    ( proplist ( \n")
    out_file.write("    ) )\n")
    for node_index in sorted(section_brushes_by_node.keys()):
        section_list = section_brushes_by_node[node_index]
        for section_index, (label, _, shader_str, texture_effect) in enumerate(section_list):
            out_file.write("    ( proplist ( \n")
            out_file.write(f'        ( string "Name" (  ) ( data "RN{node_index}_S{section_index}" ) )\n')
            out_file.write("        ( vector \"Pos\" ( distance ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )\n")
            out_file.write("        ( rotation \"Rotation\" (  ) ( data ( eulerangles (0.000000 0.000000 0.000000) ) ) )\n")
            out_file.write("        ( longint \"RenderGroup\" (  ) ( data 0.000000 ) )\n")
            out_file.write("        ( string \"Type\" ( staticlist ) ( data \"Normal\" ) )\n")
            out_file.write(f'        ( string "Lighting" ( staticlist ) ( data "{shader_str}" ) )\n')
            out_file.write("        ( bool \"NotAStep\" (  ) ( data 0 ) )\n")
            out_file.write("        ( bool \"Detail\" (  ) ( data 0 ) )\n")
            out_file.write("        ( longint \"LightControl\" ( groupowner group1 ) ( data 0.000000 ) )\n")
            out_file.write(f'        ( string "TextureEffect" ( textureeffect ) ( data "{texture_effect}" ) )\n' if texture_effect else '        ( string "TextureEffect" ( textureeffect ) )\n')
            out_file.write("        ( color \"AmbientLight\" ( group1 ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )\n")
            out_file.write("        ( longint \"LMGridSize\" ( group1 ) ( data 0.000000 ) )\n")
            out_file.write("        ( bool \"ClipLight\" ( group1 ) ( data 1 ) )\n")
            out_file.write("        ( bool \"CastShadowMesh\" ( group1 ) ( data 1 ) )\n")
            out_file.write("        ( bool \"ReceiveLight\" ( group1 ) ( data 1 ) )\n")
            out_file.write("        ( bool \"ReceiveShadows\" ( group1 ) ( data 1 ) )\n")
            out_file.write("        ( bool \"ReceiveSunlight\" ( group1 ) ( data 1 ) )\n")
            out_file.write("        ( real \"LightPenScale\" ( group1 ) ( data 0.000000 ) )\n")
            out_file.write("        ( real \"CreaseAngle\" ( group1 ) ( data 45.000000 ) )\n")
            out_file.write("    ) )\n")

    for obj_type in sorted(objects_by_type.keys()):
        for i, props in objects_by_type[obj_type]:
            meta = world_objects_meta[i]

            def format_vec3(v):
                return f"( vector ({v.x:.6f} {v.y:.6f} {v.z:.6f}) )"

            def format_rotation(q):
                return f"(eulerangles ({q.x:.6f} {q.y:.6f} {q.z:.6f}))"

            out_file.write("\t\t\t\t( proplist (\n")
            for key, value in props.items():
                if key == "__Type":
                    continue

                field_type = meta.get(key, {}).get("type")

                if field_type == "string":
                    if value is not None:
                        out_file.write(f'\t\t\t\t\t(  string "{key}" (  ) ( data "{value}") )\n')
                    else:
                        out_file.write(f'\t\t\t\t\t(  string "{key}" ( hidden  ) )\n')

                elif field_type == "vector":
                    if key == "Pos":
                        adjusted_v = apply_world_offset(value)
                        out_file.write(f'\t\t\t\t\t(  vector "{key}" ( distance  ) ( data ( vector ( {write_vec3(adjusted_v)} ) ) ) )\n')
                    else:
                        out_file.write(f'\t\t\t\t\t(  vector "{key}" ( distance  ) ( data {format_vec3(value)} ) )\n')

                elif field_type == "color":
                    out_file.write(f'\t\t\t\t\t(  color "{key}" (  ) ( data {format_vec3(value)} ) )\n')

                elif field_type == "rotation":
                    out_file.write(f'\t\t\t\t\t(  rotation "{key}" (  ) ( data {format_rotation(value)} ) )\n')

                elif field_type == "bool":
                    out_file.write(f'\t\t\t\t\t(  bool "{key}" (  ) ( data {1 if value else 0} ))\n')

                elif field_type == "real":
                    out_file.write(f'\t\t\t\t\t(  real "{key}" (  ) ( data {value:.6f} ))\n')

                elif field_type == "longint":
                    out_file.write(f'\t\t\t\t\t(  longint "{key}" (  ) ( data {float(value):.6f} ))\n')

                else:
                    if isinstance(value, tuple) and len(value) == 3:
                        out_file.write(f'\t\t\t\t\t(  vector "{key}" (  ) ( data ( vector ({value[0]:.6f} {value[1]:.6f} {value[2]:.6f}) ) ) )\n')
                    elif value is None:
                        out_file.write(f'\t\t\t\t\t(  string "{key}" ( hidden  ) )\n')

            out_file.write("\t\t\t\t)\t)\n")
            
            obj_name = props.get("Name")
            if obj_name in all_wm_data:
                for section in wm_section_props[obj_name]:
                    label          = section["label"]
                    shader_str     = section["shader_str"]
                    texture_effect = section["texture_effect"]
                    is_bsp         = section.get("is_bsp", False)

                    out_file.write("\t\t\t\t( proplist (\n")
                    out_file.write(f'\t\t\t\t\t( string "Name" (  ) ( data "{label}" ) )\n')
                    out_file.write("\t\t\t\t\t( vector \"Pos\" ( distance ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )\n")
                    out_file.write("\t\t\t\t\t( rotation \"Rotation\" (  ) ( data ( eulerangles (0.000000 0.000000 0.000000) ) ) )\n")
                    out_file.write("\t\t\t\t\t( longint \"RenderGroup\" (  ) ( data 0.000000 ) )\n")

                    if is_bsp:
                        flags_raw = section.get("flags_raw", 0)
                        flags_set = set(decode_surface_flags(flags_raw))
                        lighting  = get_lighting_type(flags_set)
                        surf_type = get_surface_type(flags_set)
                        def bool_flag(n): return 1 if n in flags_set else 0

                        out_file.write(f'\t\t\t\t\t( string "Type" ( staticlist ) ( data "{surf_type}" ) )\n')
                        out_file.write(f'\t\t\t\t\t( string "Lighting" ( staticlist ) ( data "{lighting}" ) )\n')
                        out_file.write(f'\t\t\t\t\t( bool "NotAStep" (  ) ( data {bool_flag("NOTASTEP")} ) )\n')
                        out_file.write("\t\t\t\t\t( bool \"Detail\" (  ) ( data 0 ) )\n")
                        out_file.write("\t\t\t\t\t( longint \"LightControl\" ( groupowner group1 ) ( data 0.000000 ) )\n")
                        out_file.write("\t\t\t\t\t( string \"TextureEffect\" ( textureeffect ) )\n")
                        out_file.write("\t\t\t\t\t( color \"AmbientLight\" ( group1 ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )\n")
                        out_file.write("\t\t\t\t\t( longint \"LMGridSize\" ( group1 ) ( data 0.000000 ) )\n")
                        out_file.write(f'\t\t\t\t\t( bool "ClipLight" ( group1 ) ( data {bool_flag("CLIPLIGHT")} ) )\n')
                        out_file.write(f'\t\t\t\t\t( bool "CastShadowMesh" ( group1 ) ( data {bool_flag("CASTSHADOWMESH")} ) )\n')
                        out_file.write(f'\t\t\t\t\t( bool "ReceiveLight" ( group1 ) ( data {bool_flag("RECEIVELIGHT")} ) )\n')
                        out_file.write(f'\t\t\t\t\t( bool "ReceiveShadows" ( group1 ) ( data {bool_flag("RECEIVESHADOWS")} ) )\n')
                        out_file.write(f'\t\t\t\t\t( bool "ReceiveSunlight" ( group1 ) ( data {bool_flag("RECEIVESUNLIGHT")} ) )\n')
                    else:
                        out_file.write("\t\t\t\t\t( string \"Type\" ( staticlist ) ( data \"Normal\" ) )\n")
                        out_file.write(f'\t\t\t\t\t( string "Lighting" ( staticlist ) ( data "{shader_str}" ) )\n')
                        out_file.write("\t\t\t\t\t( bool \"NotAStep\" (  ) ( data 0 ) )\n")
                        out_file.write("\t\t\t\t\t( bool \"Detail\" (  ) ( data 0 ) )\n")
                        out_file.write("\t\t\t\t\t( longint \"LightControl\" ( groupowner group1 ) ( data 0.000000 ) )\n")
                        if texture_effect:
                            out_file.write(f'\t\t\t\t\t( string "TextureEffect" ( textureeffect ) ( data "{texture_effect}" ) )\n')
                        else:
                            out_file.write("\t\t\t\t\t( string \"TextureEffect\" ( textureeffect ) )\n")
                        out_file.write("\t\t\t\t\t( color \"AmbientLight\" ( group1 ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )\n")
                        out_file.write("\t\t\t\t\t( longint \"LMGridSize\" ( group1 ) ( data 0.000000 ) )\n")
                        out_file.write("\t\t\t\t\t( bool \"ClipLight\" ( group1 ) ( data 1 ) )\n")
                        out_file.write("\t\t\t\t\t( bool \"CastShadowMesh\" ( group1 ) ( data 1 ) )\n")
                        out_file.write("\t\t\t\t\t( bool \"ReceiveLight\" ( group1 ) ( data 1 ) )\n")
                        out_file.write("\t\t\t\t\t( bool \"ReceiveShadows\" ( group1 ) ( data 1 ) )\n")
                        out_file.write("\t\t\t\t\t( bool \"ReceiveSunlight\" ( group1 ) ( data 1 ) )\n")

                    out_file.write("\t\t\t\t\t( real \"LightPenScale\" ( group1 ) ( data 0.000000 ) )\n")
                    out_file.write("\t\t\t\t\t( real \"CreaseAngle\" ( group1 ) ( data 45.000000 ) )\n")
                    out_file.write("\t\t\t\t) )\n")
                    
    if keyframer_keys_map:
        for key_name, key_list in keyframer_keys_map.items():
            for key_data in key_list:
                pos = key_data["pos"]
                if not isinstance(pos, Vec3):
                    pos = Vec3(*pos)
                
                rot_deg = key_data["rot_deg"]
                if not isinstance(rot_deg, Vec3):
                    rot_deg = Vec3(*rot_deg)
                
                bez_prev = key_data.get("bez_prev") or Vec3(0, 0, 0)
                bez_next = key_data.get("bez_next") or Vec3(0, 0, 0)
                sound_name = key_data.get("sound_name", "")
                command = key_data.get("command", "")

                out_file.write("        ( proplist ( \n")
                out_file.write(f'            ( string "Name" (  ) ( data "{key_data["full_name"]}") ) \n')
                pos = apply_world_offset(pos)
                out_file.write(f'            ( vector "Pos" ( distance ) ( data ( vector ( {write_vec3(pos)} ) ) ) ) \n')
                out_file.write(f'            ( rotation "Rotation" (  ) ( data ( eulerangles ({rot_deg.x:.6f} {rot_deg.y:.6f} {rot_deg.z:.6f}) ) ) ) \n')
                out_file.write(f'            ( longint "RenderGroup" (  ) ( data 0.000000 ) ) \n')
                out_file.write(f'            ( real "TimeStamp" (  ) ( data {key_data["timestamp"]:.6f} ) ) \n')

                if sound_name:
                    out_file.write(f'            ( string "SoundName" (  ) ( data "{sound_name}" ) ) \n')
                else:
                    out_file.write(f'            ( string "SoundName" (  ) ) \n')

                out_file.write(f'            ( real "SoundRadius" ( radius ) ( data {key_data["sound_radius"]:.6f} ) ) \n')

                if command:
                    out_file.write(f'            ( string "Command" ( notifychange ) ( data "{command}" ) ) \n')
                else:
                    out_file.write(f'            ( string "Command" ( notifychange ) ) \n')

                out_file.write(f'            ( vector "BezierPrev" ( bezierprevtangent ) ( data ( vector ({bez_prev.x:.6f} {bez_prev.y:.6f} {bez_prev.z:.6f}) ) ) ) \n')
                out_file.write(f'            ( vector "BezierNext" ( beziernexttangent ) ( data ( vector ({bez_next.x:.6f} {bez_next.y:.6f} {bez_next.z:.6f}) ) ) ) \n')
                out_file.write("        ) )\n\n")

    if len(skyportals_all) > 0:
        out_file.write("    ( proplist ( \n")
        out_file.write(f'        ( string "Name" (  ) ( data "SkyPortal" ) )\n')
        out_file.write("        ( vector \"Pos\" ( distance ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )\n")
        out_file.write("        ( rotation \"Rotation\" (  ) ( data ( eulerangles (0.000000 0.000000 0.000000) ) ) )\n")
        out_file.write("        ( longint \"RenderGroup\" (  ) ( data 0.000000 ) )\n")
        out_file.write("        ( string \"Type\" ( staticlist ) ( data \"SkyPortal\" ) )\n")
        out_file.write("        ( string \"Lighting\" ( staticlist ) ( data \"Gouraud\" ) )\n")
        out_file.write("        ( bool \"NotAStep\" (  ) ( data 0 ) )\n")
        out_file.write("        ( bool \"Detail\" (  ) ( data 0 ) )\n")
        out_file.write("        ( longint \"LightControl\" ( groupowner group1 ) ( data 0.000000 ) )\n")
        out_file.write("        ( string \"TextureEffect\" ( textureeffect ) )\n")
        out_file.write("        ( color \"AmbientLight\" ( group1 ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )\n")
        out_file.write("        ( longint \"LMGridSize\" ( group1 ) ( data 0.000000 ) )\n")
        out_file.write("        ( bool \"ClipLight\" ( group1 ) ( data 1 ) )\n")
        out_file.write("        ( bool \"CastShadowMesh\" ( group1 ) ( data 1 ) )\n")
        out_file.write("        ( bool \"ReceiveLight\" ( group1 ) ( data 1 ) )\n")
        out_file.write("        ( bool \"ReceiveShadows\" ( group1 ) ( data 1 ) )\n")
        out_file.write("        ( bool \"ReceiveSunlight\" ( group1 ) ( data 1 ) )\n")
        out_file.write("        ( real \"LightPenScale\" ( group1 ) ( data 0.000000 ) )\n")
        out_file.write("        ( real \"CreaseAngle\" ( group1 ) ( data 45.000000 ) )\n")
        out_file.write("    ) )\n")
    if len(occluders_all) > 0:
        for idx, (verts, plane_normal, plane_dist, occ_hashcode) in enumerate(occluders_all):
            occ_name = "Occluder"
            for nm, hsh in occluder_names.items():
                if hsh == occ_hashcode:
                    occ_name = nm
                    break   

            out_file.write("    ( proplist ( \n")
            out_file.write(f'        ( string "Name" (  ) ( data "{occ_name}" ) )\n')
            out_file.write("        ( vector \"Pos\" ( distance ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )\n")
            out_file.write("        ( rotation \"Rotation\" (  ) ( data ( eulerangles (0.000000 0.000000 0.000000) ) ) )\n")
            out_file.write("        ( longint \"RenderGroup\" (  ) ( data 0.000000 ) )\n")
            out_file.write("        ( string \"Type\" ( staticlist ) ( data \"Occluder\" ) )\n")
            out_file.write("        ( string \"Lighting\" ( staticlist ) ( data \"Flat\" ) )\n")
            out_file.write("        ( bool \"NotAStep\" (  ) ( data 0 ) )\n")
            out_file.write("        ( bool \"Detail\" (  ) ( data 0 ) )\n")
            out_file.write("        ( longint \"LightControl\" ( groupowner group1 ) ( data 0.000000 ) )\n")
            out_file.write("        ( string \"TextureEffect\" ( textureeffect ) )\n")
            out_file.write("        ( color \"AmbientLight\" ( group1 ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )\n")
            out_file.write("        ( longint \"LMGridSize\" ( group1 ) ( data 0.000000 ) )\n")
            out_file.write("        ( bool \"ClipLight\" ( group1 ) ( data 0 ) )\n")
            out_file.write("        ( bool \"CastShadowMesh\" ( group1 ) ( data 0 ) )\n")
            out_file.write("        ( bool \"ReceiveLight\" ( group1 ) ( data 0 ) )\n")
            out_file.write("        ( bool \"ReceiveShadows\" ( group1 ) ( data 0 ) )\n")
            out_file.write("        ( bool \"ReceiveSunlight\" ( group1 ) ( data 0 ) )\n")
            out_file.write("        ( real \"LightPenScale\" ( group1 ) ( data 0.000000 ) )\n")
            out_file.write("        ( real \"CreaseAngle\" ( group1 ) ( data 45.000000 ) )\n")
            out_file.write("    ) )\n")

    if polyBlocks_count > 0:
        out_file.write("    ( proplist ( \n")
        out_file.write(f'        ( string "Name" (  ) ( data "Blocker" ) )\n')
        out_file.write("        ( vector \"Pos\" ( distance ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )\n")
        out_file.write("        ( rotation \"Rotation\" (  ) ( data ( eulerangles (0.000000 0.000000 0.000000) ) ) )\n")
        out_file.write("        ( longint \"RenderGroup\" (  ) ( data 0.000000 ) )\n")
        out_file.write("        ( string \"Type\" ( staticlist ) ( data \"Blocker\" ) )\n")
        out_file.write("        ( string \"Lighting\" ( staticlist ) ( data \"Flat\" ) )\n")
        out_file.write("        ( bool \"NotAStep\" (  ) ( data 0 ) )\n")
        out_file.write("        ( bool \"Detail\" (  ) ( data 0 ) )\n")
        out_file.write("        ( longint \"LightControl\" ( groupowner group1 ) ( data 0.000000 ) )\n")
        out_file.write("        ( string \"TextureEffect\" ( textureeffect ) )\n")
        out_file.write("        ( color \"AmbientLight\" ( group1 ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )\n")
        out_file.write("        ( longint \"LMGridSize\" ( group1 ) ( data 0.000000 ) )\n")
        out_file.write("        ( bool \"ClipLight\" ( group1 ) ( data 0 ) )\n")
        out_file.write("        ( bool \"CastShadowMesh\" ( group1 ) ( data 0 ) )\n")
        out_file.write("        ( bool \"ReceiveLight\" ( group1 ) ( data 0 ) )\n")
        out_file.write("        ( bool \"ReceiveShadows\" ( group1 ) ( data 0 ) )\n")
        out_file.write("        ( bool \"ReceiveSunlight\" ( group1 ) ( data 0 ) )\n")
        out_file.write("        ( real \"LightPenScale\" ( group1 ) ( data 0.000000 ) )\n")
        out_file.write("        ( real \"CreaseAngle\" ( group1 ) ( data 45.000000 ) )\n")
        out_file.write("    ) )\n")
    if polypParticleblock_count > 0:
        out_file.write("    ( proplist ( \n")
        out_file.write(f'        ( string "Name" (  ) ( data "ParticleBlocker" ) )\n')
        out_file.write("        ( vector \"Pos\" ( distance ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )\n")
        out_file.write("        ( rotation \"Rotation\" (  ) ( data ( eulerangles (0.000000 0.000000 0.000000) ) ) )\n")
        out_file.write("        ( longint \"RenderGroup\" (  ) ( data 0.000000 ) )\n")
        out_file.write("        ( string \"Type\" ( staticlist ) ( data \"ParticleBlocker\" ) )\n")
        out_file.write("        ( string \"Lighting\" ( staticlist ) ( data \"Flat\" ) )\n")
        out_file.write("        ( bool \"NotAStep\" (  ) ( data 0 ) )\n")
        out_file.write("        ( bool \"Detail\" (  ) ( data 0 ) )\n")
        out_file.write("        ( longint \"LightControl\" ( groupowner group1 ) ( data 0.000000 ) )\n")
        out_file.write("        ( string \"TextureEffect\" ( textureeffect ) )\n")
        out_file.write("        ( color \"AmbientLight\" ( group1 ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )\n")
        out_file.write("        ( longint \"LMGridSize\" ( group1 ) ( data 0.000000 ) )\n")
        out_file.write("        ( bool \"ClipLight\" ( group1 ) ( data 0 ) )\n")
        out_file.write("        ( bool \"CastShadowMesh\" ( group1 ) ( data 0 ) )\n")
        out_file.write("        ( bool \"ReceiveLight\" ( group1 ) ( data 0 ) )\n")
        out_file.write("        ( bool \"ReceiveShadows\" ( group1 ) ( data 0 ) )\n")
        out_file.write("        ( bool \"ReceiveSunlight\" ( group1 ) ( data 0 ) )\n")
        out_file.write("        ( real \"LightPenScale\" ( group1 ) ( data 0.000000 ) )\n")
        out_file.write("        ( real \"CreaseAngle\" ( group1 ) ( data 45.000000 ) )\n")
        out_file.write("    ) )\n")
        
    out_file.write(") )\n")
    out_file.write(")\n")

def get_surface_type(flags: set[str]) -> str:
    if "NONEXISTENT" in flags:
        if "VISBLOCKER" in flags:
            return "Occluder"
        elif "RBSPLITTER" in flags:
            return "RBSplitter"
        else:
            return "RenderOnly"
    if "SOLID" not in flags:
        return "NonSolid"
    if "SKY" in flags:
        return "SkyPortal"
    return "Normal"

def get_lighting_type(flags: set[str]) -> str:
    if "SHADOWMESH" in flags and "GOURAUDSHADE" in flags:
        return "ShadowMesh"
    if "LIGHTMAP" in flags:
        return "Lightmap"
    if "GOURAUDSHADE" in flags:
        return "Gouraud"
    if "FLATSHADE" in flags:
        return "Flat"
    return "Gouraud"


_OPQ_IDENTITY_MODULE = (Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 1))


def _run_opq_match(args):
    """Module-level wrapper — нужен чтобы ThreadPoolExecutor мог его пикловать."""
    if args is None:
        return None
    (n_obj, raw_dist, tex_key, phys_arr, phys_center_arr, n_arr,
     primary_pool, primary_mat, fallback_pool, fallback_mat) = args
    return find_best_donor(n_obj, raw_dist, tex_key, phys_arr, phys_center_arr, n_arr,
                           primary_pool, primary_mat, fallback_pool, fallback_mat)
                           
def get_box_mapping_axes(normal):
    """Возвращает стандартные немасштабированные векторы P и Q (Box Mapping) по нормали."""
    nx, ny, nz = normal.x, normal.y, normal.z
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    from lth_types import Vec3
    
    if ax >= ay and ax >= az:
        if nx > 0: return Vec3(0, 0, 1), Vec3(0, -1, 0)
        else:      return Vec3(0, 0, -1), Vec3(0, -1, 0)
    elif ay >= ax and ay >= az:
        if ny > 0: return Vec3(1, 0, 0), Vec3(0, 0, -1)
        else:      return Vec3(1, 0, 0), Vec3(0, 0, 1)
    else:
        if nz > 0: return Vec3(-1, 0, 0), Vec3(0, -1, 0)
        else:      return Vec3(1, 0, 0), Vec3(0, -1, 0)

def format_clean_vec3(v):
    """Форматирует вектор, убирая минус нули (-0.000000)."""
    def clean0(val): return 0.0 if abs(val) < 1e-8 else val
    return f"{clean0(v.x):.6f} {clean0(v.y):.6f} {clean0(v.z):.6f}"

def write_surfaces_to_lta(f, header, worldtree, out_file, map_name,
                          main_pool_by_tex=None, main_pool_matrices=None,
                          wm_pool_by_tex=None,   wm_pool_matrices=None,
                          occluder_names=None,   keyframer_keys_map=None):
    from collections import defaultdict
    import numpy as np

    _has_pools = (main_pool_by_tex is not None and main_pool_matrices is not None)

    _OPQ_IDENTITY = (
        Vec3(0, 0, 0),
        Vec3(1, 0, 0),
        Vec3(0, 0, 1),
    )

    def _opq_from_donor(donor, tex_name, normal):
        if donor is None:
            # --- Чистый Box Mapping для ненайденных поверхностей (SkyPortal) ---
            P_box, Q_box = get_box_mapping_axes(normal)
            
            def clean0(val): return 0.0 if abs(val) < 1e-8 else val
            
            P = Vec3(clean0(P_box.x), clean0(P_box.y), clean0(P_box.z))
            Q = Vec3(clean0(Q_box.x), clean0(Q_box.y), clean0(Q_box.z))
            O = Vec3(0.0, 0.0, 0.0)
            
            return (O, P, Q), _OPQ_IDENTITY, "Default"

        tex_w, tex_h = get_dtx_texture_size(tex_name)
        tex1_name = donor.get("tex1", "Default") or "Default"
        tex1_w, tex1_h = get_dtx_texture_size(tex1_name)
        dv0 = apply_world_offset(donor["v0"])
        dv1 = apply_world_offset(donor["v1"])
        dv2 = apply_world_offset(donor["v2"])
        O, P, Q = generate_opq_exact(dv0, dv1, dv2,
                                     donor["uv0"], donor["uv1"], donor["uv2"],
                                     tex_w, tex_h)
        O1, P1, Q1 = _OPQ_IDENTITY
        if "uv1_0" in donor:
            O1, P1, Q1 = generate_opq_exact(dv0, dv1, dv2,
                                             donor["uv1_0"], donor["uv1_1"], donor["uv1_2"],
                                             tex1_w, tex1_h)
        return (O, P, Q), (O1, P1, Q1), tex1_name

    def _get_opq_dual(n_obj, raw_dist, tex_name, raw_phys_verts, wm_name):
        if not _has_pools:
            return _OPQ_IDENTITY, _OPQ_IDENTITY, "Default"

        tex_key = clean_tex_key(tex_name)
        primary_pool = (wm_pool_by_tex or {}).get(wm_name, main_pool_by_tex)
        primary_mat  = (wm_pool_matrices or {}).get(wm_name, main_pool_matrices)

        phys_arr        = np.array([[v.x, v.y, v.z] for v in raw_phys_verts], dtype=np.float64)
        phys_center_arr = phys_arr.mean(axis=0)
        n_arr           = np.array([n_obj.x, n_obj.y, n_obj.z], dtype=np.float64)

        donor = find_best_donor(
            n_obj, raw_dist, tex_key,
            phys_arr, phys_center_arr, n_arr,
            primary_pool, primary_mat,
            main_pool_by_tex, main_pool_matrices)

        if donor is None:
            return _OPQ_IDENTITY, _OPQ_IDENTITY, "Default"

        tex_w, tex_h = get_dtx_texture_size(tex_name)
        tex1_name = donor.get("tex1", "Default")
        if not tex1_name:
            tex1_name = "Default"
        tex1_w, tex1_h = get_dtx_texture_size(tex1_name)

        dv0 = apply_world_offset(donor["v0"])
        dv1 = apply_world_offset(donor["v1"])
        dv2 = apply_world_offset(donor["v2"])
        
        O, P, Q = generate_opq_exact(
            dv0, dv1, dv2,
            donor["uv0"], donor["uv1"], donor["uv2"],
            tex_w, tex_h)
            
        O1, P1, Q1 = _OPQ_IDENTITY
        if "uv1_0" in donor:
            O1, P1, Q1 = generate_opq_exact(
                dv0, dv1, dv2,
                donor["uv1_0"], donor["uv1_1"], donor["uv1_2"],
                tex1_w, tex1_h)

        return (O, P, Q), (O1, P1, Q1), tex1_name

    def _poly_plane(local_points):
        """Compute (nx, ny, nz, dist) of the plane containing local_points.

        Iterates pairs of edges until a non-degenerate cross product is found.
        Returns None if all edges are collinear.
        """
        for i in range(len(local_points) - 2):
            p0, p1, p2 = local_points[i], local_points[i + 1], local_points[i + 2]
            e1x, e1y, e1z = p1.x - p0.x, p1.y - p0.y, p1.z - p0.z
            e2x, e2y, e2z = p2.x - p0.x, p2.y - p0.y, p2.z - p0.z
            nx = e1y * e2z - e1z * e2y
            ny = e1z * e2x - e1x * e2z
            nz = e1x * e2y - e1y * e2x
            length = math.sqrt(nx * nx + ny * ny + nz * nz)
            if length > 1e-6:
                nx, ny, nz = nx / length, ny / length, nz / length
                dist = nx * p0.x + ny * p0.y + nz * p0.z
                return nx, ny, nz, dist
        return None

    def _barycentric_interp(px, py, pz, tri):
        """Return barycentric-interpolated alpha for point (px,py,pz) in tri.

        Returns (alpha, inside) where inside=True when the point is within
        the triangle (with a small tolerance).
        """
        v0a, v1a, v2a = tri["v0a"], tri["v1a"], tri["v2a"]
        e1x, e1y, e1z = v1a[0] - v0a[0], v1a[1] - v0a[1], v1a[2] - v0a[2]
        e2x, e2y, e2z = v2a[0] - v0a[0], v2a[1] - v0a[1], v2a[2] - v0a[2]
        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        area_sq = nx * nx + ny * ny + nz * nz
        if area_sq < 1e-12:
            return int(tri["c0"][3]), False

        b0x, b0y, b0z = v0a[0] - px, v0a[1] - py, v0a[2] - pz
        b1x, b1y, b1z = v1a[0] - px, v1a[1] - py, v1a[2] - pz
        b2x, b2y, b2z = v2a[0] - px, v2a[1] - py, v2a[2] - pz

        c0x = b1y * b2z - b1z * b2y
        c0y = b1z * b2x - b1x * b2z
        c0z = b1x * b2y - b1y * b2x

        c1x = b2y * b0z - b2z * b0y
        c1y = b2z * b0x - b2x * b0z
        c1z = b2x * b0y - b2y * b0x

        u = (nx * c0x + ny * c0y + nz * c0z) / area_sq
        v = (nx * c1x + ny * c1y + nz * c1z) / area_sq
        w = 1.0 - u - v

        inside = u >= -0.01 and v >= -0.01 and w >= -0.01

        # Clamp and renormalise so we always get a valid interpolation.
        u = max(0.0, u)
        v = max(0.0, v)
        w = max(0.0, w)
        s = u + v + w
        if s > 1e-8:
            u, v, w = u / s, v / s, w / s

        a = u * tri["c0"][3] + v * tri["c1"][3] + w * tri["c2"][3]
        return int(round(a)), inside

    def _find_coplanar_dual_tris(local_points, candidates, epsilon=25.0):
        """Return DualTexture render triangles coplanar with the physics polygon.

        If nothing is coplanar within epsilon, falls back to the 10 nearest
        DualTexture triangles by centroid distance (last-resort approximation).
        """
        plane = _poly_plane(local_points)
        if plane is None or not candidates:
            return []

        nx, ny, nz, dist = plane
        dual = [t for t in candidates if t.get("shader_code") in (8, 9)]
        if not dual:
            return []

        coplanar = [
            t for t in dual
            if all(
                abs(nx * v[0] + ny * v[1] + nz * v[2] - dist) < epsilon
                for v in (t["v0a"], t["v1a"], t["v2a"])
            )
        ]

        if coplanar:
            return coplanar

        # Last resort: nearest dual-texture triangles regardless of coplanarity.
        cx = sum(p.x for p in local_points) / len(local_points)
        cy = sum(p.y for p in local_points) / len(local_points)
        cz = sum(p.z for p in local_points) / len(local_points)
        dual.sort(key=lambda t: (
            (t["center"][0] - cx) ** 2 +
            (t["center"][1] - cy) ** 2 +
            (t["center"][2] - cz) ** 2
        ))
        return dual[:10]

    def _get_alpha_map_for_surface(local_points, tex_name, wm_name):
        if not _has_pools:
            return {}, []

        def pk(v):
            return (round(v.x, 2), round(v.y, 2), round(v.z, 2))

        phys_keys    = {pk(v) for v in local_points}
        tex_key      = clean_tex_key(tex_name)
        primary_pool = (wm_pool_by_tex or {}).get(wm_name, main_pool_by_tex)
        render_tris  = primary_pool.get(tex_key, [])

        alpha_map    = {}
        matched_tris = []

        def _collect(tris):
            for tri in tris:
                k0, k1, k2 = pk(tri["v0"]), pk(tri["v1"]), pk(tri["v2"])
                if k0 in phys_keys and k1 in phys_keys and k2 in phys_keys:
                    matched_tris.append(tri)
                    if tri.get("shader_code") in (8, 9):
                        if k0 not in alpha_map: alpha_map[k0] = tri["c0"][3]
                        if k1 not in alpha_map: alpha_map[k1] = tri["c1"][3]
                        if k2 not in alpha_map: alpha_map[k2] = tri["c2"][3]

        _collect(render_tris)
        if not matched_tris and primary_pool is not main_pool_by_tex:
            _collect(main_pool_by_tex.get(tex_key, []))

        # Fallback: no exact vertex matches (BSP split the polygon differently).
        # Search for coplanar DualTexture triangles in the same texture pool so
        # _barycentric_alpha_from_tris can still produce a meaningful value.
        if not matched_tris:
            all_candidates = render_tris or main_pool_by_tex.get(tex_key, [])
            matched_tris   = _find_coplanar_dual_tris(local_points, all_candidates)

        return alpha_map, matched_tris

    def _barycentric_alpha_from_tris(pt, matched_tris):
        """Interpolate alpha at pt from the nearest / containing render triangle.

        Strategy:
          1. Find the first triangle that actually contains pt (barycentric test).
             If found, return its interpolated alpha immediately.
          2. If no triangle contains pt, fall back to the nearest by centroid
             and project pt onto it via clamped barycentric coordinates.
        """
        if not matched_tris:
            return 255

        px, py, pz = pt.x, pt.y, pt.z

        # Pass 1: containment test — most accurate result.
        for tri in matched_tris:
            a, inside = _barycentric_interp(px, py, pz, tri)
            if inside:
                return a

        # Pass 2: nearest centroid — clamped projection fallback.
        best_tri = min(
            matched_tris,
            key=lambda t: (
                (t["center"][0] - px) ** 2 +
                (t["center"][1] - py) ** 2 +
                (t["center"][2] - pz) ** 2
            ),
        )
        a, _ = _barycentric_interp(px, py, pz, best_tri)
        return a
        
    f.seek(header.object_data_pos)
    world_objects, world_objects_meta = read_world_objects_lta(f, header)

    objects_by_type = defaultdict(list)
    for i, obj in enumerate(world_objects):
        obj_type = obj.get("__Type", "Unknown")
        objects_by_type[obj_type].append((i, obj))

    out_file.write(f"( world\n\t( header (\n\t\t( versioncode 2 )\n\t\t( infostring \"{lth_header.info_string}\" )\n\t) )\n\t( polyhedronlist (\n")

    brush_props = {}
    wm_brushes = defaultdict(list)
    brush_tex_effects = {}  # Словарь для хранения TextureEffect каждого браша
    brushindex = 0

    bsp_plan = []   
    bsp_args = []   

    for world_model in worldtree.world_models:
        wm_name = world_model.world_name
        surface_to_polies = defaultdict(list)
        for poly_idx, (surface_idx, plane_idx, indices) in enumerate(world_model.polies):
            flags_raw = world_model.surfaces[surface_idx].get("Flags", 0)
            flags_set = set(decode_surface_flags(flags_raw))
            if "VISBLOCKER" in flags_set:
                continue
            surface_to_polies[surface_idx].append((poly_idx, plane_idx, indices))

        for surface_idx in sorted(surface_to_polies.keys()):
            polies = surface_to_polies[surface_idx]
            point_map = {}
            local_points = []
            for _, _, indices in polies:
                for idx in indices:
                    if idx not in point_map:
                        point_map[idx] = len(local_points)
                        local_points.append(world_model.points[idx])

            tex_idx  = world_model.surfaces[surface_idx]['TextureIndex']
            tex_name = world_model.texture_names[tex_idx] if 0 <= tex_idx < len(world_model.texture_names) else "Default"

            bsp_plan.append({
                "world_model":  world_model,
                "wm_name":      wm_name,
                "surface_idx":  surface_idx,
                "polies":       polies,
                "tex_name":     tex_name,
                "point_map":    point_map,
                "local_points": local_points,
            })

            for _, plane_idx, poly_indices in polies:
                n, dist = world_model.planes[plane_idx]
                raw_phys_verts = [world_model.points[idx] for idx in poly_indices]
                if _has_pools:
                    tex_key         = clean_tex_key(tex_name)
                    primary_pool    = (wm_pool_by_tex or {}).get(wm_name, main_pool_by_tex)
                    primary_mat     = (wm_pool_matrices or {}).get(wm_name, main_pool_matrices)
                    phys_arr        = np.array([[v.x, v.y, v.z] for v in raw_phys_verts], dtype=np.float64)
                    phys_center_arr = phys_arr.mean(axis=0)
                    n_arr           = np.array([n.x, n.y, n.z], dtype=np.float64)
                    bsp_args.append((n, dist, tex_key, phys_arr, phys_center_arr, n_arr,
                                     primary_pool, primary_mat,
                                     main_pool_by_tex, main_pool_matrices))
                else:
                    bsp_args.append(None)

    num_workers = os.cpu_count() or 4
    print(f"UV matching BSP ({len(bsp_args)} polygons, {num_workers} streams)...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        bsp_donors = list(tqdm(
            executor.map(_run_opq_match, bsp_args),
            total=len(bsp_args), desc="Matching", unit="poly",
            ascii=True, ncols=80))

    donor_iter = iter(bsp_donors)
    for entry in bsp_plan:
        world_model  = entry["world_model"]
        wm_name      = entry["wm_name"]
        surface_idx  = entry["surface_idx"]
        polies       = entry["polies"]
        tex_name     = entry["tex_name"]
        point_map    = entry["point_map"]
        local_points = entry["local_points"]

        brush_props[brushindex] = (world_model, surface_idx)
        wm_brushes[wm_name].append(brushindex)

        alpha_map, matched_tris = _get_alpha_map_for_surface(
            local_points, tex_name, wm_name)

        out_file.write("\t\t( polyhedron (\n")
        out_file.write("\t\t\t( color 255 255 255 )\n")
        out_file.write("\t\t\t( pointlist \n")
        for pt in local_points:
            adjusted = apply_world_offset(pt)
            pk = (round(pt.x, 2), round(pt.y, 2), round(pt.z, 2))
            if pk in alpha_map:
                a = alpha_map[pk]                          
            elif matched_tris:
                a = _barycentric_alpha_from_tris(pt, matched_tris)  
            else:
                a = 255                                    
            out_file.write(
                f"\t\t\t\t( {adjusted.x:.6f} {adjusted.y:.6f} {adjusted.z:.6f} {a} {a} {a} {a} )\n"
            )
        out_file.write("\t\t\t)\n")

        out_file.write("\t\t\t( polylist (\n")
        surface_effect = ""  # Переменная для сохранения TextureEffect всей поверхности
        
        for _, plane_idx, indices in polies:
            donor = next(donor_iter)
            
            # Извлекаем эффект из донора (если есть)
            if donor and not surface_effect:
                surface_effect = donor.get("texture_effect", "")

            n, dist = world_model.planes[plane_idx]
            local_indices = [point_map[i] for i in indices]
            f_str = " ".join(str(i) for i in local_indices)

            (O, P, Q), (O1, P1, Q1), tex1_name = _opq_from_donor(donor, tex_name, n)

            out_file.write("\t\t\t\t( editpoly \n")
            out_file.write(f"\t\t\t\t\t( f {f_str} )\n")
            out_file.write(f"\t\t\t\t\t( n {n.x:.6f} {n.y:.6f} {n.z:.6f} )\n")
            out_file.write(f"\t\t\t\t\t( dist {dist:.6f} )\n")
            out_file.write("\t\t\t\t\t( textureinfo \n")
            out_file.write(f"\t\t\t\t\t\t( {write_vec3(O)} )\n")
            out_file.write(f"\t\t\t\t\t\t( {write_vec3(P)} )\n")
            out_file.write(f"\t\t\t\t\t\t( {write_vec3(Q)} )\n")
            out_file.write("\t\t\t\t\t\t( sticktopoly 1 )\n")
            out_file.write(f'\t\t\t\t\t\t( name "{tex_name}" )\n')
            out_file.write("\t\t\t\t\t)\n")
            out_file.write("\t\t\t\t\t( flags )\n")
            out_file.write("\t\t\t\t\t( shade 0 0 0 )\n")
            out_file.write("\t\t\t\t\t( physicsmaterial \"Default\" )\n")
            out_file.write("\t\t\t\t\t( surfacekey \"\" )\n")
            out_file.write("\t\t\t\t\t( textures ( \n")
            out_file.write("\t\t\t\t\t\t( 1 ( textureinfo \n")
            out_file.write(f"\t\t\t\t\t\t\t( {write_vec3(O1)} )\n")
            out_file.write(f"\t\t\t\t\t\t\t( {write_vec3(P1)} )\n")
            out_file.write(f"\t\t\t\t\t\t\t( {write_vec3(Q1)} )\n")
            out_file.write("\t\t\t\t\t\t\t( sticktopoly 1 )\n")
            out_file.write(f'\t\t\t\t\t\t\t( name "{tex1_name}" )\n')
            out_file.write("\t\t\t\t\t\t) )\n")
            out_file.write("\t\t\t\t\t) )\n")
            out_file.write("\t\t\t\t)\n")

        out_file.write("\t\t\t) )\n")
        out_file.write("\t\t) )\n")
        
        # Сохраняем эффект для свойств браша
        brush_tex_effects[brushindex] = surface_effect
        brushindex += 1

    polyBlocks_count = write_physics_polygons(f, header, out_file)
    polypParticleblock_count = write_particleblockers_polygons(f, header, out_file)

    if occluders_all:
        tex_occ = "Tex\\Occluder.dtx"
        for idx, (verts, plane_normal, plane_dist, occ_hashcode) in enumerate(occluders_all):
            out_file.write("\t\t( polyhedron (\n")
            out_file.write("\t\t\t( color 255 255 255 )\n")
            out_file.write("\t\t\t( pointlist \n")
            for v in verts:
                adjusted_v = apply_world_offset(v)
                out_file.write(f"\t\t\t\t( {format_clean_vec3(adjusted_v)} 255 255 255 255 )\n")
            out_file.write("\t\t\t)\n")
            out_file.write("\t\t\t( polylist (\n")
            out_file.write("\t\t\t\t( editpoly \n")
            vcount = len(verts)
            out_file.write(f"\t\t\t\t\t( f {' '.join(str(j) for j in range(vcount))} )\n")
            out_file.write(f"\t\t\t\t\t( n {format_clean_vec3(plane_normal)} )\n")
            out_file.write(f"\t\t\t\t\t( dist {plane_dist:.6f} )\n")
            
            P0, Q0 = get_box_mapping_axes(plane_normal)
            O0 = Vec3(0.0, 0.0, 0.0)
            
            out_file.write("\t\t\t\t\t( textureinfo \n")
            out_file.write(f"\t\t\t\t\t\t( {format_clean_vec3(O0)} )\n")
            out_file.write(f"\t\t\t\t\t\t( {format_clean_vec3(P0)} )\n")
            out_file.write(f"\t\t\t\t\t\t( {format_clean_vec3(Q0)} )\n")
            out_file.write("\t\t\t\t\t\t( sticktopoly 1 )\n")
            out_file.write(f'\t\t\t\t\t\t( name "{tex_occ}" )\n')
            out_file.write("\t\t\t\t\t)\n")
            out_file.write("\t\t\t\t\t( flags )\n")
            out_file.write("\t\t\t\t\t( shade 0 0 0 )\n")
            out_file.write('\t\t\t\t\t( physicsmaterial "Default" )\n')
            out_file.write('\t\t\t\t\t( surfacekey "" )\n')
            out_file.write("\t\t\t\t\t( textures ( \n")
            out_file.write("\t\t\t\t\t\t( 1 ( textureinfo \n")
            out_file.write("\t\t\t\t\t\t\t( 0.000000 0.000000 0.000000 )\n")
            out_file.write("\t\t\t\t\t\t\t( 1.000000 0.000000 0.000000 )\n")
            out_file.write("\t\t\t\t\t\t\t( 0.000000 0.000000 1.000000 )\n")
            out_file.write("\t\t\t\t\t\t\t( sticktopoly 1 )\n")
            out_file.write('\t\t\t\t\t\t\t( name "Default" )\n')
            out_file.write("\t\t\t\t\t\t) )\n")
            out_file.write("\t\t\t\t\t) )\n")
            out_file.write("\t\t\t\t)\n")
            out_file.write("\t\t\t) )\n")
            out_file.write("\t\t) )\n")

    out_file.write("\t) )\n") 

    proplist_blocks = []
    
    def add_prop(prop_str):
        proplist_blocks.append(prop_str)
        return len(proplist_blocks)

    # --- Обновлено для поддержки TextureEffect ---
    def generate_surface_prop(wm, surface_idx, name, texture_effect=""):
        surface = wm.surfaces[surface_idx]
        flags = set(decode_surface_flags(surface.get("Flags", 0)))
        surf_type = get_surface_type(flags)
        lighting = get_lighting_type(flags)
        def bool_flag(flg): return 1 if flg in flags else 0
        
        effect_line = f'( string "TextureEffect" ( textureeffect ) ( data "{texture_effect}" ) )' if texture_effect else '( string "TextureEffect" ( textureeffect ) )'
        
        return f"""    ( proplist ( 
        ( string "Name" (  ) ( data "{name}" ) )
        ( vector "Pos" ( distance ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )
        ( rotation "Rotation" (  ) ( data ( eulerangles (0.000000 0.000000 0.000000) ) ) )
        ( longint "RenderGroup" (  ) ( data 0.000000 ) )
        ( string "Type" ( staticlist ) ( data "{surf_type}" ) )
        ( string "Lighting" ( staticlist ) ( data "{lighting}" ) )
        ( bool "NotAStep" (  ) ( data {bool_flag("NOTASTEP")} ) )
        ( bool "Detail" (  ) ( data 0 ) )
        ( longint "LightControl" ( groupowner group1 ) ( data 0.000000 ) )
        {effect_line}
        ( color "AmbientLight" ( group1 ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )
        ( longint "LMGridSize" ( group1 ) ( data 0.000000 ) )
        ( bool "ClipLight" ( group1 ) ( data {bool_flag("CLIPLIGHT")} ) )
        ( bool "CastShadowMesh" ( group1 ) ( data {bool_flag("CASTSHADOWMESH")} ) )
        ( bool "ReceiveLight" ( group1 ) ( data {bool_flag("RECEIVELIGHT")} ) )
        ( bool "ReceiveShadows" ( group1 ) ( data {bool_flag("RECEIVESHADOWS")} ) )
        ( bool "ReceiveSunlight" ( group1 ) ( data {bool_flag("RECEIVESUNLIGHT")} ) )
        ( real "LightPenScale" ( group1 ) ( data 0.000000 ) )
        ( real "CreaseAngle" ( group1 ) ( data 45.000000 ) )
    ) )"""

    def format_vec3(v): return f"( vector ({v.x:.6f} {v.y:.6f} {v.z:.6f}) )"
    def format_rotation(q): return f"(eulerangles ({q.x:.6f} {q.y:.6f} {q.z:.6f}))"

    def generate_object_prop(props, meta):
        lines = ["    ( proplist ("]
        for key, value in props.items():
            if key == "__Type": continue
            field_type = meta.get(key, {}).get("type")
            if field_type == "string":
                if value is not None: lines.append(f'        (  string "{key}" (  ) ( data "{value}") )')
                else: lines.append(f'        (  string "{key}" ( hidden  ) )')
            elif field_type == "vector":
                if key == "Pos":
                    adjusted_v = apply_world_offset(value)
                    lines.append(f'        (  vector "{key}" ( distance  ) ( data ( vector ( {write_vec3(adjusted_v)} ) ) ) )')
                else:
                    lines.append(f'        (  vector "{key}" ( distance  ) ( data {format_vec3(value)} ) )')
            elif field_type == "color":
                lines.append(f'        (  color "{key}" (  ) ( data {format_vec3(value)} ) )')
            elif field_type == "rotation":
                lines.append(f'        (  rotation "{key}" (  ) ( data {format_rotation(value)} ) )')
            elif field_type == "bool":
                lines.append(f'        (  bool "{key}" (  ) ( data {1 if value else 0} ))')
            elif field_type == "real":
                lines.append(f'        (  real "{key}" (  ) ( data {value:.6f} ))')
            elif field_type == "longint":
                lines.append(f'        (  longint "{key}" (  ) ( data {float(value):.6f} ))')
            else:
                if isinstance(value, tuple) and len(value) == 3:
                    lines.append(f'        (  vector "{key}" (  ) ( data ( vector ({value[0]:.6f} {value[1]:.6f} {value[2]:.6f}) ) ) )')
                elif value is None:
                    lines.append(f'        (  string "{key}" ( hidden  ) )')
        lines.append("    ) )")
        return "\n".join(lines)


    # ======================== ИЕРАРХИЯ УЗЛОВ ========================
    base_nodeid = 1

    out_file.write("( nodehierarchy\n")
    out_file.write("\t( worldnode\n")
    out_file.write("\t\t( type null )\n")
    out_file.write(f'\t\t( label "{map_name}_PhysicsDATA" )\n')
    out_file.write(f"\t\t( nodeid {base_nodeid} )\n")
    out_file.write("\t\t( flags ( worldroot expanded ) )\n")
    out_file.write("\t\t( properties\n")
    out_file.write(f"\t\t\t( propid 0 )\n")
    out_file.write("\t\t)\n")
    out_file.write("\t\t( childlist (\n")
    base_nodeid += 1

    wm_in_objects = set()
    for obj_type, obj_list in objects_by_type.items():
        for _, obj in obj_list:
            name = obj.get("Name")
            if name in wm_brushes:
                wm_in_objects.add(name)

    # 1. WorldModels
    for world_model in worldtree.world_models:
        wm_name = world_model.world_name
        if wm_name in wm_in_objects or wm_name not in wm_brushes:
            continue

        out_file.write("\t\t\t( worldnode\n")
        out_file.write("\t\t\t\t( type null )\n")
        out_file.write(f'\t\t\t\t( label "{wm_name}" )\n')
        out_file.write(f"\t\t\t\t( nodeid {base_nodeid} )\n")
        out_file.write("\t\t\t\t( flags ( ) )\n")
        out_file.write("\t\t\t\t( properties\n")
        out_file.write(f"\t\t\t\t\t( propid 0 )\n")
        out_file.write("\t\t\t\t)\n")
        out_file.write("\t\t\t\t( childlist (\n")
        base_nodeid += 1

        grouped_brushes = defaultdict(list)
        for local_idx, b_idx in enumerate(wm_brushes[wm_name]):
            wm, surf_idx = brush_props[b_idx]
            surface = wm.surfaces[surf_idx]
            flags_set = set(decode_surface_flags(surface.get("Flags", 0)))
            category = get_surface_type(flags_set)
            grouped_brushes[category].append((local_idx, b_idx, wm, surf_idx))

        for category in sorted(grouped_brushes.keys()):
            brushes = grouped_brushes[category]
            out_file.write("\t\t\t\t\t( worldnode\n")
            out_file.write("\t\t\t\t\t\t( type null )\n")
            out_file.write(f'\t\t\t\t\t\t( label "{category}" )\n')
            out_file.write(f"\t\t\t\t\t\t( nodeid {base_nodeid} )\n")
            out_file.write("\t\t\t\t\t\t( flags ( ) )\n")
            out_file.write("\t\t\t\t\t\t( properties\n")
            out_file.write(f"\t\t\t\t\t\t\t( propid 0 )\n")
            out_file.write("\t\t\t\t\t\t)\n")
            out_file.write("\t\t\t\t\t\t( childlist (\n")
            base_nodeid += 1

            for sub_idx, (local_idx, b_idx, wm, surf_idx) in enumerate(brushes):
                brush_name = f"{surf_idx}_{category}{sub_idx}"
                # Передаем найденный TextureEffect в генератор
                prop_str = generate_surface_prop(wm, surf_idx, brush_name, brush_tex_effects.get(b_idx, ""))
                pid = add_prop(prop_str)

                out_file.write("\t\t\t\t\t\t\t( worldnode\n")
                out_file.write("\t\t\t\t\t\t\t\t( type brush )\n")
                out_file.write(f"\t\t\t\t\t\t\t\t( brushindex {b_idx} )\n")
                out_file.write(f"\t\t\t\t\t\t\t\t( nodeid {base_nodeid} )\n")
                out_file.write("\t\t\t\t\t\t\t\t( flags ( ) )\n")
                out_file.write("\t\t\t\t\t\t\t\t( properties\n")
                out_file.write(f'\t\t\t\t\t\t\t\t\t( name "Brush" )\n')
                out_file.write(f"\t\t\t\t\t\t\t\t\t( propid {pid} )\n")
                out_file.write("\t\t\t\t\t\t\t\t)\n")
                out_file.write("\t\t\t\t\t\t\t)\n")
                base_nodeid += 1
            
            out_file.write("\t\t\t\t\t\t) )\n")
            out_file.write("\t\t\t\t\t)\n")

        out_file.write("\t\t\t\t) )\n")
        out_file.write("\t\t\t)\n")

    # 2. Objects and KeyFramers
    has_objects = len(world_objects) > 0
    has_keyframers = bool(keyframer_keys_map)

    if has_objects or has_keyframers:
        out_file.write("\t\t\t( worldnode\n")
        out_file.write("\t\t\t\t( type null )\n")
        out_file.write(f'\t\t\t\t( label "Objects" )\n')
        out_file.write(f'\t\t\t\t( nodeid {base_nodeid} )\n')
        out_file.write('\t\t\t\t( flags ( ) )\n')
        out_file.write('\t\t\t\t( properties\n')
        out_file.write(f'\t\t\t\t\t( propid 0 )\n')
        out_file.write('\t\t\t\t)\n')
        out_file.write('\t\t\t\t( childlist (\n')
        base_nodeid += 1

        if has_objects:
            for obj_type in sorted(objects_by_type.keys()):
                out_file.write('\t\t\t\t\t( worldnode\n')
                out_file.write('\t\t\t\t\t\t( type null )\n')
                out_file.write(f'\t\t\t\t\t\t( label "{obj_type}" )\n')
                out_file.write(f'\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
                out_file.write('\t\t\t\t\t\t( flags ( ) )\n')
                out_file.write('\t\t\t\t\t\t( properties\n')
                out_file.write(f'\t\t\t\t\t\t\t( propid 0 )\n')
                out_file.write('\t\t\t\t\t\t)\n')
                out_file.write('\t\t\t\t\t\t( childlist (\n')
                base_nodeid += 1

                for i, obj in objects_by_type[obj_type]:
                    name = obj.get("Name", f"{obj_type}_{i}")
                    label = obj.get("__Type", "Object")
                    meta = world_objects_meta[i]

                    prop_str = generate_object_prop(obj, meta)
                    pid_obj = add_prop(prop_str)

                    out_file.write('\t\t\t\t\t\t\t( worldnode\n')
                    out_file.write('\t\t\t\t\t\t\t\t( type object )\n')
                    out_file.write(f'\t\t\t\t\t\t\t\t( label "{name}" )\n')
                    out_file.write(f'\t\t\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
                    out_file.write('\t\t\t\t\t\t\t\t( flags ( ) )\n')
                    out_file.write('\t\t\t\t\t\t\t\t( properties\n')
                    out_file.write(f'\t\t\t\t\t\t\t\t\t( name "{label}" )\n')
                    out_file.write(f'\t\t\t\t\t\t\t\t\t( propid {pid_obj} )\n')
                    out_file.write('\t\t\t\t\t\t\t\t)\n')
                    base_nodeid += 1

                    if name in wm_brushes:
                        out_file.write('\t\t\t\t\t\t\t\t( childlist (\n')
                        
                        grouped_brushes = defaultdict(list)
                        for local_idx, b_idx in enumerate(wm_brushes[name]):
                            wm, surf_idx = brush_props[b_idx]
                            surface = wm.surfaces[surf_idx]
                            flags_set = set(decode_surface_flags(surface.get("Flags", 0)))
                            category = get_surface_type(flags_set)
                            grouped_brushes[category].append((local_idx, b_idx, wm, surf_idx))

                        for category in sorted(grouped_brushes.keys()):
                            brushes = grouped_brushes[category]
                            out_file.write('\t\t\t\t\t\t\t\t\t( worldnode\n')
                            out_file.write('\t\t\t\t\t\t\t\t\t\t( type null )\n')
                            out_file.write(f'\t\t\t\t\t\t\t\t\t\t( label "{category}" )\n')
                            out_file.write(f'\t\t\t\t\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
                            out_file.write('\t\t\t\t\t\t\t\t\t\t( flags ( ) )\n')
                            out_file.write('\t\t\t\t\t\t\t\t\t\t( properties\n')
                            out_file.write(f'\t\t\t\t\t\t\t\t\t\t\t( propid 0 )\n')
                            out_file.write('\t\t\t\t\t\t\t\t\t\t)\n')
                            out_file.write('\t\t\t\t\t\t\t\t\t\t( childlist (\n')
                            base_nodeid += 1

                            for sub_idx, (local_idx, b_idx, wm, surf_idx) in enumerate(brushes):
                                brush_name = f"{surf_idx}_{category}{sub_idx}"
                                # Передаем найденный TextureEffect в генератор
                                prop_str = generate_surface_prop(wm, surf_idx, brush_name, brush_tex_effects.get(b_idx, ""))
                                pid_surf = add_prop(prop_str)

                                out_file.write('\t\t\t\t\t\t\t\t\t\t\t( worldnode\n')
                                out_file.write('\t\t\t\t\t\t\t\t\t\t\t\t( type brush )\n')
                                out_file.write(f'\t\t\t\t\t\t\t\t\t\t\t\t( brushindex {b_idx} )\n')
                                out_file.write(f'\t\t\t\t\t\t\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
                                out_file.write('\t\t\t\t\t\t\t\t\t\t\t\t( flags ( ) )\n')
                                out_file.write('\t\t\t\t\t\t\t\t\t\t\t\t( properties\n')
                                out_file.write(f'\t\t\t\t\t\t\t\t\t\t\t\t\t( name "Brush" )\n')
                                out_file.write(f'\t\t\t\t\t\t\t\t\t\t\t\t\t( propid {pid_surf} )\n')
                                out_file.write('\t\t\t\t\t\t\t\t\t\t\t\t)\n')
                                out_file.write('\t\t\t\t\t\t\t\t\t\t\t)\n')
                                base_nodeid += 1
                                
                            out_file.write('\t\t\t\t\t\t\t\t\t\t) )\n')
                            out_file.write('\t\t\t\t\t\t\t\t\t)\n')
                        out_file.write('\t\t\t\t\t\t\t\t) )\n')
                    out_file.write('\t\t\t\t\t\t\t)\n')
                out_file.write('\t\t\t\t\t\t) )\n')
                out_file.write('\t\t\t\t\t)\n')

        if has_keyframers:
            out_file.write('\t\t\t\t\t( worldnode\n')
            out_file.write('\t\t\t\t\t\t( type null )\n')
            out_file.write(f'\t\t\t\t\t\t( label "KeyFramers" )\n')
            out_file.write(f'\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
            out_file.write('\t\t\t\t\t\t( flags ( ) ) \n')
            out_file.write('\t\t\t\t\t\t( properties\n')
            out_file.write(f'\t\t\t\t\t\t\t( propid 0 )\n')
            out_file.write('\t\t\t\t\t\t)\n')
            out_file.write('\t\t\t\t\t\t( childlist (\n')
            base_nodeid += 1
            
            for key_name in keyframer_keys_map:
                out_file.write('\t\t\t\t\t\t\t( worldnode\n')
                out_file.write('\t\t\t\t\t\t\t\t( type null )\n')
                out_file.write(f'\t\t\t\t\t\t\t\t( label "{key_name}" )\n')
                out_file.write(f'\t\t\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
                out_file.write('\t\t\t\t\t\t\t\t( flags ( path ) ) \n')
                out_file.write('\t\t\t\t\t\t\t\t( properties\n')
                out_file.write(f'\t\t\t\t\t\t\t\t\t( propid 0 )\n')
                out_file.write('\t\t\t\t\t\t\t\t)\n')
                out_file.write('\t\t\t\t\t\t\t\t( childlist (\n')
                base_nodeid += 1

                for key_data in keyframer_keys_map[key_name]:
                    pos = key_data["pos"]
                    if not isinstance(pos, Vec3): pos = Vec3(*pos)
                    rot_deg = key_data["rot_deg"]
                    if not isinstance(rot_deg, Vec3): rot_deg = Vec3(*rot_deg)
                    bez_prev = key_data.get("bez_prev") or Vec3(0, 0, 0)
                    bez_next = key_data.get("bez_next") or Vec3(0, 0, 0)
                    sound_name = key_data.get("sound_name", "")
                    command = key_data.get("command", "")
                    
                    adj_pos = apply_world_offset(pos)
                    
                    kf_prop_str = f"""    ( proplist ( 
        ( string "Name" (  ) ( data "{key_data["full_name"]}") ) 
        ( vector "Pos" ( distance ) ( data ( vector ( {write_vec3(adj_pos)} ) ) ) ) 
        ( rotation "Rotation" (  ) ( data ( eulerangles ({rot_deg.x:.6f} {rot_deg.y:.6f} {rot_deg.z:.6f}) ) ) ) 
        ( longint "RenderGroup" (  ) ( data 0.000000 ) ) 
        ( real "TimeStamp" (  ) ( data {key_data["timestamp"]:.6f} ) ) 
        ( string "SoundName" (  ) {f'( data "{sound_name}" )' if sound_name else ''} ) 
        ( real "SoundRadius" ( radius ) ( data {key_data["sound_radius"]:.6f} ) ) 
        ( string "Command" ( notifychange ) {f'( data "{command}" )' if command else ''} ) 
        ( vector "BezierPrev" ( bezierprevtangent ) ( data ( vector ({bez_prev.x:.6f} {bez_prev.y:.6f} {bez_prev.z:.6f}) ) ) ) 
        ( vector "BezierNext" ( beziernexttangent ) ( data ( vector ({bez_next.x:.6f} {bez_next.y:.6f} {bez_next.z:.6f}) ) ) ) 
    ) )"""
                    pid_kf = add_prop(kf_prop_str)

                    out_file.write('\t\t\t\t\t\t\t\t\t( worldnode\n')
                    out_file.write('\t\t\t\t\t\t\t\t\t\t( type object )\n')
                    out_file.write(f'\t\t\t\t\t\t\t\t\t\t( label "Key" )\n')
                    out_file.write(f'\t\t\t\t\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
                    out_file.write('\t\t\t\t\t\t\t\t\t\t( flags ( ) )\n')
                    out_file.write('\t\t\t\t\t\t\t\t\t\t( properties\n')
                    out_file.write(f'\t\t\t\t\t\t\t\t\t\t\t( name "Key" )\n')
                    out_file.write(f'\t\t\t\t\t\t\t\t\t\t\t( propid {pid_kf} )\n')
                    out_file.write('\t\t\t\t\t\t\t\t\t\t)\n')
                    out_file.write('\t\t\t\t\t\t\t\t\t)\n')
                    base_nodeid += 1

                out_file.write('\t\t\t\t\t\t\t\t) )\n') 
                out_file.write('\t\t\t\t\t\t\t)\n') 
                
            out_file.write('\t\t\t\t\t\t) )\n') 
            out_file.write('\t\t\t\t\t)\n') 

        out_file.write('\t\t\t\t) )\n')
        out_file.write('\t\t\t)\n')

    # 3. Blockers
    if polyBlocks_count > 0:
        out_file.write("\t\t\t( worldnode\n")
        out_file.write("\t\t\t\t( type null )\n")
        out_file.write(f'\t\t\t\t( label "Blockers" )\n')
        out_file.write(f'\t\t\t\t( nodeid {base_nodeid} )\n')
        out_file.write('\t\t\t\t( flags ( ) )\n')
        out_file.write('\t\t\t\t( properties\n')
        out_file.write(f'\t\t\t\t\t( propid 0 )\n')
        out_file.write('\t\t\t\t)\n')
        out_file.write('\t\t\t\t( childlist (\n')
        base_nodeid += 1

        pid_blocker = add_prop("""    ( proplist ( 
        ( string "Name" (  ) ( data "Blocker" ) )
        ( vector "Pos" ( distance ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )
        ( rotation "Rotation" (  ) ( data ( eulerangles (0.000000 0.000000 0.000000) ) ) )
        ( longint "RenderGroup" (  ) ( data 0.000000 ) )
        ( string "Type" ( staticlist ) ( data "Blocker" ) )
        ( string "Lighting" ( staticlist ) ( data "Flat" ) )
        ( bool "NotAStep" (  ) ( data 0 ) )
        ( bool "Detail" (  ) ( data 0 ) )
        ( longint "LightControl" ( groupowner group1 ) ( data 0.000000 ) )
        ( string "TextureEffect" ( textureeffect ) )
        ( color "AmbientLight" ( group1 ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )
        ( longint "LMGridSize" ( group1 ) ( data 0.000000 ) )
        ( bool "ClipLight" ( group1 ) ( data 0 ) )
        ( bool "CastShadowMesh" ( group1 ) ( data 0 ) )
        ( bool "ReceiveLight" ( group1 ) ( data 0 ) )
        ( bool "ReceiveShadows" ( group1 ) ( data 0 ) )
        ( bool "ReceiveSunlight" ( group1 ) ( data 0 ) )
        ( real "LightPenScale" ( group1 ) ( data 0.000000 ) )
        ( real "CreaseAngle" ( group1 ) ( data 45.000000 ) )
    ) )""")

        b_idx = len(brush_props)
        for _ in range(polyBlocks_count):
            out_file.write('\t\t\t\t\t( worldnode\n')
            out_file.write('\t\t\t\t\t\t( type brush )\n')
            out_file.write(f'\t\t\t\t\t\t( brushindex {b_idx} )\n')
            out_file.write(f'\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
            out_file.write('\t\t\t\t\t\t( flags ( ) )\n')
            out_file.write('\t\t\t\t\t\t( properties\n')
            out_file.write('\t\t\t\t\t\t\t( name "Brush" )\n')
            out_file.write(f'\t\t\t\t\t\t\t( propid {pid_blocker} )\n')
            out_file.write('\t\t\t\t\t\t)\n')
            out_file.write('\t\t\t\t\t)\n')
            b_idx += 1
            base_nodeid += 1
        out_file.write('\t\t\t\t) )\n')
        out_file.write("\t\t\t)\n")

    # 4. ParticleBlockers
    if polypParticleblock_count > 0:
        out_file.write("\t\t\t( worldnode\n")
        out_file.write("\t\t\t\t( type null )\n")
        out_file.write(f'\t\t\t\t( label "ParticleBlockers" )\n')
        out_file.write(f'\t\t\t\t( nodeid {base_nodeid} )\n')
        out_file.write('\t\t\t\t( flags ( ) )\n')
        out_file.write('\t\t\t\t( properties\n')
        out_file.write(f'\t\t\t\t\t( propid 0 )\n')
        out_file.write('\t\t\t\t)\n')
        out_file.write('\t\t\t\t( childlist (\n')
        base_nodeid += 1

        pid_pblocker = add_prop("""    ( proplist ( 
        ( string "Name" (  ) ( data "ParticleBlocker" ) )
        ( vector "Pos" ( distance ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )
        ( rotation "Rotation" (  ) ( data ( eulerangles (0.000000 0.000000 0.000000) ) ) )
        ( longint "RenderGroup" (  ) ( data 0.000000 ) )
        ( string "Type" ( staticlist ) ( data "ParticleBlocker" ) )
        ( string "Lighting" ( staticlist ) ( data "Flat" ) )
        ( bool "NotAStep" (  ) ( data 0 ) )
        ( bool "Detail" (  ) ( data 0 ) )
        ( longint "LightControl" ( groupowner group1 ) ( data 0.000000 ) )
        ( string "TextureEffect" ( textureeffect ) )
        ( color "AmbientLight" ( group1 ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )
        ( longint "LMGridSize" ( group1 ) ( data 0.000000 ) )
        ( bool "ClipLight" ( group1 ) ( data 0 ) )
        ( bool "CastShadowMesh" ( group1 ) ( data 0 ) )
        ( bool "ReceiveLight" ( group1 ) ( data 0 ) )
        ( bool "ReceiveShadows" ( group1 ) ( data 0 ) )
        ( bool "ReceiveSunlight" ( group1 ) ( data 0 ) )
        ( real "LightPenScale" ( group1 ) ( data 0.000000 ) )
        ( real "CreaseAngle" ( group1 ) ( data 45.000000 ) )
    ) )""")

        b_idx = len(brush_props) + polyBlocks_count
        for _ in range(polypParticleblock_count):
            out_file.write('\t\t\t\t\t( worldnode\n')
            out_file.write('\t\t\t\t\t\t( type brush )\n')
            out_file.write(f'\t\t\t\t\t\t( brushindex {b_idx} )\n')
            out_file.write(f'\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
            out_file.write('\t\t\t\t\t\t( flags ( ) )\n')
            out_file.write('\t\t\t\t\t\t( properties\n')
            out_file.write('\t\t\t\t\t\t\t( name "Brush" )\n')
            out_file.write(f'\t\t\t\t\t\t\t( propid {pid_pblocker} )\n')
            out_file.write('\t\t\t\t\t\t)\n')
            out_file.write('\t\t\t\t\t)\n')
            b_idx += 1
            base_nodeid += 1
        out_file.write('\t\t\t\t) )\n')
        out_file.write("\t\t\t)\n")

    # 5. Occluders
    if occluders_all:
        out_file.write("\t\t\t( worldnode\n")
        out_file.write("\t\t\t\t( type null )\n")
        out_file.write(f'\t\t\t\t( label "Occluders" )\n')
        out_file.write(f'\t\t\t\t( nodeid {base_nodeid} )\n')
        out_file.write('\t\t\t\t( flags ( ) )\n')
        out_file.write('\t\t\t\t( properties\n')
        out_file.write(f'\t\t\t\t\t( propid 0 )\n')
        out_file.write('\t\t\t\t)\n')
        out_file.write('\t\t\t\t( childlist (\n')
        base_nodeid += 1

        b_idx = len(brush_props) + polyBlocks_count + polypParticleblock_count
        for idx, (verts, plane_normal, plane_dist, occ_hashcode) in enumerate(occluders_all):
            occ_name = f"Occluder_{occ_hashcode:08X}" 
            
            if occluder_names:
                for nm, hsh in occluder_names.items():
                    if hsh == occ_hashcode:
                        occ_name = nm
                        break

            pid_occ = add_prop(f"""    ( proplist ( 
        ( string "Name" (  ) ( data "{occ_name}" ) )
        ( vector "Pos" ( distance ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )
        ( rotation "Rotation" (  ) ( data ( eulerangles (0.000000 0.000000 0.000000) ) ) )
        ( longint "RenderGroup" (  ) ( data 0.000000 ) )
        ( string "Type" ( staticlist ) ( data "Occluder" ) )
        ( string "Lighting" ( staticlist ) ( data "Flat" ) )
        ( bool "NotAStep" (  ) ( data 0 ) )
        ( bool "Detail" (  ) ( data 0 ) )
        ( longint "LightControl" ( groupowner group1 ) ( data 0.000000 ) )
        ( string "TextureEffect" ( textureeffect ) )
        ( color "AmbientLight" ( group1 ) ( data ( vector (0.000000 0.000000 0.000000) ) ) )
        ( longint "LMGridSize" ( group1 ) ( data 0.000000 ) )
        ( bool "ClipLight" ( group1 ) ( data 0 ) )
        ( bool "CastShadowMesh" ( group1 ) ( data 0 ) )
        ( bool "ReceiveLight" ( group1 ) ( data 0 ) )
        ( bool "ReceiveShadows" ( group1 ) ( data 0 ) )
        ( bool "ReceiveSunlight" ( group1 ) ( data 0 ) )
        ( real "LightPenScale" ( group1 ) ( data 0.000000 ) )
        ( real "CreaseAngle" ( group1 ) ( data 45.000000 ) )
    ) )""")
            out_file.write('\t\t\t\t\t( worldnode\n')
            out_file.write('\t\t\t\t\t\t( type brush )\n')
            out_file.write(f'\t\t\t\t\t\t( brushindex {b_idx} )\n')
            out_file.write(f'\t\t\t\t\t\t( nodeid {base_nodeid} )\n')
            out_file.write('\t\t\t\t\t\t( flags ( ) )\n')
            out_file.write('\t\t\t\t\t\t( properties\n')
            out_file.write('\t\t\t\t\t\t\t( name "Brush" )\n')
            out_file.write(f'\t\t\t\t\t\t\t( propid {pid_occ} )\n')
            out_file.write('\t\t\t\t\t\t)\n')
            out_file.write('\t\t\t\t\t)\n')
            b_idx += 1
            base_nodeid += 1
        out_file.write('\t\t\t\t) )\n')
        out_file.write('\t\t\t)\n')

    out_file.write("\t\t) )\n")
    out_file.write("\t)\n")
    out_file.write(") )\n")

    # ======================== GLOBAL PROPLIST ========================
    out_file.write("( globalproplist (\n")
    out_file.write("    ( proplist ( \n    ) )\n")
    for prop in proplist_blocks:
        out_file.write(prop + "\n")
    out_file.write(") )\n")
    
