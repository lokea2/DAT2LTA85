"""
lth_types.py — Shared primitive types for the LithTech DAT parser.

Provides Vec3, Vec2, Vec3u, Color, Quaternion, Vertex, and helpers for
reading length-prefixed strings and computing the LithTech case-insensitive
hash used by DynamicOccluderVolume names.
"""

import os
import sys
import struct
from collections import namedtuple


def _get_parent_dir() -> str:
    """Return the parent of the directory that contains this file (or the executable)."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(base)


#: Root search directory used by get_dtx_texture_size in lth_export.py.
current_dir: str = _get_parent_dir()


# ---------------------------------------------------------------------------
# String I/O
# ---------------------------------------------------------------------------

def read_lithtech_string(f) -> str:
    """Read a uint16-prefixed UTF-8 string from *f*."""
    (length,) = struct.unpack("<H", f.read(2))
    if length == 0:
        return ""
    return f.read(length).decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------

def st_gethash_ic(s: str) -> int:
    """Case-insensitive LithTech string hash (used for occluder name lookup)."""
    n = 0
    for ch in s:
        val = ord(ch.upper()) - ord('A')
        n = (n * 29 + (val & 0xFFFF_FFFF)) & 0xFFFF_FFFF
    return n


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

class Vec3:
    """Three-component float vector."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x, self.y, self.z = x, y, z

    @staticmethod
    def read(f) -> "Vec3":
        return Vec3(*struct.unpack("<3f", f.read(12)))

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def inverse(self) -> "Vec3":
        def _safe_inv(v: float) -> float:
            return 1.0 / v if v != 0.0 else 0.0
        return Vec3(_safe_inv(self.x), _safe_inv(self.y), _safe_inv(self.z))

    def __repr__(self) -> str:
        return f"({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __str__(self) -> str:
        return f"({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"


class Vec3u:
    """Three-component unsigned-integer vector."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: int = 0, y: int = 0, z: int = 0):
        self.x, self.y, self.z = x, y, z

    @classmethod
    def read(cls, f) -> "Vec3u":
        return cls(*struct.unpack("<3I", f.read(12)))

    def __repr__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"


class Vec2:
    """Two-component float vector (typically UV coordinates)."""

    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x, self.y = x, y

    @classmethod
    def read(cls, f) -> "Vec2":
        return cls(*struct.unpack("<2f", f.read(8)))

    def __repr__(self) -> str:
        return f"({self.x:.4f}, {self.y:.4f})"


class Color:
    """RGB colour stored as a Vec3 (components in [0, 1])."""

    @classmethod
    def read(cls, f) -> Vec3:
        return Vec3.read(f)


class Quaternion:
    """Unit quaternion (x, y, z, w)."""

    __slots__ = ("x", "y", "z", "w")

    def __init__(self, x: float, y: float, z: float, w: float):
        self.x, self.y, self.z, self.w = x, y, z, w

    @classmethod
    def read(cls, f) -> "Quaternion":
        return cls(*struct.unpack("<4f", f.read(16)))

    def __repr__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f}, {self.w:.3f})"


# ---------------------------------------------------------------------------
# Composite types
# ---------------------------------------------------------------------------

class SectionInfo:
    """Lightweight container for a render-section's texture and triangle count."""

    def __init__(self, texture: str, triangle_count: int):
        self.texture = texture
        self.triangle_count = triangle_count


Vertex = namedtuple("Vertex", ["pos", "normal", "uv", "uv1", "color"])
