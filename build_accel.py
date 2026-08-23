"""
build_accel.py — Compile lth_accel.c into a shared library.

Supports gcc/clang (Linux, macOS, Windows via MinGW-w64) and MSVC.
The output is placed next to this script as ``lth_accel.so`` (Unix) or
``lth_accel.dll`` (Windows).

Usage:
    python build_accel.py
"""

import os
import platform
import shutil
import struct
import subprocess
import sys

HERE        = os.path.dirname(os.path.abspath(__file__))
SRC         = os.path.join(HERE, "lth_accel.c")
PLATFORM    = platform.system()
PYTHON_BITS = struct.calcsize("P") * 8
OUT         = os.path.join(HERE, "lth_accel.dll" if PLATFORM == "Windows" else "lth_accel.so")


def _find_compiler() -> tuple[str, str]:
    """Return ``(compiler_path, kind)`` where *kind* is ``"gcc_like"`` or ``"msvc"``."""
    for cc in ("gcc", "clang", "cc"):
        if shutil.which(cc):
            return cc, "gcc_like"
    if PLATFORM == "Windows" and shutil.which("cl"):
        return "cl", "msvc"
    raise RuntimeError(
        "No C compiler found.\n\n"
        "  Recommended (Windows): install MinGW-w64 via MSYS2\n"
        "    winget install -e --id=MSYS2.MSYS2\n"
        "    Then in MSYS2: pacman -S mingw-w64-x86_64-gcc\n"
        "    Add its bin/ directory to PATH.\n\n"
        "  Alternative (Windows): open the correct Visual Studio Native Tools prompt\n"
        "    64-bit Python → 'x64 Native Tools Command Prompt for VS 20xx'\n"
        "    32-bit Python → 'x86 Native Tools Command Prompt for VS 20xx'\n\n"
        "  Linux: sudo apt install gcc\n"
        "  macOS: xcode-select --install"
    )


def _check_msvc_bitness() -> None:
    """Verify that the active cl.exe target matches the running Python bitness."""
    result = subprocess.run(["cl"], capture_output=True, text=True, errors="replace")
    banner = (result.stderr + result.stdout).lower()
    cl_bits = 32 if " for x86" in banner else 64

    print(f"[build_accel] cl.exe target: {cl_bits}-bit")
    if cl_bits != PYTHON_BITS:
        print(
            f"\n[build_accel] ERROR: Python is {PYTHON_BITS}-bit but cl.exe targets "
            f"{cl_bits}-bit — ctypes cannot load the resulting DLL.\n\n"
            f"  Open the correct Developer Command Prompt and retry:\n"
            f"    64-bit Python → 'x64 Native Tools Command Prompt for VS 20xx'\n"
            f"    32-bit Python → 'x86 Native Tools Command Prompt for VS 20xx'\n"
        )
        sys.exit(1)


def _build_gcc(cc: str) -> list[str]:
    flags = [f"-m{PYTHON_BITS}", "-O3", "-ffast-math", "-shared", "-fPIC"]
    if PLATFORM == "Darwin":
        flags.append("-dynamiclib")
    if PLATFORM != "Windows":
        flags.append("-march=native")
    return [cc] + flags + ["-o", OUT, SRC, "-lm"]


def _build_msvc() -> list[str]:
    _check_msvc_bitness()
    # /MD  — dynamic CRT (required for ctypes compatibility)
    # /LD  — produce a DLL
    # /GL  — whole-program optimisation
    return ["cl", "/O2", "/GL", "/MD", f"/Fe:{OUT}", "/LD", SRC]


def build() -> None:
    cc, kind = _find_compiler()
    print(f"[build_accel] Compiler:  {cc} ({kind})")
    print(f"[build_accel] Python:    {PYTHON_BITS}-bit")
    print(f"[build_accel] Platform:  {PLATFORM}")
    print(f"[build_accel] Source:    {SRC}")
    print(f"[build_accel] Output:    {OUT}")

    cmd = _build_gcc(cc) if kind == "gcc_like" else _build_msvc()
    print(f"[build_accel] Command:   {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        print("=== BUILD FAILED ===")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    if not os.path.isfile(OUT):
        print("[build_accel] Output file was not created.")
        sys.exit(1)

    size_kb = os.path.getsize(OUT) / 1024
    print(f"[build_accel] Done — {size_kb:.1f} KB → {OUT}")


if __name__ == "__main__":
    build()
