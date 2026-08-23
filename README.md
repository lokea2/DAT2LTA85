# dat2lta85

**dat2lta85** is a converter from **DAT → LTA** map format for games built on the **Lithtech Jupiter v85 engine** (Crossfire, Combat Arms, NOLF2, Face of Mankind, Tron 2.0, Contract Jack, DOB and others).

The project is written in **Python** and inspired by:
- [Jldevictoria/lithdat](https://github.com/Jldevictoria/lithdat)
- [jsj2008/lithtech PreProcessor](https://github.com/jsj2008/lithtech/tree/master/tools/PreProcessor)

I am not a professional programmer, so the code is kept in a single file. My goal was simply to make it possible to convert DAT maps into LTA. I hope this tool helps other developers create more advanced converters and supports the Lithtech community.

---

## Requirements

If you're using the **exe build**, nothing to install — all dependencies are already bundled.

If you're running the **Python script** (`dat2lta85.py`), you'll need:

- **Python 3.10** or newer (the code uses modern type-hint syntax).
- Packages:
  ```bash
  pip install numpy tqdm
  ```

The rest of the modules (`lth_*.py`) rely only on the Python standard library.

---

## Building the accelerator (lth_accel)

Finding physics-data donors for the render geometry (matching triangles, restoring UVs and brush properties) is the heaviest part of the conversion. To speed it up, the project includes a C module, `lth_accel.c`, which compiles into a shared library and is loaded via ctypes.

If the library isn't built, the converter still works, just noticeably slower (significantly so on large maps). Build it once with:

```bash
python build_accel.py
```

The script auto-detects a compiler (`gcc`/`clang`/`cc` on Linux and macOS, MinGW-w64 or `cl` from Visual Studio on Windows) and places the result next to the scripts:

- **lth_accel.so** — on Linux/macOS;
- **lth_accel.dll** — on Windows.

Requirements:
- **Linux**: `sudo apt install gcc`.
- **macOS**: `xcode-select --install`.
- **Windows**: MinGW-w64 (recommended, via MSYS2: `pacman -S mingw-w64-x86_64-gcc`) or Visual Studio — in that case run the script from the matching Native Tools console (x64/x86), matching your Python's bitness.

Once built, just run the converter as usual — the library is loaded automatically.

---

## Usage

1. Extract resources from the game (rez) and place them into the **Game** folder of the DEdit editor.
2. Copy maps into the **Worlds** folder.
3. Place the converter itself into the **Worlds** folder as well.
4. Open a console and navigate to this folder.
5. Run:

If using the exe build:
```bash
dat2lta85 %mapname%.dat -v1
```

If using the Python script:
```bash
python dat2lta85.py %mapname%.dat -v1
```

`-v1` — use this for DAT maps with tangent and binormal (Combat Arms, Crossfire, NOLF2, etc).

`-v2` — use this for DAT maps without tangent and binormal (Contract Jack, Tron 2.0, etc).

---

## Output

After conversion, three files will appear in the folder:

- **mapname.lta** — main triangulated map with UVs. This file is for **debugging only**: it's a convenient way to look at the render section of the map and spot problem areas, if any, but it should not be edited in DEdit.
- **mapname_PhysicsDATA.lta** — contains the WorldTree section. It is a non-triangulated version of the original with correct brush properties. UV coordinates are restored for it as well, **including the second UV layer**, if the surface has one (used for blending two texture layers), so this file is fully usable for editing. **This is the file you should use for editing in DEdit** — treat it as the primary file.
- **mapname.txt** — debug information about the world, useful for identifying conversion issues.

---

## Important details

- `mapname.lta` preserves **TextureEffect** for each brush if present.
- `mapname_PhysicsDATA.lta` contains the remaining brush properties along with restored UVs (including a second UV layer for blending texture layers on a surface, if it was present in the original) and is the primary file for editing — there's no longer any need to manually copy anything over from `mapname.lta`.
- All **RenderNode** and **WorldModel** objects are automatically assigned `Detail 0` and `AmbientLight 0 0 0`. Below is a way to compute proper AmbientLight values instead of zeros.

---

## Computing AmbientLight

AmbientLight can now be calculated instead of being left at zero. Steps:

1. Convert the original DAT map to LTA the normal way (`dat2lta85 %mapname%.dat -v1`) — this produces a `mapname.txt` with the original's data (in the example below — `JUNK_FLEA.txt`).
2. Open `mapname_PhysicsDATA.lta` in DEdit. Since every object already has `AmbientLight 0 0 0`, this version of the map will have zero ambient across all sections.
3. Compile this map back into a `.dat` from DEdit.
4. Decompile the resulting `.dat` with the converter again — this creates a new `.txt` (in the example — `jftest.txt`), which will contain, among other things, the AmbientLight for each surface (already zero).
5. Compare both `.txt` files with:

```bash
dat2lta85 -comp JUNK_FLEA.txt jftest.txt
```

where:
- `JUNK_FLEA.txt` — the file obtained from the original map (step 1);
- `jftest.txt` — the file obtained from the zero-AmbientLight version (step 4).

This produces a `JUNK_FLEA_ambient.txt` file — a report of the recommended AmbientLight values for each surface. Example content:

```
Surface 2       map3/train/tra_ob02_01_op.dtx                 [1356 verts]
  => Recommended Ambient: RGB(0.3608, 0.2627, 0.1882)  [B:48 G:67 R:92]
     1. B: 48  G: 67  R: 92   ( 744 verts)
     2. B:  0  G:  0  R:  0   ( 354 verts) - ZERO
     3. B: 46  G: 63  R: 86   ( 159 verts)   <-- rejected (noise)
     4. B: 38  G:  7  R:  0   (  10 verts)   <-- rejected (noise)
     5. B: 48  G: 67  R: 67   (   4 verts)   <-- rejected (noise)
```

The surface number (`Surface N`) matches the surface numbering in `mapname_PhysicsDATA.lta`, which lets you unambiguously map the recommended values to the correct objects and restore AmbientLight manually.

---

## Restoring LMGridSize

`LMGridSize` controls the lightmap density on a surface and, as a result, the thickness/sharpness of shadows on the map. By default the converter sets `LMGridSize 0` on every brush, but the real value can be estimated and restored manually.

`LMGridSize` is estimated automatically for every render section that uses a lightmap, and the estimate is written into the regular `mapname.txt` (the one produced when converting the original DAT map — step 1 of the AmbientLight section above). Look for blocks like this:

```
  [LMGridSize] Section 0 (map3/lightanim01, 256×256) — 812 tris
    Poly   12 | Edges: [4.2 (134.2u/32.1px), 3.8 (118.7u/31.4px), SKIP] -> Avg: 4.00
    ...
    -> LMGridSize: median=4.12  (min=3.80, max=4.60)  -> rounded to 4
```

Where:
- **Section N** — the render-section index within a `RenderNode`/`WorldModelNode` (matches the texture that uses the lightmap);
- **median / min / max** — statistics across all triangles in the section;
- **rounded to N** — the final recommended `LMGridSize` value, to be set manually on the corresponding brush in `mapname_PhysicsDATA.lta` while editing in DEdit.

Sections without a lightmap (regular Gouraud/DualTexture, etc.) are not included in this estimate — `LMGridSize` isn't computed or needed for them.

---

## Working with FX

For proper FX handling, place additional files into the **Game** folder of DEdit.

### For Combat Arms

This repository provides example FX files and maps to illustrate proper usage:

- **JUNK_FLEA.7z** — original map file for Combat Arms.
  [Download JUNK_FLEA.7z](https://www.mediafire.com/file/kwp9cdcf6pef9yl/JUNK_FLEA.7z/file)

- **CAclassic_DEdit.7z** — DEdit project files for Combat Arms.
  [Download CAclassic_DEdit.7z](https://www.mediafire.com/file/7zbv05ozz6fwrea/CAclassic_DEdit.7z/file)

Place these files into your DEdit project folder to test and convert maps.

### For Crossfire

This repository provides example FX files and maps to illustrate proper usage:

- **MEXICO.lta** — example map file for Crossfire. This is not the original map, but a version adjusted by me to serve as a reference for usage and testing in DEdit.
  [Download TRANSPORTSHIP_PhysicsDATA.lta](https://www.mediafire.com/file/bpvijbcfs8uwove/TRANSPORTSHIP_PhysicsDATA.lta/file)

- **Crossfire_Dedit.7z** — DEdit project files for Crossfire.
  [Download Crossfire_Dedit.7z](https://www.mediafire.com/file/8im90f64u0q5497/Crossfire_Dedit.7z/file)

Place these files into your DEdit project folder to test and convert maps.

### For Other Lithtech Jupiter Games

For other Lithtech Jupiter games, place the corresponding files from your game into the DEdit project, for example:

- `ClientFX.fxd`
- `CShell.dll`
- `Object.lto`
- Folders `ClientFX` and `Attributes` (if present)

All files must be in decrypted form to work correctly.

- If **CLIENTFX.FXF** and **CLIENTFX.FCF** contain valid FX, you can safely click **Yes** when opening the map in DEdit.
- If those FX are missing, choosing **Yes** will make DEdit delete all FX names from the map.
- If you choose **No**, the FX names will remain untouched.

---

## Dynamic Occluders

The converter can also detect **Occluder** names.
- If the occluder is named **"Occluder"**, it is a regular occluder.
- If the name differs, it means it is used by a **DynamicOccluder** object.

---

I hope this tool will be a helpful step for modders and bring new life into projects built on Lithtech Jupiter.
