# dat2lta85

**dat2lta85** is a converter from **DAT → LTA** map format for games built on the **Lithtech Jupiter v85 engine** (Crossfire, Combat Arms, NOLF2, Face of Mankind, Tron 2.0, Contract Jack, Dead or Alive Online and others).

The project is written in **Python** and inspired by:  
- [Jldevictoria/lithdat](https://github.com/Jldevictoria/lithdat)  
- [jsj2008/lithtech PreProcessor](https://github.com/jsj2008/lithtech/tree/master/tools/PreProcessor)  

I am not a professional programmer, so the code is kept in a single file. My goal was simply to make it possible to convert DAT maps into LTA. I hope this tool helps other developers create more advanced converters and supports the Lithtech community.  

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

-v1 — use this for DAT maps with tangent and binormal (Combat Arms, Crossfire, NOLF2, etc).

-v2 — use this for DAT maps without tangent and binormal (Contract Jack, Tron 2.0, etc).  

---

## Output

After conversion, three files will appear in the folder:  

- **mapname.lta** — main triangulated map with UVs. This is the primary file for editing and compiling.  
- **mapname_PhysicsDATA.lta** — contains the WorldTree section. It is a non-triangulated version of the original with proper brush properties but without UVs. Ideally, brush properties should be copied from here into the main LTA manually.  
- **mapname.txt** — debug information about the world, useful for identifying conversion issues.  

---

## Important details

- `mapname.lta` preserves **TextureEffect** for each brush if present.  
- `mapname_PhysicsDATA.lta` contains other brush properties.  
- All **RenderNode** and **WorldModel** objects are automatically assigned `Detail 0` and `AmbientLight 0 0 0`. For more accurate values, refer to the original **JUNK_FLEA** map which is available in open form.  
<img width="882" height="570" alt="13" src="https://github.com/user-attachments/assets/26ce72f5-ee19-4e55-9c20-4585195c293c" />

---

## Working with FX

For proper FX handling, place additional files into the **Game** folder of DEdit.  

For **Crossfire** and **Combat Arms**, this repository provides example FX files and maps to illustrate proper usage. You can place these files directly into your DEdit project for testing and conversion.  

For other Lithtech Jupiter games, you should place the corresponding files from your game into the DEdit project, for example:  
- `ClientFX.fxd`  
- `CShell.dll`  
- `Object.lto`  
- Folders `ClientFX` and `Attributes` (if present)  

All files must be in decrypted form to work correctly.  

- If **CLIENTFX.FXF** and **CLIENTFX.FCF** contain valid FX, you can safely click **Yes** when opening the map in DEdit.  
- If those FX are missing, choosing **Yes** will make DEdit delete all FX names from the map.  
- If you choose **No**, the FX names will remain untouched. 
<img width="262" height="119" alt="12" src="https://github.com/user-attachments/assets/b4f47250-1088-4b6a-ad8b-5eb03a6c586f" />

---

## Dynamic Occluders

The converter can also detect **Occluder** names.  
- If the occluder is named **"Occluder"**, it is a regular occluder.  
- If the name differs, it means it is used by a **DynamicOccluder** object.  

---

I hope this tool will be a helpful step for modders and bring new life into projects built on Lithtech Jupiter.
