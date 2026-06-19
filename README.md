# GMCHIIR

Statistical analysis tools for studying molecular gas (GMCs) around HII regions in nearby galaxies.

These scripts were written to support the paper:

> **"The impact of H II regions on giant molecular cloud properties in nearby galaxies sampled by PHANGS ALMA and MUSE"
> Zakardjian, A., Pety J., Herrera C. N, et al. A&A 2023

---

## Context: the PHANGS survey

This code is tightly coupled to the data products of the **PHANGS** (Physics at High Angular resolution in Nearby GalaxieS) survey.  It is **not** a general-purpose toolkit — the column names, catalog formats, file-naming conventions, and physical assumptions all follow PHANGS conventions.  If you want to apply similar analysis to other data, you will need to adapt the catalog readers and column mappings.

### What is PHANGS?

PHANGS is a multi-wavelength survey of ~100 nearby star-forming galaxies at high angular resolution.  The two data products consumed here are:

| Product | Description | Used here as |
|---|---|---|
| **PHANGS-ALMA GMC catalogs** (CPROPS) | Molecular cloud properties extracted from CO(2-1) cubes | `dirgmc` |
| **PHANGS-Muse HII-region catalogs** | HII-region properties from optical IFU data | `dirhii` |

### PHANGS-specific conventions in the code

- **GMC catalog filenames** follow the pattern `<galaxy>_co21_<resolution>_props.fits` (e.g. `ngc0628_co21_120pc_props.fits`).  The suffix is read from `name_gmc.txt` so it can be changed without editing the scripts.
- **Resolution variants** supported: native, 120 pc homogenized (`_co21_120pc_props`), 150 pc (`_co21_v150pc_props`).  Set `typegmc` at the top of `Extract_info_plot_per_gal.py`.
- **GMC catalog columns** (CPROPS output): `XCTR_DEG`, `YCTR_DEG`, `SIGV_KMS`, `RAD_PC`, `MLUM_MSUN`, `VIRPARAM`, `SURFDENS`, `TMAX_K`, `TFF_MYR`, `FLUX_KKMS_PC2`, `S2N`, `DISTANCE_PC`, …
- **HII-region catalog columns**: `RA`, `DEC`, `CLHA` (extinction-corrected Hα luminosity), `SIZE`, `HA_SIG`, `METAL_SCAL`, `OFF_SCAL`, `BPTFLAG`, `DISTMPC`, `PA`, `INCL`, `RA_CENTER`, `DEC_CENTER`, `PHANGS_INDEX`, …
- **BPTFLAG == 0** selects unambiguously star-forming HII regions (as defined in the PHANGS-Muse catalog paper).
- Two galaxies (**NGC 1672** and **IC 5332**) are missing the `MLUM_MSUN` column in the CPROPS catalogs used for this work; the code derives mass from `FLUX_KKMS_PC2` with an α_CO conversion factor of 4.3 M☉ (K km/s pc²)⁻¹ / 0.69.
- For **NGC 1672** the central bar region (within 15 arcsec of the nucleus) is excluded from the analysis.

---

## Scripts

### `Extract_info_plot_per_gal.py`
Main pipeline script.  For each galaxy it:
1. Reads the HII-region and GMC catalogs.
2. Deprojects sky coordinates into the galaxy plane (correcting for inclination and position angle).
3. Pairs each GMC with its nearest HII region using `checkdist()`.
4. Flags pairs where the GMC and HII region physically overlap (separation < sum of radii).
5. Produces per-galaxy histogram PDFs and correlation-plot PDFs.
6. Saves intermediate results to pickle files for use by the plotting scripts.

**Outputs:**
- `Histograms_all_GMC<suffix>.pdf`
- `Correlations_galbygal_GMC<suffix>.pdf`
- `Clouds_HIIregions_positions_<galaxy><suffix>.pickle` (one per galaxy)
- `Galaxies_paired_GMC<suffix>.pickle`
- `Galaxies_variables_all_GMC<suffix>.pickle`
- `Galaxies_variables_notover_GMC<suffix>.pickle`
- `Table1.txt` (LaTeX-formatted summary table)
- DS9 region files under `../ds9tables/`

### `get_limits.py`
Pre-computes the axis limits shared across all per-galaxy correlation plots so that all galaxies use the same scale.  Must be run once before `Extract_info_plot_per_gal.py` if you want consistent axes.

**Output:** `limits_properties.pickle`

### `plot_corr_point_all_gals.py`
Reads the pickle files produced by `Extract_info_plot_per_gal.py` and makes the all-galaxy combined correlation plots: scatter points, binned medians, and linear regression.

**Output:** `Correlations_allgals_GMC<suffix>.pdf`

### `plots_correlations_2d_contours.py`
Same as above but uses 2D density contours instead of individual points, useful for crowded parameter spaces.

### `matching.py`
Drop-in replacement for the `checkdist()` function that uses `scipy.spatial.cKDTree` for nearest-neighbour matching instead of a Python loop.  **10–200× faster** on typical PHANGS catalog sizes with identical output.  See [Performance](#performance) below.

### `name_gmc.txt`
Plain-text file containing the GMC catalog filename suffix (e.g. `_12m+7m+tp_co21_120pc_props`).  Edit this file to switch between catalog versions without touching any Python script.

---

## How to run

```bash
# 1. Compute shared plot limits (optional but recommended)
python get_limits.py

# 2. Main extraction and per-galaxy plots
python Extract_info_plot_per_gal.py

# 3. All-galaxy combined plots
python plot_corr_point_all_gals.py
python plots_correlations_2d_contours.py
```

Before running, if needed, edit the path variables near the top of `Extract_info_plot_per_gal.py`:

```python
dirhii  = "../../Catalogs-HIIregions/hii_region_enhanced_catalogs/"
dirgmc  = "../../Catalogs-CPROPS/"
typegmc = "ST1p5/homogenized"   # or "ST1p5/native"
```

And set the catalog suffix in `name_gmc.txt`, e.g.:
```
_12m+7m+tp_co21_120pc_props
```

---

## Requirements

These scripts were written and run with **Python 2.7** using the package versions available in 2019–2020.

> ⚠️ **Python 2 is end-of-life.** If you want to run this code today you will need to port it to Python 3.  The main changes required are: `print` statements → `print()` calls, opening pickle files in binary mode (`"rb"`/`"wb"`), and a few integer-division edge cases.  `matching.py` is already compatible with both Python 2.7 and Python 3.x.


See `requirements.txt` for a modern Python 3 equivalent.

---

