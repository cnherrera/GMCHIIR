"""
matching.py -- Spatial pairing of GMCs and HII regions.

Compatible with Python 2.7 and Python 3.x.

Usage
-----
Replace the checkdist() function definition in Extract_info_plot_per_gal.py
with an import at the top of that file:

    from matching import checkdist

No other changes needed.
"""

from __future__ import print_function, division  # Python 2/3 compatibility

import numpy as np
from scipy.spatial import cKDTree


def _arcsec_to_pc(delta_arcsec, distance_kpc):
    """Convert angular separation in arcsec to physical distance in pc."""
    return np.radians(delta_arcsec / 3600.0) * distance_kpc * 1e6


def checkdist(xgalhii, ygalhii, xgalgmc, ygalgmc, sizehii, radgmc, distance):
    """
    Match GMCs to their nearest HII region and flag spatially overlapping pairs.

    For each GMC the nearest HII region is found.  When multiple GMCs map to
    the same HII region only the closest one is retained.  A pair is considered
    *overlapping* when the GMC-HII separation is smaller than the sum of their
    effective radii (sizehii + radgmc).

    Parameters
    ----------
    xgalhii : array_like, shape (N_hii,)
        Deprojected galactic-plane X position of HII regions in arcsec.
    ygalhii : array_like, shape (N_hii,)
        Deprojected galactic-plane Y position of HII regions in arcsec.
    xgalgmc : array_like, shape (N_gmc,)
        Deprojected galactic-plane X position of GMCs in arcsec.
    ygalgmc : array_like, shape (N_gmc,)
        Deprojected galactic-plane Y position of GMCs in arcsec.
    sizehii : array_like, shape (N_hii,)
        Effective radius of each HII region in pc.
    radgmc : array_like, shape (N_gmc,)
        Effective radius of each GMC in pc.
    distance : float
        Distance to the galaxy in kpc.

    Returns
    -------
    mindist : ndarray, shape (N_gmc,)
        Distance in pc from each GMC to its nearest HII region.
    inddist : ndarray, shape (N_gmc,), dtype int
        Index of the nearest HII region for each GMC.
    idovergmc : list of int
        Indices (into GMC arrays) of GMCs that overlap an HII region.
    idoverhii : list of int
        Indices (into HII arrays) of HII regions that overlap a GMC.
    idgmcalone : list of int
        Indices of GMCs with no overlapping HII region.
    idhiialone : list of int
        Indices of HII regions with no overlapping GMC.
    """
    xgalhii = np.asarray(xgalhii, dtype=float)
    ygalhii = np.asarray(ygalhii, dtype=float)
    xgalgmc = np.asarray(xgalgmc, dtype=float)
    ygalgmc = np.asarray(ygalgmc, dtype=float)
    sizehii = np.asarray(sizehii, dtype=float)
    radgmc  = np.asarray(radgmc,  dtype=float)

    n_gmc = len(xgalgmc)
    n_hii = len(xgalhii)

    # ------------------------------------------------------------------
    # 1. Nearest-neighbour query (vectorised -- no Python loop over GMCs)
    # ------------------------------------------------------------------
    hii_coords = np.column_stack([xgalhii, ygalhii])  # (N_hii, 2) arcsec
    gmc_coords = np.column_stack([xgalgmc, ygalgmc])  # (N_gmc, 2) arcsec

    tree = cKDTree(hii_coords)
    sep_arcsec, inddist = tree.query(gmc_coords, k=1)  # k=1: nearest neighbour

    inddist = inddist.astype(int)
    mindist = _arcsec_to_pc(sep_arcsec, distance)      # (N_gmc,) in pc

    # ------------------------------------------------------------------
    # 2. Resolve conflicts: multiple GMCs -> same HII region.
    #    Keep only the closest GMC for each HII region.
    # ------------------------------------------------------------------
    # best[hii_idx] = (best_gmc_idx, best_dist)
    best = {}
    for gmc_idx in range(n_gmc):
        hii_idx = inddist[gmc_idx]
        dist    = mindist[gmc_idx]
        if hii_idx not in best or dist < best[hii_idx][1]:
            best[hii_idx] = (gmc_idx, dist)

    # Unpack to ordered lists (sorted by hii_idx for reproducibility)
    idhii   = []
    idgmc   = []
    distmin = []
    for hii_idx in sorted(best.keys()):
        gmc_idx, dist = best[hii_idx]
        idhii.append(hii_idx)
        idgmc.append(gmc_idx)
        distmin.append(dist)

    idhii   = np.array(idhii,   dtype=int)
    idgmc   = np.array(idgmc,   dtype=int)
    distmin = np.array(distmin, dtype=float)

    # ------------------------------------------------------------------
    # 3. Flag overlapping pairs: separation < sizehii + radgmc
    # ------------------------------------------------------------------
    addsize   = sizehii[idhii] + radgmc[idgmc]
    over_mask = distmin < addsize

    idovergmc = idgmc[over_mask].tolist()
    idoverhii = idhii[over_mask].tolist()

    # ------------------------------------------------------------------
    # 4. Unpaired objects
    # ------------------------------------------------------------------
    idgmcalone = np.setdiff1d(np.arange(n_gmc), idovergmc).tolist()
    idhiialone = np.setdiff1d(np.arange(n_hii), idoverhii).tolist()

    return mindist, inddist, idovergmc, idoverhii, idgmcalone, idhiialone
