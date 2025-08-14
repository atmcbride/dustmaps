from __future__ import print_function, division

import numpy as np
import h5py
import os

import astropy.coordinates as coordinates
import astropy.units as units

from .std_paths import *
from .map_base import DustMap, ensure_flat_galactic, ensure_flat_coords
# from .unstructured_map import UnstructuredDustMap
from . import fetch_utils

class Vergely2022Query(DustMap):
    """
    The 3D dust map produced by Vergely et al. (2022). 

    For details on how to use this map, see the original publication:
    https://ui.adsabs.harvard.edu/abs/2022A%26A...664A.174V/abstract

    We default to using the highest spatial resolution map. 

    The data is deposited at Vizier: https://cdsarc.u-strasbg.fr/viz-bin/cat/J/A+A/664/A174


    """

    def __init__(self, map_fname = None):
        if map_fname is None:
            map_fname = '/uufs/astro.utah.edu/common/home/u1371365/dustmaps_data/vergely2022/vergely22_extinction_density_resol_010pc.h5'
        
        with h5py.File(map_fname, 'r') as f:
            data = np.array(f['explore']['cube_datas'])
            self._data = data #np.swapaxes(data, 0, 1)

        
        self._xyz0 = (-1500, -1500, -400) # has resolution of 10pc
        self._shape = self._data.shape


    def _coords2idx(self, coords):
        c = coords.transform_to('galactic').represent_as('cartesian')
        
        idx = np.empty((3,) + c.shape, dtype='i4')
        mask = np.zeros(c.shape, dtype=np.bool)

        for i,x in enumerate((c.x, c.y, c.z)):
            idx[i,...] = (np.floor((x.to('pc').value - self._xyz0[i]) / 5)).astype(int)
            mask |= (idx[i] < 0) | (idx[i] >= self._shape[i])

        for i in range(3):
            idx[i, mask] = -1

        return idx, mask
    
    @ensure_flat_coords
    def query(self, coords):
        """
        Returns the extinction density (in e-foldings / kpc, in Gaia 550 nm)
        at the given coordinates.

        Args:
            coords (:obj:`astropy.coordinates.SkyCoord`): Coordinates at which
                to query the extinction. Must be 3D (i.e., include distance
                information).
            component (str): Which component to return. Allowable values are
                'mean' (for the mean extinction density) and 'std' (for the
                standard deviation of extinction density). Defaults to 'mean'.

        Returns:
            The extinction density, in units of e-foldings / pc, as either a
            numpy array or float, with the same shape as the input
            :obj:`coords`.
        """
        idx, mask = self._coords2idx(coords) 

        v = self._data[idx[0], idx[1], idx[2]].astype(float)
        
        if np.any(mask):
            # Set extinction to NaN for out-of-bounds (x, y, z)
            v[mask] = np.nan

        return v