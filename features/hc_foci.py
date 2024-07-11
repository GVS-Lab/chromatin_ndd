import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import scipy.ndimage as ndi
import skimage.measure as measure
from scipy.ndimage import binary_erosion
from skimage.morphology import disk, remove_small_objects
from skimage.filters import threshold_multiotsu
from skimage.measure import regionprops_table, marching_cubes

from features import gamma_foci as gf

def measure_hc_foci_features(
        intensity: np.ndarray, 
        mask: np.ndarray,
        foci_thresh:int=256,
        channel:int=0,
        alpha:float = 1.0):
    """Compute all hc_foci features
    This function computes all features of the hc foci by calling the gamma_foci function with adjusted parameters. 
    Args:
        intensity : intensity image
        mask  : mask of nucleus
        foci_thresh : threshold for foci detection
        channel : dapi channel in intensity
        alpha: for calculating heterochromatin threshold
    Returns: A pandas dataframe with all the features for the given image
    """

    int_dapi = intensity[:,:,:,channel]
    foci_thresh = np.mean(int_dapi[mask]) + (alpha * np.std(int_dapi[mask]))

    feat = gf.measure_foci_features(intensity, mask, foci_thresh, foci_name = 'hc_foci', seg_watershed=False, channel = channel, foci_radial_dist=False, foci_lamin_coloc=False, foci_hc_coloc=False)
   
    return pd.DataFrame([feat])