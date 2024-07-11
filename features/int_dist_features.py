'''
adapted from chrometrics: https://github.com/GVS-Lab/chrometrics and chromark (radial distribution) 

'''

# Import modules
import numpy as np
import pandas as pd
from scipy import stats
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis, skew
from skimage import measure
from scipy.interpolate import interp1d
import scipy.ndimage as ndi

import warnings
warnings.filterwarnings("ignore")

def hetero_euchro_measures(regionmask: np.ndarray, intensity: np.ndarray, alpha: float = 1):
    """Computes Heterochromatin to Euchromatin features
    
    This functions obtains the Heterochromatin (high intensity) and Euchromatin (low intensity)
    and computes features that describe the relationship between the two
    
    Args:
        regionmask : binary background mask
        intensity  : intensity image
        alpha     : threshold for calculating heterochromatin intensity
    """
    high, low = np.percentile(intensity[regionmask], q=(80, 20))
    hc = np.mean(intensity[regionmask]) + (alpha * np.std(intensity[regionmask]))

    feat = {
        "i80_i20": high / low,
        "nhigh_nlow": np.sum(intensity[regionmask] >= high)/ np.sum(intensity[regionmask] <= low),
        "hc_area_ec_area": np.sum(intensity[regionmask] >= hc) / np.sum(intensity[regionmask] < hc),
        "hc_area_nuc_area": np.sum(intensity[regionmask] >= hc) / np.sum(intensity[regionmask] > 0),
        "hc_content_ec_content": np.sum(np.where(intensity[regionmask] >= hc, intensity[regionmask], 0))
            / np.sum(np.where(intensity[regionmask] < hc, intensity[regionmask], 0)),
        "hc_content_dna_content": np.sum(np.where(intensity[regionmask] >= hc, intensity[regionmask], 0))
            / np.sum(np.where(intensity[regionmask] > 0, intensity[regionmask], 0))
    }
    return feat


def intensity_histogram_measures(regionmask: np.ndarray, intensity: np.ndarray):
    """Computes Intensity Distribution features
    
    This functions computes features that describe the distribution characteristic of the intensity.
    
    Args:
        regionmask : binary background mask
        intensity  : intensity image

    """
    feat = {
        "int_min": np.percentile(intensity[regionmask], 0),
        "int_d25": np.percentile(intensity[regionmask], 25),
        "int_median": np.percentile(intensity[regionmask], 50),
        "int_d75": np.percentile(intensity[regionmask], 75),
        "int_max": np.percentile(intensity[regionmask], 100),
        "int_mean": np.mean(intensity[regionmask]),
        "int_mode": stats.mode(intensity[regionmask], axis=None, keepdims=True)[0][0],
        "int_sd": np.std(intensity[regionmask]),
        "kurtosis": float(kurtosis(intensity[regionmask].ravel())),
        "skewness": float(skew(intensity[regionmask].ravel())),
        "entropy": shannon_entropy((intensity * regionmask)),

    }
    return feat

def radial_distribution_measures(image, mask):
    def get_selem_z_xy_resolution(k: int = 7):
        selem = np.zeros([2 * k + 1, 2 * k + 1])
        selem[k, :] = 1
        selem[:, k] = 1
        selem_2 = np.zeros([2 * k + 1, 2 * k + 1])
        selem_2[k, k] = 1
        selem = np.stack([selem_2, selem, selem_2], axis=0)
        return selem


    def get_radial_distribution(image, mask, bins=10):

        selem = get_selem_z_xy_resolution(7)
        rp_masks = get_radial_profile_masks(mask, selem=selem)

        rdp = []
        for i in range(len(rp_masks)):
            rdp.append(np.sum(image * np.uint8(rp_masks[i])))
        if 0 in rdp:
            rdp = np.array(rdp[: rdp.index(0) + 1])
        else:
            rdp = np.array(rdp)
        total_int = rdp[0]
        rdp = rdp / total_int
        rdp = rdp[::-1]

        radii = np.linspace(0, bins, num=len(rdp))
        radii_new = np.linspace(0, bins, num=bins)
        spl = interp1d(radii, rdp)
        rdp_interpolated = spl(radii_new)
        return rdp_interpolated


    def get_radial_profile_masks(mask, selem):
        masks = [mask]
        for i in range(len(mask)):
            masks.append(ndi.binary_erosion(masks[-1], selem))
        return masks
    
    rd = get_radial_distribution(image, mask, bins=10)

    feat = {f"rd_{i}": rd[i] - rd[i-1] for i in range(1, len(rd))}

    return(feat)



def measure_intensity_features(regionmask: np.ndarray, intensity: np.ndarray, measure_int_dist:bool = True, measure_hc_ec_ratios:bool = True, measure_radial_dist:bool = True, hc_alpha: int = 1):
    """Compute all intensity distribution features
    This function computes all features that describe the distribution of the gray levels. 
    Args:
        regionmask : binary background mask
        intensity  : intensity image
    Returns: A pandas dataframe with all the features for the given image
    """

    feat ={}

    if(measure_int_dist):
        feat.update(intensity_histogram_measures(regionmask, intensity))
    if(measure_hc_ec_ratios):
        feat.update(hetero_euchro_measures(regionmask, intensity, hc_alpha))
    if(measure_radial_dist):
        feat.update(radial_distribution_measures(intensity, regionmask))

    return pd.DataFrame([feat])
