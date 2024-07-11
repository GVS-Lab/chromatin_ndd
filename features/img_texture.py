#Basic structure adapted from chrometrics and implemented glcm calculation for 3D images

# Import modules
import numpy as np
import pandas as pd
from skimage.feature import graycoprops
from skimage import measure 
from features.utils.graycomatrix import graycomatrix_3d

import warnings
warnings.filterwarnings("ignore")

def gclm_textures(regionmask: np.ndarray, intensity: np.ndarray, lengths=[1, 5, 20], angles=np.arange(4), resolution:float = [0.7, 0.09 ,0.09]):
    """ Compute GLCM features at given lengths
    
    Args:
        regionmask : binary background mask
        intensity  : intensity image
        lengths    : length scales 
     """
    # Contruct GCL matrix at given pixels lengths

    glcm = graycomatrix_3d(intensity * regionmask,
        distances=lengths,
        angles_1 = angles, angles_2 = angles
    )

    contrast = pd.DataFrame(np.mean(graycoprops(glcm, "contrast"), axis=1).tolist()).T
    contrast.columns = ["contrast_" + str(col) for col in lengths]
    
    dissimilarity = pd.DataFrame(
        np.mean(graycoprops(glcm, "dissimilarity"), axis=1).tolist()
    ).T
    dissimilarity.columns = ["dissimilarity_" + str(col) for col in lengths]
    
    homogeneity = pd.DataFrame(
        np.mean(graycoprops(glcm, "homogeneity"), axis=1).tolist()
    ).T
    homogeneity.columns = ["homogeneity_" + str(col) for col in lengths]
    
    ASM = pd.DataFrame(np.mean(graycoprops(glcm, "ASM"), axis=1).tolist()).T
    ASM.columns = ["asm_" + str(col) for col in lengths]
    
    energy = pd.DataFrame(np.mean(graycoprops(glcm, "energy"), axis=1).tolist()).T
    energy.columns = ["energy_" + str(col) for col in lengths]
    
    correlation = pd.DataFrame(
        np.mean(graycoprops(glcm, "correlation"), axis=1).tolist()
    ).T
    correlation.columns = ["correlation_" + str(col) for col in lengths]

    feat = pd.concat(
        [
            contrast.reset_index(drop=True),
            dissimilarity.reset_index(drop=True),
            homogeneity.reset_index(drop=True),
            ASM.reset_index(drop=True),
            energy.reset_index(drop=True),
            correlation.reset_index(drop=True),
        ],
        axis=1,
    )

    return feat


def center_mismatch(regionmask: np.ndarray, intensity: np.ndarray, resolution):
    """ Compute distance between centroid and center of mass
    
    Args:
        regionmask : binary background mask
        intensity  : intensity image
    """
    regionmask=regionmask.astype('uint8')
    measures = measure.regionprops_table(regionmask,intensity, 
                         properties=['centroid','weighted_centroid'])
    dist = np.sqrt(
        np.multiply(np.square(measures['centroid-0']-measures['weighted_centroid-0']), resolution[0])
        + np.multiply(np.square(measures['centroid-1']-measures['weighted_centroid-1']), resolution[1])
        + np.multiply(np.square(measures['centroid-2']-measures['weighted_centroid-2']), resolution[2])
        [0])
    
    feat = pd.DataFrame({"center_mismatch": dist})
    return feat


def measure_texture_features(regionmask: np.ndarray, intensity: np.ndarray, lengths=[1, 5, 20], resolution:float = [0.7, 0.09 ,0.09]):
    """Compute all texture features
    This function computes all features that describe the image texture 
    Args:
        regionmask : binary background mask
        intensity  : intensity image
        lengths    : length scales 
    Returns: A pandas dataframe with all the features for the given image
    """
    # compute features
    all_features = pd.DataFrame()
    all_features = pd.concat([all_features, gclm_textures(regionmask, intensity, lengths, resolution).reset_index(drop=True)], axis = 1)
    all_features = pd.concat([all_features, center_mismatch(regionmask, intensity, resolution).reset_index(drop=True)], axis = 1)

    return all_features