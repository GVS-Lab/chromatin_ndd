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


def calc_boundary_features(image, mask):
    #boundary mask
    bm = np.copy(mask)
    img_lamin = np.copy(image)

    #subtract previous dilation of dapi mask
    for z in range(bm.shape[0]):
        bm[z] = binary_erosion(bm[z], structure=disk(10))

    selem = get_selem_z_xy_resolution()
    bm2 = binary_erosion(bm, structure=selem)
    edge = np.subtract(bm * 1, bm2 * 1)
    edge_int = img_lamin * edge

    #print(len(np.unique(edge_int)))

    if len(np.unique(edge_int)) > 2:
        thresh = threshold_multiotsu(edge_int)[1]
        lamin_bm = (edge_int > thresh) *1
        lamin_bm = remove_small_objects(lamin_bm, min_size = 5)
        label = measure.label(lamin_bm)

        feat = {
            'lamin_boundary_patches' : np.max(np.unique(label)),
            'ratio_lamin_boundary_patches_volume' : np.sum(lamin_bm)/np.max(np.unique(label)),
            'lamin_boundary_center_int' : np.mean(img_lamin) / np.mean(edge_int)
        }
    else:

        feat = {
            'lamin_boundary_patches' : np.nan,
            'ratio_lamin_boundary_patches_volume' : np.nan,
            'lamin_boundary_center_int' : np.mean(img_lamin) / np.mean(edge_int)
        }

    return(feat)



def get_selem_z_xy_resolution(k: int = 7):
        selem = np.zeros([2 * k + 1, 2 * k + 1])
        selem[k, :] = 1
        selem[:, k] = 1
        selem_2 = np.zeros([2 * k + 1, 2 * k + 1])
        selem_2[k, k] = 1
        selem = np.stack([selem_2, selem, selem_2], axis=0)
        return selem

def radial_distribution_measures(image, mask):

    def get_radial_distribution(image, mask, bins=10):

        selem = get_selem_z_xy_resolution(7)
        rp_masks = get_radial_profile_masks(mask, selem=selem)

        rdp = []
        for i in range(len(rp_masks)):
            rdp.append(np.sum(image * np.uint8(rp_masks[i])))
        rdp = np.array(rdp[: rdp.index(0) + 1])
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

    feat = {f"rd_lamin_{i}": rd[i] - rd[i-1] for i in range(1, len(rd))}

    return(feat)


def measure_lamin_features(
        intensity: np.ndarray, 
        mask: np.ndarray,
        foci_thresh:int=256,
        foci_features:bool = True,
        hole_features:bool = True,
        radial_dist:bool = False,
        boundary_features:bool = True,
        channel:int=2, 
        dapi_channel:int=0,
        resolution:float = [0.7, 0.09 ,0.09]):
    """Compute all lamin features
    This function computes all features of the Lamin channel. 
    Args:
        intensity : intensity image
        mask  : mask of nucleus
        boundary_size : pixel size of boundary, used to construct boundary
        channel : lamin channel in intensity
        dapi_channel : dapi channel in intensity
    Returns: A pandas dataframe with all the features for the given image
    """

    feat = {}

    if (boundary_features):
        feat.update(calc_boundary_features(np.copy(intensity[:,:,:,channel]), np.copy(mask)))
    if (radial_dist):
        feat.update(radial_distribution_measures(np.copy(intensity[:,:,:,channel]), np.copy(mask)))
    if (foci_features):
        feat.update(gf.measure_foci_features(np.copy(intensity), np.copy(mask), foci_thresh, foci_name = 'lamin_foci', seg_watershed=False, channel = channel, foci_radial_dist=False, foci_lamin_coloc=False, foci_hc_coloc=False))
    if (hole_features):
        if len(np.unique(intensity[:,:,:,channel])) > 2:
            hole_thresh = threshold_multiotsu(intensity[:,:,:,channel])[0]
            mask2 = np.copy(mask)
            for z in range(mask2.shape[0]):
                mask2[z] = binary_erosion(mask2[z], structure=disk(10))
            img_holes = np.copy(intensity)
            img_holes[:,:,:,channel] = ((img_holes[:,:,:,channel] < hole_thresh) * mask2) * 1
            feat.update(gf.measure_foci_features(img_holes, mask, foci_thresh=0, foci_name = 'lamin_hole', seg_watershed=False, channel = channel, foci_radial_dist=False, foci_lamin_coloc=False, foci_hc_coloc=False))
        else:
            h_feat = {
                'lamin_hole_count': 0,
                'lamin_hole_volume_mean':  np.nan,
                'lamin_hole_volume_variance':  np.nan,
                'lamin_hole_mean_intensity_mean':  np.nan,
                'lamin_hole_mean_intensity_variance':  np.nan,
                'lamin_hole_d2b_mean': np.nan,
                'lamin_hole_d2b_variance': np.nan,
                'lamin_hole_ratio_d2b_volume_mean' : np.nan,
                'lamin_hole_ratio_d2b_volume_variance' : np.nan,
                'lamin_hole_d_nearest_neighbor_mean': np.nan,
                'lamin_hole_d_nearest_neighbor_variance': np.nan,
                'lamin_hole_ratio_nearest_neighbor_volume_mean': np.nan,
                'lamin_hole_ratio_nearest_neighbor_volume_variance': np.nan,
                'lamin_hole_d_nearest_neighbor(verts)_mean': np.nan,
                'lamin_hole_d_nearest_neighbor(verts)_variance': np.nan,
                'lamin_hole_ratio_nearest_neighbor(verts)_mean_count' : np.nan,
                'lamin_hole_ratio_nearest_neighbor(verts)_variance_count' : np.nan
            }
            feat.update(h_feat)
    
    return pd.DataFrame([feat])