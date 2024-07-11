#Foci detection adapted from chromark

import pandas as pd
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops_table, marching_cubes
from skimage import morphology, measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.interpolate import interp1d
from scipy.spatial import distance, distance_matrix
from features.utils.utils import normalize_axis, normalize_coordinates



def foci_detection(intensity: np.ndarray, mask: np.ndarray, foci_thresh:float, foci_size:int = 4, seg_watershed:bool=True):
    """Detect foci in the image
    This function detects foci in the image by thresholding the intensity image.
    Args:
        intensity  : intensity image
        foci_thresh: threshold for binarization
        foci_size  : min size of foci
    Returns: A binary image with the detected foci
    """


    foci_mask = (intensity > foci_thresh)
    foci_mask = morphology.remove_small_objects(foci_mask, min_size=foci_size)
    if seg_watershed:
        filtered = np.ma.array(intensity, mask=np.logical_not(mask))

        # Find local maxima in the filtered image 
        coords = peak_local_max(filtered, min_distance=1)
        peak_mask = np.zeros(filtered.shape, dtype=bool)
        peak_mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(peak_mask)

        # Watershed segment the images using the local maxima
        labels = watershed(-filtered, markers, mask=foci_mask)
        labels = morphology.remove_small_objects(labels, min_size=foci_size)
        labels = measure.label(labels)
    else:
        labels = measure.label(foci_mask)

    foci = regionprops_table(labels, intensity, properties=('label', 'area', 'slice', 'mean_intensity','centroid', 'equivalent_diameter', 'image', 'intensity_image'))
    
    return (foci, labels)


def foci_features_basic(foci, foci_name:str):

    feat = {
        f'{foci_name}_count': len(foci['label']),
        f'{foci_name}_volume_mean': np.mean(foci['area']),
        f'{foci_name}_volume_variance': np.var(foci['area']),
        f'{foci_name}_mean_intensity_mean': np.mean(foci['mean_intensity']),
        f'{foci_name}_mean_intensity_variance': np.var(foci['mean_intensity'])
    }


    return(feat)


def foci_features_spatial(foci, intensity: np.ndarray, mask: np.ndarray, foci_name:str, resolution=[0.7, 0.09, 0.09]):

    #Get boundary of the mask
    bw = mask > 0
    edge = np.subtract(bw * 1, morphology.erosion(bw) * 1)
    (boundary_z, boundary_x, boundary_y) = [np.where(edge > 0)[0], np.where(edge > 0)[1], np.where(edge > 0)[2]]
    boundary_z = normalize_axis(axis=boundary_z, resolution=resolution[0])
    boundary_x = normalize_axis(axis=boundary_x, resolution=resolution[1])
    boundary_y = normalize_axis(axis=boundary_y, resolution=resolution[2])

    #Create list of foci with centroid coordinates and equivalent diameter
    foci_list = [
        {
            'coords' : normalize_coordinates([row['centroid-0'], row['centroid-1'], row['centroid-2']], resolution),
            'eq_rad' : ((np.sum(np.multiply(row['area'], resolution)) * 3) / (4 * np.pi)) **(1/3),
            'area' : row['area'],
            'image' : row['intensity_image'],
            'mask' : row['image']
            }
         for index, row in pd.DataFrame(foci).iterrows() #converting foci into pd.DataFrame since it's easier to iterate over df than over multiple list of dict values
         ]


    #Compute pairwise distances between foci(centroid) and boundary, choose the minimum distance

    min_dist = [
        np.min(
        np.sqrt(np.square(boundary_z - f['coords'][0]) + np.square(boundary_x - f['coords'][1]) + np.square(boundary_y - f['coords'][2]))) - f['eq_rad']
        if np.min(np.sqrt(np.square(boundary_z - f['coords'][0]) + np.square(boundary_x - f['coords'][1]) + np.square(boundary_y - f['coords'][2]))) > f['eq_rad']
        else 0
        for f in foci_list
        ] 

    min_dist_scaled = np.multiply(min_dist, np.mean(foci['area'])) #scaling by mean area so ratio doesn't get too small
    ratio_min_dist_volume = np.array(min_dist_scaled) / np.array([foci['area'] for foci in foci_list])


    #compute pairwise distances of foci (centroids)
    if len(foci_list) < 2 :
        feat = {
            f'{foci_name}_d2b_mean' : np.mean(min_dist),
            f'{foci_name}_d2b_variance': np.var(min_dist),
            f'{foci_name}_ratio_d2b_volume_mean' : np.mean(ratio_min_dist_volume),
            f'{foci_name}_ratio_d2b_volume_variance' : np.var(ratio_min_dist_volume),
            f'{foci_name}_d_nearest_neighbor_mean': np.nan,
            f'{foci_name}_d_nearest_neighbor_variance': np.nan,
            f'{foci_name}_ratio_nearest_neighbor_volume_mean': np.nan,
            f'{foci_name}_ratio_nearest_neighbor_volume_variance': np.nan,
            f'{foci_name}_d_nearest_neighbor(verts)_mean': np.nan,
            f'{foci_name}_d_nearest_neighbor(verts)_variance': np.nan,
            f'{foci_name}_ratio_nearest_neighbor(verts)_mean_count' : np.nan,
            f'{foci_name}_ratio_nearest_neighbor(verts)_variance_count' : np.nan
        }
        return(feat)

    cords = np.vstack([foci['coords'] for foci in foci_list])
    dist_matrix = distance.squareform(distance.pdist(cords, "euclidean"))
    
    eq_rad_matrix = np.zeros_like(dist_matrix)
    for i in range(len(foci_list)):
        eq_rad_matrix[i] = foci_list[i]['eq_rad']
    
    #subtract the sum of the equivalent radii from the distance matrix as an estimate of boundary - boundary distances
    #turns out assuming the foci to be spheres and subtracting the radius from the distance matrix is too innaccurate
    #dist_matrix = dist_matrix - eq_rad_matrix - eq_rad_matrix.T
    #dist_matrix[dist_matrix < 0] = 0

    np.fill_diagonal(dist_matrix, np.inf)

    #get min distance for each foci
    d_nearest_foci = np.min(dist_matrix, axis = 1)


    vol_matrix = np.zeros_like(dist_matrix)
    for i in range(len(foci_list)):
        vol_matrix[i] = foci_list[i]['area']

    #scaling by mean volume and then taking ratio of dist to volume so that numbers dont get too small
    dist_matrix = np.multiply(np.multiply(dist_matrix, np.mean(foci['area'])), np.mean(foci['area']))
    dist_matrix = dist_matrix / vol_matrix
    dist_matrix = dist_matrix / vol_matrix.T

    ratio_nearest_foci_volume = np.min(dist_matrix, axis = 1)

    #get verts for each foci
    verts = []
    for foci in foci_list:
        if (foci['mask'].shape[0] >= 2 and foci['mask'].shape[1] >= 2 and foci['mask'].shape[2] >= 2) and (np.array_equal(np.unique(foci['mask']), [0, 1])):
            local_verts = marching_cubes(foci['mask'], level=0)[0]
            local_verts_scaled = normalize_coordinates(local_verts, resolution)
            global_verts = local_verts_scaled + foci['coords']
            verts.append(global_verts)
        else: 
            verts.append(foci['coords'].reshape(1, -1))

    # Compute distance matrix
    nearest_neighbors = []
    for i, verts_i in enumerate(verts):
        distances = []
        for j, verts_j in enumerate(verts):
            if i != j:  # Exclude same region
                distance_matrix = distance.cdist(verts_i, verts_j)
                distances.append(np.min(distance_matrix, axis=1))
        if distances:
            distances = np.concatenate(distances)
            nearest_neighbors.append(np.min(distances))
        else:
            nearest_neighbors.append(None)  # No other regions to compare with


    feat = {
        f'{foci_name}_d2b_mean': np.mean(min_dist),
        f'{foci_name}_d2b_variance': np.var(min_dist),
        f'{foci_name}_ratio_d2b_volume_mean' : np.mean(ratio_min_dist_volume),
        f'{foci_name}_ratio_d2b_volume_variance' : np.var(ratio_min_dist_volume),
        f'{foci_name}_d_nearest_neighbor_mean': np.mean(d_nearest_foci),
        f'{foci_name}_d_nearest_neighbor_variance': np.var(d_nearest_foci),
        f'{foci_name}_ratio_nearest_neighbor_volume_mean': np.mean(ratio_nearest_foci_volume),
        f'{foci_name}_ratio_nearest_neighbor_volume_variance': np.var(ratio_nearest_foci_volume),
        f'{foci_name}_d_nearest_neighbor(verts)_mean': np.mean(nearest_neighbors),
        f'{foci_name}_d_nearest_neighbor(verts)_variance': np.var(nearest_neighbors),
        f'{foci_name}_ratio_nearest_neighbor(verts)_mean_count' : np.mean(nearest_neighbors) / len(foci_list),
        f'{foci_name}_ratio_nearest_neighbor(verts)_variance_count' : np.var(nearest_neighbors) / len(foci_list)
    }
    
    return(feat)

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
        rdp = np.array(rdp[: rdp.index(0) + 1])
        total_int = rdp[0]
        rdp = rdp / total_int
        rdp = rdp[::-1]

        radii = np.linspace(0, bins, num=len(rdp))
        radii_new = np.linspace(0, bins, num=bins)
        spl = interp1d(radii, rdp, fill_value=1, bounds_error=False)
        rdp_interpolated = spl(radii_new)
        return rdp_interpolated


    def get_radial_profile_masks(mask, selem):
        masks = [mask]
        for i in range(len(mask)):
            masks.append(ndimage.binary_erosion(masks[-1], selem))
        return masks
    
    rd = get_radial_distribution(image, mask, bins=10)

    feat = {f"rd_foci_{i}": rd[i] - rd[i-1] for i in range(1, len(rd))}

    return(feat)

def foci_lamin_colocalization(foci, intensity):

    coloc_ratios = [
        np.divide(np.sum(image * foci_mask), np.sum(intensity[slice] * foci_mask)) 
        for slice, image, foci_mask in zip(foci['slice'], foci['intensity_image'], foci['image'])
    ]

    if len(coloc_ratios) == 0:
        feat = {
            'gamma_lamin_coloc_mean': 0,
            'gamma_lamin_coloc_variance': 0,
            'gamma_lamin_coloc_p25': 0,
            'gamma_lamin_coloc_p75': 0
        }
    else:
        feat = {
            'gamma_lamin_coloc_mean': np.mean(coloc_ratios),
            'gamma_lamin_coloc_variance': np.var(coloc_ratios),
            'gamma_lamin_coloc_p25': np.percentile(coloc_ratios, 25),
            'gamma_lamin_coloc_p75': np.percentile(coloc_ratios, 75)
        }

    return feat

def foci_hc_colocalization(foci, intensity, mask):
    
    coloc_ratios = [
        np.divide(np.sum(image * foci_mask), np.sum(intensity[slice] * foci_mask)) 
        for slice, image, foci_mask in zip(foci['slice'], foci['intensity_image'], foci['image'])
    ]

    if len(coloc_ratios) == 0:
        feat = {
            'gamma_hc_coloc_mean': 0,
            'gamma_hc_coloc_variance': 0,
            'gamma_hc_coloc_p25': 0,
            'gamma_hc_coloc_p75': 0,
            'p_of_hc_with_foci': 0,
            'p_of_foci_in_hc': 0,
            'hc_gamma_coloc_(volumescaled_hc_gamma)': 0
        }
    else:
        hc_thresh = np.mean(intensity[mask]) + (1 * np.std(intensity[mask]))
        hc_mask = (intensity > hc_thresh) * 1

        coloc_sum = sum([
            np.sum(hc_mask[slice] * foci_mask) #should use hc_mask instead of intensity[slice]? change back to intensity if error appears
            for slice, foci_mask in zip(foci['slice'], foci['image'])
        ])

        feat  = {
            'gamma_hc_coloc_mean': np.mean(coloc_ratios),
            'gamma_hc_coloc_variance': np.var(coloc_ratios),
            'gamma_hc_coloc_p25' : np.percentile(coloc_ratios, 25),
            'gamma_hc_coloc_p75' : np.percentile(coloc_ratios, 75),
            'p_of_hc_with_foci' : np.divide(coloc_sum, np.sum(hc_mask)), #dont overinterpret biologically due to resolution
            'p_of_foci_in_hc' : np.divide(coloc_sum, sum([np.sum(foci_mask) for foci_mask in foci['image']])),
            'hc_gamma_coloc_(volumescaled_hc_gamma)' : coloc_sum / ((np.sum(hc_mask) + sum([np.sum(foci_mask) for foci_mask in foci['image']])) / 2),
        }

    return(feat)


def measure_foci_features(
        intensity: np.ndarray, 
        mask: np.ndarray, 
        foci_thresh:float = None, 
        foci_name:str = 'gamma_foci',
        foci_size:int = 4, 
        seg_watershed:bool=True,
        foci_spatial:bool = True, 
        foci_basic:bool = True, 
        foci_radial_dist:bool = True, 
        foci_lamin_coloc:bool = True, 
        foci_hc_coloc:bool = True,
        channel:int=1, 
        lamin_channel:int=2,
        dapi_channel:int=0):
    """Compute all foci features
    This function detects and computes all features of the gammaH2AX foci. 
    Args:
        intensity : intensity image
        mask  : mask of nucleus
        foci_thresh : threshold for binarization
        foci_size : min size of foci in pixels
        seg_watershed: bool, whether to use watershed segmentation for foci detection or not
        channel : gammaH2AX channel in intensity
        lamin_channel : lamin channel in intensity
    Returns: A pandas dataframe with all the features for the given image
    """

    if not foci_thresh:
        int = intensity[:,:,:,channel]
        foci_thresh = np.mean(int[mask]) + (2.5 * np.std(int[mask]))


    feat = {}

    foci, foci_mask = foci_detection(intensity[:,:,:,channel], mask, foci_thresh, foci_size, seg_watershed=seg_watershed)

    foci_mask = foci_mask > 0

    if len(foci) == 0:
        foci_basic, foci_spatial, foci_radial_dist, foci_lamin_coloc, foci_hc_coloc = False, False, False, False, False
        feat = {
            f'{foci_name}_count': 0,
            f'{foci_name}_volume_mean':  np.nan,
            f'{foci_name}_volume_variance':  np.nan,
            f'{foci_name}_mean_intensity_mean':  np.nan,
            f'{foci_name}_mean_intensity_variance':  np.nan,
            f'{foci_name}_d2b_mean': np.nan,
            f'{foci_name}_d2b_variance': np.nan,
            f'{foci_name}_ratio_d2b_volume_mean' : np.nan,
            f'{foci_name}_ratio_d2b_volume_variance' : np.nan,
            f'{foci_name}_d_nearest_neighbor_mean': np.nan,
            f'{foci_name}_d_nearest_neighbor_variance': np.nan,
            f'{foci_name}_ratio_nearest_neighbor_volume_mean': np.nan,
            f'{foci_name}_ratio_nearest_neighbor_volume_variance': np.nan,
            f'{foci_name}_d_nearest_neighbor(verts)_mean': np.nan,
            f'{foci_name}_d_nearest_neighbor(verts)_variance': np.nan,
            f'{foci_name}_ratio_nearest_neighbor(verts)_mean_count' : np.nan,
            f'{foci_name}_ratio_nearest_neighbor(verts)_variance_count' : np.nan
        }
        if (foci_lamin_coloc):
            feat.update({
                'gamma_lamin_coloc_mean':  np.nan,
                'gamma_lamin_coloc_variance':  np.nan,
                'gamma_lamin_coloc_p25' :  np.nan,
                'gamma_lamin_coloc_p75' :  np.nan
            })
        if (foci_hc_coloc):
            feat.update({
                'gamma_hc_coloc_mean':  np.nan,
                'gamma_hc_coloc_variance':  np.nan,
                'gamma_hc_coloc_p25' :  np.nan,
                'gamma_hc_coloc_p75' :  np.nan,
                'p_of_hc_with_foci' :  np.nan,
                'p_of_foci_in_hc' :  np.nan,
                'hc_gamma_coloc_(volumescaled_hc_gamma)' : np.nan
            })

    #range-normalize lamin channel
    #intensity[:,:,:,lamin_channel] = ((intensity[:,:,:,lamin_channel] - np.min(intensity[:,:,:,lamin_channel])) * 255) / (np.max(intensity[:,:,:,lamin_channel]) - np.min(intensity[:,:,:,lamin_channel]))

    if (foci_basic):
        feat.update(foci_features_basic(foci, foci_name))
    if (foci_spatial):
        feat.update(foci_features_spatial(foci, intensity[:,:,:,channel], mask, foci_name=foci_name))
    if (foci_radial_dist):
        feat.update(radial_distribution_measures(foci_mask, mask))
    if (foci_lamin_coloc):
        feat.update(foci_lamin_colocalization(foci, intensity[:,:,:,lamin_channel]))
    if (foci_hc_coloc):
        feat.update(foci_hc_colocalization(foci, intensity[:,:,:,dapi_channel], mask))
   
    return (feat)