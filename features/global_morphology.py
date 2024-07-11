'''
adapted from chrometrics and adapted for 3D images: https://github.com/GVS-Lab/chrometrics
'''

from skimage.morphology import erosion
from scipy import stats
from scipy.spatial import distance, distance_matrix
import numpy as np
from skimage.transform import rotate
import pandas as pd
from skimage import measure
from skimage.measure import marching_cubes, mesh_surface_area


from features.utils.utils import normalize_axis, normalize_coordinates, generate_max_z


def radii_features_3D(binary_mask: np.ndarray, centroid0, centroid1, centroid2, resolution=np.array([0.7, 0.09 ,0.09], dtype=float)):
    """Describing centroid to boundary distances(radii)
    This function obtains radii from the centroid to all the points along the edge and 
    using this computes features that describe the morphology of the given object.
    """
    
    # obtain the edge pixels
    bw = binary_mask > 0
    edge = np.subtract(bw * 1, erosion(bw) * 1)
    (boundary_z, boundary_x, boundary_y) = [np.where(edge > 0)[0], np.where(edge > 0)[1], np.where(edge > 0)[2]]
    boundary_z = normalize_axis(axis=boundary_z, resolution=resolution[0])
    boundary_x = normalize_axis(axis=boundary_x, resolution=resolution[1])
    boundary_y = normalize_axis(axis=boundary_y, resolution=resolution[2])
    (centroid0, centroid1, centroid2) = normalize_coordinates(coords=(centroid0, centroid1, centroid2))


    # calculate radii
    #print(centroid0, centroid1, centroid2)
    dist_b_c = np.sqrt(np.square(boundary_z - centroid0) + np.square(boundary_x - centroid1) + np.square(boundary_y - centroid2))
    cords = np.column_stack((boundary_z, boundary_x, boundary_y))
    dist_matrix = distance.squareform(distance.pdist(cords, "euclidean"))
    # Compute features
    feret = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]  # offset from diagonal

    feat ={ "min_radius": np.min(dist_b_c),
            "max_radius": np.max(dist_b_c),
            "med_radius": np.median(dist_b_c),
            "avg_radius": np.mean(dist_b_c),
            "mode_radius": stats.mode(np.array(dist_b_c, dtype=int), axis=None, keepdims=True).mode[0],
            "d25_radius": np.percentile(dist_b_c, 25),
            "d75_radius": np.percentile(dist_b_c, 75),
            "std_radius": np.std(dist_b_c),
            "feret_max": np.max(feret)
          }
    
    del dist_matrix, dist_b_c, cords, feret, edge, bw, boundary_z, boundary_x, boundary_y

    return feat

def simple_morphology(prop):
    """ Compute simple morphological features
    Args:
        prop: (pandas dataframe) regionprops dataframe row
    """

    feat = { "concavity" : (prop["convex_area"] - prop["area"]) / prop["convex_area"],
            "solidity" : prop["area"] / prop["convex_area"],
            "a_r" : prop["minor_axis_length"] / prop["major_axis_length"],
            "area_bbarea" : prop["area"] / prop["bbox_area"]
    }
    
    return feat

def calliper_sizes(binary_mask: np.ndarray, angular_resolution:int = 10):
    """Obtains the min and max Calliper distances
    
    This functions calculates min and max the calliper distances by rotating the image
    by the given angular resolution
    
    Args: 
        binary_mask:(image_arrray)
        angular_resolution:(integer) value between 1-359 to determine the number of rotations
        
    """
    img = binary_mask > 0
    callipers = []
    for angle in range(1, 360, 10):
        rot_img = rotate(img, angle, resize=True)
        callipers.append(max(np.sum(rot_img, axis=0)))
        rot_img = None
        
    feat = { "min_calliper": min(callipers), 
             "max_calliper": max(callipers),
             "smallest_largest_calliper": min(callipers)/max(callipers)
           }
    
    return feat  

def get_nuclear_surface_area(nucleus_mask: np.ndarray, resolution=np.array([0.7, 0.09 ,0.09], dtype=float)):
    verts, faces, _, _ = marching_cubes(nucleus_mask, 0.0, spacing=resolution)
    surface_area = mesh_surface_area(verts, faces)
    del verts, faces
    return {"surface_area" : surface_area}


def measure_global_morphometrics(prop, angular_resolution:int = 10, measure_calliper:bool = True, measure_radii:bool = True, measure_simple:bool = True, measure_surface_area:bool = True):
    """Compute all boundary features
    This function computes all features that describe the boundary features
    Args:
        prop: (pandas dataframe) regionprops dataframe row
        angular_resolution:(integer) value between 1-179 to determine the number of rotations
    Returns: A pandas dataframe with all the features for the given image
    """

    feat ={}


    centroids = [float(prop['local_centroid-0']), float(prop['local_centroid-1']), float(prop['local_centroid-2'])]
    #print(centroids)
    binary_mask = prop['image'] * 1


    if(measure_calliper):
        max_z_mask = generate_max_z(binary_mask)
        feat.update(calliper_sizes(max_z_mask, angular_resolution))
        del max_z_mask
    if(measure_radii):
        feat.update(radii_features_3D(binary_mask, centroids[0], centroids[1], centroids[2]))
    if(measure_simple):
        feat.update(simple_morphology(prop))
    if(measure_surface_area):
        feat.update(get_nuclear_surface_area(binary_mask))

    del prop, binary_mask

    return (pd.DataFrame([feat]))