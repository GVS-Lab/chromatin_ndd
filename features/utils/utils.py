import numpy as np

def normalize_coordinates(coords:np.array, resolution:np.array = np.array([0.7, 0.09 ,0.09], dtype=float)):
    return(np.multiply(coords,resolution))

def normalize_axis(axis:np.array, resolution:float):
    return(np.multiply(axis,resolution))

def generate_max_z(image:np.array, axis:int = 0):
    return(np.amax(image, axis=axis))



''' not needed when spacing is provided for marching cubes

def mesh_surface_area_scaled(verts,faces, resolution:np.array=np.array([0.7, 0.09, 0.09])):
    actual_verts = verts[faces]
    a = actual_verts[:, 0, :] - actual_verts[:, 1, :]
    b = actual_verts[:, 0, :] - actual_verts[:, 2, :]
    del actual_verts

    # Compute the area of each triangle
    cross = np.cross(a, b)
    cross_scaled = np.multiply(cross, resolution)

    return ((cross_scaled ** 2).sum(axis=1) ** 0.5).sum() / 2.0
'''