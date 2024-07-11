import numpy as np


def graycomatrix_3d(img:np.ndarray, distances:float, angles_1, angles_2, resolution:float = [0.5, 0.09 ,0.09]):
    
    def calc_offset(angle_pair, distance:float):
        z:float = np.multiply(distance, np.sin(angle_pair[1]), dtype=float)*resolution[0]
        x2:float = np.multiply(np.multiply(distance, np.cos(angle_pair[1]),dtype=float), np.sin(angle_pair[0]),dtype=float)*resolution[1]
        y2:float = np.multiply(np.multiply(distance, np.cos(angle_pair[1]),dtype=float), np.cos(angle_pair[0]),dtype=float)*resolution[2]
        return(np.array([int(z),int(x2),int(y2)], dtype=int))
    
    img:int = (np.rint(img)).astype(np.uint8)
    levels:int = img.max() + 1 
    
    grid1, grid2 = np.meshgrid(angles_1, angles_2)
    angles:int = np.vstack([grid1.ravel(), grid2.ravel()]).T
    
    # Get the shape of the image
    z_max:int  = img.shape[0]
    x_max:int = img.shape[1]
    y_max:int = img.shape[2]

    glcm:int = np.zeros((levels, levels, len(distances), len(angles)), dtype=int)

    for i,distance in enumerate(distances):
        for j,angle_pair in enumerate(angles):
            offset = calc_offset(angle_pair, distance)
            for z in np.arange(z_max, dtype=int):
                for x in np.arange(x_max, dtype=int):
                    for y in np.arange(y_max, dtype=int):
                        curr = np.array([z,x,y], dtype=int)
                        neighbor = np.add(curr,offset, dtype=int)
                        if neighbor[1] >= 0 and neighbor[1] < x_max and neighbor[2] >= 0 and neighbor[2] < y_max and neighbor[0] >= 0 and neighbor[0] < z_max:
                            glcm[img[z,x,y], img[neighbor[0],neighbor[1],neighbor[2]], i, j] += 1
                        else:
                            continue
    
    return(glcm)