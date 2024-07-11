import os
from os import listdir
from os.path import isfile, join

import tifffile
from readlif.reader import LifFile
import readlif.reader
import json


from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.morphology import area_opening, area_closing, label
from skimage.segmentation import clear_border
from skimage.measure import regionprops
import numpy as np
#from stardist.models import StarDist2D
#from csbdeep.utils import normalize

from tqdm.notebook import tqdm
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import disk, remove_small_holes

from joblib import Parallel, delayed


class Segmenter:

    def __init__(self, n_jobs:int = 1):
        self.cwd = os.getcwd()
        self.n_jobs = n_jobs
    
    def save_image(self, images, filename:str, output_dir:str):
        output_path = join(self.cwd, output_dir)
       
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        if isinstance(images, np.ndarray):
            output_path = join(output_path, filename + '.tiff')
            if len(images.shape) in [2,3]:
                tifffile.imsave(output_path, images)
            else:
                print('Unknown shape for numpy.ndarray')

        elif isinstance(images, list):
            for i,img in enumerate(images):
                output_path = join(self.cwd, output_dir, filename + f'_{i}.tiff')
                #print(output_path)
                img.save(output_path)

        elif isinstance(images, readlif.reader.LifImage):
            output_path = join(self.cwd, output_dir, filename + '.json')
            meta = {**images.info, **images.settings}
            with open(output_path, 'w') as json_file:
                json.dump(meta, json_file)
            
        else:
            print('Unknown type for image')

        #print(f'{filename} saved at {output_path}')

        return()
    
    def load_images(self, input_dir:str, inclusion1:str = None, inclusion2:str = None, exclusion:str = '@#$!', filetype:str = '.tiff'):
        '''
        Loads images from input directory by inclusion and exclusion criteria and filetype
        Input_dir: str, path to input directory from current working directory
        inclusion1: str, string that has to be included in filename to be loaded
        inclusion2: str, same as inclusion1
        exclusion: str, files including this string in the filename won't be loaded
        filetype: str, ending of filename (filetype)

        Returns: list of dictionaries with filename and image
        '''
        if inclusion2 == None:
            inclusion2 = inclusion1

        input_path = join(self.cwd, input_dir)
        files = [{'filename': f, 'image': tifffile.imread(join(input_path, f))}
            for f in listdir(input_path) 
            if isfile(join(input_path, f)) and inclusion1 in f and inclusion2 in f and exclusion not in f and filetype in f]

        return(files)
    
    def load_images_from_list(self, input_dir:str, filelist:list):
        
        input_path = join(self.cwd, input_dir)
        files = [{'filename': f, 'image': tifffile.imread(join(input_path, f))}
            for f in filelist]
        return(files)
    
    def clear_masks(self, images, output_path, allow_z=True):

        def clear_border_2d(img, allow_z = True):
                if allow_z:
                    # Adding zeros at the beginning and end of the z axis
                    zeros_stack = np.zeros_like(img[0])  # Generating a zero-filled array of the same shape as a slice of img
                    img = np.concatenate(([zeros_stack], img, [zeros_stack]), axis=0)

                '''# Iterate through each 2D slice
                for z in range(img.shape[0]):
                    # Apply clear_border to each 2D slice
                    img[z] = clear_border(img[z])''' # This is the original code, but it doesn't work for 3D images correctly
                
                img = clear_border(img)

                if allow_z:
                    # Remove the zeros at the beginning and end of the z axis
                    img = img[1:-1] 

                return (img)
        
        def remove_2d_regions(image):
            labels = np.unique(image)[1::]
            regions_2d = []

            for region in labels:
                dimensions = np.where(image == region)
                if (len(np.unique(dimensions[0])) == 1) or (len(np.unique(dimensions[1])) == 1) or (len(np.unique(dimensions[2])) == 1):
                    regions_2d.append(region)
                #print(region,(len(np.unique(dimensions[0])) == 1) or (len(np.unique(dimensions[1])) == 1) or (len(np.unique(dimensions[2])) == 1), len(np.unique(dimensions[0])) == 1, len(np.unique(dimensions[1])) == 1, len(np.unique(dimensions[2])) == 1)

            for region in regions_2d:
                image[image == region] = 0

            return(image)

        def clmn(img, output_path, allow_z):
            filename = '_'.join(img['filename'].rsplit('_')[0:-2])
            processed_img = clear_border_2d(img['image'], allow_z)
            processed_img = remove_2d_regions(processed_img)
                        
            self.save_image(processed_img, filename + '_masks_cleared', output_path)


        Parallel(n_jobs=self.n_jobs)(delayed(clmn)(img, output_path, allow_z) for img in images)

        return()
    
    def create_otsu_masks(self, images, thresh_value:int=1, filename:str = None, output_path:str = None):
        '''
        Create otsu_mask for images in Parallel
        images:list of dictionaries containing filename and image for each image
        thresh_value:int, 0 or 1 to decide for lower or upper thresh of multiotsu

        Returns: nothing, call to save_image in inner function to save images
        '''

        def com(img, thresh_value:int, filename:str, output_path:str):
            '''
            Helper function to get multiotsu threshold and create mask - can be called from Parallel
            img:dictionary with filename and image
            thresh_value:int, 0 or 1 to decide for lower or upper thresh of multiotsu
            filename:str, filename to append to
            output_path: where to save

            Returns: nothing, calls save_image to save images at output_path
            '''
            filename = '_'.join(img['filename'].rsplit('_')[0:-2])
            thresh = threshold_multiotsu(img['image'])
            mask = (img['image'] > thresh[thresh_value]) *1

            for z in range(mask.shape[0]):
                mask[z] = binary_dilation(mask[z], structure=disk(5))
                mask[z] = remove_small_holes(mask[z], 1000) #gives warning because image is not binary but works as intended

            self.save_image(mask, filename + '_otsu_masks', output_path)

        Parallel(n_jobs=self.n_jobs)(delayed(com)(img, thresh_value, filename, output_path) for img in images)

    def label_3d(self, images, filename:str = None, output_path:str = None):

        def l3d(img, filename:str, output_path:str):
            filename = '_'.join(img['filename'].rsplit('_')[0:-2])
            img = label(img['image'])
            self.save_image(img, filename + '_otsu_mask_labeled', output_path)
        
        Parallel(n_jobs=self.n_jobs)(delayed(l3d)(img, filename, output_path) for img in images)
    
    




    
    