import os
from os import listdir
from os.path import isfile, join

import tifffile


from skimage.filters import threshold_otsu
from skimage.morphology import area_opening, area_closing, label
from skimage.segmentation import clear_border
from skimage.measure import regionprops
import numpy as np



class Segmenter:
    
    def __init__(self):
        self.cwd = os.getcwd()
    
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
            
        else:
            print('Unknown type for image')

        print(f'{filename} saved at {output_path}')

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



    def subtract_mask(self, images, series:bool = True):

        if series:
            for i, img in enumerate(images):
                images[i]['nuclei'], images[i]['background'] = self.apply_threshold(img, False)
            return(images)
        else:
            images['background'] = np.where(images['2d_mask'] == 0, images['dapi_max_z'], 255)
            images['nuclei'] = np.where(images['2d_mask'] == 1, images['dapi_max_z'], 0)

            return(images['nuclei'], images['background'])


    def apply_threshold(self, images, series:bool = True):

        if series:
            for i, img in enumerate(images):
                images[i]['2d_mask'] = self.apply_threshold(img, False)
            return(images)
        else:
            thresh_otsu = threshold_otsu(images['dapi_max_z'])
            images['2d_mask'] = np.where(images['dapi_max_z'] >= thresh_otsu,1,0)

            return(images['2d_mask'])

    def open_close_img(self, images, series:bool = True):
        
        if series:
            for i, img in enumerate(images):
                images[i]['2d_mask'] = self.open_close_img(img, False)
            return(images)
        else:
            images['2d_mask'] = area_opening(images['2d_mask'])
            images['2d_mask'] = area_closing(images['2d_mask'])
            return(images['2d_mask'])

    def label_segments_2d(self, images, series:bool = True):

        if series:
            for i, img in enumerate(images):
                images[i]['2d_mask'] = self.label_segments_2d(img, False)
            return(images)
        else:
            images['2d_mask'] = label(images['2d_mask'])
            return(images['2d_mask'])

    def remove_border_cells(self, images, series:bool = True):

        if series:
            for i, img in enumerate(images):
                images[i]['2d_mask'] = self.remove_border_cells(img, False)
            return(images)
        else:
            images['2d_mask'] = clear_border(images['2d_mask'])
            return(images['2d_mask'])
        
    def extract_nuclei(self, images, series:bool = True):

        if series:
            for i, img in enumerate(images):
                images[i]['nuclei'] = self.extract_nuclei(img, False)
            return(images)
        else:
            images['nuclei'] = regionprops(images['2d_mask'], images['dapi_max_z'])
            return(images['nuclei'])

    def filter_nuclei_2d(self, images, upper:int, lower:int, series:bool = True):

        if series:
            for i, img in enumerate(images):
                images[i]['nuclei'] = self.filter_nuclei_2d(img, upper, lower, False)
            return(images)
        else:
            images['nuclei'] = [nucleus for nucleus in images['nuclei'] if nucleus.area > lower and nucleus.area < upper]            
            return(images['nuclei'])





    
    