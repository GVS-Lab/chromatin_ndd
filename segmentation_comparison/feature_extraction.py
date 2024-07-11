import os
from os import listdir
from os.path import isfile, join

import tifffile
from readlif.reader import LifFile
import readlif.reader
import json

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

from tqdm.notebook import tqdm


class Extractor:

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

        elif isinstance(images, readlif.reader.LifImage):
            output_path = join(self.cwd, output_dir, filename + '.json')
            meta = {**images.info, **images.settings}
            with open(output_path, 'w') as json_file:
                json.dump(meta, json_file)
            
        else:
            print('Unknown type for image')

        print(f'{filename} saved at {output_path}')

        return()

    def load_images(self, input_dir:str, inclusion1:str = None, inclusion2:str = None, exclusion:str = '@#$!', filetype:str = '.tiff'):
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
    

    def extract_regionprops_3d(self, input_dir, mask_dir):  

        input_path = join(self.cwd, input_dir)

        mask_files = os.listdir(join(input_path, mask_dir))
        dapi_files = ['_'.join(filename.rsplit('_')[0:6]) + '_gamma_adjusted.tiff' for filename in mask_files]


        masks = self.load_images_from_list(join(input_path, mask_dir), mask_files)
        images = self.load_images_from_list(join(input_path, 'gamma_adjusted'), dapi_files)

        props=['label', 'area', 'bbox', 'bbox_area', 'centroid', 'convex_area', 'convex_image', 'coords', 'equivalent_diameter', 'euler_number', 'extent', 'filled_area', 'filled_image', 'image', 'inertia_tensor', 'inertia_tensor_eigvals', 'intensity_image',  'local_centroid', 'major_axis_length', 'max_intensity', 'mean_intensity', 'min_intensity', 'minor_axis_length', 'moments', 'moments_central', 'moments_normalized', 'slice', 'solidity', 'weighted_centroid', 'weighted_local_centroid', 'weighted_moments', 'weighted_moments_central', 'weighted_moments_normalized']

        df_all = pd.DataFrame()

        for mask, img in tqdm(zip(masks, images), total=len(masks), desc="Extracting Region Properties"):

            try:
                df_raw = regionprops_table(mask['image'], img['image'], properties=props)
                df_raw['full_label'] = []
                for i,label in enumerate(df_raw['label']):
                    df_raw['full_label'].append('_'.join(mask['filename'].rsplit('_')[0:3]) + '_nucleus_' + str(label))

                df = pd.DataFrame(df_raw)
                df.set_index('full_label', inplace=True)

                df_all = pd.concat([df_all, df], ignore_index=False)
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                try:
                    for label in np.unique(mask['image']):
                        if label != 0:
                            mask_label = np.where(mask['image'] == label, mask['image'], 0)
                            df_raw = regionprops_table(mask_label, img['image'], properties=props)
                            df_raw['full_label'] = []
                            for i,label in enumerate(df_raw['label']):
                                df_raw['full_label'].append('_'.join(mask['filename'].rsplit('_')[0:3]) + '_nucleus_' + str(label))

                            df = pd.DataFrame(df_raw)
                            df.set_index('full_label', inplace=True)

                            df_all = pd.concat([df_all, df], ignore_index=False)
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    continue

        return(df_all)