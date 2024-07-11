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

from joblib import Parallel, delayed

from features import img_texture as IT
from features import int_dist_features as IDF
from features import global_morphology as GM
from features import gamma_foci as GF
from features import lamin as LF
from features import hc_foci as HF
from features import boundary_local_curvature as BLC

import gc

import warnings

warnings.filterwarnings("ignore")


class Extractor:

    def __init__(self, n_jobs:int = 1):
        self.cwd = os.getcwd()
        self.n_jobs = n_jobs

    def save_image(self, images, filename:str, output_dir:str):
        output_path = join(self.cwd, output_dir)
       
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        if isinstance(images, np.ndarray):
            output_path = join(output_path, filename + '.tiff')
            if len(images.shape) in [2,3,4]:
                tifffile.imsave(output_path, images)
                print(f'{filename} saved at {output_path}')
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


        return()

    def load_images(self, input_dir:str, inclusion1:str = None, inclusion2:str = None, exclusion:str = '@#$!', filetype:str = '.tiff'):
        if inclusion2 == None:
            inclusion2 = inclusion1

        input_path = join(self.cwd, input_dir)
        files = [{'filename': f.rsplit('.')[0], 'image': tifffile.imread(join(input_path, f))}
            for f in listdir(input_path) 
            if isfile(join(input_path, f)) and inclusion1 in f and inclusion2 in f and exclusion not in f and filetype in f]

        if len(files) == 0:
            print(f'No files found in {input_path} with inclusion1: {inclusion1}, inclusion2: {inclusion2}, exclusion: {exclusion}, filetype: {filetype}')
        return(files)
    
    def load_images_from_list(self, input_dir:str, filelist:list):
        
        input_path = join(self.cwd, input_dir)
        files = ({'filename': f, 'image': tifffile.imread(join(input_path, f))}
            for f in filelist)
        return(files)
     
    def get_merged_images(self, input_dir, dapi_dir):  

        input_path = join(self.cwd, input_dir)

        dapi_files = os.listdir(join(input_path, dapi_dir))
        gamma_files = ['_'.join(filename.rsplit('_')[0:-3]) + '_gamma.tiff' for filename in dapi_files]
        lamin_files = ['_'.join(filename.rsplit('_')[0:-3]) + '_lamin.tiff' for filename in dapi_files]


        images_dapi = self.load_images_from_list(join(input_path, 'gamma_adjusted'), dapi_files)
        images_gamma = self.load_images_from_list(join(input_path, 'original_files'), gamma_files)
        images_lamin = self.load_images_from_list(join(input_path, 'original_files'), lamin_files)

        merged_images = []
        def merge_images(dapi, gamma, lamin):
            merged_image = np.stack([dapi['image'], gamma['image'], lamin['image']], axis=-1)
            return {'filename': dapi['filename'], 'image': merged_image}

        merged_images = Parallel(n_jobs=self.n_jobs)(delayed(merge_images)(dapi, gamma, lamin) for dapi, gamma, lamin in zip(images_dapi, images_gamma, images_lamin))

        return(merged_images)
    
    def extract_nuclei_3d(self, input_dir, mask_dir):
        '''Extract Nuclei from 3D masks

        Keyword arguments:
        input_dir -- str, path to the input directory
        mask_dir -- str, name of the mask directory

        Returns:
        List of dictionaries containing extracted properties
        '''

        input_path = join(self.cwd, input_dir)

        mask_files = os.listdir(join(input_path, mask_dir))
        dapi_files = ['_'.join(filename.rsplit('_')[0:-3]) + '_gamma_adjusted.tiff' for filename in mask_files]
        gamma_files = ['_'.join(filename.rsplit('_')[0:-4]) + '_gamma.tiff' for filename in mask_files]
        lamin_files = ['_'.join(filename.rsplit('_')[0:-4]) + '_lamin.tiff' for filename in mask_files]

        def merge_images(dapi_file, gamma_file, lamin_file):
            '''Merge DAPI, Gamma and Lamin channel
            
            Keyword arguments:
            dapi_file -- str, DAPI filename
            gamma_file -- str, Gamma filename
            lamin_file -- str, Lamin filename
            '''
            dapi = self.load_images(join(input_path, 'gamma_adjusted'), dapi_file.split('.')[0])
            gamma = self.load_images(join(input_path, 'original_files'), gamma_file.split('.')[0])
            lamin = self.load_images(join(input_path, 'original_files'), lamin_file.split('.')[0])
            merged_image = np.stack([dapi[0]['image'], gamma[0]['image'], lamin[0]['image']], axis=-1)
            #self.save_image(merged_image, '_'.join(dapi[0]['filename'].rsplit('_')[0:-3]), join(input_path, 'merged_images'))
            return (merged_image)

        props=['label','image','intensity_image','convex_area', 'convex_image', 'bbox', 'bbox_area', 'area']

        def parallel_nuclei_extraction(mask_file, dapi_file, gamma_file, lamin_file, props):
            '''Helper function to parallelize nuclei extraction per image

            Keyword arguments:
            mask_file -- str, mask filename
            dapi_file -- str, DAPI filename
            gamma_file -- str, Gamma filename
            lamin_file -- str, Lamin filename
            props -- list, list of properties to extract

            Returns:
            Dictionary of extracted properties
            '''

            mask = self.load_images(join(input_path, mask_dir), mask_file)
            img = merge_images(dapi_file, gamma_file, lamin_file)
            
            df_raw = regionprops_table(mask[0]['image'], img, properties=props)
            
            full_label_list = []
            #create full_label to keep track of condition and original image
            for label in df_raw['label']:
                full_label_list.append('_'.join(mask[0]['filename'].rsplit('_')[0:-4]) + '_nucleus_' + str(label))
            print(full_label_list)
            df_raw['full_label'] = full_label_list
            
            return(df_raw)
        

        dfs_raw = Parallel(n_jobs=self.n_jobs)(delayed(parallel_nuclei_extraction)(mask, dapi, gamma, lamin, props) for mask, dapi, gamma, lamin in zip(mask_files, dapi_files, gamma_files, lamin_files))

        df_all_raw = []
        # Iterate over each dictionary in dfs_raw to split dictionaries with multiple entries into seperate dictionaries
        for df_raw in dfs_raw:
            # Get the number of items in the lists (assuming all lists are the same length)
            num_items = len(next(iter(df_raw.values())))
            
            # Iterate over each index
            for i in range(num_items):
                # Create a new dictionary for the i-th set of values
                single_dict = {key: value[i] for key, value in df_raw.items()}
                df_all_raw.append(single_dict)

        return(df_all_raw)
    
    def save_nucleicrops(self, df, output_dir_mask, output_dir_intensity):
        '''Save extracted nuclei as tiff files
        
        Keyword arguments:
        df -- List of dictionaries, extracted nuclei
        output_dir_mask -- str, output directory for masks
        output_dir_intensity -- str, output directory for intensity images
        '''

        output_path_mask = join(self.cwd, output_dir_mask)
        output_path_intensity = join(self.cwd, output_dir_intensity)
        if not os.path.exists(output_path_mask):
            os.makedirs(output_path_mask, exist_ok=True)
        if not os.path.exists(output_path_intensity):
            os.makedirs(output_path_intensity, exist_ok=True)

        for row in df:
            try:
                filename = str(row['full_label']) + '.tiff'
                output_path_mask = join(self.cwd, output_dir_mask, filename)
                output_path_intensity = join(self.cwd, output_dir_intensity, filename)
                tifffile.imsave(output_path_mask, row['image'])
                tifffile.imsave(output_path_intensity, row['intensity_image'])
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                continue

        return()    

    def feature_extraction(
            self, 
            props, 
            regionprops_features:bool = True, 
            dapi_basic:bool = True, 
            gamma_basic:bool = False, 
            lamin_basic:bool = True,
            morphology:bool = False,
            gamma_foci:bool = True, 
            lamin:bool = True, 
            hc_foci:bool = True, 
            normalize_channels:list = [0,1,2], 
            foci_thresh=None, 
            foci_thresh_lamin=None,
            conditional_foci_thresh = False,
            dapi_only = False
            ):
        
        '''Extracts features from nuclei

        Keyword arguments:
        props -- List of dictionaries, extracted nuclei
        regionprops_features -- bool, extract regionprops features
        dapi_basic -- bool, extract DAPI basic features (default: True)
        lamin_basic -- bool, extract Lamin basic features (default: True)
        gamma_basic -- bool, extract Gamma basic features (default: False)
        gamma_foci -- bool, extract Gamma foci features (default: True)
        lamin -- bool, extract Lamin features (default: True)
        hc_foci -- bool, extract HC foci features (default: True)
        normalize_channels -- list, list of channels to normalize (default: [0,1,2])
        foci_thresh -- float, threshold for foci detection (default: None)
        foci_thresh_lamin -- float, threshold for lamin foci detection (default: None)

        Returns:
        DataFrame containing extracted features
        '''

        df_raw = props

        if dapi_only:
            normalize_channels = [0]
            gamma_basic = False
            lamin_basic = False 
            gamma_foci = False
            lamin = False


        def parallel_feauture_extraction_basics(prop, channel:int = 0, extension:str = ''):
            '''Helper function to extract basic features for each nucleus, possible in parallel

            Keyword arguments:
            prop -- dict, extracted nucleus
            channel -- int, channel to extract features from (default: 0)
            extension -- str, extension for column names (default: '')

            Returns:
            DataFrame containing extracted features
            '''
            
            df_IT = IT.measure_texture_features(prop['image'], prop['intensity_image'][:,:,:,channel]).rename(columns=lambda x: x + extension)
            df_IDF = IDF.measure_intensity_features(prop['image'], prop['intensity_image'][:,:,:,channel]).rename(columns=lambda x: x + extension)


            df = pd.DataFrame(prop).transpose()
            df = pd.concat([df.reset_index(drop=True), df_IT.reset_index(drop=True), df_IDF.reset_index(drop=True)], axis=1)
            #df.rename(columns={'index': 'full_label'}, inplace=True)
            return(df)

        def parallel_feauture_extraction_morphology(prop):
            '''Helper function to extract morphological features for each nucleus, possible in parallel

            Keyword arguments:
            prop -- dict, extracted nucleus
        
            Returns:
            DataFrame containing extracted features
            '''
            
            df_GM = GM.measure_global_morphometrics(prop)
            df_BLC = BLC.measure_curvature_features(prop['image'])

            df = pd.DataFrame(prop).transpose()
            df = pd.concat([df.reset_index(drop=True), df_GM.reset_index(drop=True), df_BLC.reset_index(drop=True)], axis=1)
            #df.rename(columns={'index': 'full_label'}, inplace=True)
            return(df)
        
        def parallel_feauture_extraction_gammafoci(prop, foci_thresh = None, channel:int = 1):
            '''Helper function to extract gamma foci features for each nucleus, possible in parallel

            Keyword arguments:
            prop -- pd.Series, extracted nucleus
            foci_thresh -- float, threshold for foci detection
            channel -- int, channel to extract features from (default: 1)
            '''
            
            df_GF = pd.DataFrame([GF.measure_foci_features(prop['intensity_image'], prop['image'], foci_thresh=foci_thresh, channel = channel)])
            
            df = pd.DataFrame(prop).transpose()
            df = pd.concat([df.reset_index(drop=True), df_GF.reset_index(drop=True)], axis=1)
            #df.rename(columns={'index': 'full_label'}, inplace=True)

            return(df)

        def parallel_feauture_extraction_lamin(prop, foci_thresh = None, channel:int = 2):
            '''Helper function to extract lamin foci features for each nucleus, possible in parallel
            
            Keyword arguments:
            prop -- pd.Series, extracted nucleus
            foci_thresh -- float, threshold for foci detection
            channel -- int, channel to extract features from (default: 2)
            '''
            
            df_LF = LF.measure_lamin_features(prop['intensity_image'], prop['image'], foci_thresh=foci_thresh)
            
            df = pd.DataFrame(prop).transpose()
            df = pd.concat([df.reset_index(drop=True), df_LF.reset_index(drop=True)], axis=1)
            #df.rename(columns={'index': 'full_label'}, inplace=True)
            gc.collect()
            return(df)
        
        def parallel_feauture_extraction_hc_foci(prop, channel:int = 0):
            
            df_HF = HF.measure_hc_foci_features(prop['intensity_image'], prop['image'])
            
            df = pd.DataFrame(prop).transpose()
            df = pd.concat([df.reset_index(drop=True), df_HF.reset_index(drop=True)], axis=1)
            #df.rename(columns={'index': 'full_label'}, inplace=True)
            gc.collect()
            return(df)
       
        if normalize_channels:
            #range-normalize each nucleus and reconstruct dataframe
            df_raw = Parallel(n_jobs=self.n_jobs, return_as='list')(delayed(self.range_normalize_channel_indf)(prop, channels = normalize_channels) for prop in props)
            #df_all = pd.concat([pd.DataFrame()] + list_rows, ignore_index=False, axis=1).transpose()


        if regionprops_features:
            features=['coords', 'equivalent_diameter', 'euler_number', 'extent', 'inertia_tensor', 'inertia_tensor_eigvals', 'local_centroid', 'major_axis_length', 'max_intensity', 'mean_intensity', 'min_intensity', 'minor_axis_length', 'moments', 'moments_central', 'moments_normalized', 'slice', 'solidity', 'weighted_centroid', 'weighted_local_centroid', 'weighted_moments', 'weighted_moments_central', 'weighted_moments_normalized']
            #print(type(df_raw) == list, type(df_raw) == dict)
            if not dapi_only:
                df_raw = Parallel(n_jobs=self.n_jobs)(delayed(lambda row: row.update(regionprops_table(row['image']*1, row['intensity_image'], properties=features)) or row)(row) for row in df_raw)
            else:
                df_raw = Parallel(n_jobs=self.n_jobs)(delayed(lambda row: row.update(regionprops_table(row['image']*1, row['intensity_image'][:,:,:,0], properties=features)) or row)(row) for row in df_raw)


        df_all = pd.DataFrame(df_raw)
    
        if dapi_basic:
            df_all_raw = Parallel(n_jobs=self.n_jobs)(delayed(parallel_feauture_extraction_basics)(prop, channel = 0) for index, prop in df_all.iterrows())

            df_all = pd.concat([pd.DataFrame()] + df_all_raw, ignore_index=False, axis=0)

        if gamma_basic:
            df_all_raw = Parallel(n_jobs=self.n_jobs)(delayed(parallel_feauture_extraction_basics)(prop, channel = 1, extension='_gamma') for index, prop in df_all.iterrows())

            df_all = pd.concat([pd.DataFrame()] + df_all_raw, ignore_index=False, axis=0)

        if lamin_basic:
            df_all_raw = Parallel(n_jobs=self.n_jobs)(delayed(parallel_feauture_extraction_basics)(prop, channel = 2, extension='_lamin') for index, prop in df_all.iterrows())

            df_all = pd.concat([pd.DataFrame()] + df_all_raw, ignore_index=False, axis=0)

        if morphology:
            #causing memory issues when running for many images
            df_all_raw = Parallel(n_jobs=self.n_jobs)(delayed(parallel_feauture_extraction_morphology)(prop) for index, prop in df_all.iterrows())

            df_all = pd.concat([pd.DataFrame()] + df_all_raw, ignore_index=False, axis=0)

        if gamma_foci:

            #measure mean and std of channel to determine threshold
            if conditional_foci_thresh:
                pixel_intensities = Parallel(n_jobs=self.n_jobs, return_as='list')(delayed(lambda x: x.flatten())(prop['intensity_image'][:,:,:,1]) for index, prop in df_all.iterrows())
                pixel_intensities = np.concatenate(pixel_intensities)
                if not foci_thresh:
                    foci_thresh = np.mean(pixel_intensities) + 2.5*np.std(pixel_intensities)

            df_all_raw = Parallel(n_jobs=self.n_jobs)(delayed(parallel_feauture_extraction_gammafoci)(prop, foci_thresh) for index, prop in df_all.iterrows())

            df_all = pd.concat([pd.DataFrame()] + df_all_raw, ignore_index=False, axis=0)
            #df_all.set_index('full_label', inplace=True)

        if lamin:
            #measure mean and std of channel to determine threshold
            if conditional_foci_thresh:
                pixel_intensities = Parallel(n_jobs=self.n_jobs, return_as='list')(delayed(lambda x: x.flatten())(prop['intensity_image'][:,:,:,2]) for index, prop in df_all.iterrows())
                pixel_intensities = np.concatenate(pixel_intensities)
                if not foci_thresh_lamin:
                    foci_thresh_lamin = np.mean(pixel_intensities) + 2.5*np.std(pixel_intensities)

            df_all_raw = Parallel(n_jobs=self.n_jobs)(delayed(parallel_feauture_extraction_lamin)(prop, foci_thresh_lamin) for index, prop in df_all.iterrows())

            df_all = pd.concat([pd.DataFrame()] + df_all_raw, ignore_index=False, axis=0)
            #df_all.set_index('full_label', inplace=True)

        if hc_foci:
            df_all_raw = Parallel(n_jobs=self.n_jobs)(delayed(parallel_feauture_extraction_hc_foci)(prop) for index, prop in df_all.iterrows())

            df_all = pd.concat([pd.DataFrame()] + df_all_raw, ignore_index=False, axis=0)
            #df_all.set_index('full_label', inplace=True)

        #df_all['full_label'] = df_all['full_label'].apply(lambda x: x[-1])
        df_all.set_index('full_label', inplace=True)

        #print(df_all.head(5))

        return(df_all)

    def range_normalize_channel_indf(self, row, channels):
        for channel in channels:
            img = row['intensity_image'].copy()
            mask = row['image'].astype(bool)
            img[:,:,:,channel] = img[:,:,:,channel] * row['image']
            masked_values = img[:,:,:,channel][mask]
            min_val_inside_mask = masked_values.min()
            max_val_inside_mask = masked_values.max()
            img[:,:,:,channel] = ((img[:,:,:,channel] - min_val_inside_mask) / (max_val_inside_mask- min_val_inside_mask)) * 255
            img[:,:,:,channel] = img[:,:,:,channel].clip(min=0)
            row['intensity_image'] = img
        return row


    