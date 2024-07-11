import argparse
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from readlif.reader import LifFile 
import sys
import pandas as pd
import pickle



import preprocessing
import segmentation
import feature_extraction
import time

import warnings

warnings.filterwarnings("ignore")


def run_preprocessing(input_dir:str):
    '''Run preprocessing pipeline

    Keyword arguments:
    input_dir -- str, path to the input directory
    '''

    channel_map=['dapi', 'gamma', 'lamin']
    Preprocessor = preprocessing.Preprocessor(channel_map, n_jobs=int(args.n_jobs))

    input_path = join(os.getcwd(), input_dir)
    #get all filenames (.lif files) in the input directory
    filenames = set([f.split('.')[0] for f in listdir(input_path) if (isfile(join(input_path, f)) and f.split('.')[1] == 'lif')])
    #create a folder for each lif file
    for filename in filenames:
        folder_path = join(input_path, filename)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
    
    #for each lif file, extract images, save as numpy array, normalize and adjust gamma of dapi channel
    for filename in tqdm(filenames, desc=f'Preprocessing {filename}'):
        #get images from lif file
        print(filename)
        input_path = join(os.getcwd(), input_dir, filename)
        print('     - Extracting images from lif file...')
        new = LifFile(input_path + '.lif')
        img_list = [i for i in new.get_iter_image()]

        #for each image, extract channels and save in list dictionaries (channel:imagedata), then safe as numpy array
        print('     - Extracting channels and saving as numpy array...')
        images = Preprocessor.extract_map_channels(img_list)
        Preprocessor.pillow_to_numpy_3d(images, filename=filename, output_path=input_path)

        print('     - Normalizing images...')
        images = Preprocessor.load_images(join(input_path, 'original_files'), 'dapi')
        Preprocessor.range_normalize(images, join(input_path, 'normalized'))

        print('     - Adjusting gamma...')
        images = Preprocessor.load_images(join(input_path, 'normalized'), 'dapi')
        Preprocessor.adjust_gamma(images, join(input_path, 'gamma_adjusted'))

    return()


def run_segmentation(input_dir:str):
    '''Run segmentation pipeline

    Keyword arguments:
    input_dir -- str, path to the input directory
    '''

    Segmenter = segmentation.Segmenter(n_jobs=int(args.n_jobs))
    input_path = join(os.getcwd(), input_dir)
    filenames = set([f.split('.')[0] for f in listdir(input_path) if (isfile(join(input_path, f)) and f.split('.')[1] == 'lif')])

    for filename in tqdm(filenames, desc='Segmentation'):

        input_path_curr = join(input_path, filename)

        print('     - Creating otsu masks...')
        images = Segmenter.load_images(join(input_path_curr, 'gamma_adjusted'), 'dapi')
        Segmenter.create_otsu_masks(images,thresh_value=1, output_path=join(input_path_curr, 'otsu_masks'))

        print('     - Labeling masks...')
        images = Segmenter.load_images(join(input_path_curr, 'otsu_masks'), '')
        Segmenter.label_3d(images, output_path=join(input_path_curr, 'otsu_masks_labeled'))

        print('     - Clearing masks...')
        images = Segmenter.load_images(join(input_path_curr, 'otsu_masks_labeled'), '')
        Segmenter.clear_masks(images, output_path=join(input_path_curr, 'otsu_masks_cleared'), allow_z = False)


    return()


def run_nuclear_crop_extraction(input_dir:str):
    '''Run nuclear crop extraction pipeline

    Keyword arguments:
    input_dir -- str, path to the input directory
    '''

    input_path = join(os.getcwd(), input_dir)
    filenames = set([f.split('.')[0] for f in listdir(input_path) if (isfile(join(input_path, f)) and f.split('.')[1] == 'lif')])

    Extractor = feature_extraction.Extractor(n_jobs=int(args.n_jobs))
    
    for filename in tqdm(filenames, desc='Nuclear crop extraction'):

        input_path_curr = join(input_path, filename)
        print('     - Extracting nuclei...')
        df_raw = Extractor.extract_nuclei_3d(input_path_curr, 'otsu_masks_cleared')
        with open (join(input_path_curr, 'regionprops_3d_otsu1.pkl'), 'wb') as f:
            pickle.dump(df_raw, f)

        print('     - Saving nuclear crops...')
        Extractor.save_nucleicrops(df_raw, join(input_path_curr, 'nuclei_masks'), join(input_path_curr, 'nuclei_images'))

    return()

def run_nuclear_chromatin_feat_ext(input_dir:str, output_dir:str):
    input_path = join(os.getcwd(), input_dir)
    output_path = join(os.getcwd(), output_dir)
    filenames = set([f.split('.')[0] for f in listdir(input_path) if (isfile(join(input_path, f)) and f.split('.')[1] == 'lif')])

    Extractor = feature_extraction.Extractor(n_jobs=int(args.n_jobs))

    
    for filename in tqdm(filenames, desc='Feature extraction'):

        input_path_curr = join(input_path, filename)
        output_path_curr = join(output_path, filename)
        print('     - Extracting features...')
        with open(join(input_path_curr, 'regionprops_3d_otsu1.pkl'), 'rb') as f:
            props = pickle.load(f)
        df = Extractor.feature_extraction(props, dapi_only=True)
        df.to_pickle(join(output_path_curr, 'regionprops_3d_extended.pkl'))

    return()


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="Relative path to the input directory")
parser.add_argument("--output_dir", help="Relative path to the output directory (only for feature extraction)")
parser.add_argument("--n_jobs", help="Number of jobs to run in parallel")
parser.add_argument("--run_preprocessing", help="Run preprocessing pipeline", action='store_true')
parser.add_argument("--run_segmentation", help="Run segmentation pipeline", action='store_true')
parser.add_argument("--run_nuclear_crop_extraction", help="Run nuclear crop extraction pipeline", action='store_true')
parser.add_argument("--run_nuclear_chromatin_feat_ext", help="Run nuclear chromatin feature extraction pipeline", action='store_true')
parser.add_argument("--run_all", help="Run all pipelines", action='store_true')
args = parser.parse_args()

if args.n_jobs == 'all':
    args.n_jobs = os.cpu_count()
    

if args.run_preprocessing or args.run_all:
    print('Running preprocessing...')
    start_time = time.time()
    run_preprocessing(args.input_dir)
    end_time = time.time()
    print('Preprocessing completed in {:.2f} seconds'.format(end_time - start_time))

if args.run_segmentation or args.run_all:
    print('Running segmentation...')
    start_time = time.time()
    run_segmentation(args.input_dir)
    end_time = time.time()
    print('Segmentation completed in {:.2f} seconds'.format(end_time - start_time))

if args.run_nuclear_crop_extraction or args.run_all:
    print('Running nuclear crop extraction...')
    start_time = time.time()
    run_nuclear_crop_extraction(args.input_dir)
    end_time = time.time()
    print('Nuclear crop extraction completed in {:.2f} seconds'.format(end_time - start_time))

if args.run_nuclear_chromatin_feat_ext or args.run_all:
    print('Running nuclear chromatin feature extraction...')
    start_time = time.time()
    run_nuclear_chromatin_feat_ext(args.input_dir, args.output_dir)
    end_time = time.time()
    print('Nuclear chromatin feature extraction completed in {:.2f} seconds'.format(end_time - start_time))

print('All requested pipelines completed at ' + str(args.input_dir))