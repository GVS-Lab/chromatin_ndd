import os
from os import listdir
from os.path import isfile, join


import numpy as np
import PIL
import tifffile
import skimage #
from skimage.exposure import adjust_gamma
from readlif.reader import LifFile
import readlif.reader
import json
from joblib import Parallel, delayed



class Preprocessor:
    def __init__(self, channel_map:list, n_jobs:int = 1):
        self.channel_map = channel_map
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
        if inclusion2 == None:
            inclusion2 = inclusion1

        input_path = join(self.cwd, input_dir)
        files = [{'filename': f, 'image': tifffile.imread(join(input_path, f))}
            for f in listdir(input_path) 
            if isfile(join(input_path, f)) and inclusion1 in f and inclusion2 in f and exclusion not in f and filetype in f]

        return(files)


    def extract_map_channels(self, img_list, series:bool = True):
        if series:
            img_dict_channels_series = []
            img_dict_channels_series = Parallel(n_jobs=self.n_jobs)(delayed(self.extract_map_channels)(entry, False) for entry in img_list)
            return img_dict_channels_series
        else:
            img_dict_channels = dict()
            for i,ch in enumerate(self.channel_map):
                img_dict_channels[ch+'_pil'] = [img for img in img_list.get_iter_z(t=0, c=i)]
            img_dict_channels['original'] = img_list
            return(img_dict_channels)

    def pillow_to_numpy_3d(self, image_pil, filename:str, output_path:str, series: bool = True):
        if series:
            Parallel(n_jobs=self.n_jobs)(delayed(self.pillow_to_numpy_3d)(img, filename + f'_{i}', output_path, False) for i, img in enumerate(image_pil))
            return()

        else:
            for channel in self.channel_map:
                image_np = np.ndarray(shape=(len(image_pil[channel+'_pil']), image_pil[channel+'_pil'][0].size[0], image_pil[channel+'_pil'][0].size[1]))
                for i, slice in enumerate(image_pil[channel+'_pil']):
                    image_np[i,:,:] = slice
                self.save_image(image_np, filename + f'_{channel}', join(output_path, 'original_files'))
                self.save_image(image_pil['original'], filename, join(output_path, 'original_files'))
            return()



    def project_max_z(self, images_3d, datatype:str = 'pillow', series:bool = True, channels:list = None):
        if channels == None:
            channels = self.channel_map

        if series:
            for i,img in enumerate(images_3d):
                images_3d[i] = {**images_3d[i], **self.project_max_z(img, datatype, False, channels)}
            return(images_3d)
        else:
            return({f'{channel}_max_z' : self.project_max_z_single(images_3d[channel], datatype) for channel in channels})


    def project_max_z_single(self, image_3d, datatype):
        if datatype =='numpy':
                #input a single 3d image as np.ndarray
                max_z_image = np.amax(image_3d, axis=0)
                return max_z_image
        elif datatype == 'pillow':
              return(
                   self.project_max_z_single( 
                        self.pillow_to_numpy_3d(image_3d),
                        'numpy'
                        )
                    )
                
    def adjust_gamma(self, images, output_path, gamma = 0.7):

        def adjg(img, output_path, gamma):
            filename = '_'.join(img['filename'].rsplit('_')[0:-1]) + '_gamma_adjusted'
            gamma_adjusted = adjust_gamma(img['image'], gamma)
            self.save_image(gamma_adjusted, filename, output_path)

        Parallel(n_jobs=self.n_jobs)(delayed(adjg)(img, output_path, gamma) for img in images)

        return()

    def range_normalize(self, images, output_path):

        def normalize_image(img, output_path):
            filename = img['filename'].split('.')[0] + '_normalized'
            normalized = (255 * (img['image'] - np.min(img['image'] ))) / (np.max(img['image'] ) - np.min(img['image'] ))
            self.save_image(normalized, filename, output_path)

        Parallel(n_jobs=self.n_jobs)(delayed(normalize_image)(img, output_path) for img in images)
        return()


