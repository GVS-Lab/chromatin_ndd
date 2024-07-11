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
from stardist.models import StarDist2D
from csbdeep.utils import normalize

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

    def open_close_img_old(self, images, series:bool = True):
        
        if series:
            for i, img in enumerate(images):
                images[i]['2d_mask'] = self.open_close_img(img, False)
            return(images)
        else:
            images['2d_mask'] = area_opening(images['2d_mask'])
            images['2d_mask'] = area_closing(images['2d_mask'])
            return(images['2d_mask'])

    def create_open_close_masks(self, images, filename:str = None, output_path:str = None):
        for img in tqdm(images):
            filename = '_'.join(img['filename'].rsplit('_')[0:3])
            
            img = area_opening(img['image'])
            img = area_closing(img)

            self.save_image(img, filename + '_open_close_masks', output_path)
        return()

    def label_segments_2d(self, images, series:bool = True):

        if series:
            for i, img in enumerate(images):
                images[i]['2d_mask'] = self.label_segments_2d(img, False)
            return(images)
        else:
            images['2d_mask'] = label(images['2d_mask'])
            return(images['2d_mask'])

    def clear_masks(self, images, output_path, filename_extension:str = '_cleared', allow_z=True):

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

        for img in images:
            filename = '_'.join(img['filename'].rsplit('_')[0:5])
            processed_img = clear_border_2d(img['image'], allow_z)
            processed_img = remove_2d_regions(processed_img)
            processed_img = label(processed_img)
            self.save_image(processed_img, filename + '_' + filename_extension, output_path)
        return()
    
    def clear_masks_new(self, images, output_path, allow_z=True):

        
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


    def subtract_background():
        return()
    

    def match_stacks(self, series_of_img, thresh=0.5, first_pass=True):
    
        def calc_iou(region_a, region_b):

            intersection = len(list(set(region_a) & set(region_b)))
            region_a.extend(region_b)
            union = len(set(region_a))    
        
            return(intersection/union)


        def match_single(curr, next, thresh): #return pairs of matching labels and scores for those, only if above threshold
            matches = []
            scores = []

            #get labels in curr and next image
            curr_labels, next_labels = np.unique(curr)[1::], np.unique(next)[1::]
            #curr_regions will be a list of [cl,[pixel-indexes where label cl is true]]
            curr_regions, next_regions = [[cl,[(y,x) for y,x in zip(np.where(curr == cl)[0], np.where(curr == cl)[1])]] for cl in curr_labels], [[nl,[(y,x) for y,x in zip(np.where(next == nl)[0], np.where(next == nl)[1])]] for nl in next_labels]
            #return(next_regions)
        
            for region_a in curr_regions:
                for region_b in next_regions:
                    iou = calc_iou(region_a[1], region_b[1]) #[1] accesses the list of pixel-indexes, [0] would access the label-value
                    if iou > thresh:
                        matches.append((region_a[0], region_b[0]))
                        scores.append(iou)
        
            return(matches,scores)
        
        #produce new masks with continuing labels
        original_imgs = np.asarray(series_of_img)
        imgs = np.copy(original_imgs)

        amount_labels=0
        for z_stack in original_imgs:
            amount_labels += max(np.unique(z_stack))
        
        new_labels = [i+1 for i in range(amount_labels)]

        for i,z_stack in enumerate(imgs):
            curr_labels = np.unique(z_stack)[1::] #exclude first entry (value 0 -> background)
            for label in curr_labels:
                imgs[i,:,:][z_stack == label] = new_labels.pop(0)



        i=1
        while i < imgs.shape[0]:
            matches, scores = match_single(imgs[i-1], imgs[i], thresh) 
            for match in matches:
                imgs[i,:,:][imgs[i] == match[1]] = min(match)
            i+=1
            
        if first_pass:
            imgs = self.match_stacks(imgs[::-1, :, :],thresh, False) #pass reversed z-stack 
            imgs = imgs[::-1, :, :] #reverse z stack order back

            
        #return(match_single(imgs[0], imgs[1]), thresh)
        
        return(imgs)
    
    def create_stardist_masks(self, images, thresh=0.5, filename:str = None, output_path:str = None, series:bool = True, model=None):

        if series:
            model = StarDist2D.from_pretrained('2D_versatile_fluo')
            for img in tqdm(images, desc='Processing images'):
                print(f"Processing {img['filename']}")
                filename = '_'.join(img['filename'].rsplit('_')[0:5])
                self.create_stardist_masks(img, thresh, filename, output_path, False, model=model)
            return(images)
        else:
            if model == None:
                model = StarDist2D.from_pretrained('2D_versatile_fluo')
                print('Singleprediction')

            stardist_masks = list()

            for z in images['image']:
                labels, _ = model.predict_instances(normalize(z))
                stardist_masks.append(labels)
            
            stardist_masks = self.match_stacks(stardist_masks, thresh)

            self.save_image(stardist_masks, filename + '_stardist_masks', output_path)

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
                mask[z] = remove_small_holes(mask[z], 1000)

            self.save_image(mask, filename + '_otsu_masks', output_path)

        Parallel(n_jobs=self.n_jobs)(delayed(com)(img, thresh_value, filename, output_path) for img in images)

    def label_3d(self, images, filename:str = None, output_path:str = None):

        def l3d(img, filename:str, output_path:str):
            filename = '_'.join(img['filename'].rsplit('_')[0:-2])
            img = label(img['image'])
            self.save_image(img, filename + '_otsu_mask_labeled', output_path)
        
        Parallel(n_jobs=self.n_jobs)(delayed(l3d)(img, filename, output_path) for img in images)
    
    




    
    