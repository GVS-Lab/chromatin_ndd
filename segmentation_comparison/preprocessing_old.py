import numpy as np
import PIL
import skimage
from skimage.exposure import adjust_gamma
from readlif.reader import LifFile



class Preprocessor:
    def __init__(self, channel_map:list):
        self.channel_map = channel_map

    def extract_map_channels(self, img_list, series:bool = True):
        if series:
             img_dict_channels_series = []
             for entry in img_list:
                 img_dict_channels_series.append(self.extract_map_channels(entry, False))
             return(img_dict_channels_series)
        else:
            img_dict_channels = dict()
            for i,ch in enumerate(self.channel_map):
                img_dict_channels[ch+'_pil'] = [img for img in img_list.get_iter_z(t=0, c=i)]
            img_dict_channels['original'] = img_list
            return(img_dict_channels)


    def pillow_to_numpy_3d(self, image_pil, series: bool = True):
        if series:
            for i,img in enumerate(image_pil):
                image_pil[i] = {**image_pil[i], **self.pillow_to_numpy_3d(img, False)}
            return(image_pil)

        else:
            dict_images_np = {}
            for channel in self.channel_map:
                image_np = np.ndarray(shape=(len(image_pil[channel+'_pil']), image_pil[channel+'_pil'][0].size[0], image_pil[channel+'_pil'][0].size[1]))
                for i, slice in enumerate(image_pil[channel+'_pil']):
                    image_np[i,:,:] = slice
                dict_images_np[channel] = image_np
            return(dict_images_np)



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
                
    def adjust_gamma(self, images_3d, gamma:float, series:bool = True, channels:list = None):
        if channels == None:
            channels = self.channel_map

        if series:
            for i,img in enumerate(images_3d):
                        images_3d[i] = {**images_3d[i], **self.adjust_gamma(img, gamma, False, channels)}
            return(images_3d)
        else:
            for channel in channels:
                images_3d[channel] = adjust_gamma(images_3d[channel], gamma=gamma)
            return(images_3d)


