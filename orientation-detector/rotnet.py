# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 1:52:27 2018

@author: Cagri

rotnet.py:
Custom data generator for canonical orientation cerrection classifier. Generates
rotated images on the fly. Receives batches of images and returns batches of 
rotated images with their tags. The input can be either a directory which
contains images, or a list of arrays (like mnist). If the input is a directory,
the code iterates through sub-folders and finds all the [*.png, *.jpg, *.jpeg] 
files in the given directory.

The code can be tested by runnig the main function.

Reference: https://d4nst.github.io/2017/01/12/image-orientation/
"""

# FIXME: A known bug in image padding part. (line 299). Padding might not be necessary

from __future__ import division

import os, math, cv2, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint
from keras.preprocessing.image import Iterator
from keras.utils.np_utils import to_categorical
import keras.backend as K

#%% additional functions-------------------------------------------------------
def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - abs(abs(x - y) - 180)


def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))


def angle_error_regression(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a float number between 0 and 1.
    """
    return K.mean(angle_difference(y_true * 360, y_pred * 360))


def binarize_images(x):
    """
    Convert images to range 0-1 and binarize them by making
    0 the values below 0.1 and 1 the values above 0.1.
    """
    x /= 255
    x[x >= 0.1] = 1
    x[x < 0.1] = 0
    return x


def rotate(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
    
    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [(np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
                      (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
                      (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
                      (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]])
    
    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform    
    return cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR) 


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (bb_w - 2 * x,
            bb_h - 2 * y)


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def crop_largest_rectangle(image, angle, height, width):
    """
    Crop around the center the largest possible rectangle
    found with largest_rotated_rect.
    """
    return crop_around_center(image, *largest_rotated_rect(width, height,
                                                           height, 
                                                           math.radians(angle)))


def rand_saturation(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s += randint(0, value)
    final_hsv = cv2.merge((h, s, v))
    
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def contrast_brightness(image, cont=False, bright=False):
    if cont: alpha = random.uniform(1.0, cont)
    else: alpha = 1.0
    if bright: beta = random.uniform(-bright, bright)
    else: beta = 0    
    
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        

def add_gaussian_noise(image, sigma): 
    sigma = random.uniform(0, sigma)
    height, width, chan =  image.shape
    mean = 0
    gaussian_noise = np.random.normal(mean, sigma, (height, width, chan))
    gaussian_noise = gaussian_noise.reshape(height, width, chan)
    
    return image + gaussian_noise


def add_gaussian_blur(image, ksize):
    ksize = random.randrange(3, ksize, 2)
    xsigma = 0
    ysigma = 0

    return cv2.GaussianBlur(image, (ksize, ksize), xsigma, ysigma)


def add_color(image, color_range):
    channel = random.choice(['R', 'G', 'B'])
    value = random.randrange(color_range)
    if channel == 'R':
        image[..., 0] = np.where((255 - image[..., 0]) < value,255,image[..., 0]+value)
        
    elif channel == 'G':
        image[..., 1] = np.where((255 - image[..., 1]) < value,255,image[..., 1]+value)
        
    elif channel == 'B':
        image[..., 2] = np.where((255 - image[..., 2]) < value,255,image[..., 2]+value)   
    
    return image



def rand_crop(image, height, width, rate):
    
    return image[int(height*random.uniform(0, rate)):int(height*random.uniform(1, 1-rate)),
                 int(width*random.uniform(0, rate)):int(width*random.uniform(1, 1-rate))]


def generate_image(image, angle, size=None, padding=False, crop_center=False,
                   crop_largest_rect=False, random_crop=False, 
                   gaussian_noise=False, brightness=False, color=False, 
                   contrast=False, saturation=False, blur=False):
    """
    Generate a valid rotated image for the RotNetDataGenerator. If the
    image is rectangular, the crop_center option should be used to make
    it square. To crop out the black borders after rotation, use the
    crop_largest_rect option. To resize the final image, use the size
    option.
    """
    height, width = image.shape[:2]
    if crop_center:
        if width < height:
            height = width
        else:
            width = height

    if random_crop:
        image = rand_crop (image, height, width, random_crop)
        
    if color:
        image = add_color (image, color)
        
    if blur:
        image = add_gaussian_blur(image, blur)
    
    if gaussian_noise:
        image = add_gaussian_noise(image, gaussian_noise)
        
    if brightness or contrast:
        image = contrast_brightness(image, cont=contrast, bright=brightness)
        
    if saturation:
        image = rand_saturation(image, saturation)

    if crop_largest_rect:
        image = crop_largest_rectangle(image, angle, height, width)

    # add black borders to make the image square
    if padding:
        max_size = max(height, width)
        tb = int((max_size-height)/2)
        lr = int((max_size-width)/2)
        image = cv2.copyMakeBorder(image, tb, tb , lr, lr, cv2.BORDER_CONSTANT, value=[0,0,0])
        image = cv2.copyMakeBorder(image, tb, tb , lr, lr, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    if not isinstance(angle, str) and angle !=0: image = rotate(image, angle)
    
    if size: image = cv2.resize(image,(size[1], size[0]), interpolation=cv2.INTER_CUBIC)

    return image

#%% image data generator-------------------------------------------------------
class RotNetGen(Iterator):
    """
    Given a NumPy array of images or a list of image paths,
    generate batches of rotated images and rotation angles on-the-fly.
    """

    def __init__(self, input, target_size=(224,224), target_classes=[0, 90, 180, 270],
                 color_mode='rgb', class_mode='categorical', batch_size=32,                 
                 preprocessing_function=None, shuffle=False, seed=None, 
                 check_images=True, save_to_dir=False, padding=False, crop_center=False, 
                 crop_largest_rect=False, random_crop=False, gauss_noise=False,
                 brightness=False, contrast=False, saturation=False, blur=False,
                 add_color=False):

        self.images = None
        self.root_folder_path = None
        self.filenames = None
        self.target_size = target_size
        self.target_classes = target_classes
        self.color_mode = color_mode
        self.class_mode = class_mode
        self.batch_size = batch_size
        self.preprocessing_function = preprocessing_function
        self.shuffle = shuffle
        self.seed = seed
        self.check_images = check_images
        self.save_to_dir = save_to_dir
        self.padding = padding
        self.crop_center = crop_center
        self.crop_largest_rect = crop_largest_rect
        self.random_crop = abs(random_crop)
        self.gaussian_noise = abs(gauss_noise)
        self.brightness = abs(brightness)
        self.contrast = abs(contrast)
        self.saturation = abs(saturation)
        self.blur = abs(blur)
        self.color = abs(add_color)
        
        if brightness and brightness < 1.0:
            raise ValueError('Random brightness range [{}] cannot be smaller than 1.0.'.format(brightness))
        
        if blur:
            if blur%2 == 0 or not isinstance(blur, int): 
                raise ValueError('Random blur kernel size [{}] must be an odd integer number.'.format(blur))
                    
        if self.color and self.color > 255: self.color = 255
        
        if int(self.color_mode == 'rgb'): self.target_size = target_size + (3,)
        
        rot_angs = []
        non_rots_dict = {}
        for clss in self.target_classes:
            if isinstance(clss, int):
                rot_angs.append(clss)
            elif isinstance(clss, str):
                non_rots_dict[clss.lower()] = []
            else:
                raise ValueError('Class labels must be either integer or string.')
        
        if self.color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode. [{}] is expected to be "rgb" or "grayscale".'.format(self.color_mode))
                             

        # check whether the input is a NumPy array or a list of paths
        if isinstance(input, (np.ndarray)):
            self.images = input
            self.N = self.images.shape[0]
            if not self.target_size:
                self.target_size = self.images.shape[1:]
                # add dimension if the images are greyscale
                if len(self.target_size) == 2:
                    self.target_size = self.target_size + (1,)
        else:
            root_img_fold = input          
            rot_imgs = []
            non_rot_classes = non_rots_dict.keys()
            self.filenames = []
            path_list = []
            if self.check_images: print('\n[INFO#{}]: Unreadible files being checked. This might take several minutes.'. format(os.path.basename(__file__)))
            for subdir, dirs, image_files in os.walk(root_img_fold):
                for file in image_files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path_list.append(os.path.join(subdir, file))
            
            print('\nImage indexing...')          
            for path in tqdm(path_list, desc='is in progress'):            
                splitted_path = path.split(os.path.sep)
                splitted_path = [string.lower() for string in splitted_path]
                non_rot_folder = list(set(non_rot_classes)&set(splitted_path))
                if len(non_rot_folder)>1:
                    raise ValueError ('Class folders cannot be nested: ', non_rot_folder)
                if self.check_images:
                    self.unreadibles = []
                    image = cv2.imread(path, 1)
                    if not image is None:
                        if non_rot_folder:
                            non_rots_dict[non_rot_folder[0]].append(path)                                                                            
                        elif path not in rot_imgs:
                            rot_imgs.append(path)
                    else:
                        self.unreadibles.append(path)
                else:
                    if non_rot_folder:
                            non_rots_dict[non_rot_folder[0]].append(path)                                                                            
                    elif path not in rot_imgs: 
                        rot_imgs.append(path)
            
            for i in range(len(rot_angs)): self.filenames.extend(rot_imgs)
            print('Found {}(*{}) images to be rotated.'.format(len(rot_imgs), len(rot_angs)))            
            for k in non_rots_dict.keys():
                print('Found {} images belonging to class ({}).'.format(len(non_rots_dict[k]), k))
                self.filenames.extend(non_rots_dict[k])                                     
            self.N = len(self.filenames)
        
        self.class_list = np.concatenate([i*np.ones(len(rot_imgs)) for i in range(len(rot_angs))])
        for k in non_rots_dict.keys():
            unique = np.unique (self.class_list)
            self.class_list = np.concatenate([self.class_list, len(unique)*np.ones(len(non_rots_dict[k]))])        
        self.class_list = self.class_list.astype(dtype='uint8')
        
        self.classes = np.array([], dtype='uint8')
        if not self.shuffle: self.classes=self.class_list
        else: self.classes = np.array([], dtype='uint8')
           
        self.class_indices = dict(zip([str(i) for i in (self.target_classes)], 
                                       list(range(len(self.target_classes)))))        
        
        self.samples =self.N        
        super(RotNetGen, self).__init__(self.N, batch_size, shuffle, seed)

    
    def _get_batches_of_transformed_samples(self, index_array):
        # create array to hold the images and corresponding labels
        batch_x = np.zeros((len(index_array),) + self.target_size, dtype='float32')
        batch_y = np.zeros(len(index_array), dtype='float32')
        
        # generate rotated images and corresponding labels
        for i, j in enumerate(index_array):            
            if self.filenames is None:
                image = self.images[j]
                if len(image.shape) == 2: image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            else:
                is_color = int(self.color_mode == 'rgb')
                image = cv2.imread(self.filenames[j], is_color)
                if is_color and not image is None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                
            # do nothing if the image is none
            if not image is None:
                generated_im = generate_image(image, 
                                              self.target_classes[self.class_list[j]], 
                                              self.target_size[:2],
                                              padding=self.padding,
                                              crop_center=self.crop_center,
                                              crop_largest_rect=self.crop_largest_rect,
                                              random_crop=self.random_crop,
                                              gaussian_noise=self.gaussian_noise,
                                              brightness=self.brightness,
                                              contrast=self.contrast,
                                              saturation=self.saturation,
                                              blur=self.blur, color=self.color)
                                                    
                if self.save_to_dir:
                    splitted = self.filenames[j].split(os.path.sep)
                    save_dir = os.path.join(self.save_to_dir, str(self.class_list[j])+'_'+splitted[-1])
                    cv2.imwrite(save_dir, cv2.cvtColor(generated_im, cv2.COLOR_BGR2RGB))
                    
                if self.preprocessing_function: generated_im = self.preprocessing_function(generated_im)
                
                # add dimension to account for the channels if the image is greyscale
                if generated_im.ndim == 2: generated_im = np.expand_dims(generated_im, axis=2)
                
                batch_x[i] = generated_im
                batch_y[i] = self.class_list[j]
        
        if self.shuffle: 
            self.classes = np.concatenate([self.classes, batch_y])
            if len(self.classes) > len(self.filenames): 
                self.classes = self.classes[-len(self.filenames):]
        
        batch_y = to_categorical(batch_y, len(self.target_classes))            
        return batch_x, batch_y
    
    def next(self):
        with self.lock:
            # get input data index and size of the current batch
            index_array = next(self.index_generator)
        # create array to hold the images
        return self._get_batches_of_transformed_samples(index_array)


def display_examples(model, input, num_images=5, size=None, crop_center=False,
                     crop_largest_rect=False, preprocess_func=None, save_path=None):
    """
    Given a model that predicts the rotation angle of an image,
    and a NumPy array of images or a list of image paths, display
    the specified number of example images in three columns:
    Original, Rotated and Corrected.
    """

    if isinstance(input, (np.ndarray)):
        images = input
        N, h, w = images.shape[:3]
        if not size:
            size = (h, w)
        indexes = np.random.choice(N, num_images)
        images = images[indexes, ...]
    else:
        images = []
        filenames = input
        N = len(filenames)
        indexes = np.random.choice(N, num_images)
        for i in indexes:
            image = cv2.imread(filenames[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        images = np.asarray(images)

    x = []
    y = []
    for image in images:
        rotation_angle = np.random.randint(360)
        rotated_image = generate_image(
            image,
            rotation_angle,
            size=size,
            crop_center=crop_center,
            crop_largest_rect=crop_largest_rect)
        x.append(rotated_image)
        y.append(rotation_angle)

    x = np.asarray(x, dtype='float32')
    y = np.asarray(y, dtype='float32')

    if x.ndim == 3:
        x = np.expand_dims(x, axis=3)

    y = to_categorical(y, 360)

    x_rot = np.copy(x)

    if preprocess_func:
        x = preprocess_func(x)

    y = np.argmax(y, axis=1)
    y_pred = np.argmax(model.predict(x), axis=1)

    plt.figure(figsize=(10.0, 2 * num_images))

    title_fontdict = {'fontsize': 14, 'fontweight': 'bold'}

    fig_number = 0
    for rotated_image, true_angle, predicted_angle in zip(x_rot, y, y_pred):
        original_image = rotate(rotated_image, -true_angle)
        if crop_largest_rect:
            original_image = crop_largest_rectangle(original_image, -true_angle, *size)

        corrected_image = rotate(rotated_image, -predicted_angle)
        if crop_largest_rect:
            corrected_image = crop_largest_rectangle(corrected_image, -predicted_angle, *size)

        if x.shape[3] == 1:
            options = {'cmap': 'gray'}
        else:
            options = {}

        fig_number += 1
        ax = plt.subplot(num_images, 3, fig_number)
        if fig_number == 1:
            plt.title('Original\n', fontdict=title_fontdict)
        plt.imshow(np.squeeze(original_image).astype('uint8'), **options)
        plt.axis('off')

        fig_number += 1
        ax = plt.subplot(num_images, 3, fig_number)
        if fig_number == 2:
            plt.title('Rotated\n', fontdict=title_fontdict)
        ax.text(0.5, 1.03, 'Angle: {0}'.format(true_angle),
                horizontalalignment='center', transform=ax.transAxes, fontsize=11)
        
        plt.imshow(np.squeeze(rotated_image).astype('uint8'), **options)
        plt.axis('off')

        fig_number += 1
        ax = plt.subplot(num_images, 3, fig_number)
        corrected_angle = angle_difference(predicted_angle, true_angle)
        if fig_number == 3:
            plt.title('Corrected\n', fontdict=title_fontdict)
        ax.text(0.5, 1.03, 'Angle: {0}'.format(corrected_angle),
                horizontalalignment='center', transform=ax.transAxes, fontsize=11)
        
        plt.imshow(np.squeeze(corrected_image).astype('uint8'), **options)
        plt.axis('off')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    if save_path:
        plt.savefig(save_path)

#%% main function to check the output------------------------------------------
from keras.applications.vgg16 import preprocess_input

if __name__ == '__main__':
    # Any folder directory which contains some images
    os.chdir(os.path.dirname(__file__))
    wdir = os.getcwd()
    directory = 'D:\\Lectures\\Thesis\\orienter\\cats'
    savepath = 'D:\\Lectures\\Thesis\\orienter\\rotnet_sample_out'
    batch_size = 5
    target_angles = [0, 90, 180, 270, 'itler', 'undef']
    generator = RotNetGen(directory, target_size=(224,224), target_classes=target_angles,
                          batch_size=batch_size, preprocessing_function=preprocess_input, 
                          check_images=True, gauss_noise=30, brightness=50, contrast=1.5, 
                          save_to_dir=savepath, shuffle=True, padding=True)                                                         
    
    for i in range(10):
        images, classes = generator.next()
#    for j in range(0,20):
#        image = images[j]
#        plt.imshow(image)
#        plt.show()