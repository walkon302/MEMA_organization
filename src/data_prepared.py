import glob
import os
import numpy as np
import cv2
from PIL import Image
import math
from sklearn.utils import shuffle

class ImagePreProcess(object):
    '''
    Collections of methods for preprocessing the images.
    '''
    @staticmethod
    def image_bw(old_image_folder='ori_organized',
                 new_image_folder='bw'
                 ):
        """
        Load old RGB image and convert it to black and white.

        Parameters:
        -----------
        old_image_folder: str
            The name of folder containing training image.
        new_image_folder: str
            The name of folder for processed images.

        Return:
        -------
        Generate new folders containing black and white images.
        """
        curdir = os.path.dirname(os.getcwd())

        directory = '{}/input/{}_{}'.format(curdir,
                                            new_image_folder,
                                            old_image_folder)
        if not os.path.exists(directory):
            os.makedirs(directory)

        filelist = glob.glob('{}/input/{}/*.{}'.format(curdir,
                                                       old_image_folder,
                                                       '*'))

        for file_name in filelist:
            im = Image.open(file_name)
            im_bw = im.convert('1')
            im_bw.save('{}/bw_{}'.format(directory,os.path.basename(file_name)))

    @staticmethod
    def image_resize(old_image_folder='bw_ori_organized',
                     new_image_folder='resize',
                     canvas_size=300
                     ):
        """
        Resize the canvas of old_image_path and store the new image in
        new_image_path. Center the image on the new canvas.

        Parameters:
        -----------
        old_image_folder: str
            The name of folder containing training image.
        new_image_folder: str
            The name of folder for processed images.
        canvas_size: int
            The pixel size of canvas.

        Returns:
        --------
            Generate new folders containing resized images.
        """
        curdir = os.path.dirname(os.getcwd())

        directory = '{}/input/{}_{}'.format(curdir,
                                            new_image_folder,
                                            old_image_folder)
        if not os.path.exists(directory):
            os.makedirs(directory)

        filelist = glob.glob('{}/input/{}/*.{}'.format(curdir,
                                                       old_image_folder,
                                                       '*'))

        for file_name in filelist:
            im = Image.open(file_name)
            canvas_size = max(canvas_size, im.size[0], im.size[1])

        for file_name in filelist:
            im = Image.open(file_name)
            old_width, old_height = im.size

            # Center the image
            x1 = int(math.floor((canvas_size - old_width) / 2))
            y1 = int(math.floor((canvas_size - old_height) / 2))

            mode = im.mode
            if len(mode) == 1:  # L, 1
                new_background = (0)
            if len(mode) == 3:  # RGB
                new_background = (0, 0, 0)
            if len(mode) == 4:  # RGBA, CMYK
                new_background = (0, 0, 0, 0)

            new_image = Image.new(mode,
                                  (canvas_size, canvas_size),
                                  new_background)
            new_image.paste(im, (x1, y1, x1 + old_width, y1 + old_height))
            new_image.save('{}/resize_{}'.format(directory,
                                                 os.path.basename(file_name)))

class ImageAugmentation(object):
    '''
    Collections of methods for converting images into numpy array and
    augmenting the training images.
    '''
    @staticmethod
    def image_to_array(image_file_list):
        '''
        Convert images to numpy array.

        Parameters:
        -----------
        image_file_list: list
            The list of path of image files.

        Returns:
        --------
        image_array: numpy array
            A numpy array containing image information.
        '''
        image_array = np.array([cv2.imread(file_name, 0)
                        for file_name in image_file_list])

        return image_array

    @staticmethod
    def image_rotation(image_array, degree=90):
        '''
        Rotate image array.

        Parameters:
        -----------
        image_array: numpy array
            A numpy array containing image information.
        degree: int
            The degree of image rotation.

        Returns:
        --------
        result: numpy array
            A numpy array containing rotated image information.
        '''
        result = []
        rows, cols = image_array[0].shape
        for image in image_array:
            r_m = cv2.getRotationMatrix2D((cols/2, rows/2),
                                          degree ,1)
            image_modified = cv2.warpAffine(image, r_m,
                                            (cols, rows))
            result.append(image_modified)

        result = np.array(result)

        return result

    @staticmethod
    def image_flip(image_array):
        '''
        Flip the image vertically and horizontally.

        Parameters:
        -----------
        image_array: numpy array
            A numpy array containing image information.

        Returns:
        --------
        result: numpy array
            A numpy array containing flipped image information.
        '''
        result = []
        for image in image_array:
            x_flip = cv2.flip(image, 0)
            y_flip = cv2.flip(image, 1)
            result.append(x_flip)
            result.append(y_flip)

        result = np.array(result)

        return result

    @staticmethod
    def cnn_image_resize(image_array):
        '''
        Resize image array to 128x128 for CNN.

        Parameters:
        -----------
        image_array: numpy array
            A numpy array containing image information.

        Returns:
        --------
        result: numpy array
            A numpy array containing image information that is 128x128 size.
        '''
        result = []
        dim = (128, 128)
        for image in image_array:
            resized = cv2.resize(image, dim,
                                 interpolation=cv2.INTER_AREA)
            result.append(resized)

        result = np.array(result)

        return result

    @staticmethod
    def image_augmentation(image_array):
        '''
        The process of image augmentation.

        Parameters:
        -----------
        image_array: numpy array
            A numpy array containing image information.

        Returns:
        --------
        result: numpy array
            A numpy array containing information combining all augmented
            images.
        '''
        flip = ImageAugmentation.image_flip(image_array)
        rotation = ImageAugmentation.image_rotation(image_array)

        result = np.concatenate([image_array, flip, rotation])

        return result

class CNNDataPreProcess(object):
    '''
    A collections of methods for preprocessing data for CNN model.
    '''
    @staticmethod
    def get_file_list(folder):
        '''
        A method for getting the list of files in the targeted folder.

        Parameters:
        -----------
        folder: str
            The name of the targeted folder

        Returns:
        file_list: list
            A list of paths of files in the targeted folder.
        '''
        cur_path = '{}/input'.format(os.path.dirname(os.getcwd()))
        work_path = '{}/{}'.format(cur_path, folder)
        file_list = glob.glob('{}/*'.format(work_path))

        return file_list

    @staticmethod
    def augmented_prepared(folder):
        '''
        A method for processing training data, which need to be augmented.

        Parameters:
        -----------
        folder: str
            The name of the targeted folder (where training data belongs)

        Returns:
        --------
        image_array_aug_resize: numpy array
            A numpy array containing information that are augmented and resized
            to 128x128.
        '''
        file_list = CNNDataPreProcess.get_file_list(folder)
        image_array = ImageAugmentation.image_to_array(file_list)
        image_array_aug = ImageAugmentation.image_augmentation(image_array)
        image_array_aug_resize = ImageAugmentation.cnn_image_resize(image_array_aug)

        return image_array_aug_resize

    @staticmethod
    def normal_prepared(folder):
        '''
        A method for processing training data without augmentation.

        Parameters:
        -----------
        folder: str
            The name of the targeted folder (where testing data belongs)

        Returns:
        --------
        image_array_resize: numpy array
            A numpy array containing information that are resized to 128x128.
        '''
        file_list = CNNDataPreProcess.get_file_list(folder)
        image_array = ImageAugmentation.image_to_array(file_list)
        image_array_resize = ImageAugmentation.cnn_image_resize(image_array)

        return image_array_resize

    @staticmethod
    def cnn_preprocess(image_array):
        '''
        Rescale the pixel intensity to 0-1 for CNN model.

        Parameters:
        -----------
        image_array: numpy array
            A numpy array containing image information.

        Returns:
        image_array: numpy array
            A numpy array containing image information that pixel intensities
            are scaled to 0-1.
        '''
        image_array = np.array(image_array/255., dtype = 'float32')
        image_array = image_array.reshape([len(image_array),
                                    image_array.shape[1]*image_array.shape[2]])

        return image_array

    @staticmethod
    def data_generate(pos, neg):
        '''
        A process to combine two patterns of image data and generate data as
        numpy array and labels.

        Parameters:
        -----------
        pos: numpy_array
            A numpy array containing image information from pattern one folder
            (e.g., organized).
        neg: numpy_array
            A numpy array containing image information from pattern two folder
            (e.g., disorganized).

        Returns:
        --------
        train_sample: numpy array
            A numpy array containing shuffled image information from two
            training data folders.
        train_label: numpy array
            A numpy array containing labels of train_sample.
        '''
        train_sample = np.concatenate([pos, neg])
        train_label = np.concatenate([np.repeat(0, len(pos)),
                                      np.repeat(1, len(neg))])

        train_sample, train_label = shuffle(train_sample,
                                            train_label,
                                            random_state=0)

        return train_sample, train_label

    @staticmethod
    def train_eval_prep(folder_1='ori_organized',
                        folder_2='ori_disorganized',
                        mode='train'):
        '''
        The process of preparation for training or testing data.

        Parameters:
        -----------
        folder_1: str
            The name of targeted folder one.
        folder_2: str
            The name of targeted forder two.
        mode: str
            The mode of this preparation. train will use augmentation, and eval
            will not.

        Returns:
        --------
        sample: numpy array
            A numpy array containing image information that are prepared for
            CNN.
        label: numpy array
            A numpy array containing labels of sample.
        '''
        ImagePreProcess.image_bw(old_image_folder = folder_1)
        ImagePreProcess.image_bw(old_image_folder = folder_2)
        ImagePreProcess.image_resize(
        old_image_folder='bw_{}'.format(folder_1)
        )
        ImagePreProcess.image_resize(
        old_image_folder='bw_{}'.format(folder_2)
        )

        if mode == 'train':

            good = (
            CNNDataPreProcess.augmented_prepared('resize_bw_{}'.format(folder_1))
            )
            bad = (
            CNNDataPreProcess.augmented_prepared('resize_bw_{}'.format(folder_2))
            )
        elif mode == 'eval':

            good = (
            CNNDataPreProcess.normal_prepared('resize_bw_{}'.format(folder_1))
            )
            bad = (
            CNNDataPreProcess.normal_prepared('resize_bw_{}'.format(folder_2))
            )

        good_array = CNNDataPreProcess.cnn_preprocess(good)
        bad_array = CNNDataPreProcess.cnn_preprocess(bad)

        sample, label = CNNDataPreProcess.data_generate(good_array, bad_array)

        return sample, label

    @staticmethod
    def predict_prep(folder='predict'):
        '''
        The process of preparation for data that needs to be classified.

        Parameters:
        -----------
        folder: str
            The name of targeted folder, e.g., 'predict'.

        Returns:
        image_array_resize: numpy array
            A numpy array containing image information of data that needs to
            be classified.
        file_name: numpy array
            A numpy array containing file name of each images.
        '''
        ImagePreProcess.image_bw(old_image_folder=folder)
        ImagePreProcess.image_resize(old_image_folder='bw_predict')

        image_array_resize = (
        CNNDataPreProcess.normal_prepared('resize_bw_{}'.format(folder))
        )

        file_list = CNNDataPreProcess.get_file_list(folder)

        file_name = []
        for f in file_list:
            file_name.append(os.path.basename(f))
        file_name = np.array(file_name)

        return image_array_resize, file_name
