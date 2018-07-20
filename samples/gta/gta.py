import os
import sys
import time
import numpy as np
import zipfile
import urllib.request
import shutil
import samples.gta.gta_classes as gc
import skimage

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
GTA_MASKS_PATH = os.path.join(ROOT_DIR, "..\\gta_data\\train\\inst")
GTA_IMAGE_PATH = os.path.join(ROOT_DIR, "..\\gta_data\\train\\img")




class GTAConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "gta"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 30  # CITYSCAPES has 80 classes

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.0005
    LEARNING_MOMENTUM = 0.9


config = GTAConfig()
config.display()


############################################################
#  Dataset
############################################################

class GTADataset(utils.Dataset):

    def load_gta(self, class_ids=None):

        for class_id in gc.labels:
            self.add_class("gta", class_id.id, class_id.name)
        for root, dirs, files in os.walk(GTA_IMAGE_PATH):
            '''
            for name in files:                
                self.add_image("gta",image_id=os.path.join(root, name))
                i += 1
            '''
            path = root.split(os.sep)
            dir_path = os.path.abspath(root)
            for file in files:
                self.add_image("gta", image_id=(os.path.splitext(file)[0]), path=os.path.join(dir_path, file))

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
                """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        tmp_img_id = self.image_info[image_id]
        mask = skimage.io.imread(GTA_MASKS_PATH + str(tmp_img_id)[:3] + str(tmp_img_id))
        self.extracting_object_instances_from_image(mask=mask)

    def extracting_object_instances_from_image(self, mask):
        '''

        :param mask: Input Mask,encoded as array of shape (height, with, 3 (RGB)) first channel (R) encodes the class ID,
         and the two remaining channels encode the instance ID (= 256 * G + B).
        :return:  class_ids: of all found object and
                  masks: of these with shape (height, width, num_of_instances)
        '''

        objects_on_img = list()
        list_of_masks = list()
        class_ids = list()
        id, row_no = 0, 0
        for row in mask:
            column_no = 0
            for column in row:
                # New Object found
                if column_no[0] != 0 and column_no not in objects_on_img:
                    list_of_masks.append(np.empty(np.shape(mask)))
                    class_ids.append(column_no[0])
                    objects_on_img.append(column_no)
                # Object already detected
                if column_no[0] != 0 and column_no in objects_on_img:
                    # Get the index of the depending mask and set it at (row_no|column_no) to 1
                    (list_of_masks[objects_on_img.index(column_no)])[row_no][column_no] = 1

                column_no += 1
            row_no += 1
        first_arr = True
        masks = None
        for arr in list_of_masks:
            if first_arr:
                masks = arr.astype(bool)
                first_arr = False
            else:
                np.dstack((masks, arr.astype(bool)))

        return class_ids, masks

        # Cannot specify end of Iterations



config = GTADataset()
config.load_gta("001_00001")
config.load_image("001_00001")
config.load_mask("001_00001")
