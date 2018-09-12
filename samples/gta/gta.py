import os
import sys
import time
import numpy as np
import zipfile
import urllib.request
import shutil
import gta_classes as gc
import skimage
from mrcnn import visualize
import cv2

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

# WINDOWS
"""
GTA_TRAIN_MASKS_PATH = os.path.join(ROOT_DIR, "..\\gta_data\\train\\inst")
GTA_TRAIN_IMAGE_PATH = os.path.join(ROOT_DIR, "..\\gta_data\\train\\img")

GTA_VAL_MASKS_PATH = os.path.join(ROOT_DIR, "..\\gta_data\\val\\inst")
GTA_VAL_IMAGE_PATH = os.path.join(ROOT_DIR, "..\\gta_data\\val\\img")
"""

GTA_TRAIN_MASKS_PATH = "/home/koffi/Projects/gta data/train/inst"
GTA_TRAIN_IMAGE_PATH = "/home/koffi/Projects/gta data/train/img"

GTA_VAL_MASKS_PATH = "/home/koffi/Projects/gta data/val/inst"
GTA_VAL_IMAGE_PATH = "/home/koffi/Projects/gta data/val/img"


class GTAConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "gta"

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 32  # gta dataset has 32 classes

    STEPS_PER_EPOCH = 150

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 30

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.002
    LEARNING_MOMENTUM = 0.9

    WEIGHT_DECAY = 0.0005

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

config = GTAConfig()
config.display()


############################################################
#  Dataset
############################################################

class GTADataset(utils.Dataset):
    def load_gta(self, class_ids=None, dataset=None):
        id = 0
        path_to_dataset = ""
        for class_id in gc.labels:
            self.add_class("gta", class_id.id, class_id.name)

        if dataset == "training":
            path_to_dataset = GTA_TRAIN_IMAGE_PATH
        elif dataset == "validation":
            path_to_dataset = GTA_VAL_IMAGE_PATH

        for root, dirs, files in os.walk(path_to_dataset):
            dir_path = os.path.abspath(root)
            for file in files:
                id += 1
                self.add_image("gta", image_id=id, path=os.path.join(dir_path, file))

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
        mask = skimage.io.imread(((self.image_info[image_id])['path'].replace('img', 'inst')).replace('jpg', 'png'))
        hash_of_objects_on_img = list()
        list_of_masks = list()
        class_ids = list()
        id, row_no = 0, 0
        for row in mask:
            column_no = 0
            for column in row:
                # New Object found
                if column[0] != 0 and hash(frozenset(column)) not in hash_of_objects_on_img:
                    list_of_masks.append(np.zeros((np.shape(mask)[0], np.shape(mask)[1])))
                    class_ids.append(column[0])
                    hash_of_objects_on_img.append(hash(frozenset(column)))
                # Object already detected
                if column[0] != 0 and hash(frozenset(column)) in hash_of_objects_on_img:
                    # Get the index of the depending mask and set it at (row_no|column_no) to 1
                    (list_of_masks[hash_of_objects_on_img.index(hash(frozenset(column)))])[row_no][column_no] = 1
                # print("({}|{})".format(column_no, row_no))
                column_no += 1
            row_no += 1
        first_arr = True
        masks = None
        for arr in list_of_masks:
            if first_arr:
                masks = arr.astype(bool)
                first_arr = False
            else:
                masks = np.dstack((masks, arr.astype(bool)))
        class_ids = np.asarray(class_ids)
        return masks.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        return self.image_info[image_id]['path']


    def train_gta(self):
        dataset_train = GTADataset()
        dataset_train.load_gta(dataset="training")
        dataset_train.prepare()

        dataset_val = GTADataset()
        dataset_val.load_gta(dataset="validation")
        dataset_val.prepare()


        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

        # Which weights to start with?
        init_with = "coco"  # imagenet, coco, or last

        if init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif init_with == "coco":
            # Load weights trained on MS COCO, but skip layers that
            # are different due to the different number of classes
            # See README for instructions to download the COCO weights
            model.load_weights(COCO_MODEL_PATH, by_name=True,
                               exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                        "mrcnn_bbox", "mrcnn_mask"])
        elif init_with == "last":
            # Load the last model you trained and continue training
            model.load_weights(model.find_last(), by_name=True)

        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train by name pattern.
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=150,
                    layers='heads')

'''
gta_data = GTADataset()
gta_data.load_gta()
gta_data.train_gta()
'''