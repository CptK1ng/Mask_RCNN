import os
import sys
import numpy as np
import json
from PIL import Image
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

VISTAS_TRAIN_MASKS_PATH = "/home/koffi/Projects/mapillary-vistas-dataset_public_v1.1/training/instances"
VISTAS_TRAIN_IMAGE_PATH = "/home/koffi/Projects/mapillary-vistas-dataset_public_v1.1/training/images"

VISTAS_VAL_MASKS_PATH = "/home/koffi/Projects/mapillary-vistas-dataset_public_v1.1/validation/instances"
VISTAS_VAL_IMAGE_PATH = "/home/koffi/Projects/mapillary-vistas-dataset_public_v1.1/validation/images"
with open('config.json') as config_file:
    config = json.load(config_file)
    # in this example we are only interested in the labels
labels = config['labels']


# read in config file


class VistasConfig(Config):
    """Configuration for training on Mapillary Vistas.
       Derives from the base Config class and overrides values specific
       to the Mapillary Vistas.
       """
    # Give the configuration a recognizable name
    NAME = "vistas"

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 66  # gta dataset has 32 classes

    STEPS_PER_EPOCH = 50

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 10

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.002
    LEARNING_MOMENTUM = 0.9
    DETECTION_MIN_CONFIDENCE = 0.5

    WEIGHT_DECAY = 0.0005


config = VistasConfig()
config.display()


############################################################
#  Dataset
############################################################
class VistasDataset(utils.Dataset):
    def load_vistas(self, dataset=None):
        id = 0
        path_to_dataset = ""
        # read in config file
        with open('config.json') as config_file:
            config = json.load(config_file)
        # in this example we are only interested in the labels
        labels = config['labels']
        class_id = 0
        for label in labels:
            #if label['instances'] is True:
            self.add_class("vistas", labels.index(label), label['readable'])

        if dataset == "training":
            path_to_dataset = VISTAS_TRAIN_IMAGE_PATH
        elif dataset == "validation":
            path_to_dataset = VISTAS_VAL_IMAGE_PATH

        for root, dirs, files in os.walk(path_to_dataset):
            dir_path = os.path.abspath(root)
            for file in files:
                id += 1
                self.add_image("vistas", image_id=id, path=os.path.join(dir_path, file))

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
        # mask = skimage.io.imread(((self.image_info[image_id])['path'].replace('images', 'instances')).replace('jpg', 'png'))
        instance_image = Image.open(
            ((self.image_info[image_id])['path'].replace('images', 'instances')).replace('jpg', 'png'))
        instance_array = np.array(instance_image, dtype=np.uint16)
        instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
        instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)

        hash_of_objects_on_img = list()
        list_of_masks = list()
        class_ids = list()
        id, row_no = 0, 0
        for row in instance_label_array:
            column_no = 0
            for column in row:
                # New Object found
                is_object_with_instance = labels[column]['instances']
                hash_of_object = hash(frozenset([column, instance_ids_array[row_no][column_no]]))
                if is_object_with_instance and hash_of_object not in hash_of_objects_on_img:
                    class_ids.append(column)
                    list_of_masks.append(np.zeros((np.shape(instance_image)[0], np.shape(instance_image)[1])))
                    hash_of_objects_on_img.append(hash_of_object)
                # Object is already in list
                if is_object_with_instance and hash_of_object in hash_of_objects_on_img:
                    (list_of_masks[hash_of_objects_on_img.index(hash_of_object)])[row_no][column_no] = 1
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


    def train_vistas(self):
        dataset_train = VistasDataset()
        dataset_train.load_vistas(dataset="training")
        dataset_train.prepare()

        dataset_val = VistasDataset()
        dataset_val.load_vistas(dataset="validation")
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
                    epochs=100,
                    layers='heads')

'''
vistas_data = VistasDataset()
# vistas_data.load_vistas()
vistas_data.train_vistas()
'''
