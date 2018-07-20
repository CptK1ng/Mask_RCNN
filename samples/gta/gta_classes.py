#!/usr/bin/python
#
# Gta labels
#

from collections import namedtuple

# --------------------------------------------------------------------------------
# Definitions
# --------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'color',  # The color of this label
])

# --------------------------------------------------------------------------------
# A list of all labels
# --------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name  |id| trainId | category |catId | hasInstances | ignoreInEval | color
    Label('unlabeled', 0, 255, (0, 0, 0)),
    Label('ambiguous', 1, 255, (111, 74, 0)),
    Label('sky', 2, 0, (70, 130, 180)),
    Label('road', 3, 1, (128, 64, 128)),
    Label('sidewalk', 4, 2, (244, 35, 232)),
    Label('rail track', 5, 255, (230, 150, 140)),
    Label('terrain', 6, 3, (152, 251, 152)),
    Label('tree', 7, 4, (87, 182, 35)),
    Label('vegetation', 8, 5, (35, 142, 35)),
    Label('building', 9, 6, (70, 70, 70)),
    Label('infrastructure', 10, 7, (153, 153, 153)),
    Label('fence', 11, 8, (190, 153, 153)),
    Label('billboard', 12, 9, (150, 20, 20)),
    Label('traffic light', 13, 10, (250, 170, 30)),
    Label('traffic sign', 14, 11, (220, 220, 0)),
    Label('mobile barrier', 15, 12, (180, 180, 100)),
    Label('fire hydrant', 16, 13, (173, 153, 153)),
    Label('chair', 17, 14, (168, 153, 153)),
    Label('trash', 18, 15, (81, 0, 21)),
    Label('trash can', 19, 16, (81, 0, 81)),
    Label('person', 20, 17, (220, 20, 60)),
    Label('animal', 21, 255, (255, 0, 0)),
    Label('bicycle', 22, 255, (119, 11, 32)),
    Label('motorcycle', 23, 18, (0, 0, 230)),
    Label('car', 24, 19, (0, 0, 142)),
    Label('van', 25, 20, (0, 80, 100)),
    Label('bus', 26, 21, (0, 60, 100)),
    Label('truck', 27, 22, (0, 0, 70)),
    Label('trailer', 28, 255, (0, 0, 90)),
    Label('train', 29, 255, (0, 80, 100)),
    Label('plane', 30, 255, (0, 100, 100)),
    Label('boat', 31, 255, (50, 0, 90)),
]

# --------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
# --------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label = {label.name: label for label in labels}
# id to label object
id2label = {label.id: label for label in labels}
# trainId to label object
trainId2label = {label.trainId: label for label in reversed(labels)}
# category to list of label objects



# --------------------------------------------------------------------------------
# Assure single instance name
# --------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName(name):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name


# --------------------------------------------------------------------------------
# Main for testing
# --------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7}".format('name', 'id', 'trainId', ))
    print("    " + ('-' * 98))
    for label in labels:
        print(
            "    {:>21} | {:>3} | {:>7} | ".format(label.name, label.id, label.trainId))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id = name2label[name].id
    print("ID of label '{name}': {id}".format(name=name, id=id))

    # Map from ID to label
    print("Category of label with ID '{id}'".format(id=id, ))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format(id=trainId, name=name))
