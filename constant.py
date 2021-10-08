import os

# Path to jsons
JSON_DIR = '../data/jsons'

# Path to mask
MASK_DIR  = '../data/masks'
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)

# Path to output images
IMAGE_OUT_DIR = '../data/masked_images'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)

# Path to original images
IMAGE_DIR = '../data/images'


AUG_IMAGE_DIR = os.path.join('../data/aug_img')
AUG_MASK_DIR = os.path.join('../data/aug_mask') 


TEST_IMAGE_DIR = os.path.join('../data/test_images')
TEST_JSON_DIR = os.path.join('./data/test_jsons')
TEST_MASK_DIR = os.path.join('../data/test_mask')
if not os.path.exists(TEST_MASK_DIR):
    os.mkdir(TEST_MASK_DIR)


# In order to visualize masked-image(s), change "False" with "True"
VISUALIZE = True

# Bacth size
BACTH_SIZE = 4

# Input dimension
HEIGHT = 224
WIDTH = 224

# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS= 2