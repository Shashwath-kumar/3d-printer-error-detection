import os

# Image transforms
IMG_WIDTH = 512
IMG_HEIGHT = 512

# Data path
DATA_DIR = 'data/'
IMG_DIR = os.path.join(DATA_DIR, 'images/')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV_PATH = os.path.join(DATA_DIR, 'test.csv')

# Model checkpoints

CHECKPOINT_DIR = 'model_checkpoints/'
LDC_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, '16_model.pth')

# Edge detection dir

LDC_OUTPUT_DIR = 'LDC/'
LDC_IMAGE_FOLDER = 'Images/'
LDC_TRAIN_FOLDER = 'train/'
LDC_TEST_FOLDER = 'test/'