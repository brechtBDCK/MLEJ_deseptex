TESTING = True

ORIGINAL_IMAGE_SHAPE = (5588, 8232, 3) # fixed by camera resolution, I don't know how to change this to e.g. get images faster
SHARED_IMAGE_SHAPE = (2067, 3046, 3)
GUI_IMAGE_SHAPE = (1200, 1600, 3)

classification_image_shape = 224 # fixed by model training
segmentation_image_shape = 1280 # fixed by model training



if TESTING:
    ORIGINAL_IMAGE_SHAPE = SHARED_IMAGE_SHAPE # to avoid downsizing by undistorting the image