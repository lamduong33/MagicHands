# Author: Lam Duong

import numpy
import keras
import sklearn
import scipy

def DataSet():
    def __init__(self, t_file_name):
        self.file_name = t_file_name

def main():
    data_generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                                                                  width_shift_range=0.2,
                                                                  height_shift_range=0.2,
                                                                  rescale=1/255,
                                                                  shear_range=0.2,
                                                                  zoom_range=0.2,
                                                                  horizontal_flip=True,
                                                                  fill_mode='nearest')

    # TRAINING DATA IN BATCH
    training_data = data_generator.flow_from_directory('data/train', class_mode='binary', batch_size=64)

    # Single data test
    image = keras.preprocessing.image.load_img("data/train/A/A158.jpg")

    x = keras.preprocessing.image.img_to_array(image)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in data_generator.flow(x, batch_size=1, save_to_dir="preview", save_prefix="A", save_format="jpeg"):
        i += 1
        if i > 20:
            break

main()