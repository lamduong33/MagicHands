# Author: Lam Duong

import numpy
import keras
import tensorflow

""" Ensuring that GPUs are used """
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tensorflow.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tensorflow.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

def DataSet():
    def __init__(self, t_train_data_dir, t_test_data_dir):
        self.test_data_dir = t_test_data_dir
        self.train_data_dir = t_train_data_dir

def main():
    train_data_generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                                                                  width_shift_range=0.2,
                                                                  height_shift_range=0.2,
                                                                  rescale=1./255,
                                                                  shear_range=0.2,
                                                                  zoom_range=0.2,
                                                                  horizontal_flip=True,
                                                                  fill_mode="nearest")

main()