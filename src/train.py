# Author: Lam Duong

import numpy
import keras
import sklearn
import scipy

def DataSet():
    def __init__(self, t_file_name):
        self.file_name = t_file_name

def main():
    train_data_generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                                                                  width_shift_range=0.2,
                                                                  height_shift_range=0.2,
                                                                  rescale=1./255,
                                                                  shear_range=0.2,
                                                                  zoom_range=0.2,
                                                                  horizontal_flip=True,
                                                                  fill_mode="nearest")

    batch_size = 16

    # TESTING DATA IN BATCH
    test_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # TRAINING DATA IN BATCH
    training_data = train_data_generator.flow_from_directory('data/train', target_size=(200,200), class_mode='binary', batch_size=batch_size)

    # VALIDATION DATA
    validation_generator = test_data_generator.flow_from_directory("data/validation", target_size=(200,200), batch_size=batch_size, class_mode="binary")

    # Single data test
    image = keras.preprocessing.image.load_img("data/train/A/A158.jpg")

    # Training model with 3 layers for height, width, and features
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(3, 200, 200)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

    model.add(keras.layers.Conv2D(32, (3, 3)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(32, (3, 3)))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation("sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer="rmsprop",
                  metric=["accuracy"])
    
    model.fit_generator(train_data_generator, steps_per_epoch=2000 // batch_size, epochs=50, validation_data=validation_generator, validation_steps=800//batch_size)
    model.save_weights("first_try.h5")

main()