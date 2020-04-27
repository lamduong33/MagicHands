import cv2
import os
import numpy # For dealing with arrays
import sklearn # For dealing with data wrangling
import keras # For convnet
import tensorflow # Keras's backbone

""" Ensuring that GPU is used """
GPUS = tensorflow.config.experimental.list_physical_devices('GPU')
if GPUS:
    # Restrict TensorFlow to only use the first GPU
    try:
        tensorflow.config.experimental.set_visible_devices(GPUS[0], 'GPU')
        LOGICAL_GPUS = tensorflow.config.experimental.list_logical_devices('GPU')
        print(len(GPUS), "Physical GPUs,", len(LOGICAL_GPUS), "Logical GPU")
    except RuntimeError as error:
        # Visible devices must be set before GPUs have been initialized
        print(error)

class DataSet:
    """ Class to handle training, validation, and testing data, along with any associated
        data that might be used for evaluation purposes and further classification. """
    def __init__(self, t_train_data_dir, t_test_data_dir, t_batch_size):
        """ Constructor for the image data """
        self.classes = ['A', 'B', 'C', 'D', "del", 'E', 'F', 'G', 'H', 'I', 'J']
        self.classes = self.classes + ['K', 'L', 'M', 'N', "nothing", 'O', 'P', 'Q', 'R']
        self.classes = self.classes + ['S', "space", 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        self.image_height, self.image_width = 200, 200
        self.batch_size = t_batch_size
        self.test_data_dir = t_test_data_dir
        self.train_data_dir = t_train_data_dir
        self.train_data_generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                width_shift_range=0.2, height_shift_range=0.2, rescale=1./255, shear_range=0.2,
                zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
        self.test_data_generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                width_shift_range=0.2, height_shift_range=0.2, rescale=1./255, shear_range=0.2,
                zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
        self.training_data = self.train_data_generator.flow_from_directory(self.train_data_dir,
                target_size=(self.image_height, self.image_width), batch_size=self.batch_size, 
                class_mode="binary", classes=self.classes, color_mode="rgb")
        self.testing_data = self.test_data_generator.flow_from_directory(self.test_data_dir, 
                target_size=(self.image_height, self.image_width), batch_size=self.batch_size,
                class_mode="binary", classes=self.classes, color_mode="rgb")

class ConvNet:
    """ The convolutional neural net model """
    def __init__(self, t_dataset, t_first_activation="relu", t_height=200, t_width=200):
        """ t_dataset : a DataSet object containing the training and testing data
            t_first_activation : a string of the name of the activation method for the first layer """
        self.image_height, self.image_width = t_height, t_width
        self.model = keras.models.Sequential()
        self.data = t_dataset
        if (self.data.image_height != self.image_height) and (self.data.image_width != self.image_width):
            raise Exception("ERR: Data does not have the same dimensions as training model")
        self.init_layers(t_first_activation)

    def init_layers(self, t_first_activation):
        """ FIRST layer HAS to know about the input shape (in this case, should be (64, 200,200,3)
         (64,200,200,3) - batch_size=64, height and width = 200, and 3 RGB values """
        self.model.add(keras.layers.Dense(self.data.batch_size, activation=t_first_activation, 
                input_shape=(self.image_height, self.image_width, 3)))
    
    def train(self):
        """ Add final layers, compile, and train the model """
        #self.model.add(keras.layers.Dense(10, activation="softmax"))
        self.model.add(keras.layers.Conv2D(64, kernel_size=10, activation="relu",
                input_shape=(self.data.image_height, self.data.image_width, 3)))
        self.model.add(keras.layers.Conv2D(32, kernel_size=10, activation="relu"))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(1, activation="softmax"))

        # Compilation process
        self.model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # Training process
        self.model.fit_generator(self.data.training_data, steps_per_epoch=2000, epochs=50,
                validation_data=self.data.testing_data, validation_steps=800)
    
    def predict(self):
        self.model.predict_generator(self.data.testing_data, use_multiprocessing=True, verbose=1)
    
def main():
    data = DataSet("data/train", "data/test", t_batch_size=64)
    cnn = ConvNet(data)
    cnn.train()
    #cnn.model.fit_generator(cnn.data.train_data_generator, steps_per_epoch=2000, epochs=50, validation_data=cnn.data.test_data_generator, validation_steps=800)
    print("Done!")

main()