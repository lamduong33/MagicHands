import numpy
import keras
import tensorflow

""" Ensuring that GPU is used """
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tensorflow.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tensorflow.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as error:
        # Visible devices must be set before GPUs have been initialized
        print(error)

class DataSet:
    """ Class to handle training, validation, and testing data, along with any associated
        data that might be used for evaluation purposes and further classification. """
    def __init__(self, t_train_data_dir, t_test_data_dir):
        """ Constructor for the image data """
        self.image_height, self.image_width = 200, 200
        self.batch_size = 64
        self.test_data_dir = t_test_data_dir
        self.train_data_dir = t_train_data_dir
        self.train_data_generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                width_shift_range=0.2, height_shift_range=0.2, rescale=1./255, shear_range=0.2,
                zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
        self.test_data_generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                width_shift_range=0.2, height_shift_range=0.2, rescale=1./255, shear_range=0.2,
                zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
        self.training_data = self.train_data_generator.flow_from_directory(self.train_data_dir, target_size=(self.image_height, self.image_width), batch_size=self.batch_size, class_mode="binary")
        self.testing_data = self.test_data_generator.flow_from_directory(self.test_data_dir, target_size=(self.image_height, self.image_width), batch_size=self.batch_size, class_mode="binary")

class ConvNet:
    """ The convolutional neural net model """
    def __init__(self, t_dataset):
        """ Construct the layers """
        self.model = keras.models.Sequential()
        self.data = t_dataset


def main():
    data = DataSet("data/train", "data/test")
    cnn = ConvNet(data)
    #cnn.model.fit_generator(cnn.data.train_data_generator, steps_per_epoch=2000, epochs=50, validation_data=cnn.data.test_data_generator, validation_steps=800)
    print("Done!")

main()
