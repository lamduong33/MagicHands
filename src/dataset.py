import keras

class DataSet:
    """ Class to handle training, validation, and testing data, along with any associated
        data that might be used for evaluation purposes and further classification. """
    def __init__(self, t_train_data_dir, t_test_data_dir, t_batch_size):
        """ Constructor for the image data """

        self.classes = ['A', 'B', 'C', 'D', "del", 'E', 'F', 'G', 'H', 'I', 'J']
        self.classes = self.classes + ['K', 'L', 'M', 'N', "nothing", 'O', 'P', 'Q', 'R']
        self.classes = self.classes + ['S', "space", 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        self.image_height, self.image_width = 64, 64
        self.batch_size = t_batch_size
        self.test_data_dir = t_test_data_dir
        self.train_data_dir = t_train_data_dir

        # Data generators
        self.train_data_generator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True,
                samplewise_std_normalization=True, validation_split=0.1)
        """
        self.train_data_generator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True,
                samplewise_std_normalization=True, validation_split=0.1, rotation_range=40,
                width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                horizontal_flip=True, fill_mode="nearest")
        """ 
        self.training_data = self.train_data_generator.flow_from_directory(self.train_data_dir,
                target_size=(self.image_height, self.image_width), batch_size=64, shuffle=True,
                subset="training")
        self.testing_data = self.train_data_generator.flow_from_directory(self.test_data_dir,
                target_size=(self.image_height, self.image_width), batch_size=64, subset="validation")
    
    def generate_previews(self):
        test_image = keras.preprocessing.image.load_img("data/test/A/A_test.jpg")
        image_set = keras.preprocessing.image.img_to_array(test_image)
        image_set = image_set.reshape((1,) + image_set.shape)
        i = 0
        for batch in self.train_data_generator.flow(image_set, batch_size=1,
            save_to_dir="preview", save_prefix="A", save_format="jpeg"):
            i += 1
            if i > 20:
                break