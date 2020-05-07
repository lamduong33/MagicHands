import keras
class DataSet:
    """ Class to handle training, validation, and testing data, along with any associated
        data that might be used for evaluation purposes and further classification. """
    def __init__(self, t_train_data_dir, t_test_data_dir, t_batch_size):
        """ Constructor for the image data """

        self.classes = ['A', 'B', 'C', 'D', "del", 'E', 'F', 'G', 'H', 'I', 'J']
        self.classes = self.classes + ['K', 'L', 'M', 'N', "nothing", 'O', 'P', 'Q', 'R']
        self.classes = self.classes + ['S', "space", 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        self.image_height, self.image_width = 64,64
        self.batch_size = t_batch_size
        self.test_data_dir = t_test_data_dir
        self.train_data_dir = t_train_data_dir

        # Data generators, to choose from either non-transformed or transformed data respectively
        self.regular_data_generator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True,
            samplewise_std_normalization=True, validation_split=0.1)
        self.transformed_data_generator = keras.preprocessing.image.ImageDataGenerator(samplewise_center=True,
            samplewise_std_normalization=True, validation_split=0.1, rotation_range=40,
            width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.5,
            horizontal_flip=True, fill_mode="nearest")

        # Data sets
        self.training_data = self.transformed_data_generator.flow_from_directory(self.train_data_dir,
            target_size=(self.image_height, self.image_width), batch_size=64, shuffle=True,
            classes=self.classes, class_mode="categorical", color_mode="rgb", seed=42, subset="training")
        self.validation_data = self.transformed_data_generator.flow_from_directory(self.train_data_dir,
            target_size=(self.image_height, self.image_width), batch_size=64, shuffle=True,
            classes=self.classes, class_mode="categorical", color_mode="rgb", seed=42, subset="validation")
        self.testing_data = self.transformed_data_generator.flow_from_directory(self.test_data_dir,
            target_size=(self.image_height, self.image_width), shuffle=False, batch_size=1)
        print("Finished processing data")
    
    def generate_transformed_data(self):
        for each_class in self.classes:
            class_directory = "data/test/test_folder/"
            class_directory += each_class
            class_directory += "_test.jpg"
            test_image = keras.preprocessing.image.load_img(class_directory)
            image_set = keras.preprocessing.image.img_to_array(test_image)
            image_set = image_set.reshape((1,) + image_set.shape)
            i = 0
            for each_batch in self.transformed_data_generator.flow(image_set, batch_size=1,
                save_to_dir="data/test_transformed/test_folder", save_prefix=each_class, save_format="jpeg"):
                i += 1
                if i > 20:
                    break