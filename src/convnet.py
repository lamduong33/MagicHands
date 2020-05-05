import keras

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
        self.init_layers()

    def init_layers(self):
        """ FIRST layer HAS to know about the input shape (in this case, should be (64, 200,200,3)
         (64,200,200,3) - batch_size=64, height and width = 200, and 3 RGB values """

        # First layer of the model
        # 32 filters, of size 140x140, using relu activation with the given input size and 3 colors RGB
        self.model.add(keras.layers.Conv2D(64, kernel_size=10, strides=1,
            activation="relu", input_shape=(64,64,3)))
    
    def build(self):
        """ Add final layers, compile, and train the model """

        self.model.add(keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Conv2D(128, kernel_size=4, strides=1, activation='relu'))
        self.model.add(keras.layers.Conv2D(128, kernel_size=4, strides=2, activation='relu'))
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Conv2D(256, kernel_size=4, strides=1, activation='relu'))
        self.model.add(keras.layers.Conv2D(256, kernel_size=4, strides=2, activation='relu'))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(512, activation='relu'))
        self.model.add(keras.layers.Dense(29, activation='softmax'))

        print("CURRENT MODEL: ")
        print(self.model.summary())
    
    
    def train(self):

        # Compilation process
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # Training process
        self.model.fit_generator(self.data.training_data, epochs=5,
                validation_data=self.data.testing_data)
    
    def predict_generator(self):
        self.model.predict_generator(self.data.testing_data, use_multiprocessing=True, verbose=1)
 