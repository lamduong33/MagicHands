import keras

class ConvNet:
    """ The convolutional neural net model """
    def __init__(self, t_dataset, t_first_activation="relu"):
        """ t_dataset : a DataSet object containing the training and testing data
            t_first_activation : a string of the name of the activation method for the first layer """
        self.model = keras.models.Sequential()
        self.data = t_dataset
        self.init_layers()

    def init_layers(self):
        """ FIRST layer HAS to know about the input shape (in this case, should be (200,200,3)
         (200,200,3) height and width = 200, and 3 RGB values """

        # First layer of the model
        # 32 filters, of size 140x140, using relu activation with the given input size and 3 colors RGB
        self.model.add(keras.layers.Conv2D(64, kernel_size=10, strides=1,
            activation="relu", input_shape=(self.data.image_height,self.data.image_width,3)))
    
    def build(self):
        """ Add final layers, compile, and train the model """

        """
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
        """

        # ADJUST STRIDES OR PADDING
        # Original strides = 2, no padding specified

        #self.model.add(keras.layers.Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
        self.model.add(keras.layers.Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
        self.model.add(keras.layers.MaxPool2D(pool_size = [3,3]))
    
        self.model.add(keras.layers.Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
        self.model.add(keras.layers.Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu'))
        self.model.add(keras.layers.MaxPool2D(pool_size = [3,3]))
    
        self.model.add(keras.layers.Conv2D(128, kernel_size = [3,3], padding = 'same', activation = 'relu'))
        self.model.add(keras.layers.Conv2D(256, kernel_size = [3,3], padding = 'same', activation = 'relu'))
        self.model.add(keras.layers.MaxPool2D(pool_size = [3,3]))
    
        self.model.add(keras.layers.BatchNormalization())
    
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(512, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.001)))
        self.model.add(keras.layers.Dense(29, activation = 'softmax'))


        print("CURRENT MODEL: ")
        print(self.model.summary())
    
    
    def train(self):

        # Compilation process
        #self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

        # Training process
        self.model.fit_generator(self.data.training_data, epochs=4,
                validation_data=self.data.testing_data)
        
        self.model.save("magichands_model.h5")
    
    def predict_generator(self):
        self.model.predict_generator(self.data.testing_data, use_multiprocessing=True, verbose=1)
    
    def predict(self):
        self.model.predict()