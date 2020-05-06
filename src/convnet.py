import keras
import numpy
import pandas

class ConvNet:
    """ The convolutional neural net model """
    def __init__(self, t_dataset, t_first_activation="relu"):
        """ t_dataset : a DataSet object containing the training and testing data
            t_first_activation : a string of the name of the activation method for the first layer """
        self.model = keras.models.Sequential()
        self.predictions = numpy.array([])
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

        # ADJUST STRIDES OR PADDING
        # Original strides = 2, no padding specified

        self.model.add(keras.layers.Conv2D(32, kernel_size=[3, 3], padding='same', activation='relu'))
        self.model.add(keras.layers.MaxPool2D(pool_size=[3, 3]))
    
        self.model.add(keras.layers.Conv2D(32, kernel_size=[3, 3], padding='same', activation='relu'))
        self.model.add(keras.layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'))
        self.model.add(keras.layers.MaxPool2D(pool_size=[3, 3]))
    
        self.model.add(keras.layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'))
        self.model.add(keras.layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'))
        self.model.add(keras.layers.MaxPool2D(pool_size=[3, 3]))
    
        self.model.add(keras.layers.BatchNormalization())
    
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(29, activation='softmax'))


        print("CURRENT MODEL: ")
        print(self.model.summary())
    
    def load_model(self, model_name):
        self.model = keras.models.load_model(model_name)
    
    def train(self):

        steps_for_validation = self.data.validation_data.n//self.data.validation_data.batch_size
        steps_for_train = self.data.training_data.n//self.data.training_data.batch_size

        # Compilation process
        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

        # Training process
        self.model.fit_generator(self.data.training_data, epochs=5, steps_per_epoch=steps_for_train,
                validation_data=self.data.validation_data, validation_steps=steps_for_validation)

        # Evaluation process
        evaluation = self.model.evaluate_generator(generator=self.data.validation_data, steps=steps_for_validation)
        print(f"EVALUATION:\n Loss {evaluation[0]}\nAccuracy: {evaluation[1]}\n")
        
        self.model.save("magichands_model.h5")
    
    def predict_generator(self):
        steps_for_test = self.data.testing_data.n//self.data.testing_data.batch_size

        self.predictions = self.model.predict_generator(self.data.testing_data,
                                                        steps=steps_for_test,
                                                        use_multiprocessing=True,
                                                        verbose=1)

        """
        self.data.testing_data.reset() # Reset in order to get the right labeling
        predictions_indices = numpy.argmax(self.predictions, axis=1)

        labels = (self.data.training_data.class_indices)
        labels = dict((value, key) for key,value in labels.items())
        final_predictions = [labels[k] for k in predictions_indices]

        filenames = self.data.testing_data.filenames
        results = pandas.DataFrame({"Filename:":filenames, "Predictions":final_predictions})
        results.to_csv("results.csv", index=False)
        """

    
    def predict(self):
        self.model.predict()