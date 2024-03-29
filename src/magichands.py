import tensorflow # Keras's backbone
import keras

import dataset
import convnet

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
   
def main():
    """ NOTE: Uncomment the build and train to run an existing model with load_model
              else, build and train the model with those 2 lines and make sure to
              uncomment the "load_model" line """
    data = dataset.DataSet("data/train", "data/test_transformed", t_batch_size=64)
    #data.generate_previews()
    cnn = convnet.ConvNet(data)
    #cnn.build()
    #cnn.train()
    cnn.load_model("transformed.h5")
    cnn.predict_generator()
    exit(0)

main()