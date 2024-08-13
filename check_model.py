import os
from keras.models import load_model
import tensorflow as tf
import keras

# Function to check the current working directory and list files
def check_directory():
    cwd = os.getcwd()
    files = os.listdir(cwd)
    
    print("Current working directory:", cwd)
    print("Files in the current directory:", files)

# Function to check TensorFlow and Keras versions
def check_versions():
    tf_version = tf.__version__
    keras_version = keras.__version__
    
    print("TensorFlow version:", tf_version)
    print("Keras version:", keras_version)

# Function to attempt loading the model
def load_rnn_model(model_path):
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

if __name__ == "__main__":
    # Step 1: Check the current directory and list files
    check_directory()
    
    # Step 2: Check TensorFlow and Keras versions
    check_versions()
    
    # Step 3: Attempt to load the model
    model_path = 'rnn_model.h5'  # Adjust this path if your model is stored elsewhere
    load_rnn_model(model_path)
