import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def create_model():
    """Creates and returns a CNN model for classification."""
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(5, activation="softmax")  # Output layer for 5 classes
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    model = create_model()
    model.save("./models/primitives_model.h5")
    print("Model saved at ./models/primitives_model.h5")
