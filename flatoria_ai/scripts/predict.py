import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

class_names = {0: 'Window', 1: 'Door', 2: 'Wall', 3: 'Floor', 4: 'Ceiling'}

def predict_image(img_path, model_path):
    """Loads a trained model and predicts the class of the given image."""
    
    model = tf.keras.models.load_model(model_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    
    return class_names.get(class_index, "Unknown")  # Return class name

if __name__ == "__main__":
    MODEL_PATH = "./models/primitives_model.h5"
    IMG_PATH = "./dataset/test/sample.jpg"

    predicted_class = predict_image(IMG_PATH, MODEL_PATH)
    print(f"Predicted class: {predicted_class}")
