import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .model import create_model 


datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

def train_model(model_path):
    """Train the model on the dataset and save it to the specified path."""
    
    train_data = datagen.flow_from_directory(
        "dataset", target_size=(224, 224), batch_size=32, class_mode="sparse", subset="training"
    )
    print("Class indices for training data:", train_data.class_indices)
    
    val_data = datagen.flow_from_directory(
        "dataset", target_size=(224, 224), batch_size=32, class_mode="sparse", subset="validation"
    )
    
    model = create_model()
    model.fit(train_data, validation_data=val_data, epochs=20)
    
    model.save(model_path)
    print(f"Trained model saved at {model_path}")

