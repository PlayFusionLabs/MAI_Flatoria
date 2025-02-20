from tensorflow.keras.preprocessing import image_dataset_from_directory

train_dataset = image_dataset_from_directory(
    "dataset/train",
    image_size=(224, 224), 
    batch_size=32,         
    shuffle=True
)

test_dataset = image_dataset_from_directory(
    "dataset/test",
    image_size=(224, 224),
    batch_size=32
)
