from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), batch_size=32, class_mode="sparse", subset="training")

val_data = datagen.flow_from_directory(
    "dataset", target_size=(224, 224), batch_size=32, class_mode="sparse", subset="validation")

# Запуск навчання
model.fit(train_data, validation_data=val_data, epochs=20)
