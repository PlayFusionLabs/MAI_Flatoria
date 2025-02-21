import os
import tensorflow as tf
from scripts.model import create_model
from scripts.train import train_model
from scripts.predict import predict_image

MODEL_PATH = "./models/primitives_model.h5"

def main():
    while True:
        print("\n=== MENU ===")
        print("1. Create a new model")
        print("2. Train the model")
        print("3. Make a prediction")
        print("4. Exit")

        choice = input("Select an option (1-4): ")

        if choice == "1":
            # Create and save a new model
            model = create_model()
            model.save(MODEL_PATH)
            print(f"Model saved at {MODEL_PATH}")

        elif choice == "2":
            # Train the model if it exists
            if os.path.exists(MODEL_PATH):
                train_model(MODEL_PATH)
                print(f"Trained model saved at {MODEL_PATH}")
            else:
                print("Error: You must create a model first (Option 1).")

        elif choice == "3":
            # Perform image classification
            img_path = input("Enter the path to the image: ")
            if os.path.exists(MODEL_PATH):
                predicted_class = predict_image(img_path, MODEL_PATH)
                print(f"Predicted class: {predicted_class}")
            else:
                print("Error: The model has not been created or trained yet.")

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid selection, please try again.")

if __name__ == "__main__":
    main()
