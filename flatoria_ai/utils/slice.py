import cv2  
import os  
import uuid  


input_folder = "input_images"
output_folder = "cropped_output"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(input_folder):
    os.makedirs(input_folder)    

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)  
    image = cv2.imread(image_path)  

    if image is None:
        print(f"No FILE {image_file}, skipping...")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_gray.png"), gray)  # Зберігаємо ч/б копію

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

    image_output_folder = os.path.join(output_folder, os.path.splitext(image_file)[0])
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    found_count = 0

    for i, cnt in enumerate(contours):  
        x, y, w, h = cv2.boundingRect(cnt)  

        if w > 50 and h > 50:
            roi = gray[y:y+h, x:x+w]  
            unique_id = str(uuid.uuid4())[:8]
            output_path = os.path.join(image_output_folder, f"cropped_{i}_{unique_id}.png")
            cv2.imwrite(output_path, roi)  
            found_count += 1

    print(f"Processed: {image_file}, Found {found_count} objects.")

print("✅ All images processed! Cropped elements saved in 'cropped_output'.")
