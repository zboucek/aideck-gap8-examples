import os
import shutil

# Source and target directories
source_dir = "examples/ai/classification/dataset-klasifikace"
target_dir = "examples/ai/classification/images"

# Iterate through folders in the source directory
for class_folder in os.listdir(source_dir):
    source_class_path = os.path.join(source_dir, class_folder)
    target_class_path = os.path.join(target_dir, class_folder)

    # Create the target folder if it doesn't exist
    os.makedirs(target_class_path, exist_ok=True)

    # Iterate through files in the class folder
    for file_name in os.listdir(source_class_path):
        source_file_path = os.path.join(source_class_path, file_name)
        target_file_path = os.path.join(target_class_path, file_name)

        # If the file already exists, add a number to the name
        if os.path.exists(target_file_path):
            base, ext = os.path.splitext(file_name)
            counter = 1
            while os.path.exists(target_file_path):
                new_file_name = f"{base}_{counter}{ext}"
                target_file_path = os.path.join(target_class_path, new_file_name)
                counter += 1

        # Copy the file
        shutil.copy2(source_file_path, target_file_path)

print("Copying completed.")