import os
import shutil
import tensorflow as tf
import numpy as np

ROOT_PATH = (
    f"{os.path.abspath(os.curdir)}/examples/ai/classification/"
)
IMAGES_FOLDER = f"{ROOT_PATH}{'images'}"

# Folder structure
base_dir = f"{ROOT_PATH}{'training_data'}"
train_dir = os.path.join(base_dir, 'train')  
val_dir = os.path.join(base_dir, 'validation')

# Make folder structure
os.makedirs(train_dir)
os.makedirs(val_dir)

# Get class folders
class_folders = os.listdir(IMAGES_FOLDER)
        
for cls_folder in class_folders:

  # Copy images to train folder
  cls_train_folder = os.path.join(train_dir, cls_folder)
  shutil.copytree(os.path.join(IMAGES_FOLDER, cls_folder), cls_train_folder)
  
  # Get list of images
  images = tf.data.Dataset.list_files(cls_train_folder + '/*')
  
  # Shuffle images
  images = images.shuffle(buffer_size=1000) 
  
  # Split into train and validation
  num_val = int(0.2 * tf.cast(images.cardinality(), tf.float32))
  train_ds = images.skip(num_val)
  val_ds = images.take(num_val)

  # Move validation images to separate folder
  cls_val_folder = os.path.join(val_dir, cls_folder)
  os.makedirs(cls_val_folder)
  
  for img_path in val_ds:
    img_path_bytes = np.bytes_(img_path)
    img_path_str = tf.compat.as_text(img_path_bytes)
    shutil.move(img_path_str, cls_val_folder+"/")