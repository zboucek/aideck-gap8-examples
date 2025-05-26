import os
import shutil
import random
import glob

ROOT_PATH     = os.path.abspath(os.curdir) + "/examples/ai/classification/"
IMAGES_FOLDER = os.path.join(ROOT_PATH, "images")
BASE_DIR      = os.path.join(ROOT_PATH, "training_data")
TRAIN_DIR     = os.path.join(BASE_DIR, "train")
VAL_DIR       = os.path.join(BASE_DIR, "validation")

# clean out any old data
for d in (TRAIN_DIR, VAL_DIR):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

# get all class names
classes = [d for d in os.listdir(IMAGES_FOLDER)
           if os.path.isdir(os.path.join(IMAGES_FOLDER, d))]

# find smallest class size
min_count = min(len(os.listdir(os.path.join(IMAGES_FOLDER, cls)))
                for cls in classes)

# for each class: sample, split, copy
for cls in classes:
    src = os.path.join(IMAGES_FOLDER, cls)
    all_imgs = glob.glob(os.path.join(src, "*"))
    random.shuffle(all_imgs)

    # cap to min_count
    sampled = all_imgs[:min_count]
    n_val   = int(0.2 * min_count)
    val_imgs   = sampled[:n_val]
    train_imgs = sampled[n_val:]

    # make dest dirs
    target_train = os.path.join(TRAIN_DIR, cls)
    target_val   = os.path.join(VAL_DIR, cls)
    os.makedirs(target_train, exist_ok=True)
    os.makedirs(target_val,   exist_ok=True)

    # copy files
    for p in val_imgs:
        shutil.copy(p, os.path.join(target_val,   os.path.basename(p)))
    for p in train_imgs:
        shutil.copy(p, os.path.join(target_train, os.path.basename(p)))
