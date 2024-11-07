import os
import glob
import shutil
import random
import cv2
import numpy as np

def get_splits(parent, folder, num_files, destination):
    size = len(glob.glob(os.path.join(folder, "*")))
    return {destination + "train/" + folder.replace(parent, ""): [int(size * .70), int(num_files * .70) - int(size * .70)],
            destination + "val/" + folder.replace(parent, ""): [int(size * .20), int(num_files * .20) - int(size * .20)],
            destination + "test/" + folder.replace(parent, ""): [int(size * .10), int(num_files * .10) - int(size * .10)]
            }

def random_augment(image):
    image = cv2.imread(image)
    random_num = random.randint(1, 3)
    if random_num == 1: # random rotate
        angle = random.randint(-45, 45)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, m, (w, h))
    elif random_num == 2:
        return cv2.flip(image, 1)
    else:
        noise = np.random.randn(image.shape[0], image.shape[1], image.shape[2])
        noisy_img = image + 0.1 * noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

def structure_dataset(dataset_name):
    dataset_name = "dataset/" + dataset_name + '/'

    subdir = [] # get names of subdirectories under data_dir only, not lower
    for root, folders, files in os.walk(dataset_name):
        for folder in folders:
            subdir.append(folder)
        break

    max_size = [] # get largest num of splits only
    for dirs in subdir:
        max_size.append(len(glob.glob(os.path.join(dataset_name + dirs, "*"))))

    max_size = max(max_size) # will oversample to reach max
    additional_oversample = 1000 # count of images to add, oversample all

    structured_dataset = "dataset_structured/" + dataset_name.replace("dataset/", "")
    if not os.path.isdir(structured_dataset): os.mkdir(structured_dataset)

    path_count_dict = {} # {path of folder to be restructured: {path of restructured folder: (count of files, num to be oversampled)}}
    for dirs in subdir:
        path_count_dict[dataset_name + dirs] = get_splits(dataset_name, dataset_name + dirs,
                                                          max_size + additional_oversample,
                                                          structured_dataset)

    # create backup
    shutil.copytree(dataset_name, "backup/" + dataset_name, dirs_exist_ok=True)

    for dirs in path_count_dict:
        images = os.listdir(dirs)
        random.shuffle(images)

        for dest in path_count_dict[dirs]:
            if not os.path.isdir(dest): os.makedirs(dest)

            orig_count = path_count_dict[dirs][dest][0]

            while path_count_dict[dirs][dest][0] > 0: # original images, renamed
                count = path_count_dict[dirs][dest][0]

                src_image_path = images.pop(0)
                extension = src_image_path.rsplit(".", 1)[1]

                shutil.copy2(dirs + "/" + src_image_path, dest)
                os.rename(dest + "/" + src_image_path,
                          dest + "/" + "{:0>4d}".format(count) + "_" + dest.rsplit("/", 1)[1] + "." + extension)

                path_count_dict[dirs][dest][0] = count - 1

            while path_count_dict[dirs][dest][1] > 0: # loop for generated oversampled images
                count = path_count_dict[dirs][dest][1]

                image = random.choice(os.listdir(dest))
                src_image_path = dest + "/" + image

                cv2.imwrite(dest + "/" + "{:0>4d}".format(count + orig_count) + "_aug_" + image.split("_", 1)[1],
                            random_augment(src_image_path))

                path_count_dict[dirs][dest][1] = count - 1