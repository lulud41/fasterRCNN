#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import os
import PIL
import numpy as np

DATA_PATH = "/home/lucien/Documents/ST09/stage_utt/implem/check_img_mask/img"



DATA_SIZE = int(73182) # taille des données (test/valid/test compris), max : 73K
DATA_TRAIN_SIZE = int(DATA_SIZE*0.6)
DATA_VALID_SIZE = int(DATA_SIZE*0.2)
DATA_TEST_SIZE = int(DATA_SIZE*0.2)

IMAGE_SIZE = (2790,99)  # Images resize en taille ~ moyenne : moyenne pondérée
INPUT_IMAGE_SHAPE = (99,2790,1) # shape qui rentre dans le cnn
# même taille pour toutes les parties d'images, présentes en même densité

NUM_CALLS = tf.data.experimental.AUTOTUNE

DATA_PATH = "/home/cogrannr/roues/MEFRO/grises/img_galbe_avec_defauts"
DATA_PATH_DEFAUTS_REELS = "/home/cogrannr/roues/MEFRO/images_defauts/img_galbe_avec_defauts"
BATCH_SIZE = 1

"""
    Fonction custom, car doit être fait en eager execution, sinon
    c'est un param "none" passé à Image.open
    ouvereture des photos / resize
"""
def open_image(file_name):
    name = tf.get_static_value(file_name)

    name = name.decode()
    image = PIL.Image.open(name)

    image = image.resize(IMAGE_SIZE)
    image_arr = np.array(image)

    image_arr = image_arr[:,:,np.newaxis]
    return image_arr

def parse_images(filename): # appel "open_image"
    image_arr = tf.py_function(open_image,[filename],tf.float32)
    image_arr = tf.image.convert_image_dtype(image_arr,tf.float32)
    return image_arr

def reset_shapes_gray(image):
    image.set_shape(list(INPUT_IMAGE_SHAPE))
    return image


"""
    chargement des listes de fichiers photo
"""
def load_data():

    file_names = tf.data.Dataset.list_files(DATA_PATH+"/*",shuffle=False)
    file_names = file_names.take(DATA_SIZE)

    return file_names

"""
    répartition des fichiers enn ensembles train/valid/test
"""
def generate_train_valid_test():

    data_set = load_data()

    train_dataset = data_set.take(DATA_TRAIN_SIZE)
    valid_dataset = data_set.skip(DATA_TRAIN_SIZE).take(DATA_VALID_SIZE)
    test_dataset = data_set.skip(DATA_TRAIN_SIZE+DATA_VALID_SIZE)

    datasets_list = []

    for data_set in [train_dataset, valid_dataset, test_dataset]:
            data_set = data_set.map(parse_images, num_parallel_calls=NUM_CALLS)
            data_set = data_set.map(reset_shapes_gray, num_parallel_calls=NUM_CALLS)

            data_set = data_set.batch(BATCH_SIZE)
            data_set = data_set.prefetch(1)

            datasets_list.append(data_set)

    return datasets_list

train_dataset, valid_dataset, test_dataset = generate_train_valid_test()
