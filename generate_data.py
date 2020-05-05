#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import matplotlib.pyplot as plt
import os
import PIL
import numpy as np

DATA_SIZE = int(146e3) # taille des données (test/valid/test compris), max : 146K
DATA_TRAIN_SIZE = int(DATA_SIZE*0.6)
DATA_VALID_SIZE = int(DATA_SIZE*0.2)
DATA_TEST_SIZE = int(DATA_SIZE*0.2)

IMAGE_SIZE = (2790,100)  # Images resize en taille ~ moyenne : moyenne pondérée
INPUT_IMAGE_SHAPE = (100,2790,1) # shape qui rentre dans le cnn
# même taille pour toutes les parties d'images, présentes en même densité

INPUT_IMAGE_SHAPE_RGB = (100,2790,3)
BATCH_SIZE = 30

THRESHOLD_PREDICTION = 0.5

NUM_CALLS = tf.data.experimental.AUTOTUNE

DATA_PATH = "/home/cogrannr/roues/MEFRO/grises/"
DATA_PATH_DEFAUTS_REELS = "/home/cogrannr/roues/MEFRO/images_defauts/"

"""
    Fonction custom, car doit être fait en eager execution, sinon c'est un param "none" passé à Image.open
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

def parse_images(filename,label): # appel "open_image"
    image_arr = tf.py_function(open_image,[filename],tf.float32)
    image_arr = tf.image.convert_image_dtype(image_arr,tf.float32)
    return image_arr,label

def gray_to_rgb(image,label):
    image = image[:,:,0]
    image = tf.stack([image,image,image],axis=2)
    return image,label

def data_augmentation(image,label):
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_brightness(image, max_delta=10.0/255.0)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image,label

def reset_shapes_gray(image,label):
    image.set_shape(list(INPUT_IMAGE_SHAPE))
    return image, label

def reset_shapes_rgb(image,label):
    image.set_shape(list(INPUT_IMAGE_SHAPE_RGB))
    return image, label


"""
    chargement des listes de fichiers photo
"""
def load_data(cover_dir,faulty_dir):

    file_names_cover = tf.data.Dataset.list_files(cover_dir+"/*",shuffle=False)
    file_names_cover = file_names_cover.take(DATA_SIZE//2)
    labels_0 = tf.data.Dataset.from_tensors(0).repeat(DATA_SIZE//2)
    data_set_cover = tf.data.Dataset.zip((file_names_cover,labels_0))

    file_names_faulty = tf.data.Dataset.list_files(faulty_dir+"/*",shuffle=False)
    file_names_faulty = file_names_faulty.take(DATA_SIZE//2)
    labels_1 = tf.data.Dataset.from_tensors(1).repeat(DATA_SIZE//2)
    data_set_faulty = tf.data.Dataset.zip((file_names_faulty,labels_1))

    data_set = data_set_cover.concatenate(data_set_faulty)
    data_set = data_set.shuffle(DATA_SIZE)

    return data_set

"""
    répartition des fichiers enn ensembles train/valid/test
"""
def generate_train_valid_test(cover_dir,faulty_dir, augment_data=False, gray_to_rgb_images=False):

    data_set = load_data(cover_dir, faulty_dir)

    train_dataset = data_set.take(DATA_TRAIN_SIZE)
    valid_dataset = data_set.skip(DATA_TRAIN_SIZE).take(DATA_VALID_SIZE)
    test_dataset = data_set.skip(DATA_TRAIN_SIZE+DATA_VALID_SIZE)

    datasets_list = []

    for data_set in [train_dataset, valid_dataset, test_dataset]:

        if data_set == train_dataset:
            data_set = data_set.repeat()
            data_set = data_set.map(parse_images, num_parallel_calls=NUM_CALLS)
            if augment_data == True:
                data_set = data_set.map(reset_shapes_gray, num_parallel_calls=NUM_CALLS)
                data_set = data_set.map(data_augmentation, num_parallel_calls=NUM_CALLS)
        else:
            data_set = data_set.map(parse_images, num_parallel_calls=NUM_CALLS)

        if gray_to_rgb_images == True:
            data_set =  data_set.map(gray_to_rgb, num_parallel_calls=NUM_CALLS)
            data_set = data_set.map(reset_shapes_rgb, num_parallel_calls=NUM_CALLS)
        else:
            data_set = data_set.map(reset_shapes_gray, num_parallel_calls=NUM_CALLS)

        data_set = data_set.batch(BATCH_SIZE)
        data_set = data_set.prefetch(1)

        datasets_list.append(data_set)

    return datasets_list
