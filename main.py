#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import model
import anchors
import generate_data_sequence
# LAST


# init model, init data, lance train / test / pred / visu

GROUND_TRUTH_BBOX_PATH = "./bbox_galbe_2.csv"
DATA_PATH = "/home/lucien/Documents/ST09/stage_utt/implem/check_img_mask/img/"
#DATA_PATH = "/home/cogrannr/roues/MEFRO/grises/img_galbe_avec_defauts"


IMAGE_SHAPE = (2792,99)  # Images resize en taille ~ moyenne : moyenne pondérée
INPUT_IMAGE_SHAPE = (99,2792,1) # shape qui rentre dans le cnn
# même taille pour toutes les parties d'images, présentes en même densité

RATIO_X = 2 #4
RATIO_Y = 4  # 8

BATCH_SIZE = 1
ANCHOR_BATCH_SIZE = 16  # 4 pos et 4 neg
ratio_batch = 0.3

"""anchor_sizes = ((10,20),
                (20,40),
                (35,70),
                (50,100),
                (10,10),
                (15,15),
                (20,20),
                (25,25),
                (30,30),
                (40,40),
                (50,50))"""

anchor_sizes = ((10,20),
                (30,60),
                (50,100),
                (10,10),
                (30,30),
                (50,50))


#DATA_SIZE = int(73182) # taille des données (test/valid/test compris), max : 73K

DATA_SIZE = int(50)

LEARNING_RATE = 0.000001
NUM_EPOCHS = 200


anchors_array = anchors.generate(INPUT_IMAGE_SHAPE,RATIO_X,RATIO_Y,anchor_sizes)
print(anchors_array.shape)

train_dataset = generate_data_sequence.Dataset_sequence("train",DATA_SIZE, IMAGE_SHAPE,ANCHOR_BATCH_SIZE,
    ratio_batch, DATA_PATH, GROUND_TRUTH_BBOX_PATH, anchors_array)

valid = generate_data_sequence.Dataset_sequence("valid",DATA_SIZE, IMAGE_SHAPE,ANCHOR_BATCH_SIZE,
    ratio_batch, DATA_PATH, GROUND_TRUTH_BBOX_PATH, anchors_array)




model_creator = model.Model_creator(INPUT_IMAGE_SHAPE,anchors_array,LEARNING_RATE,"base_model_galbe.h5",BATCH_SIZE)
rpn = model_creator.init_RPN_model()




rpn.summary()

rpn.fit(train_dataset,epochs=NUM_EPOCHS, validation_data=valid   )





"""
valid_dataset = generate_data_sequence.Dataset_sequence("valid",DATA_SIZE, IMAGE_SHAPE,ANCHOR_BATCH_SIZE,
    ratio_batch, DATA_PATH, GROUND_TRUTH_BBOX_PATH, anchors_array)

test_dataset  = generate_data_sequence.Dataset_sequence("test",DATA_SIZE, IMAGE_SHAPE,ANCHOR_BATCH_SIZE,
    ratio_batch, DATA_PATH, GROUND_TRUTH_BBOX_PATH, anchors_array)

rpn_model = model.init_RPN_model(input_shape=INPUT_IMAGE_SHAPE,nb_anchors=len(anchor_sizes),
    learning_rate=LEARNING_RATE)


call_backs_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath ='rpn.h5', monitor ="val_accuracy", save_best_only = True)]

rpn_model.fit(train_dataset, epochs=NUM_EPOCHS, callbacks=call_backs_list, validation_data=valid_dataset,
        max_queue_size=10, workers=1, use_multiprocessing=True)

        # use_multiprocessing  : pas obligé de le mettre e True, peut etre freeze, a voir
"""
