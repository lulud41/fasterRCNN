#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import model
import anchor_tools
import generate_data_sequence


ANNOTATION_PATH = "/content/VOCdevkit/VOC2012/Annotations/"
DATA_PATH = "/content/VOCdevkit/VOC2012/JPEGImages/"

IMAGE_SHAPE = (224,224)  # Images resize en taille ~ moyenne : moyenne pondérée
INPUT_IMAGE_SHAPE = (224,224,3) # shape qui rentre dans le cnn

RATIO_X = 8
RATIO_Y = 8

# pour size
anchor_sizes = ((80,80),
                (120,120),
                (200,200),

                (40,80),
                (60,120),
                (100,200),

                (80,40),
                (120,60),
                (200,100))

BATCH_SIZE = 256

DATA_SIZE = int(17120*0.1)# 17125 max

LEARNING_RATE = 0.0001
NUM_EPOCHS = 10



anchor_array = anchor_tools.generate(INPUT_IMAGE_SHAPE, RATIO_X, RATIO_Y, anchor_sizes )

train_dataset =  generate_data_sequence.FasterRCNN_Dataset_Sequence("train",DATA_SIZE, INPUT_IMAGE_SHAPE ,DATA_PATH,
    ANNOTATION_PATH, batch_size=BATCH_SIZE, anchor_array = anchor_array)

valid = generate_data_sequence.FasterRCNN_Dataset_Sequence("valid", DATA_SIZE, INPUT_IMAGE_SHAPE ,DATA_PATH,
    ANNOTATION_PATH, batch_size=BATCH_SIZE, anchor_array = anchor_array)

#filepath="/content/drive/My\ Drive/fasterRCNN/rpn_1_chkpnt.h5",

callback = [
  tf.keras.callbacks.ModelCheckpoint(filepath="/content/rpn_2_chkpnt.h5",
      save_best_only=True),

  tf.keras.callbacks.TensorBoard("tensor_board"),

  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
      patience=2, min_lr=0.00001 , verbose=1, min_delta=0.05)
]




m = model.ModelCreator(anchor_array.shape, "resnet.h5", trainable_backbone=False, cut_layer=80)
rpn = m.init_RPN_model(LEARNING_RATE)

print("test 0", rpn(np.random.rand(1,224,224,3)))

rpn.fit(train_dataset,epochs=NUM_EPOCHS)#, validation_data=valid, callbacks=callback)


print("test 1", rpn(np.random.rand(1,224,224,3)))
print("test 2", rpn(np.random.rand(1,224,224,3)))



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
