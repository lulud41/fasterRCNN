#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import model
import anchor_tools
import generate_data_sequence


ANNOTATION_PATH = "../VOCdevkit/VOC2012/Annotations/"
DATA_PATH = "../VOCdevkit/VOC2012/JPEGImages/"

IMAGE_SHAPE = (224,224)
INPUT_IMAGE_SHAPE = (224,224,3) # shape qui rentre dans le cnn

RATIO_X = 4
RATIO_Y = 4

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

DATA_SIZE = int(17120*0.001)# 17125 max

LEARNING_RATE = 0.00001
NUM_EPOCHS = 40

SAVED_MODEL_NAME = "rpn_model.h5"


anchor_array = anchor_tools.generate(INPUT_IMAGE_SHAPE, RATIO_X, RATIO_Y, anchor_sizes )
anchor_array_not_clean = anchor_tools.generate(INPUT_IMAGE_SHAPE, RATIO_X, RATIO_Y, anchor_sizes, clean=False )

train_dataset =  generate_data_sequence.FasterRCNN_Dataset_Sequence("train",DATA_SIZE, INPUT_IMAGE_SHAPE ,DATA_PATH,
    ANNOTATION_PATH, batch_size=BATCH_SIZE, anchor_array = anchor_array)


# cut 80 : 28,28
# cut 38 : 56,56 : ration 4
m = model.ModelCreator(anchor_array.shape, "resnet.h5", trainable_backbone=False, cut_layer=38)
rpn, rpn_cls_loss, rpn_reg_loss = m.RPN_model()


optimizer = tf.keras.optimizers.SGD(LEARNING_RATE, momentum=0.9, nesterov=True)

@tf.function
def train_step(rpn, x,y , rpn_cls_loss, rpn_reg_loss):
    with tf.GradientTape() as tape:

        pred = rpn(x)

        pred_cls = pred["rpn_cls"]
        pred_reg = pred["rpn_reg"]

        y_cls = y["rpn_cls"]
        y_reg = y["rpn_reg"]

        cls_loss = rpn_cls_loss(y_cls, pred_cls)
        reg_loss = rpn_reg_loss(y_reg, pred_reg)

        loss = cls_loss + reg_loss

        grad = tape.gradient(loss, rpn.trainable_variables)

        optimizer.apply_gradients(zip(grad, rpn.trainable_variables))

    return  cls_loss, reg_loss

for epoch in range(0, NUM_EPOCHS):
    print(">>>>> Epoch ",epoch)

    for x,y in train_dataset:
        cls_loss, reg_loss = train_step(rpn, x, y , rpn_cls_loss, rpn_reg_loss)
        loss = cls_loss + reg_loss

        print( "loss cls ", cls_loss.numpy(), ",  reg ",reg_loss.numpy())

    if epoch == 0:
        best_epoch_loss = loss
        rpn.save(SAVED_MODEL_NAME)
    else:
        if loss < best_epoch_loss:
            best_epoch_loss = loss
            rpn.save(SAVED_MODEL_NAME)



for i in range(0,20):
    x,y = train_dataset.__getitem__(i)
    pred = rpn(x)
    name = "../VOCdevkit/VOC2012/JPEGImages/"+str(train_dataset.file_names[i])

    model.show_prediction(pred, anchor_array_not_clean, name)
    input()
