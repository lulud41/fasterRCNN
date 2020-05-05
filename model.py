#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

def init_RPN_model(input_shape, nb_anchors):
    model_input = tf.keras.layers.Input(input_shape=input_shape)

    x = tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding="VALID",strides=(1,1))(model_input)
    cls = tf.keras.layers.Conv2D(nb_anchors,(1,1),activation="sigmoid",padding="SAME",strides=(1,1))(x)
    reg = tf.keras.layers.Conv2D(nb_anchors*4,(1,1),padding="SAME",strides=(1,1))(x)

    model = tf.keras.Model(input=model_input, outputs=[cls, reg])

    return model

if __name__=="__main__":
    RPN_model = init_RPN_model()
    print(RPN_model.summary())
