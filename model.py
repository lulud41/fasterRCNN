#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import anchor_tools
import functools
import numpy as np

class ModelCreator():
    def __init__(self, anchor_array_shape, backbone_path,
        trainable_backbone=False, cut_layer=80):

        self.anchor_array_shape = anchor_array_shape
        self.nb_anchors_sizes = self.anchor_array_shape[1]

        self.backbone = self.init_backbone(backbone_path, cut_layer, trainable_backbone)

    def init_backbone(self, backbone_path, cut_layer, trainable):
        backbone = tf.keras.models.load_model(backbone_path)
        backbone.trainable = trainable
        input_layer = backbone.input

        output_layer = backbone.layers[cut_layer].output

        backbone = tf.keras.models.Model(inputs=input_layer, outputs= output_layer)

        return backbone

    def RPN_model(self):

        input_layer = self.backbone.input
        x_1 = self.backbone.output

        x_2 = tf.keras.layers.Conv2D(256,(3,3),activation="relu",padding="SAME",strides=(1,1),name="conv2d_rpn",
            )(x_1)

        cls_1 = tf.keras.layers.Conv2D(self.nb_anchors_sizes,(1,1),activation="sigmoid",padding="SAME",strides=(1,1),name="cls_conv")(x_2)

        cls_2 = tf.keras.layers.Reshape(target_shape=(self.anchor_array_shape[0],self.nb_anchors_sizes), name="rpn_cls")(cls_1)

        reg_1 = tf.keras.layers.Conv2D(self.nb_anchors_sizes*4,(1,1),padding="SAME",strides=(1,1),name="reg_conv",
            )(x_2)
        reg_2 = tf.keras.layers.Reshape(target_shape=(self.anchor_array_shape[0],self.nb_anchors_sizes*4), name="rpn_reg")(reg_1)

        model = tf.keras.Model(inputs=input_layer, outputs={"rpn_cls":cls_2, "rpn_reg":reg_2})

        rpn_reg_loss_func = functools.partial(self.rpn_reg_loss, anchor_array=self.anchor_array_shape)

        return model, self.rpn_cls_loss, rpn_reg_loss_func

    def rpn_cls_loss(self, cls_true, cls_pred):

        num_pos_labels = tf.reduce_sum(tf.cast(cls_true==1,tf.int32))
        num_neg_labels = tf.reduce_sum(tf.cast(cls_true==-1,tf.int32))

        pos_labels = tf.ones(num_pos_labels)
        neg_labels = tf.zeros(num_neg_labels)

        labels = tf.concat((pos_labels, neg_labels), axis=0)

        pred = tf.concat((cls_pred[cls_true==1], cls_pred[cls_true==-1]),axis=0)

        loss = tf.keras.losses.binary_crossentropy(labels, pred)

        return loss

    def rpn_reg_loss(self, reg_true, reg_pred, anchor_array):

        reg_true = tf.reshape(reg_true,(1 , anchor_array[0],  anchor_array[1],4))
        reg_pred = tf.reshape(reg_pred,(1 , anchor_array[0],  anchor_array[1],4))

        labels = tf.reshape(reg_true[reg_true != 0], (-1,4))
        pred = tf.reshape(reg_pred[reg_true!= 0],(-1,4))

        dif = tf.abs(pred - labels)
        a = dif[dif < 1]
        a = tf.reduce_sum(tf.square(a))*0.5

        b = dif[dif >=1]
        b = tf.reduce_sum(b - 0.5)

        loss = (a+b)

        return loss




def show_prediction(pred,anchor_array, image_name, image_shape=(224,224)  ):

    cls_pred = pred["rpn_cls"]
    reg_pred = pred["rpn_reg"]

    cls_pred = tf.get_static_value(cls_pred)
    cls_pred = cls_pred[0]

    reg_pred = tf.get_static_value(reg_pred)
    reg_pred = reg_pred[0]
    reg_pred = reg_pred.reshape((-1,9,4))


    indexes = np.where(cls_pred >= cls_pred.max()*0.97)
    #indexes = np.where(cls_pred > 0.9)


    pred_bbox = reg_pred[indexes]

    print(anchor_array[indexes])

    pred_unorm = anchor_tools.UN_parametrize_predicition(anchor_array[indexes], pred_bbox)

    anchor_tools.draw_anchors(image_name,pred_unorm, image_shape )


if __name__=="__main__":

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

    anchors = anchor_tools.generate(INPUT_IMAGE_SHAPE, RATIO_X, RATIO_Y, anchor_sizes )



    m = RPN(anchors.shape, "resnet.h5")
    m.build(m.backbone.input_shape)
    m.compile("adam")
