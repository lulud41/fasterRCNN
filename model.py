#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import anchors

# H_out = h_in / r_x
# r_x =

class Model_creator():
    def __init__(self, input_shape, anchor_array,learning_rate,
        base_model_name, batch_size):

        self.input_shape = input_shape

        self.anchor_array = tf.convert_to_tensor(anchor_array,dtype=tf.float32)
        self.anchor_array_shape = anchor_array.shape

        self.nb_anchors = self.anchor_array_shape[1]

        self.learning_rate = learning_rate

        self.base_model_name = base_model_name
        self.batch_size = batch_size

    def base_model(self):
            model_input = tf.keras.Input(shape=self.input_shape)
            conv_1 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='SAME',strides=(1,2))(model_input)
            conv_2 = tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='SAME',strides=(1,1))(conv_1)
            bn_1 = tf.keras.layers.BatchNormalization()(conv_2)
            mxp_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(bn_1)

            conv_3 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='SAME',strides=(1,1))(mxp_1)
            conv_4 = tf.keras.layers.Conv2D(64,(3,3),activation='relu',padding='SAME',strides=(1,1))(conv_3)
            bn_2 = tf.keras.layers.BatchNormalization()(conv_4)
            mxp_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(bn_2)

            conv_5 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='SAME',strides=(1,1))(mxp_2)
            conv_6 = tf.keras.layers.Conv2D(128,(3,3),activation='relu',padding='SAME',strides=(1,1))(conv_5)
            out = tf.keras.layers.BatchNormalization()(conv_6)

            model = tf.keras.Model(inputs=model_input, outputs=out)
            print(model.summary())

            return model

    def RPN_model(self):

        base_model = tf.keras.models.load_model(self.base_model_name,compile=False)
        base_model.trainable = False

        model_input = base_model.input

        x = base_model.layers[-13].output

        #x = conv_plus = tf.keras.layers.Conv2D(256,(3,3),activation='relu',padding='SAME',strides=(1,1))(x)

        x = tf.keras.layers.Conv2D(256,(3,3),activation="relu",padding="SAME",strides=(1,1),name="conv2d_rpn",
            kernel_initializer="glorot_uniform")(x)

        cls = tf.keras.layers.Conv2D(self.nb_anchors,(1,1),activation="sigmoid",padding="SAME",strides=(1,1),name="cls_conv",
            kernel_initializer="glorot_uniform")(x)
        cls = tf.keras.layers.Reshape(target_shape=(self.anchor_array_shape[0],self.nb_anchors), name="cls_pred")(cls)

        reg = tf.keras.layers.Conv2D(self.nb_anchors*4,(1,1),padding="SAME",strides=(1,1),name="reg_conv",
            kernel_initializer="glorot_uniform")(x)
        reg = tf.keras.layers.Reshape(target_shape=(self.anchor_array_shape[0],self.nb_anchors*4), name="reg_pred")(reg)

        model = tf.keras.Model(inputs=model_input, outputs=[cls, reg])

        return model

    def init_RPN_model(self):
        rpn_model = self.RPN_model()

        #optim = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        optim = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


        losses = {"cls_pred":self.rpn_cls_loss, "reg_pred":self.rpn_reg_loss}

        precision_metric = tf.keras.metrics.Precision(thresholds=0.5)
        metrics = {"cls_pred":["accuracy",precision_metric]} #,  # "reg_pred":reg_loss}

        loss_weights = {"cls_pred":1,"reg_pred":1}

        rpn_model.compile(loss=losses,optimizer=optim, metrics=metrics)
            #loss_weights=loss_weights)

        return rpn_model


    """
    cls true (1, num pos + num neg)

    """
    def rpn_cls_loss(self,cls_true, cls_pred):
        num_pos_labels = tf.reduce_sum(tf.cast(cls_true==1,tf.int32))
        num_neg_labels = tf.reduce_sum(tf.cast(cls_true==-1,tf.int32))

        pos_labels = tf.ones(num_pos_labels)
        neg_labels = tf.zeros(num_neg_labels)

        labels = tf.concat((pos_labels, neg_labels), axis=0)

        pred = tf.concat((cls_pred[cls_true==1], cls_pred[cls_true==-1]),axis=0)

        loss = tf.keras.losses.binary_crossentropy(labels, pred)

        return loss

    """
    reg true (1, num pos, 4)
    reg pred (1, 8352, 44)

    smooth L1 loss
    """

    def rpn_reg_loss(self,reg_true, reg_pred):

        reg_true = tf.reshape(reg_true,(self.batch_size,self.anchor_array_shape[0], self.nb_anchors,4))
        reg_pred = tf.reshape(reg_pred,(self.batch_size,self.anchor_array_shape[0], self.nb_anchors,4))

        labels = tf.reshape(reg_true[reg_true != 0], (-1,4))
        pred = tf.reshape(reg_pred[reg_true!=0],(-1,4))

        #un_normalized_target_anchors = tf.reshape(self.anchor_array[reg_true[0] != 0], (-1,4))

        #normalized_pred = anchors.parametrize_prediction(un_normalized_target_anchors, pred)

        dif = tf.abs(pred - labels)
        a = dif[dif < 1]
        a = tf.reduce_sum(tf.square(a))*0.5

        b = dif[dif >=1]
        b = tf.reduce_sum(b - 0.5)

        loss = (a+b)

        return loss



if __name__=="__main__":


    m = base_model((99,2790,1))
