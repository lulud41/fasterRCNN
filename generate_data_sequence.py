#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import PIL.Image
import os
import anchor_tools
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cv2
import itertools

# essayer de baisser le threshold bas, pour des sets plus petits

"""label_name_dict = {
    0:"Background"
    1:"Aeroplanes"
    2:"Bicycles"
    3:"Birds"
    4:"Boats"
    5:"Bottles"
    6:"Buses"
    7:"Cars"
    8:"Cats"
    9:"Chairs"
    10:"Cows"
    11:"Dining tables"
    12:"Dogs"
    13:"Horses"
    14:"Motorbikes"
    15:"People"
    16:"Potted plants"
    17:"Sheep"
    18:"Sofas"
    19:"Trains"
    20:"TV/Monitors"
    }"""

"""anchor_tools.draw_anchors(DATA_PATH+dataset.file_names[i],
    anchors[( chosen_neg_indexes[:,0], chosen_neg_indexes[:,1] ) ],
     dataset.input_image_shape[:2])"""

class DatasetSequence(tf.keras.utils.Sequence):

    def __init__(self, dataset_type,dataset_size, input_image_shape,data_path, annotation_path):

        self.dataset_type = dataset_type
        self.dataset_size = dataset_size

        self.input_image_shape = input_image_shape

        self.data_path = data_path
        self.annotation_path = annotation_path

        self.file_names = self.init_files_names()

    def init_files_names(self):

        all_file_names = np.sort(os.listdir(self.data_path))

        train_size = int(self.dataset_size*0.6)
        valid_size = int(self.dataset_size*0.2)
        test_size = int(self.dataset_size*0.2)

        if self.dataset_type == "train":
            return all_file_names[:train_size]

        if self.dataset_type == "valid":
            return all_file_names[train_size:(train_size+valid_size)]

        if self.dataset_type == "test":
            return all_file_names[train_size+valid_size:] # ok

    def __len__(self):
        length = len(self.file_names)
        return length

    def load_image(self, image_name):  #ok
        image = PIL.Image.open(self.data_path+image_name)
        image = image.resize(self.input_image_shape[0:2])
        image_arr = np.array(image, dtype=np.float32)
        image_arr = image_arr[np.newaxis,:,:]

        return image_arr

    def resize_coordinates(self, bbox, image_size):
        #modif des les bbox en resize
        #format bbox = [x_min, x_max, y_min, y_max]

        bbox[0] = int(bbox[0]*self.input_image_shape[0]/image_size[0])
        bbox[1] = int(bbox[1]*self.input_image_shape[1]/image_size[0])
        bbox[2] = int(bbox[2]*self.input_image_shape[0]/image_size[1])
        bbox[3] = int(bbox[3]*self.input_image_shape[1]/image_size[1])

        return bbox

    def x1x2y1y2_to_xyhw(self, bbox):
        # format bbox  = [x_min, x_max, y_min, y_max]
        x = bbox[0] + (bbox[1] - bbox[0]) // 2
        y = bbox[2] + (bbox[3] - bbox[2]) // 2
        h = bbox[1] - bbox[0]
        w = bbox[3] - bbox[2]

        return [x, y, h, w]

    def xyhw_to_x1x2y1y2(self, bbox):
        # bbox = [x, y, h, w]
        x_min = bbox[0] - bbox[2] // 2
        y_min = bbox[1] - bbox[3] // 2
        x_max = bbox[0] + bbox[2] // 2
        y_max = bbox[1] + bbox[3] // 2

        return [x_min, x_max, y_min, y_max]

    def draw_bbox(self, image, bbox_list):
        # liste de bbox au format [[x,y,h,w]]

        for i in range(0,len(bbox_list)):
            bbox = self.xywh_to_x1x2y1y2(bbox_list[i])
            # open cv : axe vertical : y, axe horiz : x
            try:
                image = cv2.rectangle(image, (bbox[2], bbox[0]), (bbox[3], bbox[1]), (255,0,0), 2)
            except :
                print("Erreur : draw bbox, peut etre a cause d'image preprocess ")

        plt.imshow(np.uint8(image))
        plt.show()

    def load_labels(self, image_name):

        label_name_list = []
        bbox_list = []
        bbox = []

        annotation_file_name = image_name[:-3]+"xml"
        xml_file = ET.parse(self.annotation_path+annotation_file_name)
        object_list =  xml_file.findall("object")

        image_size = (int(xml_file.find("size").find("height").text),
            int(xml_file.find("size").find("width").text))

        for object in object_list:

            label_name_list.append(object.find("name").text)

            bndbox = object.find("bndbox")
            #x_min : axe vertical, ymin : axe veritcal
            x_min = int(bndbox.find("ymin").text)
            y_min = int(bndbox.find("xmin").text)
            x_max = int(bndbox.find("ymax").text)
            y_max = int(bndbox.find("xmax").text)

            bbox = np.array([x_min, x_max, y_min, y_max])

            bbox = self.resize_coordinates(bbox, image_size)
            bbox = self.x1x2y1y2_to_xyhw(bbox)

            bbox_list.append(bbox)

        return [label_name_list, bbox_list]

    def __getitem__(self, idx):
        image_name = self.file_names[idx]
        label_name_list, bbox_list = self.load_labels(image_name)
        image = self.load_image(image_name)

        return (image, [label_name_list, bbox_list])

class FasterRCNN_Dataset_Sequence(DatasetSequence):
    def __init__(self, dataset_type,dataset_size, input_image_shape,data_path, annotation_path, batch_size,
        anchor_array):
        super().__init__(dataset_type,dataset_size, input_image_shape,data_path, annotation_path)
        self.batch_size = batch_size
        self.anchor_size = anchor_array.shape
        self.anchor_array = anchor_array

    def load_image(self, image_name):
        image = PIL.Image.open(self.data_path+image_name)
        image = image.resize(self.input_image_shape[0:2])
        image_arr = np.array(image, dtype=np.float32)
        image_arr = image_arr[np.newaxis,:,:]

        #preprocess std/mean pour resnet
        image_arr = tf.keras.applications.resnet50.preprocess_input(image_arr)

        return image_arr

    def select_anchors(self, positive_anchors_index_list, negative_anchors_index):

        # true si dedans sinon false
        def row_in_array(array, row):
            return np.any(np.all( array == row, axis=1))

        # retire relation spatiale entre les indices
        np.random.shuffle(negative_anchors_index)
        chosen_neg_indexes = negative_anchors_index[:self.batch_size//2]

        chosen_pos_indexes = np.empty((0,2), dtype = np.int32)
        chosen_pos_name_index = np.empty((0), dtype = np.int32)

        # donne tuple de la premiere ligne de tous, avec None sinon
        for x in itertools.zip_longest(*positive_anchors_index_list):
            for index, item in enumerate(x):
                if item is not None:
                    if not row_in_array(chosen_pos_indexes, item):
                        chosen_pos_indexes = np.concatenate((chosen_pos_indexes, [item]))
                        chosen_pos_name_index = np.concatenate((chosen_pos_name_index, [index]))

                        if len(chosen_pos_name_index) == self.batch_size//2:
                            return chosen_pos_indexes, chosen_pos_name_index, chosen_neg_indexes

        # si manque pos index ou pos index vides
        missing_indexes_number = self.batch_size - len(chosen_pos_indexes)
        chosen_neg_indexes = negative_anchors_index[:missing_indexes_number]

        return chosen_pos_indexes, chosen_pos_name_index, chosen_neg_indexes

    def generate_rpn_cls_labels(self, positive_index, negative_index):
        cls_labels = np.zeros(self.anchor_size[0:2])
        cls_labels[(positive_index[:,0], positive_index[:,1])] = 1
        cls_labels[(negative_index[:,0], negative_index[:,1])] = -1

        return np.expand_dims(cls_labels,axis=0)

    def generate_rpn_reg_labels(self, positive_index, label_list, positive_index_name):

        # chosen_pos_indexes : index dans la liste de label list
        # iterer et parametrer avec la truth correspondante
        reg_labels = np.zeros(self.anchor_size, dtype=np.float32)

        for idx, bbox_index in enumerate(positive_index):

            ground_truth_bbox = label_list[1][positive_index_name[idx]]
            anchor = self.anchor_array[(bbox_index[0], bbox_index[1])]
            anchor = anchor[np.newaxis, :]

            norm_bbox = anchor_tools.parametrize_anchors( anchor, ground_truth_bbox)

            reg_labels[(bbox_index[0], bbox_index[1])] = norm_bbox

        reg_labels = reg_labels.reshape((reg_labels.shape[0], reg_labels.shape[1]*4))

        return np.expand_dims(reg_labels, axis=0)

    # a definir bien pour avoir dictionnaire : reg et cls label_list
    # pour le rpn et detector
    def __getitem__(self, idx):
        image_name = self.file_names[idx]
        image = self.load_image(image_name)

        label_list = self.load_labels(image_name)

        positive_anchors_index_list, negative_anchors_index = anchor_tools.compute_IoU(self.anchor_array, label_list[1])

        chosen_pos_indexes, chosen_pos_name_index, chosen_neg_indexes = self.select_anchors(
            positive_anchors_index_list, negative_anchors_index)

        """anchor_tools.draw_anchors(DATA_PATH+self.file_names[i],
            anchors[( chosen_pos_indexes[:,0], chosen_pos_indexes[:,1] ) ],
             self.input_image_shape[:2])"""

        cls = self.generate_rpn_cls_labels(chosen_pos_indexes, chosen_neg_indexes)

        reg = self.generate_rpn_reg_labels(chosen_pos_indexes, label_list, chosen_pos_name_index)



        # juste test, car label : cls et reg
        return (image, {"rpn_cls":cls, "rpn_reg":reg} )

if __name__ == "__main__":

    ANNOTATION_PATH = "/home/lucien/Documents/VOCdevkit/VOC2012/Annotations/"
    DATA_PATH = "/home/lucien/Documents/VOCdevkit/VOC2012/JPEGImages/"

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

    dataset = FasterRCNN_Dataset_Sequence("train",100, INPUT_IMAGE_SHAPE ,DATA_PATH, ANNOTATION_PATH, batch_size=16,
        anchor_array = anchors)

    for i in range(0,10):

        x , y = dataset.__getitem__(i)
        input()
