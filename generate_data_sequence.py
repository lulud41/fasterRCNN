#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import PIL.Image
import glob
import anchors

"""
implem multithread possible, seul probleme est que le idx est incrémenté
de 1, et si une image n'est pas utilisaanchors_arrayble, on chope la suivante mais sans
toucher a idx, donc il est possible de faire 2 batch a la suite sur la meme
image si une negative apparait. (idx =1 , image suivante, idx=2 : encore la meme)

mais pas si grave car on selectionne un sous ensemble au hasard
des index positifs et negatifs.

"""

class Dataset_sequence(tf.keras.utils.Sequence):

    def __init__(self, dataset_type,dataset_size, image_shape, anchor_batch_size,
            ratio_batch,data_path, ground_truth_bbox_path, anchors_array):

        self.dataset_type = dataset_type
        self.dataset_size = dataset_size

        self.image_shape = image_shape

        self.anchor_batch_size = anchor_batch_size
        self.ratio_batch = ratio_batch
        self.num_positives = int(self.anchor_batch_size*ratio_batch)
        self.num_negatives = int(self.anchor_batch_size*(1-ratio_batch))

        self.anchors_array = anchors_array
        self.ground_truth_bbox_array = self.init_ground_truth_bbox(ground_truth_bbox_path)

        self.data_path = data_path
        self.file_names = self.init_files_names()

    def init_files_names(self):

        all_file_names = np.array(glob.glob(self.data_path+"*.pgm"))
        all_file_names = np.sort(all_file_names)

        train_size = int(self.dataset_size*0.6)
        valid_size = int(self.dataset_size*0.2)
        test_size = int(self.dataset_size*0.2)

        if self.dataset_type == "train":
            return all_file_names[:train_size]

        if self.dataset_type == "valid":
            return all_file_names[train_size:valid_size]

        if self.dataset_type == "test":
            return all_file_names[train_size+valid_size:]

    def init_ground_truth_bbox(self,ground_truth_bbox_path):

        train_size = int(self.dataset_size*0.6)
        valid_size = int(self.dataset_size*0.2)
        test_size = int(self.dataset_size*0.2)

        ground_truth_bbox_list = np.genfromtxt(ground_truth_bbox_path,delimiter=",",dtype=np.int32)

        if self.dataset_type == "train":
            return ground_truth_bbox_list[:train_size]

        if self.dataset_type == "valid":
            return ground_truth_bbox_list[train_size:valid_size]

        if self.dataset_type == "test":
            return ground_truth_bbox_list[train_size+valid_size:]

    def __len__(self):
        length = self.ground_truth_bbox_array.shape[0]
        return length

    def load_image(self,index):
        image = PIL.Image.open(self.file_names[index])
        image = image.resize(self.image_shape)
        image_arr = np.array(image, dtype=np.float32)
        image_arr = image_arr[np.newaxis,:,:,np.newaxis]

        return image_arr


    def __getitem__(self, idx):

        correct_image_found = 0
        current_index = idx

        while correct_image_found == False:

            image = self.load_image(current_index)

            positive_anchors_index, negative_anchors_index = anchors.compute_IoU(
                self.anchors_array,self.ground_truth_bbox_array[current_index])

            if positive_anchors_index[0].shape[0] >= int(self.anchor_batch_size*self.ratio_batch):

                positive_anchors_index, negative_anchors_index = anchors.select_anchors(
                            self.num_positives,self.num_negatives,positive_anchors_index,negative_anchors_index)

                cls_labels = anchors.generate_cls_labels(positive_anchors_index, negative_anchors_index, self.anchors_array.shape)
                reg_labels = anchors.generate_reg_labels(self.anchors_array, positive_anchors_index,
                            self.ground_truth_bbox_array[current_index])

                correct_image_found = True

            else:
                current_index = (current_index+1)%self.__len__()

        return (image, [cls_labels,reg_labels])
