#!/usr/bin/env python
# coding: utf-8

# pas encore tensor : anchors convertis a la fin, car
# ils sont constants!

#from tensorflow import convert_to_tensor, get_static_value

import numpy as np
import PIL.Image
import PIL.ImageDraw

epsilon = 1e-8

def generate(input_image_shape, ratio_x, ratio_y, anchor_sizes, clean=True):
    # la shape correspond a la shape de la prediction par le répartition
    # par le rpn. on a (n_x, n_y, k, 4)

    # avec n_x et n_y les deux premieres dim du tenseur de prediction (sortie de
    # fenetre glissante)
    # du rpn ; (n_x, n_y, 4xk) + cls
    anchors = np.zeros(((input_image_shape[0]//ratio_x),
            (input_image_shape[1]//ratio_y),
            len(anchor_sizes), 4))

    i = 0
    j = 0

    for x in range(ratio_x//2, input_image_shape[0] - 3*ratio_x//2, ratio_x):
        for y in range(ratio_y//2, input_image_shape[1] - 3*ratio_y//2, ratio_y):
            for k in range(0,len(anchor_sizes)):
                anchors[i, j, k, :] = [x, y, anchor_sizes[k][0], anchor_sizes[k][1] ]
            j=j+1
        j=0
        i=i+1

    if clean == True:
        # retirer les anchors qui sortent de l'image ( mis a 0)
        anchors[ np.where(anchors[:,:,:,0] + anchors[:,:,:,2]//2 > input_image_shape[0]) ] = np.nan
        anchors[ np.where(anchors[:,:,:,0] - anchors[:,:,:,2]//2 <= 0)] = np.nan

        anchors[ (anchors[:,:,:,1] + anchors[:,:,:,3]//2) > input_image_shape[1]] = np.nan
        anchors[ (anchors[:,:,:,1] - anchors[:,:,:,3]//2) <= 0] = np.nan

    anchors = anchors.reshape((anchors.shape[0]*anchors.shape[1],len(anchor_sizes),4))

    return anchors

def draw_bounding_box(PIL_image, bbox, color="black"):
	# PIL : x horizontal ; y vertival
	# rect : upper left corner , w, h

	# bbox : x vertical ; y horizontal
	#     x_center , y center, h , w
    x = bbox[1] - bbox[3]//2
    y = bbox[0] - bbox[2]//2
    w = x + bbox[3]
    h = y + bbox[2]

    rect = PIL.ImageDraw.Draw(PIL_image)
    rect.rectangle([x,y,w,h],outline=color,width=1)
    return PIL_image

def draw_anchors(image_name,anchor_list, image_shape, color="black"):
    # anchor_list (shape ( n,4))
    # anchor peut etre 0 donc pas dessiné, ou != 0
    PIL_image = PIL.Image.open(image_name)
    PIL_image = PIL_image.resize(image_shape)

    for anchor in anchor_list:
        PIL_image = draw_bounding_box(PIL_image,anchor, color)
    PIL_image.show()

def x1x2y1y2_to_xywh_array(bbox):
    # format bbox  = (m, n , 4)
    # avec bbox[0,0,:] = [n,x_min, x_max, y_min, y_max]
    x = bbox[...,0] + (bbox[...,1] - bbox[...,0]) // 2
    y = bbox[...,2] + (bbox[...,3] - bbox[...,2]) // 2
    h = bbox[...,1] - bbox[...,0]
    w = bbox[...,3] - bbox[...,2]

    return np.stack((x, y, h, w),axis=-1)

def xyhw_to_x1x2y1y2_array(bbox):
    # format bbox  = (m, n , 4)
    # bbox = [x, y, h, w]
    x_min = bbox[...,0] - bbox[...,2] // 2
    y_min = bbox[...,1] - bbox[...,3] // 2
    x_max = bbox[...,0] + bbox[...,2] // 2
    y_max = bbox[...,1] + bbox[...,3] // 2

    return np.stack((x_min, x_max, y_min, y_max),axis=-1)

"""
# anchors : liste d'anchor généres (784,9,4)
# ground truth box : mask [x,y,h,w]
# anchors sous forme [x,y,h,w]
"""
def compute_IoU(anchors, ground_truth_bbox_list, positive_threshold=0.5, negative_threshold=0.2):

    # calulue l'intersection entre tous les indexes negatifs suivant chaque label
    # donc pas d'intersection avec les pos_indexes
    def add_neg_indexes(negative_indexes, new_neg_indexes):

        if len(negative_indexes) == 0:
            return new_neg_indexes

        set_neg = set([ tuple(i) for i in negative_indexes])
        set_new = set([ tuple(i) for i in new_neg_indexes])

        set_neg = set_neg.intersection(set_new)

        return np.array( [i for i in set_neg ])


    anchors = xyhw_to_x1x2y1y2_array(anchors)

    # liste de tuples d'array : liste de np where
    positive_anchors_index_list = []
    negative_anchors_index = np.empty((0,2), dtype = np.int32)


    for i in range(0,len(ground_truth_bbox_list)):

        ground_truth_bbox = np.array(ground_truth_bbox_list[i])
        ground_truth_bbox = xyhw_to_x1x2y1y2_array(ground_truth_bbox)

        I_x1 = np.maximum(anchors[:,:,0], ground_truth_bbox[0] )
        I_y1 = np.maximum(anchors[:,:,2], ground_truth_bbox[2] )
        I_x2 = np.minimum(anchors[:,:,1], ground_truth_bbox[1] )
        I_y2 = np.minimum(anchors[:,:,3], ground_truth_bbox[3] )

        intersection = np.maximum((I_x2 - I_x1 + 1),0)*np.maximum((I_y2 - I_y1 +1),0)

        IoU = intersection / ( (ground_truth_bbox[3] - ground_truth_bbox[2]+1)*(ground_truth_bbox[1] - ground_truth_bbox[0]+1) +
            (anchors[:,:,1] - anchors[:,:,0]+1)*(anchors[:,:,3] - anchors[:,:,2]+1) - intersection)

        pos_indexes = np.array(np.where(IoU >= positive_threshold)).T
        positive_anchors_index_list.append(pos_indexes)

        neg_indexes = np.array(np.where( (IoU < negative_threshold) & (IoU != np.nan) )).T
        negative_anchors_index = add_neg_indexes(negative_anchors_index, neg_indexes)

    return positive_anchors_index_list, negative_anchors_index

"""
On donne positive index pour la normalisation des predictions comme pour la
normalisation des anchors positifs

return positves anchors, normalisés : juste les anchors qui servent
de label (num_positives, 4)

"""
def parametrize_anchors(anchors, ground_truth_bbox):

    param_anchors = anchors.copy()

    param_anchors[...,0] = (ground_truth_bbox[0] - anchors[...,0] )/anchors[...,2] + epsilon
    param_anchors[...,1] = (ground_truth_bbox[1] - anchors[...,1] )/anchors[...,3] + epsilon
    param_anchors[...,2] = np.log(ground_truth_bbox[2]/anchors[...,2]) + epsilon
    param_anchors[...,3] = np.log(ground_truth_bbox[3]/anchors[...,3]) + epsilon

    return param_anchors

"""

déparametre la pred : directement avec les anchors correspondants

donc shapes (n,4) pas (8000, k, 4)

"""
def UN_parametrize_predicition(anchors, norm_predictions):

    x = (norm_predictions[...,0] - epsilon) * anchors[...,2] + anchors[...,0]
    y = (norm_predictions[...,1] - epsilon) * anchors[...,3] + anchors[...,1]
    h = np.exp(norm_predictions[...,2] - epsilon )*anchors[...,2]
    w = np.exp(norm_predictions[...,3] - epsilon )*anchors[...,3]

    pred = np.array([x,y,h,w]).T

    return pred


# -------------------   DEBUG   --------------------

def debug_draw_unparametrized_anchors(anchor,positive_index, negative_index,
    ground_truth_bbox, image_name):
    positive_anchors = anchor[positive_index]
    negative_anchors = anchor[negative_index]

    draw_anchors(image_name, positive_anchors, "white")
    draw_anchors(image_name, negative_anchors, "black")
    draw_bounding_box(PIL.Image.open(image_name),ground_truth_bbox, color="black")

def debug_compute_number_positive_anchors(anchor_sizes, anchor_list, ground_truth_bbox_file):

    bbox_list = np.genfromtxt(ground_truth_bbox_file,delimiter=",",dtype=np.int32)

    positives = []
    for bbox in bbox_list:
        positive_anchors_index, negative_anchors_index = compute_IoU(anchor_list,bbox,positive_threshold=0.5)
        positives.append(positive_anchors_index[0].shape[0])

    positives = np.array(positives)

    print("% of positives anchors >= 5, for every truth bbox : "+str(np.sum(positives>=5)/positives.shape))


if __name__ == "__main__":

    INPUT_IMAGE_SHAPE = (224,224,3)

    RATIO_X = 8
    RATIO_Y = 8

    anchor_sizes = ((80,80),
                    (120,120),
                    (200,200),

                    (40,80),
                    (60,120),
                    (100,200),

                    (80,40),
                    (120,60),
                    (200,100))


    anchors_list = generate(INPUT_IMAGE_SHAPE, RATIO_X, RATIO_Y, anchor_sizes)
    #anchors_list = anchors_list[ anchors_list != 0].reshape((-1,9,4))
    ground = [100,90,50,40]


    a = parametrize_anchors(anchors_list, ground)
    b = UN_parametrize_predicition(anchors_list, a)
    #iou = compute_IoU(anchors_list, ground)
