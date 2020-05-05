#!/usr/bin/env python
# coding: utf-8

# pas encore tensor : anchors convertis a la fin, car
# ils sont constants!
from tensorflow import convert_to_tensor, get_static_value
import numpy as np
import PIL.Image
import PIL.ImageDraw

def generate(input_image_shape, ratio_x, ratio_y, anchosr_sizes):
    # la shape correspond a la shape de la prediction par le répartition
    # par le rpn. on a (n_x, n_y, k, 4)

    # avec n_x et n_y les deux premieres dim du tenseur de prediction (sortie de
    # fenetre glissante)
    # du rpn ; (n_x, n_y, 4x6) + cls
    anchors = np.zeros(((input_image_shape[0]//ratio_x-2),
            (input_image_shape[1]//ratio_y -2),
            len(anchor_sizes), 4))

    i = 0
    j = 0
    for x in range(ratio_x+ratio_x//2, input_image_shape[0] - 3*ratio_x//2, ratio_x):
        for y in range(ratio_y+ratio_y//2, input_image_shape[1] - 3*ratio_y//2, ratio_y):
            for k in range(0,len(anchor_sizes)):
                anchors[i, j, k, :] = [x, y, anchor_sizes[k][0], anchor_sizes[k][1] ]
            j=j+1
        j=0
        i=i+1

    # retirer les anchors qui sortent de l'image ( mis a 0)
    anchors[ anchors[:,:,:,0] + anchors[:,:,:,2]//2 > input_image_shape[0]] = 0
    anchors[ anchors[:,:,:,0] - anchors[:,:,:,2]//2 <= 0] = 0

    anchors[ anchors[:,:,:,1] + anchors[:,:,:,3]//2 > input_image_shape[1]] = 0
    anchors[ anchors[:,:,:,1] - anchors[:,:,:,3]//2 <= 0] = 0

    anchors = anchors.reshape((anchors.shape[0]*anchors.shape[1],len(anchor_sizes),4))
    np.random.shuffle(anchors) # retire relation spatiale, utile pour la selection par la suite
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
    #image.show()
    return PIL_image

def draw_anchors(image_name,anchor_list, color="black"):
    # anchor_list (shape ( n,4))
    # anchor peut etre 0 donc pas dessiné, ou != 0
    PIL_image = PIL.Image.open(image_name)

    for anchor in anchor_list:
        if anchor.sum() != 0:
            PIL_image = draw_bounding_box(PIL_image,anchor, color)
    PIL_image.show()

def compute_IoU(anchors, ground_truth_bbox,positive_threshold=0.7,negative_threshold=0.3):
    # anchors : liste d'anchor généres (1870,6,4)
    # ground truth box : mask [x,y,h,w]
    y = anchors[:,:,3]*ground_truth_bbox[3] - np.abs(anchors[:,:,1] - ground_truth_bbox[1])
    x = anchors[:,:,2]*ground_truth_bbox[2] - np.abs(anchors[:,:,0] - ground_truth_bbox[0])

    intersection = x*y

    IoU = intersection /
        (anchors[:,:,2]*anchors[:,:,3]+ground_truth_bbox[2]*ground_truth_bbox[3] - intersection)

    #index shape  tuple : ( arr (x), arr(y) )
    positive_anchors_index  = np.where(IoU > positive_threshold )
    negative_anchros_index = np.where(IoU < negative_threshold)

    return positive_anchors_index, negative_anchros_index

def select_anchors(nb_positive, nb_negative, positive_index, negative_index):
    # complexifier cette fonction si besoin d'avoir
    # traitemtn plus complexe : si on a pas toujours de quoi faire des batch
    # de 1:1 entre positifs et negatifs
    positive_index = (positive_index[0][:nb_positive], positive_index[1][:nb_positive])
    negative_index = (negative_index[0][:nb_negative], negative_index[1][:nb_negative])
    return positive_index,negative_index

def parametrize_anchors(anchors,positive_index, ground_truth_bbox):
    parametrized_anchors = np.zeros(anchors.shape)

    positive_anchors = parametrized_anchors[positive_index]

    positive_anchors[:,0] = (ground_truth_bbox[0] - positive_anchors[:,0])/positive_anchors[3]
    positive_anchors[:,1] = (ground_truth_bbox[1] - positive_anchors[:,1])/positive_anchors[2]
    positive_anchors[:,2] = np.log(ground_truth_bbox[2]/positive_anchors[2])
    positive_anchors[:,3] = np.log(ground_truth_bbox[3]/positive_anchors[3])

    parametrized_anchors[positive_index] = positive_index

    #reshape en (1870, 6*4) pour avoir meme format que prediction
    parametrized_anchors = parametrized_anchors.reshape(( parametrized_anchors.shape[0],
        parametrized_anchors.shape[1]*parametrize_anchors[2]))

    return parametrized_anchors

def generate_cls_labels(positive_index, negative_index, anchors_shape):
    # anchors_shape : (1870,6,4)
    # labels( nb_of_window posit°, nb_of types of anchor)
    # -1 negatif  ; 1 positifs ; 0 rien du tout
    # cls_labels(1870,6)  : avec 0,1,-1

    cls_labels = np.zeros((anchors_shape[0],anchors_shape[1]))
    cls_labels[positive_index] = 1
    cls_labels[negative_index] = -1

    return cls_labels

def debug_draw_unparametrized_anchors(anchor,positive_index, negative_index, ground_truth_bbox, image_name):
    positive_anchors = anchor[positive_index]
    negative_anchors = anchor[negative_index]

    draw_anchors(image_name, positive_anchors, "white")
    draw_anchors(image_name, negative_anchors, "black")
    draw_bounding_box(PIL.Image.open(image_name),ground_truth_bbox, color="black")



if __name__ == "__main__":

    input_image_shape = (99,2790,1)

    ratio_x = 4
    ratio_y = 32

    # size : (h, w)
    anchor_sizes = (
        (10,10),
        (30,30),
        (50,50),
        (8,16),
        (20,40),
        (34,70))

    a = generate(input_image_shape, ratio_x, ratio_y, anchor_sizes)
    draw_anchors("photo.pgm",a)
