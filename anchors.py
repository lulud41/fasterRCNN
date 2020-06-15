#!/usr/bin/env python
# coding: utf-8

# pas encore tensor : anchors convertis a la fin, car
# ils sont constants!

#from tensorflow import convert_to_tensor, get_static_value

import numpy as np
import PIL.Image
import PIL.ImageDraw
import tensorflow as tf

epsilon = 1e-8

def generate(input_image_shape, ratio_x, ratio_y, anchor_sizes):
    # la shape correspond a la shape de la prediction par le répartition
    # par le rpn. on a (n_x, n_y, k, 4)

    # avec n_x et n_y les deux premieres dim du tenseur de prediction (sortie de
    # fenetre glissante)
    # du rpn ; (n_x, n_y, 4x6) + cls
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

    # retirer les anchors qui sortent de l'image ( mis a 0)
    anchors[ np.where(anchors[:,:,:,0] + anchors[:,:,:,2]//2 > input_image_shape[0]) ] = 0
    anchors[ np.where(anchors[:,:,:,0] - anchors[:,:,:,2]//2 <= 0)] = 0

    anchors[ (anchors[:,:,:,1] + anchors[:,:,:,3]//2) > input_image_shape[1]] = 0
    anchors[ (anchors[:,:,:,1] - anchors[:,:,:,3]//2) <= 0] = 0

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
    return PIL_image

def draw_anchors(image_name,anchor_list, image_shape, color="black"):
    # anchor_list (shape ( n,4))
    # anchor peut etre 0 donc pas dessiné, ou != 0
    PIL_image = PIL.Image.open(image_name)
    PIL_image = PIL_image.resize(image_shape)

    for anchor in anchor_list:
        PIL_image = draw_bounding_box(PIL_image,anchor, color)
    PIL_image.show()

def get_bbox_from_mask(img_name):
    image = np.array(PIL.Image.open(img_name))
    indexes = np.where(image == 255)

    top_left_corner = [ indexes[0][0], indexes[1][0] ]

    bottom_right_corner = [ indexes[0][-1], indexes[1][-1] ]
    # horizontal
    bbox_w  = bottom_right_corner[1] - top_left_corner[1] +1
    # vertical
    bbox_h = bottom_right_corner[0] - top_left_corner[0] +1
    # center x coordinate
    bbox_x = top_left_corner[0] + bbox_h//2
    # center y coordinate
    bbox_y = top_left_corner[1] + bbox_w//2

    return [bbox_x, bbox_y, bbox_h, bbox_w]

def compute_IoU(anchors, ground_truth_bbox,positive_threshold=0.6, negative_threshold=0.3):
    # anchors : liste d'anchor généres (7562,11,4)
    # ground truth box : mask [x,y,h,w]
    y = (anchors[:,:,3]+ground_truth_bbox[3])/2 - np.abs(anchors[:,:,1] - ground_truth_bbox[1])
    x = (anchors[:,:,2]+ground_truth_bbox[2])/2 - np.abs(anchors[:,:,0] - ground_truth_bbox[0])

    y[np.where(y <= 0)] = 0
    x[np.where(x <= 0)] = 0

    intersection = x*y

    IoU = intersection /(anchors[:,:,2]*anchors[:,:,3]+ground_truth_bbox[2]*ground_truth_bbox[3] - intersection)
    #index shape  tuple : ( arr (x), arr(y) )
    positive_anchors_index  = np.where(IoU >= positive_threshold )

    ok_anchors = (anchors != 0)[:,:,0] #anchors qui ne sortent pas de l'image
       # un anchor negatif ne doit pas etre un qui sort de l'image
    negative_anchors_index = np.where((IoU <= negative_threshold)*ok_anchors)


    return positive_anchors_index, negative_anchors_index

def select_anchors(num_positives,num_negatives, positive_index, negative_index):

    positives_choosed = np.random.permutation(positive_index[0].shape[0])[:num_positives]
    negative_choosed = np.random.permutation(negative_index[0].shape[0])[:num_negatives]

    positive_index = (positive_index[0][positives_choosed], positive_index[1][positives_choosed])
    negative_index = (negative_index[0][negative_choosed], negative_index[1][negative_choosed])

    return positive_index,negative_index

"""
On donne positive index pour la normalisation des predictions comme pour la
normalisation des anchors positifs

return positves anchors, normalisés : juste les anchors qui servent
de label (num_positives, 4)

"""
def parametrize_anchors(anchors, ground_truth_bbox, means, std):

    #parametrized_anchors = np.zeros(anchors.shape)

    anchors[:,0] = (ground_truth_bbox[0] - anchors[:,0] + epsilon)/anchors[:,3]
    anchors[:,1] = (ground_truth_bbox[1] - anchors[:,1] + epsilon)/anchors[:,2]
    anchors[:,2] = np.log(ground_truth_bbox[2]/anchors[:,2] + epsilon)
    anchors[:,3] = np.log(ground_truth_bbox[3]/anchors[:,3] + epsilon)

    #parametrized_anchors[positive_index] = positive_anchors

    #reshape en (7562, 11*4) pour avoir meme format que prediction
    #parametrized_anchors = parametrized_anchors.reshape(( parametrized_anchors.shape[0],
    #    parametrized_anchors.shape[1]*parametrize_anchors.shape[2]))

    #normalisation
    anchors = (anchors - means)  / std


    return anchors

"""

déparametre la pred : directement avec les anchors correspondants

donc shapes (n,4) pas (8000, k, 4)

"""
def UN_parametrize_predicition(anchors, norm_predictions, means, std):

    norm_predictionsnorm_predictions = norm_predictions*std + means

    x = norm_predictions[:,0] * anchors[:,3] + anchors[:,0]
    y = norm_predictions[:,1] * anchors[:,2] + anchors[:,1]
    h = np.exp(norm_predictions[:,2])*anchors[:,2]
    w = np.exp(norm_predictions[:,3])*anchors[:,3]

    pred = np.array([x,y,h,w]).T



    return pred



"""
recupere la ground_truth_bbox a partir des anchors normalisés

osef, juste debug
"""
def UN_parametrize_anchors_bbox(anchors, param_anchors):

    x = param_anchors[:,0] * anchors[:,3] + anchors[:,0]
    y = param_anchors[:,1] * anchors[:,2] + anchors[:,1]
    h = np.exp(param_anchors[:,2])*anchors[:,2]
    w = np.exp(param_anchors[:,3])*anchors[:,3]

    return np.array([x,y,h,w]).T


"""
parametrise la prediction, avec anchors de base
( on normalise que les pred selectionnees avec positive_index)
et on return que ceux qui vont servir au loss

predictions (n,4)
pred pareil

: on fait pas ça

"""
def parametrize_prediction(target_anchors, predictions):

    """    predictions = predictions.reshape((predictions.shape[0], -1, 4))
    #shape comme anchors  : (7562,11,4)

    selected_predictions = predictions[positive_index]  #shape (n pos, 4)
    selected_anchors = anchors[positive_index]          #shape (n pos, 4)
    """

    x = (predictions[:,0] - target_anchors[:,0])/target_anchors[:,3]
    y = (predictions[:,1] - target_anchors[:,1])/target_anchors[:,2]
    h = tf.math.log(predictions[:,2]/target_anchors[:,2])
    w = tf.math.log(predictions[:,3]/target_anchors[:,3])

    normalized_predictions = tf.stack([x,y,h,w],axis=1)

    return normalized_predictions


"""
array 1d de labels, uniquement de la taille du batch, avec
positifs sur la premiere partie et neg sur la deuxieme

pos 1
neg 0

return shape (1, n)
(car custom loss input : (batch_size, n, n2 ...))
"""
"""
    > version return que les labels
def generate_cls_labels(positive_index, negative_index):

    num_positives = len(positive_index[0])
    num_negatives = len(negative_index[0])

    cls_labels = np.zeros(( 1, num_negatives+num_positives ))
    cls_labels[0,:num_positives] = 1
    cls_labels[0, num_positives:] = 0

    cls_labels = tf.convert_to_tensor(cls_labels)

    return cls_labels

def generate_reg_labels(anchors, positive_index, ground_truth_bbox):
    reg_labels = parametrize_anchors(anchors,positive_index, ground_truth_bbox)
    reg_labels = reg_labels[np.newaxis, :]

    reg_labels = tf.convert_to_tensor(reg_labels)

    return reg_labels
"""




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

    INPUT_IMAGE_SHAPE = (99,2790,1)

    RATIO_X = 4
    RATIO_Y = 8

    anchor_sizes = ((10,20),
                (20,40),
                (35,70),
                (50,100),
                (10,10),
                (15,15),
                (20,20),
                (25,25),
                (30,30),
                (40,40),
                (50,50))
        # 4,8 et 0.5   : 0.91  >=5
        # 4,8 et 0.6   : 0.86  >=5


    bbox_list = np.genfromtxt("bbox_galbe_2.csv",delimiter=",",dtype=np.int32)


    anchors = generate(INPUT_IMAGE_SHAPE, RATIO_X, RATIO_Y, anchor_sizes)

    #positive_anchors_index, negative_anchors_index = compute_IoU(anchors,bbox_list[0],positive_threshold=0.6)

    #positive_anchors_index, negative_anchors_index = select_anchors(10,10,positive_anchors_index,negative_anchors_index)
    print(anchors.shape)
    #draw_anchors("photo.pgm",np.reshape(a,(-1,4)))
    debug_compute_number_positive_anchors(anchor_sizes,anchors,"bbox_galbe_2.csv")

    #ground_truth_bbox = np.array(bbox_list[0])

    #c = generate_cls_labels(positive_anchors_index, negative_anchors_index)
