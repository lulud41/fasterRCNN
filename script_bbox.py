#!/usr/bin/env python
# coding: utf-8

import os

path_mask = "/home/lucien/Documents/ST09/stage_utt/implem/check_img_mask/mask/"

def generate_bbox_from_mask():

	with open("bbox.csv","a") as csv_file:
		for file in sorted(os.listdir(path_mask)):
			print(file)
			image = np.array(PIL.Image.open(path_mask+file))
			indexes = np.where(image == 255)

			top_left_corner = [ indexes[0][0], indexes[1][0] ]

			bottom_right_corner = [ indexes[0][-1], indexes[1][-1] ]
			# horizontald length
			bbox_w  = bottom_right_corner[1] - top_left_corner[1] +1
			# vertical length
			bbox_h = bottom_right_corner[0] - top_left_corner[0] +1
			# center x coordinate
			bbox_x = top_left_corner[0] + bbox_h//2
			# center y coordinate
			bbox_y = top_left_corner[1] + bbox_w//2

			csv_file.write(str(bbox_x)+","+str(bbox_y)+","+
				str(bbox_h)+","+str(bbox_w)+"\n")

		
