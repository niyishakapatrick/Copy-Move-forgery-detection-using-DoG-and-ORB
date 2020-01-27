#!/usr/bin/env python


from datetime import datetime
from skimage.feature import blob_dog,match_descriptors
from math import sqrt
import cv2
import numpy as np
import scipy
from scipy import ndimage
from scipy.spatial import distance
import glob, os
import math



# Initiate orb detector
orb = cv2.ORB_create(1000)
# create BFMatcher
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

def sobel_f(im1):
	image =im1.astype (int)
	# derivatives
	dx=ndimage.sobel(image, 1)
	dy=ndimage.sobel(image, 0)
	mag=np.hypot(dx, dy)
	# normalization
	mag*= 255.0 / np.max(mag)
	sobel_im1 = np.uint8(mag)
	return sobel_im1


def dog_f(im1_gray):
	blobs_dog = blob_dog(im1_gray, max_sigma=40, threshold=.1)
	blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
	return blobs_dog


def show_f(blobs_all):
	blob_area =[]
	blobs_list = [blobs_all]
	for blobs in blobs_list:
		for blob in blobs:
			y, x, r = blob
			area = [y,x,r]           
			if 2*r > 1:
				#print area
				blob_area.append(area)              
	return blob_area

if __name__=='__main__':
	i = 0
	images = [image for image in sorted(glob.glob('*.jpg'))]
	for im in images:
		print(im)
		start_time = datetime.now()
		im1 = cv2.imread (im)
		sobel_image = sobel_f(im1)
		sobel_gray =cv2.cvtColor(sobel_image, cv2.COLOR_BGR2GRAY)
		im2_gray =cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
		blobs_all = dog_f(sobel_gray)
		output = show_f(blobs_all)
		clone1 = im1.copy()
		key,des = orb.detectAndCompute(im2_gray, None)
		#print('keypoints :',len(key),'...',len(des))
		src = np.array([]).reshape(-1,1,2)
		dst = np.array([]).reshape(-1,1,2)
		geom = 0
 
		ll =[]
		for b0 in range(0,len(output)):
			b0y,b0x,b0r = output[b0]
			cv2.circle(clone1, (int(b0x),int(b0y)), int(b0r), (0, 0, 250), 1)             
			l =[]
			kp_1 =[]
			ds_1 =[]
			l3 =[]
			index= 0
			for  k,d in zip(key,des):
				if (k.pt[0] - b0x)**2 + (k.pt[1] - b0y)**2  <= (b0r **2):
					l.append(index)
					#print('l :',len(l))
					kp_1.append(k)
					ds_1.append(d)
				index+=1
			if l:
				kp_2= np.delete(key,l,axis=0)
				ds_2 = np.delete(des,l,axis=0)
				#print('k :',len(kp),'...',len(ds))
				#nn_matches = bf.match(np.array(ds_1),ds_2)
				nn_matches = matcher.knnMatch(np.array(ds_1), ds_2, 2)
				#print(nn_matches)
				good = []
				#matched1 = []
				#matched2 = []
				nn_match_ratio = 0.6 # Nearest neighbor matching ratio
				for m, n in nn_matches:
					#print(m)
					#Use 2-nn matches and ratio criterion to find correct keypoint matches
					#If the closest match distance is significantly lower than the second closest one, then the match is correct (match is not ambiguous).
					if m.distance < nn_match_ratio * n.distance:
						#print(x1,y1,x2,y2)
						good.append(m)


				MIN_MATCH_COUNT = 3
				if len(good) > MIN_MATCH_COUNT:
					src_pts = np.float32([kp_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
					dst_pts = np.float32([kp_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
					#src = np.concatenate((src,src_pts))
					#dst = np.concatenate((dst,dst_pts))
					src = np.array(src_pts).ravel()
					dst = np.array(dst_pts).ravel()
					ps =np.array(src).reshape((-1,2))
					pd =np.array(dst).reshape((-1,2))
					for k1,k2 in zip(ps,pd):
						cv2.circle(clone1, (int(k1[0]),int(k1[1])),4,(0,0,255),-1)
						cv2.circle(clone1, (int(k2[0]),int(k2[1])),4,(0,255,255),-1)
						cv2.line(clone1,(int(k1[0]),int(k1[1])),(int(k2[0]),int(k2[1])),(0,255,0),2)           
		#cv2.imshow('image',clone1)
		cv2.imwrite('detectionz-results__'+str(i)+'.png',clone1)    
		end_time = datetime.now()
		print('Duration: {}'.format(end_time - start_time))
		i += 1
	cv2.waitKey(0)
	cv2.destroyAllWindows()




