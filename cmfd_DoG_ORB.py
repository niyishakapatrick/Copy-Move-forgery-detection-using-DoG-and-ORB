#!/usr/bin/env python

'''Niyishaka Patrick, Chakravarthy Bhagvati.: "Digital Image Forensics for Copy-Move Forgery Detection using DoG and ORB",
International Conference on Computer Vision and Graphics(ICCVG 2018).To be published in Lecture Notes in Computer Science (LNCS), Springer.
'''

from datetime import datetime
from skimage.feature import blob_dog
from math import sqrt
import cv2
import numpy as np
import scipy
from scipy import ndimage
from scipy.spatial import distance
import glob, os

start_time = datetime.now()
# Initiate ORB detector
orb = cv2.ORB(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE,edgeThreshold=10)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

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

def dist_Hm(ll):     
	for i in range(0,len(ll)-1):
		for kp1,des1 in ll[i]:
			for q in range(i+1,len(ll)):
				for kp2,des2 in ll[q]:
					#match = distance.hamming(des1,des2)
					matches = bf.match(des1, des2)
					#if match < 0.6:
					cv2.line(clone1,(int(kp1.pt[0]),int(kp1.pt[1])),(int(kp2.pt[0]),int(kp2.pt[1])),(100,255,0),1)                   

if __name__=='__main__':
	im1 = scipy.misc.imread ('car.jpg')
	sobel_image = sobel_f(im1)
	im2_gray =cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im1_gray =cv2.cvtColor(sobel_image, cv2.COLOR_BGR2GRAY)
	blobs_all = dog_f(im1_gray)
	output = show_f(blobs_all)
	clone1 = im1.copy()
	kp, des = orb.detectAndCompute(im2_gray,None)
 
	ll =[]
	for b0 in range(0,len(output)):
		b0y,b0x,b0r = output[b0]
		cv2.circle(clone1, (int(b0x),int(b0y)), int(b0r), (0, 0, 255), 1)             
		l =[]
		for  k,d in zip(kp,des):
			if (k.pt[0] - b0x)**2 + (k.pt[1] - b0y)**2 < (b0r **2):
				l.append([k,d]) 
				cv2.circle(clone1, (int(k.pt[0]),int(k.pt[1])), 3, (0, 255, 0), 1)
		if l:
			ll.append(l)                    
	if ll:
		dist_Hm(ll)
	#cv2.imshow('image',clone1)
	cv2.imwrite('cmfd_output.png',clone1)    
	end_time = datetime.now()
	print('Duration: {}'.format(end_time - start_time))     
	cv2.waitKey(0)
	cv2.destroyAllWindows()

