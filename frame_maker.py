# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 18:58:46 2019

@author: uba_p
"""

import cv2

vidcap = cv2.VideoCapture('videos/c video.mp4')
success, image = vidcap.read()
count = 1
while success:
    cv2.imwrite("IPHONE/frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
