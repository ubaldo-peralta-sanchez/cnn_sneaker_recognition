# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:52:01 2019

@author: uba_p
"""

import cv2
import requests
import numpy as np

url = "http://192.168.0.162:8080/shot.jpg"


img_resp = requests.get(url)
img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
img = cv2.imdecode(img_arr, -1)
cv2.imshow("Android cam", img)

