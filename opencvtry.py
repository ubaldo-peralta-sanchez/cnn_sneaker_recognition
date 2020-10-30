# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 13:03:48 2019

@author: uba_p
"""


#importación de librerias 
import numpy as np
import requests
import glob
from PIL import Image
import cv2


cap = cv2.VideoCapture(0)
X=[]

from keras.models import model_from_json
json_file = open("network.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Cargar los pesos (weights) en un nuevo modelo
loaded_model.load_weights("network_weights.h5")
print("Modelo cargado desde el disco")
contador = 0

while(True):
    # Capture frame-by-frame
    url = "http://192.168.0.162:8080/shot.jpg"

    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    #    ret, frame = cap.read()
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imwrite("save_frames/frame%d.jpg" % contador, frame) 
    
    
    for filename in glob.glob('save_frames/frame'+ str(contador) +'.jpg'): #assuming gif
        im=Image.open(filename) #abrimos cada imagen del directorio
        im= im.resize((100, 100)).convert('L') #resize 100*100
        im= np.array(im) #metemos la imagen reescalada en un array de numpy
        X.append(im) #añadimos el array de numpy al array creado al principio 
        
    X_test = np.array(X, dtype=np.uint8) #convierto de lista a numpy
    # Antes de realizar el entrenamiento, preparar los datos transformando las imágenes 
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')
    X_test/=255
    # Predecir sobre el conjunto de test
    predicted_classes = loaded_model.predict_classes(X_test)
    
    if predicted_classes[contador] ==0:
        print("VANS")
    elif predicted_classes[contador] ==1:
        print("NIKE")
    elif predicted_classes[contador] ==2:
        print("BLANCA")
    contador=contador + 1
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()