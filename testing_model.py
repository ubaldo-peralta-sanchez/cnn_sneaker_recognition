# -*- coding: utf-8 -*-
"""
@author: uba_p
"""

# importación de librerias
import numpy as np
import glob
from PIL import Image
from keras.models import model_from_json

# Leer JSON y crear el modelo
json_file = open("network.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Cargar los pesos (weights) en un nuevo modelo
loaded_model.load_weights("network_weights.h5")
print("Modelo cargado desde el disco")
X = []

for filename in glob.glob('EVAL/*.jpg'):  # assuming gif
    im = Image.open(filename)  # abrimos cada imagen del directorio
    im = im.resize((100, 100)).convert('L')  # resize 100*100
    im = np.array(im)  # metemos la imagen reescalada en un array de numpy
    X.append(im)  # añadimos el array de numpy al array creado al principio

X_test = np.array(X, dtype=np.uint8)  # convierto de lista a numpy
# Antes de realizar el entrenamiento,
# preparar los datos transformando las imágenes
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],
                        X_test.shape[2], 1).astype('float32')
X_test /= 255
# Predecir sobre el conjunto de test
predicted_classes = loaded_model.predict_classes(X_test)

# plt.imshow(X_test[1].reshape(100,100), cmap='gray', interpolation='none')

for i in predicted_classes:
    if i == 0:
        print("VANS")
    elif i == 1:
        print("NIKE")
    elif i == 2:
        print("BLANCA")
