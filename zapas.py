# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 17:36:44 2019

@author: uba_p
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 20:59:11 2019

@author: uba_p
"""

#importación de librerias 
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import glob
from PIL import Image

X = []
Y = []
for filename in glob.glob('VANS/*.jpg'): #assuming gif
    im=Image.open(filename) #abrimos cada imagen del directorio
    im= im.resize((100, 100)).convert('L') #resize 100*100
    im= np.array(im) #metemos la imagen reescalada en un array de numpy
    X.append(im) #añadimos el array de numpy al array creado al principio 
    Y.append(0) #añadimos la etiqueta 0 al array Y creado al principio

for filename in glob.glob('NIKE/*.jpg'): #assuming gif
    im=Image.open(filename) #abrimos cada imagen del directorio
    im= im.resize((100, 100)).convert('L') #resize 200*200
    im= np.array(im) #metemos la imagen reescalada en un array de numpy
    X.append(im) #añadimos el array de numpy al array creado al principio 
    Y.append(1) #añadimos la etiqueta 1 al array Y creado al principio
    
for filename in glob.glob('BLANCA/*.jpg'): #assuming gif
    im=Image.open(filename) #abrimos cada imagen del directorio
    im= im.resize((100, 100)).convert('L') #resize 200*200
    im= np.array(im) #metemos la imagen reescalada en un array de numpy
    X.append(im) #añadimos el array de numpy al array creado al principio 
    Y.append(2) #añadimos la etiqueta 1 al array Y creado al principio
        
Y = np.array(Y) #convierto de lista a numpy
X = np.array(X, dtype=np.uint8) #convierto de lista a numpy


X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2) #asignamos 0.2 al test y 0.8 al train
print('Training data shape : ', X_train.shape, y_train.shape) #print de la forma de la parte del train
print('Testing data shape : ', X_test.shape, y_test.shape) #print de la forma de la parte del test

print(X_test) #print de X_test


# Antes de realizar el entrenamiento, preparar los datos transformando las imágenes 
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')

X_train/=255
X_test/=255

# Crear la arquitectura de la red neuronal convolucional (CNN)
# Preparar también las etiquetas en categorías:
from keras.utils import to_categorical
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


from keras import models
from keras import layers
network = models.Sequential()


network = Sequential()
#Capa de entrada
network.add(Conv2D(32, kernel_size = (2, 2), activation='relu', input_shape=(100, 100, 1)))

network.add(MaxPooling2D(pool_size=(2,2)))
network.add(BatchNormalization())

network.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(BatchNormalization())

network.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(BatchNormalization())

network.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(BatchNormalization())

network.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
network.add(MaxPooling2D(pool_size=(2,2)))
network.add(BatchNormalization())
network.add(Dropout(0.2))

network.add(Flatten())
network.add(Dense(128, activation='relu'))
#network.add(Dropout(0.3))

#Capa de salida
network.add(Dense(3, activation = 'softmax'))


#Resumen de la arquitectura
network.summary()
# Definir la función de pérdida, el optimizador y las métricas para monitorizar el entrenamiento y la prueba de validación
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

# Realizar el entrenamiento. Guardar el resultado en una variable denominada ‘historia’
history = network.fit(X_train, y_train_cat, epochs=5, batch_size=50, validation_data=(X_test, y_test_cat))

test_loss, test_acc = network.evaluate(X_test, y_test_cat)
print('test_acc:', test_acc)

# Visualizar las métricas
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Precición del modelo')
plt.ylabel('Precision')
plt.xlabel('epoch')
plt.legend(['Entrenamiento', 'Test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Pérdida del modelo')
plt.ylabel('Pérdida')
plt.xlabel('epoch')
plt.legend(['Entrenamiento', 'Test'], loc='upper right')

plt.tight_layout()
plt.show()

# Guardar el modelo en formato JSON

from keras.models import model_from_json
model_json = network.to_json()
with open("network.json", "w") as json_file:
    json_file.write(model_json)

# Guardar los pesos (weights) a formato HDF5
network.save_weights("network_weights.h5")
print("Guardado el modelo a disco")


# Leer JSON y crear el modelo
json_file = open("network.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Cargar los pesos (weights) en un nuevo modelo
loaded_model.load_weights("network_weights.h5")
print("Modelo cargado desde el disco")

# Predecir sobre el conjunto de test
predicted_classes = loaded_model.predict_classes(X_test)

# Comprobar que predicciones son correctas y cuales no
indices_correctos = np.nonzero(predicted_classes == y_test)[0]
indices_incorrectos = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(indices_correctos)," clasificados correctamente")
print(len(indices_incorrectos)," clasificados incorrectamente")

# Adaptar el tamaño de la figura para visualizar 18 subplots
plt.rcParams['figure.figsize'] = (7,14)

figure_evaluation = plt.figure()

# Visualizar 6 predicciones correctas
for i, correct in enumerate(indices_correctos[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(X_test[correct].reshape(100,100), cmap='gray', interpolation='none')
    plt.title(
      "Pred: {}, Original: {}".format(predicted_classes[correct],
                                        y_test[correct]))
    plt.xticks([])
    plt.yticks([])

# Visualizar 6 predicciones incorrectas
for i, incorrect in enumerate(indices_incorrectos[:6]):
    plt.subplot(6,3,i+10)
    plt.imshow(X_test[incorrect].reshape(100,100), cmap='gray', interpolation='none')
    plt.title(
      "Pred: {}, Original: {}".format(predicted_classes[incorrect], 
                                       y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])

figure_evaluation