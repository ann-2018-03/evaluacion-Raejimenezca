# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:36:54 2019

@author: Rafael_Jz
"""

import pandas as pd
import tensorflow as tf
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LinearRegression, LogisticRegression
#from sklearn.metrics import mean_squared_error
#from sklearn.neighbors import KNeighborsRegressor


## leer de las columnas
colnames = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'y']

## Leer el archivo CSV
dbDiabetes = pd.read_csv('diabetes.csv', names = colnames)

# Pasar columnas que nos interesan ('s6' : glucosa, 'Y' : niveles de azúcar en la sangre) del CSV a lista
# X = dbDiabetes.bmi.tolist()
X = dbDiabetes.s6.tolist()
Y = dbDiabetes.y.tolist()

## Elimina el retorno de carro de los strings
##dbDiabetes = [x.split(',') for x in dbDiabetes]

X = [float(y) for y in X[1:]]
Y = [float(z) for z in Y[1:]]
#print(X)
#print(Y)

X.sort()
Y.sort()

plt.figure(0)
plt.plot(X)

plt.figure(1)
plt.plot(Y)

##
## datos
##
x = tf.constant(X)
d = tf.constant(Y)

## parámetros
w0 = tf.Variable(2.5)
w1 = tf.Variable(2.5)

## Define el modelo
m = tf.add(tf.multiply(x, w1), w0)

## Define la función de error
sse = tf.reduce_sum(tf.square(d - m)) # sum of the squared errors

## Inicializa el optimizador
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.03)

## Minimiza la función de error
opt = optimizer.minimize(sse)

## estima el modelo
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        sess.run(opt)
        if (i % 5 == 0):
            print(sess.run(sse))
    #print('\nValores encontrados\n\n  w0 = {:f}\n  w1 = {:f}'.format(sess.run(w0), sess.run(w1)))

###
### Sumatoria del error cuadrático
###
#def SSE(w0, w1):
#    return (sum( [(v - w0 - w1*u)**2  for u, v in zip(X, Y)] ))
#
###
### Generación de una malla de puntos
### y valor del SSE en cada punto
###
#W0 = np.arange(-5.0, 5.0, 0.05)
#W1 = np.arange(-5.0, 5.0, 0.05)
#W0, W1 = np.meshgrid(W0, W1)
#F = SSE(W0, W1)
#
###
###  Superficie de error
###
#plt.figure(2)
#fig = plt.figure(figsize=(7, 7))
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(W0, W1, F, cmap=cm.coolwarm, linewidth=1, antialiased=False)
#
###
### Contorno
###
#def plot_contour():
#    fig, ax = plt.subplots(figsize=(7, 7))
#    ax.set_aspect('equal', 'box')
#    ax.contour(W0, W1, F, levels=[0, 1, 2, 3, 5, 10, 20, 40, 60, 90])
#    ax.grid()
#
#plot_contour()
#
## Computar gradiente
#def gSSE(w0, w1):
#    ## calcula el vector de errores
#    e = [(v - w0 - w1*u)  for u, v in zip(X, Y)]
#
#    ## gradientes
#    gw0 = -2 * sum(e)
#    gw1 = -2 * sum([q*v for q, v in zip(e,X)])
#
#    return (gw0, gw1)
#
# 
#def mejora(w0, w1, mu):
#    ## computa el gradiente para los parámetros actuales
#    gw0, gw1 = gSSE(w0, w1)
#
#    ## realiza la corrección de los parámetros
#    w0 = w0 - mu * gw0
#    w1 = w1 - mu * gw1
#
#    ## retorna los parámetros corregidos
#    return (w0, w1)
#
#
#### Punto de inicio
#w0 = 0.5
#w1 = 3.0
#
#history_w0 = [w0]
#history_w1 = [w1]
#history_f  = [SSE(w0, w1)]
#
#for epoch in range(20):
#    w0, w1 = mejora(w0, w1, 0.05)
#    history_w0.append(w0)
#    history_w1.append(w1)
#    history_f.append(SSE(w0, w1))
#
#print('\nValores encontrados\n\n  w0 = {:f}\n  w1 = {:f}'.format(w0, w1))
#
#plot_contour()
#plt.plot(history_w0, history_w1, color='red');

###
###  A continuación se grafican la recta encontrada.
###
#
###  Se generan los puntos
#z = np.linspace(0.0, 1.0)
#y = w0 + w1 * z
#
### se grafican los datos originales
#plt.figure(3)
#plt.plot(X, Y, 'o');
### se grafica la recta encontrada
#plt.plot(z, y, '-');
#
#plt.show();


### REGRESION LINEAL

regression = linear_model.LinearRegression()
#print("# validation", cross_val_score(m, X, Y, cv=20).mean())
regression.fit(np.array(X).reshape(-1, 1), Y)
y_pred = regression.predict(np.array(X).reshape(-1, 1))
#print(y_pred)
print("Resultado de la validación cruzada: ", cross_val_score(regression, np.array(X).reshape(-1, 1), Y, cv=20).mean())
plt.figure(4)
plt.plot(X, Y, '.r')
plt.plot(X, y_pred, '-b')

plt.show()