# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:13:56 2019

@author: rafael_jz
"""

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
#import math
import pandas as pd
#from matplotlib.pyplot import plot, show

#np.random.seed(12345)
#d =  [1.0 * math.sin(t * math.pi /  8) for t in range(64)]
#d += [0.5 * math.sin(t * math.pi /  4) for t in range(64, 134)]
#d += [0.8 * math.sin(t * math.pi / 12) for t in range(134, 250)]
#d =  [x + 0.06 * np.random.standard_normal() for x in d]

# Leer el csv
mp = pd.read_csv('allYearPrices.csv')

# generar la fráfica
d = np.log(mp['mean'])
plt.figure(figsize=(11,3))
plt.plot(d, color='black');



class Model(object):
    def __init__(self, L):
        self.w = tf.contrib.eager.Variable([0.0] * (L))

    def __call__(self, x):
        x = tf.constant(np.array([1.0] + x, dtype=np.float32))
        y_pred = tf.reduce_sum(tf.multiply(self.w, x))
        return y_pred

    def fit(self, mu, x, y_desired):
        y_pred = self(x)
        e = y_desired - y_pred
        x = tf.constant(np.array([1.0] + x, dtype=np.float32))
        self.w.assign_add(tf.scalar_mul(2 * mu * e, x))
        
##
##  Para pronosticar el valor actual se toman los `L`
##  valores previos de la serie
##
        
L = 7

##
##  Modelo (Crea la clase 'modelo')
##
model = Model(L) 

##
##  Pronosticos del modelo
##
y_pred = np.empty(len(d))
y_pred[:] = np.nan

for t in range(L, len(d)):
    x = d[t-L:t]
    y_pred[t] = model(x)
    model.fit(mu=0.0005, x=x, y_desired=d[t])

plt.figure(figsize=(14,3))
plt.plot(d, color='blue');
plt.plot(y_pred, color = 'red')
plt.show()