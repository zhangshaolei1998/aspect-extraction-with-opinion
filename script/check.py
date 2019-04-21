import numpy as np
from nltk import ngrams
import string
import os
import random
import json


ae_data = np.load("../data/prep_data/restaurant.npz")

X = ae_data['train_X']
y = ae_data['train_y']

new_data = np.load("../data/prep_data/restaurant_data.npz")
nX=new_data['sentences']
ny=new_data['aspect_tags']

for i in range(1975,1976):
    print('X',i)
    print(X[i])
    print('y',i)
    print(y[i])


for i in range(1730,1731):
    print('X',i)
    print(nX[i])
    print('y',i)
    print(ny[i])

count=0

for i in range(0,2000):
    for j in range(0,2000):
        if (X[i]==nX[j]).all() :#and (y[i]==ny[j]).all():
            #print(i,"match",j)
            count+=1
            #break

print("match:",count)
