import sys
from tkinter import S
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import plotly.express as px
import time
#to load matlab mat files
from numpy import genfromtxt
from scipy.io import loadmat
np.set_printoptions(threshold=sys.maxsize)

size = 50

def create_ml_data():
    ds = pd.read_csv('bee_dataset/relevant_bee_data.csv')
    vector_size = size*size
    features_array = []
    subspecies_array = []
    for idx,file in enumerate(ds['file']):
        img_array = cv2.imread('bee_dataset/bee_imgs/'+file, cv2.IMREAD_GRAYSCALE)
        h,w = img_array.shape
        if h <= size and w <= size:
            h_dif = size-h
            zeros = np.zeros((h_dif,w), dtype=int)
            img_array = np.vstack((img_array,zeros))
            
            w_dif = size-w
            zeros = np.zeros((size,w_dif), dtype=int)
            img_array = np.hstack((img_array,zeros))

            flat_img_array = img_array.flatten()
            flat_img_array = np.reshape(flat_img_array,(1,vector_size))
            flat_img_array = flat_img_array / 255
            flat_img_array = flat_img_array.flatten()
            str_flat_img_array = ' '.join(map(str, flat_img_array))
            features_array.append(str_flat_img_array)
            subspecies_array.append(ds['subspecies'][idx])
        
    data = {'features': features_array, 'class': subspecies_array}
    df = pd.DataFrame(data=data, columns=['features', 'class'])
    df.to_csv('bee_dataset/ml_data.csv', index=False)

def test_csv_read():
    ds = pd.read_csv('bee_dataset/ml_data.csv')
    features=ds['features']
    y=ds['class']
    n_examples = features.shape[0]
    feature_size = size*size
    X = np.zeros((n_examples,feature_size), float)
    for idx,f in enumerate(features):
        arr_f = np.fromstring(f, dtype=float, sep=' ')
        arr_f = np.reshape(arr_f, (1,arr_f.shape[0]))
        X[idx,:] = np.copy(arr_f)
        
    print(X.shape)

    return X

def show_img(X):
    print(X[0])
    a = np.reshape(X[0],(size,size))
    print(type(a))
    print(a)

    #cv2.imshow('img',a)


create_ml_data()
X = test_csv_read()
show_img(X)
