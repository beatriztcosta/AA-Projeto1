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
    max_width = 0
    max_height = 0
    mean_width = 0
    mean_height = 0
    img_dict = {}
    for idx,file in enumerate(ds['file']):
        img_array = cv2.imread('bee_dataset/bee_imgs/'+file, cv2.IMREAD_GRAYSCALE)
        h,w = img_array.shape
        mean_height += h
        mean_width += w
        if h > max_height:
            max_height = h
        if w > max_width:
            max_width = w
        if h <= size and w <= size: 
            vector_size = h*w
            flat_img_array = img_array.flatten()
            flat_img_array = np.reshape(flat_img_array,(1,vector_size))
            flat_img_array = flat_img_array / 255
            img_dict[idx] = {'features': flat_img_array, 'subspecie': ds['subspecies'][idx]}
    
    #mean_width = mean_width/i
    #mean_height = mean_height/i
    #print(mean_width)
    #print(mean_height)

    #vector_size = max_height*max_width
    vector_size = size*size
    features_array = []
    subspecies_array = []
    for key, value in img_dict.items():
        features = value['features']
        h,w = features.shape
        if h != max_height or w != max_width:
            size_dif = vector_size-(h*w)
            pad_features = np.pad(features, ((0,0),(0,size_dif)), mode='constant', constant_values=0)
        pad_features = pad_features.flatten()
        pad_features = ' '.join(map(str, pad_features))
        features_array.append(pad_features)
        subspecies_array.append(ds['subspecies'][key])
        
    data = {'features': features_array, 'class': subspecies_array}
    df = pd.DataFrame(data=data, columns=['features', 'class'])
    df.to_csv('bee_dataset/ml_data.csv', index=False)

def test_csv_read():
    ds = pd.read_csv('bee_dataset/ml_data.csv')
    features=ds['features']
    y=ds['class']
    #print(features.shape)
    n_examples = features.shape[0]
    feature_size = size*size
    X = np.zeros((n_examples,feature_size), float)
    #np.fromstring(X.tostring(),a.dtype).reshape(a.shape)
    for idx,f in enumerate(features):
        arr_f = np.fromstring(f, dtype=float, sep=' ')
        arr_f = np.reshape(arr_f, (1,arr_f.shape[0]))
        X[idx,:] = np.copy(arr_f)
        
    #X = np.copy(arr)
    print(X.shape)

    return X

def show_img(X):
    print(X[0])
    a = np.reshape(X[0],(size,size))
    print(type(a))
    print(a)

    #cv2.imshow('img',a)

#create_ml_data()
X = test_csv_read()
show_img(X)