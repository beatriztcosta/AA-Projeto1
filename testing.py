import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import plotly.express as px
#to load matlab mat files
from scipy.io import loadmat
np.set_printoptions(threshold=sys.maxsize)

ds = pd.read_csv('bee_dataset/relevant_bee_data.csv')
max_width = 0
max_height = 0
mean_width = 0
mean_height = 0
img_dict = {}
#i = 0
for idx,file in enumerate(ds['file']):
    img_array = cv2.imread('bee_dataset/bee_imgs/'+file, cv2.IMREAD_GRAYSCALE)
    h,w = img_array.shape
    mean_height += h
    mean_width += w
    if h > max_height:
        max_height = h
    if w > max_width:
        max_width = w
    if h <= 75 and w <= 75: 
        vector_size = h*w
        #flat_img_array = img_array.flatten()
        #flat_img_array = np.reshape(flat_img_array,(1,vector_size))
        #print(img_array)
        #img_dict[idx] = {'features': flat_img_array, 'subspecie': ds['subspecies'][idx]}
        img_dict[idx] = {'features': img_array, 'subspecie': ds['subspecies'][idx]}
    #i+=1

test_features = img_dict[76]['features']
#print(img_dict[76]['features'])

#mean_width = mean_width/i
#mean_height = mean_height/i
#print(mean_width)
#print(mean_height)

#vector_size = max_height*max_width
h,w = test_features.shape
height_dif = 75-h 
width_dif = 75-w
pad_features = np.pad(test_features, ((0,height_dif),(0,width_dif)), mode='constant', constant_values=0)
#print(pad_features)

features_array = []
subspecies_array = []
for key, value in img_dict.items():
    features = value['features']
    h,w = features.shape
    if h != max_height or w != max_width:
        #size_dif = vector_size-(h*w)
        #pad_features = np.pad(features, ((0,0),(0,size_dif)), mode='constant', constant_values=0)
        height_dif = 75-h 
        width_dif = 75-w
        pad_features = np.pad(test_features, ((0,height_dif),(0,width_dif)), mode='constant', constant_values=0)
    features_array.append(pad_features)
    subspecies_array.append(ds['subspecies'][key])
    
data = {'features': features_array, 'class': subspecies_array}
df = pd.DataFrame(data=data, columns=['features', 'class'])
df.to_csv('bee_dataset/ml_data.csv', index=False)