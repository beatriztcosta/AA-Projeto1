import sys
from tkinter import S
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import plotly.express as px
import time
import random

#to load matlab mat files
from numpy import genfromtxt
from scipy.io import loadmat
np.set_printoptions(threshold=sys.maxsize)

size = 100
img_class_threshold = 500 #Minimum amount of images from each class

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

def create_ml_data_resized():
    ds = pd.read_csv('bee_dataset/relevant_bee_data.csv')
    vector_size = size*size
    features_array = []
    subspecies_array = []
    for idx,file in enumerate(ds['file']):
        img_array = cv2.imread('bee_dataset/bee_imgs/'+file, cv2.IMREAD_GRAYSCALE)
        h,w = img_array.shape
        dim = (size,size)
        resized_img_array = cv2.resize(img_array, dim, interpolation = cv2.INTER_AREA)
        flat_img_array = resized_img_array.flatten()
        flat_img_array = np.reshape(flat_img_array,(1,vector_size))
        flat_img_array = flat_img_array / 255
        flat_img_array = flat_img_array.flatten()
        str_flat_img_array = ' '.join(map(str, flat_img_array))
        features_array.append(str_flat_img_array)
        subspecies_array.append(ds['subspecies'][idx])
        
    data = {'features': features_array, 'class': subspecies_array}
    df = pd.DataFrame(data=data, columns=['features', 'class'])
    df.to_csv('bee_dataset/ml_data_resized.csv', index=False)

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

def show_img():
    img = "bee_dataset/bee_imgs/003_034.png"
    img = cv2.imread(img,cv2.IMREAD_COLOR)
    
    h,w,c = img.shape

    #cv2.imshow('img',new_img)
    #cv2.waitKey(0) 

#-------------/RGB/--------------

def create_ml_data_rgb():
    ds = pd.read_csv('bee_dataset/relevant_bee_data.csv')
    vector_size = size*size
    features_array = []
    subspecies_array = []
    for idx,file in enumerate(ds['file']):
        img_array = cv2.imread('bee_dataset/bee_imgs/'+file, cv2.IMREAD_COLOR)
        h,w,c = img_array.shape
        if h <= size and w <= size:
            str_flat_img_array = ''
            for channel in range(c):
                img_array_c = np.copy(img_array[:,:,channel])
                h_dif = size-h
                zeros = np.zeros((h_dif,w), dtype=int)
                img_array_c = np.vstack((img_array_c,zeros))

                w_dif = size-w
                zeros = np.zeros((size,w_dif), dtype=int)
                
                img_array_c = np.hstack((img_array_c,zeros))
                flat_img_array = img_array_c.flatten()
                flat_img_array = np.reshape(flat_img_array,(1,vector_size))
                flat_img_array = flat_img_array / 255
                flat_img_array = flat_img_array.flatten()
                
                if not str_flat_img_array:
                    str_flat_img_array = ' '.join(map(str, flat_img_array))
                else:
                    str_flat_img_array = str_flat_img_array + ' ' + ' '.join(map(str, flat_img_array))

            features_array.append(str_flat_img_array)
            subspecies_array.append(ds['subspecies'][idx])
        
    data = {'features': features_array, 'class': subspecies_array}
    df = pd.DataFrame(data=data, columns=['features', 'class'])
    df.to_csv('bee_dataset/ml_data_rgb.csv', index=False)

def show_img_rgb():
    img = "bee_dataset/bee_imgs/003_034.png"
    img = cv2.imread(img,cv2.IMREAD_COLOR)
    
    h,w,c = img.shape
    print(img.shape)
    vector_size = size*size
    str_flat_img_array = ''

    for channel in range(c):
        img_array_c = np.copy(img[:,:,channel])
        h_dif = size-h
        zeros = np.zeros((h_dif,w), dtype=int)
        img_array_c = np.vstack((img_array_c,zeros))
        w_dif = size-w
        zeros = np.zeros((size,w_dif), dtype=int)
        
        img_array_c = np.hstack((img_array_c,zeros))
        flat_img_array = img_array_c.flatten()
        flat_img_array = np.reshape(flat_img_array,(1,vector_size))
        flat_img_array = flat_img_array #/ 255
        flat_img_array = flat_img_array.flatten()
        
        if not str_flat_img_array:
            str_flat_img_array = ' '.join(map(str, flat_img_array))
        else:
            str_flat_img_array = str_flat_img_array + ' ' + ' '.join(map(str, flat_img_array))


    #arr_f = np.fromstring(str_flat_img_array, dtype=float, sep=' ')
    arr_f = np.fromstring(str_flat_img_array, dtype=np.uint8, sep=' ')
    new_img = np.reshape(arr_f,(3,size,size)).transpose(1,2,0) #.reshape(arr_f, (size,size,3))
    print(img[0])
    print(new_img[0])
    print(new_img.shape)
    cv2.imshow('img',new_img)
    cv2.waitKey(0)

def create_balanced_ml_data():
    ds = pd.read_csv('bee_dataset/ml_data_resized.csv')
    tmp = ds['class'].value_counts()
    print(tmp)
    print(tmp.index)
    print(tmp.values)

    short_classes_index = np.where(tmp.values == np.amin(tmp.values))[0]
    short_classes = tmp.index[short_classes_index].values
    print("Minimum number of images in a class: " + str(np.amin(tmp.values)) + " from class " + str(short_classes))

    relevant_classes_index = np.where(tmp.values >= img_class_threshold)[0]
    relevant_classes = tmp.index[relevant_classes_index].values
    print("Classes with a mininum number of images equal to " + str(img_class_threshold) + ": " + str(relevant_classes))

    img_class_count = np.amin(tmp.values[relevant_classes_index])
    print("\nEach class will then have " + str(img_class_count) + " images")

    class_dict = {}
    for c in relevant_classes:
        class_dict[str(c)] = img_class_count
        
    
    features=ds['features']
    y=ds['class']
    features_array = []
    subspecies_array = []
    for idx,f in enumerate(features):
        if sum(class_dict.values()) == 0:
            break
        if y[idx] in class_dict:
            if class_dict[y[idx]] > 0:
                class_dict[y[idx]] -= 1
                features_array.append(f)
                subspecies_array.append(y[idx])
                
    data = {'features': features_array, 'class': subspecies_array}
    df = pd.DataFrame(data=data, columns=['features', 'class'])
    df.to_csv('bee_dataset/balanced_ml_data.csv', index=False)


def create_balanced_ml_data_oversampling():
    ds = pd.read_csv('bee_dataset/ml_data_resized.csv')
    features_array = []
    subspecies_array = []
    tmp = ds['class'].value_counts()

    short_classes_index = np.where(tmp.values == np.amin(tmp.values))[0]
    short_classes = tmp.index[short_classes_index].values
    print("Minimum number of images in a class: " + str(np.amin(tmp.values)) + " from class " + str(short_classes))

    relevant_classes_index = np.where(tmp.values >= img_class_threshold)[0]
    relevant_classes = tmp.index[relevant_classes_index].values
    print("Classes with a mininum number of images equal to " + str(img_class_threshold) + ": " + str(relevant_classes))

    potential_classes_index = np.where((tmp.values < img_class_threshold) & (tmp.values >= img_class_threshold/2))[0]
    potential_classes = tmp.index[potential_classes_index].values
    print("Classes to be oversampled since they display a number of images equal or above half of the threshold (" + str(img_class_threshold) + "): " + str(potential_classes))

    img_class_count = np.amin(tmp.values[relevant_classes_index])
    print("\nEach class will then have " + str(img_class_count) + " images")

    for i in potential_classes_index:
        c = tmp.index[i]
        img_count = tmp.values[i]
        add_img_count = img_class_count-img_count
        print("\nOversampling class " + str(c) + " with " + str(add_img_count) + " extra images")
        df=ds.loc[ds['class'] == c, 'features'].iloc[0:add_img_count]
        for f in df:
            n = random.randint(1,3)
            arr_f = np.fromstring(f, dtype=np.float, sep=' ')
            new_img = np.reshape(arr_f,(size,size))
            rot_img = np.rot90(new_img,n)
            flat_rot_img = rot_img.flatten()
            str_flat_rot_img = ' '.join(map(str, flat_rot_img))
            features_array.append(str_flat_rot_img)
            subspecies_array.append(c)
            
    relevant_classes = np.hstack((relevant_classes,potential_classes))
    
    class_dict = {}
    for c in relevant_classes:
        class_dict[str(c)] = img_class_count
        
    
    features=ds['features']
    y=ds['class']
    for idx,f in enumerate(features):
        if sum(class_dict.values()) == 0:
            break
        if y[idx] in class_dict:
            if class_dict[y[idx]] > 0:
                class_dict[y[idx]] -= 1
                features_array.append(f)
                subspecies_array.append(y[idx])
                
    data = {'features': features_array, 'class': subspecies_array}
    df = pd.DataFrame(data=data, columns=['features', 'class'])
    df.to_csv('bee_dataset/oversampled_balanced_ml_data.csv', index=False)


#create_ml_data()
#create_ml_data_resized()
#X = test_csv_read()
#show_img()

#create_ml_data_rgb()
#show_img_rgb()

#create_balanced_ml_data()
create_balanced_ml_data_oversampling()