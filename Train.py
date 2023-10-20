import numpy as np
import cv2
import os
import random
import pickle

directory = r'C:\Users\SHASHANK K\pythonProject\Kidney Stone Prediction DBIT\Datasets\Train'
categories = ['Kidney_stone', 'Normal']

IMG_SIZE = 110
count = 0
data = []
for category in categories:
    folder = os.path.join(directory, category)
    label = categories.index(category)
    for img in os.listdir(folder):
        try:
            img_path = os.path.join(folder, img)
            img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
            data.append([img_arr, label])
        except cv2.error as e:
            pass

print("Data Length: ", len(data))
random.shuffle(data)

X_train_list = []
y_train_list = []

for features, labels in data:
    X_train_list.append(features)
    y_train_list.append(labels)

X_train_list_arr = np.array(X_train_list)
y_train_list_arr = np.array(y_train_list)

pickle.dump(X_train_list_arr, open('X_train.pickle', 'wb'))
pickle.dump(y_train_list_arr, open('y_train.pickle', 'wb'))

print("Data Train")
