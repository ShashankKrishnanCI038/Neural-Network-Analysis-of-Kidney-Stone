import numpy as np
import cv2
import os
import random
import pickle

directory = r'C:\Users\SHASHANK K\pythonProject\Kidney Stone Prediction DBIT\Datasets\Test'
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

X_test_list = []
y_test_list = []
for features, labels in data:
    X_test_list.append(features)
    y_test_list.append(labels)

X_test_list_arr = np.array(X_test_list)
y_test_list_arr = np.array(y_test_list)

pickle.dump(X_test_list_arr, open('X_test.pickle', 'wb'))
pickle.dump(y_test_list_arr, open('y_test.pickle', 'wb'))

print("Data Test")