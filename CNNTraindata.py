import customtkinter
from tkinter import messagebox, filedialog
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.callbacks import TensorBoard
import time
from PIL import Image
import numpy as np
import cv2
import os
import random
import pickle

customtkinter.set_appearance_mode('Light')
customtkinter.set_default_color_theme('green')
windows = customtkinter.CTk()
windows.title("Kidney stone detection Application")
windows.geometry("600x600")
windows.resizable(False, False)
windows.configure(fg_color="#FFFFFF")
CNNModelTrain = None
Trained = None
########################################################################################################################
def imgpred():
    try:
        file = filedialog.askopenfilename(filetypes=[("CNN Model HDF File", "*.H5")])
        model = load_model(file)
        imagepath = filedialog.askopenfilename(filetypes=[("JPG or JPEG Format", "*.jpg"), ("PNG Format", "*.png")])
        try:
            imge = Image.open(imagepath)
            while True:
                if imge == None:
                    messagebox.showinfo(message="PLease choose the image")
                    break
                else:

                    test_image = imge.resize((110, 110))
                    test_image = np.asarray(test_image)
                    test_image = np.expand_dims(test_image, axis=0)
                    test_image = test_image / 255
                    photo = cv2.imread(imagepath)
                    cv2.imshow("image", photo)
                    classification = model.predict(test_image)
                    print("The value ", classification[0][0])
                    print("\n")
                    if classification[0][0] > 0.5:
                        print("Kidney Stone is not present")
                        messagebox.showinfo(message="Kidney Stone is not present")
                    elif classification[0][0] < 0.5:
                        print("Kidney stone is present")
                        messagebox.showinfo(message="Kidney stone is present")
                    else:
                        print("error in image")
                        messagebox.showinfo(message="error in image")
                    cv2.waitKey(0)
                break
        except AttributeError as e:
           messagebox.showinfo(message="Please choose the image")

    except ValueError as e:
        print("Error in Image value")
        pass
    except OSError as oe:
        messagebox.showinfo(message="Please choose the CNN Model to proceed")

def Traindata():
    global Trained
    Trained=0
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

    messagebox.showinfo(message=f'"Data Length: ", {len(data)}')
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
    Trained = 1
    messagebox.showinfo(message="Data Train")

def CNNModel():
    global CNNModelTrain
    while True:
        if CNNModelTrain == 1:
            messagebox.showinfo(message="CNN Model is Trained.....Please import Trained CNN Model and Predict The Kidney Stone")
        elif CNNModelTrain == None:
            Traindata()
            if Trained == 0:
                messagebox.showinfo(message="Please Train the data")
            elif Trained == 1:
                messagebox.showinfo(message="Data is ready for input to CNN Model")

                NAME = f'kidneyy-stone-prediction{int(time.time())}' # tensorboard --logdir=logs/
                tnsboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

                try:
                    X_train = pickle.load(open('X_train.pickle', 'rb'))
                    y_train = pickle.load(open('y_train.pickle', 'rb'))
                    X_train = X_train / 255
                    print(X_train.shape)

                    model = Sequential()

                    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(110, 110, 1)))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    model.add(BatchNormalization())

                    model.add(Conv2D(64, (3, 3), activation='relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    model.add(BatchNormalization())

                    model.add(Conv2D(128, (3, 3), activation='relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    model.add(BatchNormalization())

                    model.add(Flatten())

                    model.add(Dense(128, activation='relu'))
                    model.add(Dense(64, activation='relu'))
                    model.add(Dense(32, activation='relu'))
                    model.add(Dense(16, activation='relu'))
                    model.add(Dense(1, activation='sigmoid'))

                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                    model.fit(X_train, y_train, epochs=75, validation_split=0.1, callbacks=[tnsboard])

                    model.save('DBITCnnmodel.h5')
                    messagebox.showinfo(message="CNN Model is trained and saved")
                    CNNModelTrain = 1
                except MemoryError as e:
                    messagebox.showinfo(message="Memory error")
            else:
                pass
        else:
            pass
        break

def systemexit():
    exit()
########################################################################################################################

frame = customtkinter.CTkFrame(master=windows, height=450, width=340,
                               bg_color="#FFFFFF", fg_color="#F8F8F8").place(x=136, y=70)

label = customtkinter.CTkLabel(master=frame, text="Kidney Stone Detection", font=('Cambria', 29),
                               bg_color="#F8F8F8", corner_radius=25).pack(padx=39, pady=90)  # #32DFF6

button1 = customtkinter.CTkButton(master=frame, text="PREDICT KIDNEY STONE", font=('cambria', 20),
                                  bg_color="#F8F8F8", fg_color="#BAF7FC", text_color="#000000",
                                  hover_color="#FFFFFF", corner_radius=12, command=lambda: imgpred()).pack()

button2 = customtkinter.CTkButton(master=frame, text="TRAIN CNN MODEL", font=('cambria', 20),
                                  bg_color="#F8F8F8", fg_color="#BAF7FC", text_color="#000000",
                                  hover_color="#FFFFFF", corner_radius=12, command=lambda : CNNModel()).pack(padx=20, pady=30)

button4 = customtkinter.CTkButton(master=frame, text="EXIT", font=('cambria', 20),
                                  bg_color="#F8F8F8", fg_color="#BAF7FC", text_color="#000000",
                                  hover_color="#FFFFFF", corner_radius=12, command=lambda : systemexit()).pack(padx=20, pady=30)

windows.mainloop()
