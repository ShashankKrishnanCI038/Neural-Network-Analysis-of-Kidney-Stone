import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.callbacks import TensorBoard
import time

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

    loss, accuracy = model.evaluate(X_train, y_train)
    print('Validation Loss:', loss)
    print('Validation Accuracy:', accuracy)

    model.save('DBITCnnmodel.h5')
except MemoryError as e:
    print("Memory error")
