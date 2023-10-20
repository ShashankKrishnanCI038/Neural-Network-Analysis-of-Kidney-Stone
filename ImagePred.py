import tensorflow as tf
from keras.models import load_model
import keras.preprocessing.image
from PIL import Image
import numpy as np
import cv2

model = load_model('DBITCnnmodelll.h5')
try:
    imge = Image.open(r'C:\Users\SHASHANK K\pythonProject\Kidney Stone Prediction DBIT\Datasets\Test\Normal\1.3.46.670589.33.1.63716923798771028500001.5192664294078085720.png')
    test_gray = cv2.imread(imge, cv2.IMREAD_GRAYSCALE)
    test_image = imge.resize(test_gray, (110, 110))
    test_image = np.asarray(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255
    classification = model.predict(test_image)
    print("The value ", classification[0][0])
    print("\n")
    if classification[0][0] > 0.5:
        print("kidney")
    elif classification[0][0] < 0.5:
        print("kidney stone")
    else:
        print("error in image")

except ValueError as e:
    print("Error in Image value")
    pass
