import cv2
import numpy as np
import os
import train_model

# Predicts model output for an input image
#To run code type python model_predict.py <image_name>

def model_predict(img1):
    #loads trained model
    print('model loading')
    model = train_model.model()
    #if img1 is not None:
    print('model load completed')
    test_data = cv2.resize(cv2.imread("/home/gopika/Project/Sketch-Photo-Conversion-using-Deep-CNN-master/Main Project/static/img/sketch.jpg"), (100, 100))
    cv2.imwrite('/home/gopika/Project/Sketch-Photo-Conversion-using-Deep-CNN-master/Main Project/static/img/sketch'+ '_resized.jpg', test_data)
    np.shape(test_data)
    model_out = model.predict([test_data])

    x = np.reshape(model_out, (100, 100, 3))

    cv2.imwrite('/home/gopika/Project/Sketch-Photo-Conversion-using-Deep-CNN-master/Main Project/static/img/output/output.jpg', x)


if __name__ == '__main__':
    import argparse

    #parser = argparse.ArgumentParser()
    #parser.add_argument("path", help="name of image file")

    #args = parser.parse_args()
    #path = args.path
    path='/home/gopika/Project/Sketch-Photo-Conversion-using-Deep-CNN-master/Main Project/'
    model_predict(path)

