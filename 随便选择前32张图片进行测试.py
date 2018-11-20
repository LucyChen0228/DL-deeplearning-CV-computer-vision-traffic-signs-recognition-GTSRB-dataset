from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
import cv2
import pickle
from sklearn.metrics import log_loss
import numpy as np
from tensorflow import keras
from keras.models import load_model


img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3       #RGB三通道
num_classes = 43


datasets_path = "D:/traffic-signs-data/"
models_path = "./models/"


testing_file = datasets_path + 'test.p'

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_test, y_test = test['features'], test['labels']


#去均值
def normalise_image(X):
    result = []
    for img in X:
        im = cv2.resize(img,(224, 224)).astype(np.float32)#图像信道输入有问题
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        #im = im.transpose((2,0,1))
        result.append(im/255.0)
    return np.array(result)


X_test=normalise_image(X_test)
X_test_1 = X_test[0:32,:] #如何选择前32个

new_y = np.zeros((len(y_test), num_classes))
new_y[np.arange(len(y_test)), y_test] =1
y_test = new_y.astype(np.int8)
y_test_1 = y_test[0:32,:]

from keras.models import load_model
model = load_model('traffic_signs.h5')
predictions_test = model.evaluate(X_test_1, y_test_1, batch_size=32, verbose=1)
print('test loss:',predictions_test[0])
print('test accurancy:',predictions_test[1])


#看看前32张里面哪些地方预测错了

predicted_y = np.argmax(model.predict(X_test_1, batch_size=32, verbose=1), axis=1)
groundtruth_y = np.argmax(y_test_1, axis=1)
diff_idx= np.equal( np.array(predicted_y), ( np.array(groundtruth_y)))

print(predicted_y[diff_idx == False])
print(groundtruth_y[diff_idx == False])


