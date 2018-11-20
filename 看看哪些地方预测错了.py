from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
import cv2
import pickle
from sklearn.metrics import log_loss
import numpy as np



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


# test 用全量，内存好像会爆，训练完，保存完模型再跑吧
X_test=normalise_image(X_test)

new_y = np.zeros((len(y_test), num_classes))
new_y[np.arange(len(y_test)), y_test] =1
y_test = new_y.astype(np.int8)

from keras.models import load_model
model = load_model('traffic_signs.h5')


predicted_y = np.argmax(model.predict(X_test, batch_size=32, verbose=1), axis=1)
groundtruth_y = np.argmax(y_test, axis=1)
diff_idx= np.equal( np.array(predicted_y), ( np.array(groundtruth_y)))

print(predicted_y[diff_idx == False])
print(groundtruth_y[diff_idx == False])


