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


X_test = normalise_image(X_test)
X_test_1 = X_test[0:32,:]   #如何选择前32个   ###？？？？？？？  这里有问题，原来写的“ X_test_1 = X_test[0:32,:]  ”


# === 查看图片 ===========================

print(np.shape(X_test))
print(np.shape(X_test_1))

import numpy as np
import cv2
from matplotlib import pyplot as plt


r1 = X_test_1[0,:,:,0]
g1 = X_test_1[0,:,:,1]
b1 = X_test_1[0,:,:,2]

r2 = X_test_1[2,:,:,0]
g2 = X_test_1[2,:,:,1]
b2 = X_test_1[2,:,:,2]

r3 = X_test_1[1,:,:,0]
g3 = X_test_1[1,:,:,1]
b3 = X_test_1[1,:,:,2]

img1 = cv2.merge([b1,g1,r1])
img2 = cv2.merge([b2,g2,r2])
img3 = cv2.merge([b3,g3,r3])

plt.figure()
plt.subplot(1,3,1)
plt.imshow(img1)

plt.subplot(1,3,2)
plt.imshow(img2)

plt.subplot(1,3,3)
plt.imshow(img3)

plt.show()
# ====================================================


#对标签Y进行one hot 编码
new_y = np.zeros((len(y_test), num_classes))
new_y[np.arange(len(y_test)), y_test] =1
y_test = new_y.astype(np.int8)

# ===== 用于测试  ============================
y_test_1 = y_test[0:32,:]
#print(y_test_1)
#print("标签值：\n",np.size(y_test_1))
# ==============================================

from keras.models import load_model
model = load_model('traffic_signs.h5')
#predictions_test = model.evaluate(X_test_1, y_test_1, batch_size=32, verbose=1)
#print('test loss:',predictions_test[0])
#print('test accurancy:',predictions_test[1])


#看看前32张里面哪些地方预测错了
#from keras import applications
#model= applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channel))#imagenet 的权重的路径问题（需解决）

predicted_y = model.predict(X_test_1, batch_size=32, verbose=1)


#predicted_y_1 = predicted_y[0,:]
#print(predicted_y)
#print(np.size(predicted_y))
#print('真实值：\n',np.size(predicted_y))

Pre_loc = np.argmax(predicted_y,axis=1)
Lab_loc = np.argmax(y_test_1,axis=1)

Compare = np.equal(Pre_loc,Lab_loc)


print("predicted_y: ",np.shape(predicted_y))
print("Pre_loc: ",np.shape(Pre_loc))
print("Lab_loc ",np.shape(Lab_loc))


print('第 1 张图标签 ：',y_test_1[0,:])
print('第 2 张图标签 ：',y_test_1[1,:])
print('第 3 张图标签 ：',y_test_1[2,:])


print('第 1 张图类别概率 argmax ：',predicted_y[0,:])
print('第 2 张图类别概率 argmax ：',predicted_y[1,:])
print('第 3 张图类别概率 argmax ：',predicted_y[2,:])

print(Pre_loc[0:3])
print(Lab_loc[0:3])
print(Compare)


predicted_y = np.argmax(model.predict(X_test_1, batch_size=32, verbose=1), axis=1)
groundtruth_y = np.argmax(y_test_1, axis=1)
diff_idx= np.equal( np.array(predicted_y), ( np.array(groundtruth_y)))

print(predicted_y[diff_idx == False])
print(groundtruth_y[diff_idx == False])

