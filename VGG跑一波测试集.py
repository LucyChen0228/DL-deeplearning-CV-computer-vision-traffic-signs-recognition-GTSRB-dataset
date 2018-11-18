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


# test 用全量，内存好像会爆，训练完，保存完模型再跑吧
X_test=normalise_image(X_test)




new_y = np.zeros((len(y_test), num_classes))
new_y[np.arange(len(y_test)), y_test] =1
y_test = new_y.astype(np.int8)


#from keras import backend as K
#K.set_image_dim_ordering('tf')
#from keras import applications
#base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channel))

#from keras.models import Model
#x = base_model.output
#model = Model(input=base_model.input, output=predictions)

#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)#VGG 16里面仍旧采用了SGD 进行优化，优化器应该还是ADAM？
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


from keras.models import load_model
model = load_model('traffic_signs.h5')
predictions_test = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)#VGG 16里面仍旧采用了SGD 进行优化，优化器应该还是ADAM？
#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


#看看是哪些预测错了的地方
#predicted_y = np.argmax(model.predict(X_test, batch_size=32, verbose=1), axis=1)
#groundtruth_y = np.argmax(y_test, axis=1)
#diff_idx= np.equal( np.array(predicted_y), ( np.array(groundtruth_y)))

#predicted_y[diff_idx == False]
#groundtruth_y[diff_idx == False]


#keras.callbacks.Callback()
#keras.call.0backs.EarlyStopping( monitor='val_loss',min_delta=0,patience=0,verbose=0,mode='auto',baseline=None, restore_best_weights=False)#后面加的防止过拟合的东西