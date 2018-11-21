from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
import cv2
import pickle
from sklearn.metrics import log_loss
import numpy as np


img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_classes = 43


datasets_path = "D:/traffic-signs-data/"
models_path = "./models/"

training_file = datasets_path + 'train.p'
validation_file= datasets_path + 'valid.p'
testing_file = datasets_path + 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

def normalise_image(X):
    result = []
    for img in X:
        im = cv2.resize(img, (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        #im = im.transpose((2,0,1))
        result.append(im/255.0)
    return np.array(result)

r_index = np.random.choice(len(X_train), 10000)
X_train=normalise_image(X_train[r_index])
y_train=y_train[r_index]

r_index=np.random.choice(len(X_valid), 1000)
X_valid=normalise_image(X_valid[r_index])
y_valid=y_valid[r_index]

#one hot 编码
new_y = np.zeros((len(y_valid), num_classes))
new_y[np.arange(len(y_valid)), y_valid] =1
y_valid = new_y.astype(np.int8)

new_y = np.zeros((len(y_train), num_classes))
new_y[np.arange(len(y_train)), y_train] =1
y_train = new_y.astype(np.int8)


X_test=normalise_image(X_test)

new_y = np.zeros((len(y_test), num_classes))
new_y[np.arange(len(y_test)), y_test] =1
y_test = new_y.astype(np.int8)


# 查看每种类别数量是否都有
np.sum(y_valid,axis=0)
np.sum(y_train,axis=0)

#调整维度的顺序
from keras import backend as K
K.set_image_dim_ordering('tf')


#调用VGG16模型，以及自动下载imagenet 的权重
from keras import applications
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, channel))

from keras.models import Model
x = base_model.output

# 多加两层结构

x = Flatten()(x)
x = Dense(32, activation='relu')(x)
x=  Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 40
nb_epoch = 100
# Start Fine-tuning
model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          shuffle=True,
          verbose=1,
          validation_data=(X_valid, y_valid),
          )
model.save('traffic_signs.h5')
