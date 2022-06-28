import load_data
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D

def cnn(X_train):
    # 使用卷积窗口的长度为3、4、5，加最大池化层，加dropout层（防止过拟合），加softmax分类层
    model = tf.keras.Sequential()
    model.add(Convolution1D(input_shape=(X_train.shape[1], X_train.shape[2]),
                            filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPool1D())
    model.add(Convolution1D(128, 4, activation='relu'))
    model.add(MaxPool1D())
    model.add(Convolution1D(64, 5))
    model.add(Flatten())
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(2, activation='softmax'))
    print(model.summary())
    return model


def dense(X_train):
    # 3层全连接网络
    model = tf.keras.Sequential()
    model.add(layers.Dense(128,input_dim=17700,activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    print(model.summary())
    return model

import numpy as np
if __name__ =='__main__':
    x,y=load_data.get_wordVec3d_data()
    print(x.shape)
    print(type(x))
    print(y.shape)
    print(type(y))
    X_test,X_train,y_test,y_train=load_data.shuffle_split_data(x,y)

    model=cnn(x)
    #使用多分类损失函数，优化方法使用Adam，度量y_和y都是数值给出
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    #训练，epochs=10,batch_size=128
    history=model.fit(X_train,y_train,epochs=10,
                      batch_size=128,verbose=1,
                      validation_data=(X_test,y_test))

    #图示化训练和验证准确率变化，试验后发现epochs=10比较合适
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'valiation'], loc='upper left')
    plt.show()
