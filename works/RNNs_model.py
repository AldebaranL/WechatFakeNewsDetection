import load_data
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
def crnn(X_train):
    model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Convolution1D(input_shape=(X_train.shape[1], X_train.shape[2]),
                                            filters=256,
                                            kernel_size=3,
                                            activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(
        tf.keras.layers.Convolution1D(input_shape=(X_train.shape[1], X_train.shape[2]),
                                      filters=256,
                                      kernel_size=4,
                                      activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(
        tf.keras.layers.Convolution1D(input_shape=(X_train.shape[1], X_train.shape[2]),
                                      filters=256,
                                      kernel_size=5,
                                      activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.GRU(128, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))
    model.add(tf.keras.layers.GRU(128, dropout=0.2, recurrent_dropout=0.1))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    print(model.summary())
    return model


def rnns(X_train):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32,
                                                                 activation='sigmoid'),
                                            input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.summary()
    return model


if __name__ == '__main__':
    X, Y = load_data.get_wordVec3d_data()
    X_test, X_train, y_test, y_train = load_data.shuffle_split_data(X,Y)

    model = crnn(X_train)

    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam",metrics=['accuracy'])#, metrics=['sparse_categorical_accuracy'])

    batch_size = 128
    # 训练，epochs=16,batch_size=128
    history = model.fit(X_train, y_train, epochs=64,
                        batch_size=batch_size,
                        verbose=1,
                        validation_data=(X_test, y_test)
                        )
    #model.evaluate(X_test, y_test)