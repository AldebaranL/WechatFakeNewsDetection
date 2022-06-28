import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert
import pandas as pd
import numpy as np
import re
import random
import math
import os

class TEXT_MODEL(tf.keras.Model):
    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l)
        l_1 = self.pool(l_1)

        concatenated = self.dense_1(l_1)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output

def tokenize_text(text_input):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_input))

if __name__ == '__main__':

    # hyper parameters
    BATCH_SIZE = 128
    EMB_DIM = 300
    CNN_FILTERS = 100
    DNN_UNITS = 256
    OUTPUT_CLASSES = 2
    DROPOUT_RATE = 0.5
    NB_EPOCHS = 5
    max_len = 60

    # raw data
    df_raw = pd.read_csv('../datasets/WeFEND-AAAI20-master/data/train/news.csv')
    df_raw1 = pd.read_csv('../datasets/WeFEND-AAAI20-master/data/test/news.csv')
    print(type(df_raw))
    print(len(df_raw1))
    df_raw=pd.concat([df_raw,df_raw1],ignore_index=True)
    print(len(df_raw))

    y = np.array(df_raw["label"])
    # Creating a BERT Tokenizer
    BertTokenizer = bert.bert_tokenization.FullTokenizer
    bert_layer = hub.KerasLayer("bert_zh_L-12_H-768_A-12_2",trainable=False)

    vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

    # Tokenize all the text

    text = list(df_raw['Title'])

    tokenized_text = [(text[i], y[i]) for i in range(len(text))]

    # Prerparing Data For Training

    random.shuffle(tokenized_text)

    tokenized_text = tf.data.Dataset.from_generator(lambda: tokenized_text, output_types=(tf.int32, tf.int32))
    batched_dataset = tokenized_text.padded_batch(BATCH_SIZE, padded_shapes=((max_len,), ()))

    TOTAL_BATCHES = math.ceil(len(batched_dataset) / BATCH_SIZE)
    TEST_BATCHES = TOTAL_BATCHES // 3

    test_data = batched_dataset.take(TEST_BATCHES)
    train_data = batched_dataset.skip(TEST_BATCHES)
    print(type(test_data))
    #print(test_data.shape)
    #print(test_data)
    VOCAB_LENGTH = len(tokenizer.vocab)
    text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                            embedding_dimensions=EMB_DIM,
                            cnn_filters=CNN_FILTERS,
                            dnn_units=DNN_UNITS,
                            model_output_classes=OUTPUT_CLASSES,
                            dropout_rate=DROPOUT_RATE)


    text_model.compile(loss="binary_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])

    text_model.fit(train_data, epochs=NB_EPOCHS)
    # text_model.fit(train_data, epochs=NB_EPOCHS,validation_data=test_data)
    # test test data
    results = text_model.evaluate(test_data)
    print(results)