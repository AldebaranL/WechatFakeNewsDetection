from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import jieba

def get_tfidf_data():#jieba+tfidf生成词向量，X为2d,仅用title
    df_raw = pd.read_csv('../datasets/WeFEND-AAAI20-master/data/train/news.csv')
    df_raw1 = pd.read_csv('../datasets/WeFEND-AAAI20-master/data/test/news.csv')
    data = pd.concat([df_raw, df_raw1], ignore_index=True)

    X = []
    for text in data['Title']:
        X.append(" ".join(jieba.cut(text)))
    Y=data['label']
    tfidf = TfidfVectorizer(analyzer='word')
    X = tfidf.fit_transform(X)
    #print(tfidf.vocabulary_)
    return X.A,Y


def get_wordVec2d_data():#word2vec词向量，X为2d,用title #和Official Account Name
    X,Y=get_wordVec3d_data()
    X=np.array([i.flatten() for i in X])
    return X,Y

import os
def get_wordVec3d_data():#word2vec词向量，X为3d（总数据数，max（每句中单词数），300=embeding维数）
    if not os.path.isfile("./data_x.txt"):
        create_wordVec3d_data_title()
    if not os.path.isfile("./data_y.txt"):
        create_data_y()
    infile = open("./data_x.txt", "r", encoding='UTF-8')
    X = infile.read()
    X = json.loads(X)
    infile.close()
    infile = open("./data_y.txt", "r", encoding='UTF-8')
    Y = infile.read()
    Y = json.loads(Y)
    infile.close()
    return np.array(X),np.array(Y)

import json
def create_wordVec3d_data_title():#将x_title词向量数值写入文件
    infile = open("./features_train.txt", "r", encoding='UTF-8')#
    features_train = infile.read()
    features_train = json.loads(features_train)
    infile.close()
    infilet = open("./features_test.txt", "r", encoding='UTF-8')
    features_test = infilet.read()
    features_test = json.loads(features_test)
    infilet.close()
    features=features_train
    features.extend(features_test)

    max_len = max([len(i) for i in features])
    X = []
    for sen in features:
        tl = sen
        tl += [[0]*300] * (max_len - len(tl))
        X.append(tl)
    X = json.dumps(X)
    file = open("./data_x.txt", "w", encoding='UTF-8')
    file.write(X)
    file.close()

def create_data_y():#将y数值写入文件
    df_raw = pd.read_csv('../datasets/WeFEND-AAAI20-master/data/train/news.csv')
    df_raw1 = pd.read_csv('../datasets/WeFEND-AAAI20-master/data/test/news.csv')
    data = pd.concat([df_raw, df_raw1], ignore_index=True)

    Y=data['label']
    Y = json.dumps(list(Y))
    file = open("./data_y.txt", "w", encoding='UTF-8')
    file.write(Y)
    file.close()

def get_data_offName():#Ofiicial Account Name转成数字特征
    df_raw = pd.read_csv('../datasets/WeFEND-AAAI20-master/data/train/news.csv')
    df_raw1 = pd.read_csv('../datasets/WeFEND-AAAI20-master/data/test/news.csv')
    data = pd.concat([df_raw, df_raw1], ignore_index=True)
    data=data['Ofiicial Account Name']
    val_dic=list(set(data))#去重
    val_dic=dict(zip(val_dic,range(len(val_dic))))#构造对应表
    X=[[val_dic[i]] for i in data]#转成数字特征
    return X

def shuffle_split_data(X,Y):
    X=np.array(X)
    Y=np.array(Y)
    #X=X[len(X) // 2:]
    #Y=Y[len(Y)//2:]
    #重排
    np.random.seed(5)
    np.random.shuffle(X)
    np.random.seed(5)
    np.random.shuffle(Y)
    split = len(X) // 3*2
    #s2=len(X) // 2
    #split=10142
    #划分
    X_test = X[split:]
    X_train = X[:split]
    y_test = Y[split:]
    y_train = Y[:split]

    #np.random.seed(8)
    #np.random.shuffle(X_train)
    #np.random.seed(8)
    #np.random.shuffle(y_train)
    return X_test,X_train,y_test,y_train

if __name__=='__main__':
    x,y=get_wordVec2d_data()
    print(type(x))
    print(len(x))
    print(len(x[0]))
    print(len(x[0][0]))
    print(type(y))
    print(y.shape)