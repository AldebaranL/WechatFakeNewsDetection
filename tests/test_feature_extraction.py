from time import time
import sys
import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import jieba
import jieba.posseg as pseg
import pandas as pd
import os
import pdb

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB



data_split = 'train'
path = os.path.join('../datasets/WeFEND-AAAI20-master/data/', data_split)
file_name = os.path.join(path, 'news.csv')
train_frame = pd.read_csv(file_name)
train_frame['Report Content'] = train_frame['Report Content'].apply(lambda x: x.split('##'))
print("Number of training data is %d, the fake news is %d" %(train_frame.shape[0], train_frame[train_frame['label'] == 1].shape[0]))

data_split = 'test'
path = os.path.join('../datasets/WeFEND-AAAI20-master/data/', data_split)
file_name = os.path.join(path, 'news.csv')
test_frame = pd.read_csv(file_name)
test_frame['Report Content'] = test_frame['Report Content'].apply(lambda x: x.split('##'))


corpus=train_frame['Title']
print(corpus.head())
print(type(corpus))
tokenized_corpus = []
for text in corpus:
    tokenized_corpus.append(" ".join(jieba.cut(text)))

test_corpus = test_frame['Title']
tokenized_test_corpus = []
for text in test_corpus:
    tokenized_test_corpus.append(" ".join(jieba.cut(text)))

# 下面几个是HashingVectorizer， CountVectorizer+TfidfTransformer，TfidfVectorizer， FeatureHasher的正确用法。

#fh = feature_extraction.FeatureHasher(n_features=15,input_type='string')
#X_train=fh.fit_transform(tokenized_corpus)
#X_test=fh.fit_transform(tokenized_test_corpus)

# fh = feature_extraction.text.HashingVectorizer(n_features=15,non_negative=True,analyzer='word')
# X_train=fh.fit_transform(tokenized_corpus)
# X_test=fh.fit_transform(tokenized_test_corpus)

cv=CountVectorizer(analyzer='word')
transformer=TfidfTransformer()
X_train=transformer.fit_transform(cv.fit_transform(tokenized_corpus))
cv2=CountVectorizer(vocabulary=cv.vocabulary_)
transformer=TfidfTransformer()
X_test = transformer.fit_transform(cv2.fit_transform(tokenized_test_corpus))


# word=cv.get_feature_names()
# weight=X_train.toarray()
# for i in range(len(weight)):
# print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"
# for j in range(len(word)):
#            print word[j],weight[i][j]

#tfidf = TfidfVectorizer(analyzer='word')
#X_train = tfidf.fit_transform(tokenized_corpus)
#tfidf = TfidfVectorizer(analyzer='word', vocabulary=tfidf.vocabulary_)
#X_test = tfidf.fit_transform(tokenized_test_corpus)

y_train = train_frame['label']
y_test = test_frame['label']

#print(y_test.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(X_train.shape)

#ss=StandardScaler()
#X_train=ss.fit_transform(X_train)
#X_test=ss.fit_transform(X_test)


def benchmark(clf_class, params, name):
    print("parameters:", params)
    t0 = time()
    clf = clf_class(**params).fit(X_train, y_train)
    print("done in %fs" % (time() - t0))
    if hasattr(clf, 'coef_'):
        print("Percentage of non zeros coef: %f" % (np.mean(clf.coef_ != 0) * 100))
    print("Predicting the outcomes of the testing set")
    t0 = time()
    pred = clf.predict(X_test)
    print("done in %fs" % (time() - t0))
    print("Classification report on test set for classifier:")
    print(clf)
    print()
    print(classification_report(y_test, pred))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    print("Testbenching a linear classifier...")
    parameters={}
    benchmark(MultinomialNB, parameters, 'naive_bayes')
