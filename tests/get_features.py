import jieba
import pandas as pd

def get_featuers():
    #读入数据
    train_data = pd.read_csv('../datasets/WeFEND-AAAI20-master/data/train/news.csv')
    train_data['Report Content'] = [x.split('##') for x in train_data['Report Content']]
    test_data = pd.read_csv('../datasets/WeFEND-AAAI20-master/data/test/news.csv')
    test_data['Report Content'] = [x.split('##') for x in test_data['Report Content']]
    print("finish read data")

    #读入预训练word2Vec
    f = open("../datasets/sgns.sogounews.bigram-char","r",encoding='UTF-8')
    t=f.readline().split()
    n,dimension=int(t[0]),int(t[1])
    print(dimension)
    chinesewordvec=f.readlines()
    chinesewordvec=[i.split() for i in chinesewordvec]
    vectorsmap=[]
    wordtoindex={}
    indextoword={}
    for i in range(n):
        vectorsmap.append(list(map(float,chinesewordvec[i][len(chinesewordvec[i])-dimension:])))
        wordtoindex[chinesewordvec[i][0]]=i
        indextoword[i]=chinesewordvec[i][0]
    f.close()
    print("finish reading cwv")

    #jieba分词，对应vec向量，生成features_train、features_test
    #vectorsmap=DataFrame(vectorsmap)
    features_train = []
    features_test=[]
    for text in train_data['Title']:
        wordfeature=[]
        for word in jieba.cut(text):
            if word in wordtoindex:
                wordfeature.append(vectorsmap[wordtoindex[word]])
        features_train.append(wordfeature)

    for text in test_data['Title']:
        wordfeature=[]
        for word in jieba.cut(text):
            if word in wordtoindex:
                wordfeature.append(vectorsmap[wordtoindex[word]])
        features_test.append(wordfeature)
    print("finish creat features")

    #存入文件
    import json
    features_train = json.dumps(features_train)
    file_feature_train = open("./features_train.txt", "w",encoding='UTF-8')
    file_feature_train.write(features_train)
    file_feature_train.close()

    features_test = json.dumps(features_test)
    file_feature_test = open("./features_test.txt", "w",encoding='UTF-8')
    file_feature_test.write(features_test)
    file_feature_test.close()

    print("finish write festures")
if __name__=='__main__':
    get_featuers()