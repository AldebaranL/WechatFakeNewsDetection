from time import time
import sys
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

import pandas as pd
import os

data_split = 'test'
path = os.path.join('../datasets/WeFEND-AAAI20-master/data/', data_split)
file_name = os.path.join(path, 'news.csv')
test_frame = pd.read_csv(file_name)
test_frame['Report Content'] = test_frame['Report Content'].apply(lambda x: x.split('##'))


corpus=test_frame['Image Url']
print(corpus.head())
print(type(corpus))

import urllib.request

def download_img(img_url, img_name):
    header = {"Authorization": "Bearer " + "fklasjfljasdlkfjlasjflasjfljhasdljflsdjflkjsadljfljsda"} # 设置http header
    request = urllib.request.Request(img_url, headers=header)
    try:
        response = urllib.request.urlopen(request)
        filename = '../datasets/WeFEND-AAAI20-master/data/test_pics/' + img_name + '.png'
        if (response.getcode() == 200):
            with open(filename, "wb") as f:
                f.write(response.read()) # 将内容写入图片
            return filename
    except:
        return "failed"

i=0
for each in corpus:
    download_img(each,str(i))
    i+=1
print("done")