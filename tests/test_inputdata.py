import pandas as pd
import os
import pdb
import numpy as np
data_split = 'train'
path = os.path.join('../datasets/WeFEND-AAAI20-master/data/', data_split)
file_name = os.path.join(path, 'news.csv')
train_frame = pd.read_csv(file_name)
train_frame['Report Content'] = train_frame['Report Content'].apply(lambda x: x.split('##'))
print("Number of training data is %d, the fake news is %d" %(train_frame.shape[0], train_frame[train_frame['label'] == 1].shape[0]))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(type(train_frame))
print(train_frame.head(1))