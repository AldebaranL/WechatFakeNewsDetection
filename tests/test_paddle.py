import paddle

# 文件路径
from paddle import int64

path_to_file = './t8.shakespeare.txt'
test_sentence = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# 文本长度是指文本中的字符个数
print ('Length of text: {} characters'.format(len(test_sentence)))

from string import punctuation
process_dicts={i:'' for i in punctuation}
print(process_dicts)

punc_table = str.maketrans(process_dicts)
test_sentence = test_sentence.translate(punc_table)

test_sentence_list = test_sentence.lower().split()

word_dict_count = {}
for word in test_sentence_list:
    word_dict_count[word] = word_dict_count.get(word, 0) + 1

word_list = []
soted_word_list = sorted(word_dict_count.items(), key=lambda x: x[1], reverse=True)
for key in soted_word_list:
    word_list.append(key[0])

word_list = word_list[:2500]
print(len(word_list))

# 设置参数
hidden_size = 1024               # Linear层 参数
embedding_dim = 256              # embedding 维度
batch_size = 256                 # batch size 大小
context_size = 2                 # 上下文长度
vocab_size = len(word_list) + 1  # 词表大小
epochs = 2                       # 迭代轮数

trigram = [[[test_sentence_list[i], test_sentence_list[i + 1]], test_sentence_list[i + 2]]
           for i in range(len(test_sentence_list) - 2)]

word_to_idx = {word: i+1 for i, word in enumerate(word_list)}
word_to_idx['<pad>'] = 0
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

# 看一下数据集
print(trigram[:3])

import numpy as np


class TrainDataset(paddle.io.Dataset):
    def __init__(self, tuple_data):
        self.tuple_data = tuple_data

    def __getitem__(self, idx):
        data = self.tuple_data[idx][0]
        label = self.tuple_data[idx][1]
        data = np.array(list(map(lambda word: word_to_idx.get(word, 0), data)))
        label = np.array(word_to_idx.get(label, 0))
        return data, label

    def __len__(self):
        return len(self.tuple_data)


train_dataset = TrainDataset(trigram)

# 加载数据
train_loader = paddle.io.DataLoader(train_dataset, return_list=True, shuffle=True,
                                    batch_size=batch_size, drop_last=True)
import paddle.nn.functional as F

class NGramModel(paddle.nn.Layer):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramModel, self).__init__()
        self.embedding = paddle.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.linear1 = paddle.nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = paddle.nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = paddle.reshape(x, [-1, context_size * embedding_dim])
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

import paddle.nn.functional as F
losses = []
def train(model):
    model.train()
    optim = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            print(type(y_data))
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            loss.backward()
            if batch_id % 500 == 0:
                losses.append(loss.numpy())
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
            optim.step()
            optim.clear_grad()
model = NGramModel(vocab_size, embedding_dim, context_size)
train(model)