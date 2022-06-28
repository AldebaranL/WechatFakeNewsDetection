import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TRAIN_STEPS=20

# Prepare train data
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

print(train_X.shape)

w=tf.Variable(initial_value=1.0)
b=tf.Variable(initial_value=1.0)

optimizer=tf.keras.optimizers.SGD(0.1)
mse=tf.keras.losses.MeanSquaredError()

for i in range(TRAIN_STEPS):
    print("epoch:",i)
    print("w:", w.numpy())
    print("b:", b.numpy())
    #计算和更新梯度
    with tf.GradientTape() as tape:
        logit = w * train_X + b
        loss=mse(train_Y,logit)
    gradients=tape.gradient(target=loss,sources=[w,b])  #计算梯度
    #print("gradients:",gradients)
    #print("zip:\n",list(zip(gradients,[w,b])))
    optimizer.apply_gradients(zip(gradients,[w,b]))     #更新梯度


#draw
plt.plot(train_X,train_Y,"+")
plt.plot(train_X,w * train_X + b)
plt.show()
'''
# 导入pandas与numpy工具包。
import pandas as pd
import numpy as np

# 创建特征列表。
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

# 使用pandas.read_csv函数从互联网读取指定数据。
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names = column_names )

# 将?替换为标准缺失值表示。
data = data.replace(to_replace='?', value=np.nan)
# 丢弃带有缺失值的数据（只要有一个维度有缺失）。
data = data.dropna(how='any')

# 输出data的数据量和维度。
#print(data.shape)
# 使用sklearn.cross_valiation里的train_test_split模块用于分割数据。
from sklearn.model_selection import train_test_split

# 随机采样25%的数据用于测试，剩下的75%用于构建训练集合。
x_train, x_test, y_train, y_test = train_test_split(data[column_names[1:3]], data[column_names[10]], test_size=0.25, random_state=33)

#print(x_train)
x_train=np.float32(x_train.T)
x_test=np.float32(x_test.T)
y_train=np.float32(y_train.T)
y_test=np.float32(y_test.T)

#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)
import tensorflow as tf
#import tensorflow.compat.v1 as tf

b=tf.Variable(tf.zeros([1]))
w=tf.Variable(tf.random.uniform([1,2],-1.0,1.0))
y=tf.matmul(w, x_train)+b
loss=tf.reduce_mean(tf.square(y-y_train))
#tf.disable_v2_behavior()
#tf.compat.v1.disable_eager_execution()
optimizer=tf.keras.optimizers.Adam(0.01)
train=optimizer.minimize(loss,var_list=[b,w])

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
for step in xrange(0,1000):
    sess.run(train)
    if step%200==0:
        print(step,sess.run(w),sess.run(b))
test_negative=test.loc[test['Type']==0][['Clump Thickness','Cell Size']]
test_positive=test.loc[test['Type']==0][['Clump Thickness','Cell Size']]


import matplotlib.pyplot as plt
plt.scatter(test_negative['Clump Thickness'], test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(test_positive['Clump Thickness'], test_positive['Cell Size'],marker='x',s=150,c='black')
plt.xlabel('Clump Thickness')
plt.ylabel('Cell Size')
lx=np.arange(0,12)
ly=(0.5-sess.run(b)-lx*sess.run(w)[0][0])/sess.run(w)[0][1]
plt.plot(lx,ly,color='green')

plt.show()

'''