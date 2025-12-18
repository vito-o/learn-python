import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from common.functions import sigmoid, softmax   # 激活函数

# 读取数据
def get_data():
    # 1.从文件加载数据集
    data = pd.read_csv('../data/train.csv')
    # 2.划分数据集
    x = data.drop('label', axis=1)
    y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
    # 3.特征工程：归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_test, y_test

# 初始化神经网络
def init_network():
    network = joblib.load('../data/nn_sample')
    return network

# 前向传播
def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 逐层进行计算传递
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

# 主流程
# 1.获取测试数据
x, y = get_data()
print(x.shape)
print(y.shape)

# 2. 创建模型（加载参数）
network = init_network()

# 定义变量
batch_size = 100
accuracy_cnt = 0
n = x.shape[0]

# 3.循环迭代：分彼此做测试，前向传播，并累计预测准确个数
for i in range(0, n, batch_size):
    # 3.1 取出当前批次的数据
    x_batch = x[i:i+batch_size]
    # 3.2 前向传播
    y_batch = forward(network, x_batch)
    # 3.3 将输出分类概率转换为分类标签
    y_pred = np.argmax(y_batch, axis=1)
    # 3.4 累加准确个数
    accuracy_cnt += np.sum(y_pred == y[i:i+batch_size])

print("Accuracy: ", accuracy_cnt / n)