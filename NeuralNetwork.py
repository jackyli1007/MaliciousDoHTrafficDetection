import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# # 读取数据
# benign = pd.read_csv('l2-benign.csv')
# malicious = pd.read_csv('l2-malicious.csv')
#
# # 标签标注
# benign['Label'] = 0
# malicious['Label'] = 1
#
# # 数据合并，去除空值
# df = pd.concat([benign, malicious]).dropna()

df = pd.read_csv('train_set_2500.csv')
# 特征选择
X = df.iloc[:, 5:-1]  # 选择第6列到倒数第二列（不包括最后一列标签）
y = df.iloc[:, -1]  # 选择最后一列作为标签

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    # 超参数设置


input_size = X.shape[1]
hidden_size = 64
num_classes = 2

# 实例化模型和优化器
model = NeuralNetwork(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0015)
# 确保y_train是一个Tensor
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
# 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    # 梯度归零
    optimizer.zero_grad()

    # 前向传播
    # 在这里使用y_train_tensor而不是y_train
    outputs = model(torch.tensor(X_train, dtype=torch.float32))
    loss = criterion(outputs, y_train_tensor)  # 注意这里使用了y_train_tensor

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 测试模型
def evaluate(result, test_y, name='mymodel'):
    # 转换预测结果和测试标签为numpy数组（如果它们不是的话）
    result = result.numpy()
    test_y = test_y.values

    # 计算评估指标
    acc = np.mean(result == test_y)
    precision = precision_score(test_y, result, average='binary')  # 假设是二分类问题
    recall = recall_score(test_y, result, average='binary')
    f1 = f1_score(test_y, result, average='binary')

    # 打印评估结果
    print(f"{name} 准确率:", acc.round(4))
    print(f"{name} 精确率:", precision.round(4))
    print(f"{name} 召回率:", recall.round(4))
    print(f"{name} F1:", f1.round(4))

# 测试模型并评估
with torch.no_grad():
    # 获取模型预测（只获取类别，不是原始logits）
    predictions = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1)
    # 调用evaluate函数来评估模型
    evaluate(predictions, y_test, name='My Neural Network')

