import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 加载数据集
df = pd.read_csv('train_set_2500.csv')  # 替换为你的CSV文件路径

# 假设最后一列是二分类标签（'Label'），其他列是特征
X = df.iloc[:, 5:-1]  # 提取所有行，除了最后一列的所有列（即特征）
y = df.iloc[:, -1]  # 提取最后一列作为标签

# 确保标签是整数类型（通常是0和1），如果不是，则需要转换
y = y.astype(int)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化决策树分类器
clf = DecisionTreeClassifier(random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"决策树模型的准确率为: {accuracy}")

# 计算精确率、召回率和F1分数（注意需要指定平均方法，例如'binary'或'micro', 'macro', 'weighted'）
precision = precision_score(y_test, y_pred, average='binary')  # 对于二分类问题，使用'binary'
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"决策树模型的精确率为: {precision}")
print(f"决策树模型的召回率为: {recall}")
print(f"决策树模型的F1分数为: {f1}")

# （可选）保存模型
# import joblib
# joblib.dump(clf, 'decision_tree_model.pkl')

# （可选）加载保存的模型
# clf_loaded = joblib.load('decision_tree_model.pkl')