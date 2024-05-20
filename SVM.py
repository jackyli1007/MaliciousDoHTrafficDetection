import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('train_set_2500.csv')
# 特征选择
X = df.iloc[:, 5:-1]  # 选择第6列到倒数第二列（不包括最后一列标签）
y = df.iloc[:, -1]  # 选择最后一列作为标签

# 确保标签是整数类型（通常是0和1），如果不是，则需要转换
y = y.astype(int)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# 初始化SVM分类器（例如使用线性核）
clf = svm.SVC(kernel='linear', random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM模型的准确率为: {accuracy}")

# 计算精确率、召回率和F1分数（注意需要指定平均方法）
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"SVM模型的精确率为: {precision}")
print(f"SVM模型的召回率为: {recall}")
print(f"SVM模型的F1分数为: {f1}")

# （可选）如果你想要保存模型以供将来使用
# import joblib
# joblib.dump(clf, 'svm_model.pkl')

# （可选）如果你想要加载保存的模型
# clf_loaded = joblib.load('svm_model.pkl')