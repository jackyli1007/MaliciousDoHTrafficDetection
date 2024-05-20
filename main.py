import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report  # 混淆矩阵
from sklearn import metrics

from sklearn import metrics
import seaborn as sns

from sklearn.preprocessing import StandardScaler

import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import dgl
import torch

# mpl.rcParams['font.sans-serif'] = ['FangSong']SimHei
# plt.rcParams['font.sans-serif'] = ['FangSong']	# 显示中文
# plt.rcParams['axes.unicode_minus'] = False		# 显示负号

# 创建预测图目录
import os

# 获取当前工作目录
# 判断imgs文件夹是否存在 保存结果图片
imgs_path = './model'
if not os.path.exists(imgs_path):
    # 如果不存在，则创建imgs文件夹
    os.makedirs(imgs_path)

# 设置随机种子为固定值，这里设为0
torch.manual_seed(42)

#图的边的特征列名
edge_f = ['Duration', 'FlowBytesSent', 'FlowSentRate',
       'FlowBytesReceived', 'FlowReceivedRate', 'PacketLengthVariance',
       'PacketLengthStandardDeviation', 'PacketLengthMean',
       'PacketLengthMedian', 'PacketLengthMode', 'PacketLengthSkewFromMedian',
       'PacketLengthSkewFromMode', 'PacketLengthCoefficientofVariation',
       'PacketTimeVariance', 'PacketTimeStandardDeviation', 'PacketTimeMean',
       'PacketTimeMedian', 'PacketTimeMode', 'PacketTimeSkewFromMedian',
       'PacketTimeSkewFromMode', 'PacketTimeCoefficientofVariation',
       'ResponseTimeTimeVariance', 'ResponseTimeTimeStandardDeviation',
       'ResponseTimeTimeMean', 'ResponseTimeTimeMedian',
       'ResponseTimeTimeMode', 'ResponseTimeTimeSkewFromMedian',
       'ResponseTimeTimeSkewFromMode',
       'ResponseTimeTimeCoefficientofVariation']

#读取两种数据
benign = pd.read_csv('l2-benign.csv')
malicious = pd.read_csv('l2-malicious.csv')
#标签标注：良性：0  恶性：1
benign['Label'] = 0
malicious['Label'] = 1
#数据合并 去除空值  随机筛选2万条数据进行训练和测试
df = pd.concat([benign,malicious]).dropna()
#设置随机抽取数据集数量
sample = 3000
# 筛选标签为0的数据
df_label_0 = df[df['Label'] == 0]
# 筛选标签为1的数据
df_label_1 = df[df['Label'] == 1]
# 从标签为0的数据中随机取sample条
sample_0 = df_label_0.sample(n=sample, random_state=42)  # 设置random_state以确保结果的可复现性
# 从标签为1的数据中随机取sample条
sample_1 = df_label_1.sample(n=sample, random_state=42)
# 合并样本作为训练集
data = pd.concat([sample_0, sample_1])
# 如果需要，保存训练集到CSV文件
data.to_csv(f'train_set_{sample}.csv', index=False)
# data.shape#(6000, 35)
# data.head()#查看特征名
# data.isna().sum()
pd.set_option('display.float_format',lambda x:'{:.3f}'.format(x))    #控制输出为精确到小数点后三位
color = sns.color_palette()
sns.set_style('darkgrid')
#目标和特征热力图
corrmat = data.corr()
f,ax = plt.subplots(figsize=(20,20))
ax.set_xticklabels(corrmat,rotation='horizontal')
sns.heatmap(corrmat, vmax =0.9,square=True)
label_y = ax.get_yticklabels()
plt.setp(label_y , rotation = 360)
label_x = ax.get_xticklabels()
plt.setp(label_x , rotation = 90)
plt.show();
#边特征标准化 消除量纲影响
scaler = StandardScaler()
scaler.fit(data[edge_f])
data[edge_f] = scaler.transform(data[edge_f])
#生成节点字符串
data['src'] = data['SourceIP']+ '.'+data['SourcePort'].astype(str)
data['dst'] = data['DestinationIP']+ '.'+data['DestinationPort'].astype(str)
#合并所有发送和接收节点
nodes = list(set(data['src'].tolist() + data['dst'].tolist()))
nodes[-10:]#输出合并后的ip+端口
#节点特征：使用IP和PORT作为节点特征值
nodes_df = pd.DataFrame(nodes,columns=['node'])
nodes_df_split = nodes_df['node'].str.split('.', expand=True)
node_feature = pd.concat([nodes_df,nodes_df_split],axis=1)
node_feature['id'] = node_feature.index
node_feature.head()
#节点特征标准化
scaler1 = StandardScaler()
scaler1.fit(node_feature[[0,1,2,3,4]])  # data为待标准化的数据
node_feature[[0,1,2,3,4]] = scaler1.transform(node_feature[[0,1,2,3,4]])
#边的数量
len(data[['src','dst']].drop_duplicates())
#补齐节点序号，以便输入模型处理
data = data.drop_duplicates(subset=['src','dst'])
data = pd.merge(data,node_feature[['node','id']],left_on='src',right_on='node')
data = pd.merge(data,node_feature[['node','id']],left_on='dst',right_on='node')
data[data['FlowBytesSent']==0]
#构建图网络模型
src = data['id_x'].values
dst = data['id_y'].values
src = torch.tensor(src,dtype=torch.long)
dst = torch.tensor(dst,dtype=torch.long)
edge_pred_graph = dgl.graph((src, dst))
# 建立点和边特征，以及边的标签
edge_pred_graph.ndata['feature'] = torch.tensor(node_feature[[0,1,2,3,4]].values,dtype=torch.float)
edge_pred_graph.edata['feature'] = torch.tensor(data[edge_f].values,dtype=torch.float)
edge_pred_graph.edata['label'] = torch.tensor(data['Label'].values)
# 进行训练、验证和测试集划分
edge_pred_graph.edata['train_mask'] = torch.zeros(len(src), dtype=torch.bool).bernoulli(0.8)
node_feats = torch.tensor(node_feature[[0,1,2,3,4]].values,dtype=torch.float)
edge_feats = torch.tensor(data[edge_f].values,dtype=torch.float)
edge_labels = torch.tensor(data['Label'].values)
train_mask = torch.zeros(len(src), dtype=torch.bool).bernoulli(0.8)
test_mask = ~train_mask
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='gcn')
        self.conv2 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='gcn')

    def forward(self, graph, inputs):
        # 输入是节点的特征
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h
import dgl.function as fn
class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 3, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        e_f = edges.data['feature']
        # e_f = edges['feature']
        score = self.W(torch.cat([h_u, h_v, e_f], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        # self.pred = DotProductPredictor()
        self.pred = MLPPredictor(5, 2)
        self.S = nn.Softmax()

    def forward(self, g, x):
        h = self.sage(g, x)
        s = self.pred(g, h)
        return self.S(s)
class EdgeClassifier(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, hidden_size, out_size):
        super(EdgeClassifier, self).__init__()

        self.rel_graph_conv = RelGraphConv(hidden_size, hidden_size, num_rels=1)
        self.classifier = nn.Linear(hidden_size * 3, out_size)  # 考虑到节点和边的组合特征
        self.sage_n = SAGE(in_node_feats, hidden_size, hidden_size)
        #self.sage_e = SAGE(in_edge_feats, hidden_size, out_size)
        self.edge_embedding = nn.Linear(in_edge_feats, hidden_size)

    def forward(self, g, node_feats, edge_feats):
        self.node_embedding = self.sage_n(g,node_feats)
        #self.edge_embedding = self.sage_e(g,edge_feats)
        # 节点嵌入
        h_nodes = F.relu(self.node_embedding)

        # 边嵌入
        h_edges = F.relu(self.edge_embedding(edge_feats))

        # 使用DGL的RelGraphConv进行消息传递
        g.ndata['feature'] = h_nodes
        g.edata['feature'] = h_edges
        g.apply_edges(lambda edges: {'feature': torch.cat([edges.src['feature'], edges.dst['feature'], edges.data['feature']], dim=1)})
        h_edges = g.edata['feature']

        # 分类器
        #h_edges = F.relu(self.rel_graph_conv(g, h_edges))
        edge_preds = self.classifier(h_edges)

        return edge_preds
#定义交叉熵损失
criterion = nn.CrossEntropyLoss()
#训练模型，输出损失值loss
from dgl.nn import RelGraphConv
class EarlyStopping():
    def __init__(self,patience=7,verbose=False,delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self,val_loss,model,path):
        #print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
        elif score < self.best_score+self.delta:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
            self.counter = 0
    def save_checkpoint(self,val_loss,model,path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+f'model_checkpoint_{sample}.pth')
        self.val_loss_min = val_loss
early_stopping = EarlyStopping(patience=10,verbose=False)
loss_list = []
val_loss_list = []

acc = []
val_acc = []

epoch = 100
edge_label = edge_pred_graph.edata['label']
model = EdgeClassifier(5, 29, 64, 2)
opt = torch.optim.Adam(model.parameters())
for epoch in range(1, epoch + 2):
    outputs = model(edge_pred_graph, node_feats, edge_feats)
    outputs_max = np.argmax(outputs[train_mask].detach().numpy(), axis=1)
    labels_max = edge_label[train_mask].detach().numpy()
    total_number = 0
    correct_number = 0
    for i in range(len(outputs_max)):
        if outputs_max[i] == labels_max[i]:
            correct_number += 1
        total_number += 1
    acc.append(correct_number / total_number)

    loss = criterion(outputs[train_mask], edge_label[train_mask])
    opt.zero_grad()
    loss.backward()
    opt.step()
    loss_list.append(loss.item())

    model.eval()
    val_outputs = model(edge_pred_graph, node_feats, edge_feats)
    val_outputs_max = np.argmax(val_outputs[test_mask].detach().numpy(), axis=1)
    val_labels_max = edge_label[test_mask].detach().numpy()
    val_total_number = 0
    val_correct_number = 0

    for i in range(len(val_outputs_max)):
        if val_outputs_max[i] == val_labels_max[i]:
            val_correct_number += 1
        val_total_number += 1
    val_acc.append(val_correct_number / val_total_number)
    val_loss = criterion(outputs[test_mask], edge_label[test_mask])
    val_loss_list.append(val_loss.item())

    if epoch % 10 == 0:
        print(
            f'Epoch {epoch} | Train Loss: {loss.item():.4f} Accuracy: {correct_number / total_number:.4f} | Val Loss: {val_loss.item():.4f} Accuracy: {val_correct_number / val_total_number:.4f}')

    # ==================early stopping======================
    early_stopping(val_loss_list[-1], model=model, path=r'model')
    if early_stopping.early_stop:
        print("Early stopping")
        break
epochs = len(loss_list)
#绘制Accuracy图
# plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
# plt.rcParams['axes.unicode_minus'] = False  # 显示负号

plt.figure(figsize=(10, 5))
plt.grid(True, linestyle='--',which='both')  # 设置背景网格线为虚线

ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框
#ax.set_facecolor("white")

plt.plot(range(epochs), loss_list, label='Train Loss',c='r')
plt.plot(range(epochs), val_loss_list, label='Val Loss',c='b')

plt.title('Loss')
plt.xlabel('Train Epoch')
plt.ylabel('Loss')
plt.legend(loc=1) #第二象项，也就是左上角。 loc可以为1,2,3,4

# leg = plt.gca().get_legend()
# ltext = leg.get_texts()
# plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
plt.savefig('Loss.png')
plt.show();
#绘制acc图
plt.figure(figsize=(10, 5))
plt.grid(True, linestyle='--',which='both')  # 设置背景网格线为虚线

plt.plot(range(epochs), acc, label='Train Accuracy',c='r')
plt.plot(range(epochs), val_acc, label='Val Accuracy',c='b')
plt.title('Train & Val Accuracy')
plt.xlabel('Train Epoch')
plt.ylabel('Accuracy')
plt.savefig('Accuracy.png')
plt.legend(loc=4) #第二象项，也就是左上角。 loc可以为1,2,3,4
plt.show();
#读取
my_model = torch.load(f'model/model_checkpoint_{sample}.pth')
pred = model(edge_pred_graph, node_feats, edge_feats)
torch.set_printoptions(sci_mode=False)

y_pred = torch.argmax(pred[test_mask],axis=1).tolist()
y_test = edge_label[test_mask].tolist()
y_prob = F.softmax(pred[test_mask])[:,1].tolist()
print(classification_report(y_test, y_pred))


# 二分类评价指标
def evaluate(result, test_y, probas, name='mymodel'):
    # 1-1 评估指标
    acc = np.mean([result[i] == test_y[i] for i in range(len(result))])
    precision = metrics.precision_score(test_y, result)
    recall = metrics.recall_score(test_y, result)
    f1 = metrics.f1_score(test_y, result)
    # spe = specificity_loss_func(test_y, result)
    print("准确率:", acc.round(4))
    print("精确率:", precision.round(4))
    print("召回率:", recall.round(4))

    print("F1:", f1.round(4))
    # 1-2.混淆矩阵
    confusion_mat = confusion_matrix(test_y, result)

    tn, fp, fn, tp = confusion_mat.ravel()

    # 计算敏感性和特异性
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("特异度:", round(specificity, 4))
    print("敏感度:", round(sensitivity, 4))

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat)
    disp.plot(
        include_values=True,
        cmap=plt.cm.Blues,
        ax=None,
        xticks_rotation="horizontal",
        values_format=".2f"
    )
    plt.title(f"{name} confusion matrix")
    plt.grid(False)
    plt.show();

    # ROC曲线
    roc = metrics.roc_auc_score(test_y, probas)
    print("AUC值:", roc.round(4))
    fpr, tpr, thresholds = metrics.roc_curve(test_y, probas)
    plt.plot(fpr, tpr, label=f"{name} ROC curve (area={round(roc, 2)})")
    plt.plot([0, 1], [0, 1], linestyle='dashed')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} ROC")
    plt.legend(loc='lower right')
    plt.grid(linestyle='-.')
    plt.grid(True)
    plt.show();

    # 敏感性特异性曲线
    plt.plot(thresholds, 1 - fpr, label='specificity')
    plt.plot(thresholds, tpr, label='sensitivity')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Classification Proportion')
    plt.xlabel('Cutoff')
    plt.legend()
    plt.show()

    # PR曲线

    precision, recall, thresholds = metrics.precision_recall_curve(test_y, probas)
    plt.plot(recall, precision, color='y', label=name)
    plt.title(f'{name} Precision-recall curve')
    plt.legend(loc='lower left')
    plt.plot([1, 0], [0, 1], 'r--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.grid(linestyle='-.')
    plt.grid(True)
    plt.show()
    return
evaluate(y_pred, y_test, y_prob, name = 'model')
