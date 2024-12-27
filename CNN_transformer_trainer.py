import torch
import pandas as pd
import numpy as np
import os
import networkx as nx
from torch_geometric.data import Data, DataLoader
from torch.nn import functional as F
import random
# 设置随机种子
seed = 24
torch.manual_seed(seed)          # 设置 PyTorch 随机种子
np.random.seed(seed)             # 设置 NumPy 随机种子
random.seed(seed)

BASES = ['A', 'U', 'C', 'G']
BASE_TO_INDEX = {base: idx for idx, base in enumerate(BASES)}
SEQUENCE_LENGTH = 41
NUM_CLASSES = 2
from sklearn.metrics import roc_auc_score
def read_file(directory_path, label):
    file_names = os.listdir(directory_path)
    data_list=[]
    num_file=0
    for file_name in file_names:
        if num_file>=1000:
            break
        num_file = num_file + 1
        file_name = directory_path+'/'+file_name
        df = pd.read_csv(file_name,skiprows=1,header=None,sep='\t')
        df[0]=df[0]-1
        df[2]=df[2]-1
        # 添加边
        num_nodes = df.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes))
        # 填充邻接矩阵
        for idx, row in df.iterrows():
            u=row[0]
            v=row[2]
            if v!=-1:
                adj_matrix[u][v] = 1
                adj_matrix[v][u] = 1
            if u<num_nodes-1:
                adj_matrix[u][u+1] = 1
                adj_matrix[u+1][u] = 1
            adj_matrix[u][u]=1
        # 写入序列特征
        sequence="".join(df[1].values.reshape(-1).tolist())
        one_hot = torch.zeros((SEQUENCE_LENGTH, len(BASES)))
        for i, base in enumerate(sequence):
            if base in BASE_TO_INDEX:
                one_hot[i, BASE_TO_INDEX[base]] = 1.0
        adj_matrix = np.hstack([adj_matrix,one_hot])
        # data_list.append(Data(x=[adj_matrix,one_hot], y=torch.tensor([label], dtype=torch.long)))
        data_list.append(Data(
            x=torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0), y=torch.tensor([label], dtype=torch.long)))

    return data_list


import torch
import torch.nn as nn
import torch.optim as optim
num_nodes=41
class GraphCNN(nn.Module):
    def __init__(self):
        super(GraphCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)  # 输入通道为1，输出通道为16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)  # 输出通道为32
        self.fc1 = nn.Linear(8 * (num_nodes // 2) * (num_nodes // 2), 128)  # 全连接层
        self.fc2 = nn.Linear(128, 4)
        self.fc3 = nn.Linear(8, 2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=4, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, batch):
        adj_x = x[:,:, :num_nodes]
        seq_x = x[:,:, num_nodes:]
        adj_x = adj_x.reshape([batch,1,num_nodes,num_nodes])
        adj_x = self.pool(F.relu(self.conv1(adj_x)))
        adj_x = self.pool(F.relu(self.conv2(adj_x)))
        adj_x = adj_x.view(-1, 8 * (num_nodes // 2) * (num_nodes // 2))  # 展平
        adj_x = F.relu(self.fc1(adj_x))
        adj_x = F.relu(self.fc2(adj_x))
        adj_x = self.dropout(adj_x)
        seq_x = self.transformer(seq_x)  # [batch_size, seq_len, hidden_dim]
        seq_x = seq_x.mean(dim=1)  # Pool over the sequence dimension
        seq_x = self.dropout(seq_x)
        x = torch.cat((adj_x, seq_x), dim=1)
        return F.log_softmax(self.fc3(x), dim=1)



# Step 4: Train the model
def train_model(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in data_loader:
        optimizer.zero_grad()
        out = model(batch.x,batch.batch_size)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print(total_loss)

    return total_loss / len(data_loader)

# 验证函数
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    # 只取正类的概率
    #
    # # 计算 AUC-ROC
    # auc_roc = roc_auc_score(y_test, y_pred_proba)

    # return auc_roc
    with torch.no_grad():
        for batch in data_loader:
            out = model(batch.x, batch.batch_size)
            y_pred_proba = out[:, 1]
            auc_roc = roc_auc_score(batch.y, y_pred_proba)
            pred = out.argmax(dim=1)  # 预测类别
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total,auc_roc

positive_file = read_file('data/mlf/bpseq',1)
negative_file = read_file('data/neg/bpseq',0)

positive_file.extend(negative_file)

random.shuffle(positive_file)
split_index = int(0.8 * len(positive_file))
train_list = positive_file[:split_index]
test_list = positive_file[split_index:]
train_dataset = DataLoader(train_list, batch_size=20, shuffle=True)
test_dataset = DataLoader(test_list, batch_size=len(test_list), shuffle=True)
# 实例化模型
model = GraphCNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练流程
for epoch in range(100):  # 20个训练轮次
    train_loss = train_model(model, train_dataset, optimizer, criterion)
    accuracy,auc_roc = evaluate_model(model, test_dataset)
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f},AUC_ROC: {auc_roc:.4f}")

