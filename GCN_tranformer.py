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
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool

BASES = ['A', 'U', 'C', 'G']
BASE_TO_INDEX = {base: idx for idx, base in enumerate(BASES)}
SEQUENCE_LENGTH = 41
NUM_CLASSES = 2
from sklearn.metrics import roc_auc_score
def read_file(directory_path, label):
    file_names = os.listdir(directory_path)
    data_list=[]
    file_index=0
    for file_name in file_names:
        if file_index>=1000:
            break
        file_name = directory_path+'/'+file_name
        df = pd.read_csv(file_name,skiprows=1,header=None,sep='\t')
        df[0]=df[0]-1
        df[2]=df[2]-1
        G = nx.Graph()
        max_len=df.shape[0]
        # 添加边
        for idx, row in df.iterrows():
            G.add_node(row[0], content=row[1])
            if row[2]!=-1:
                G.add_edge(row[0], row[2])
            if row[0]<max_len-1:
                G.add_edge(row[0],row[0]+1)

        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        node_features = []
        for node, attr in G.nodes(data=True):
            # One-hot encoding of nucleotide content
            content_map = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
            feature = [0] * 4
            feature[content_map[attr['content']]] = 1
            node_features.append(feature)
        node_features = torch.tensor(node_features, dtype=torch.float)
        # node_features = node_features.reshape([1,SEQUENCE_LENGTH,4])
        data_list.append(Data(x=node_features, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long)))
        file_index=file_index+1
    return data_list

import torch
import torch.nn as nn
import torch.optim as optim
num_nodes=41

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1,hidden_dim2, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim1)
        self.fc1 = torch.nn.Linear(hidden_dim1, hidden_dim2)  # 图级分类输出
        self.fc2 = torch.nn.Linear(hidden_dim2+4, output_dim)  # 图级分类输出
        encoder_layer = nn.TransformerEncoderLayer(d_model=4, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        # self.dropout = nn.Dropout(p=0.2)


    def forward(self, seq_x, edge_index, batch, batch_size):
        adj_x = self.conv1(seq_x, edge_index)
        adj_x = F.relu(adj_x)
        adj_x = self.conv2(adj_x, edge_index)
        adj_x = F.relu(adj_x)
        # 全局池化
        adj_x = global_mean_pool(adj_x, batch)
        adj_x = F.relu(self.fc1(adj_x))
        seq_x = seq_x.reshape([batch_size,-1,4])
        seq_x = self.transformer(seq_x)  # [batch_size, seq_len, hidden_dim]
        seq_x = seq_x.mean(dim=1)  # Pool over the sequence dimension
        # seq_x = self.dropout(seq_x)
        x = torch.cat((adj_x, seq_x), dim=1)
        return F.log_softmax(self.fc2(x), dim=1)

# Step 4: Train the model
def train_model(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in data_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch,batch.batch_size)
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
            out = model(batch.x, batch.edge_index, batch.batch, batch.batch_size)
            y_pred_proba = out[:, 1]
            auc_roc = roc_auc_score(batch.y, y_pred_proba)
            pred = out.argmax(dim=1)  # 预测类别
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total,auc_roc


# Initialize the model, optimizer, and loss function
model = GCN(input_dim=4, hidden_dim1=64, hidden_dim2=8, output_dim=2)  # 二分类任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

positive_file = read_file('data/mlf/bpseq',1)
negative_file = read_file('data/neg/bpseq',0)

positive_file.extend(negative_file)

data_loader = DataLoader(positive_file, batch_size=5, shuffle=True)

random.shuffle(positive_file)
split_index = int(0.8 * len(positive_file))
train_list = positive_file[:split_index]
test_list = positive_file[split_index:]
train_dataset = DataLoader(train_list, batch_size=50, shuffle=True)
test_dataset = DataLoader(test_list, batch_size=len(test_list), shuffle=True)

# 训练流程
for epoch in range(100):  # 20个训练轮次
    train_loss = train_model(model, train_dataset, optimizer, criterion)
    accuracy,auc_roc = evaluate_model(model, test_dataset)
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f},AUC_ROC: {auc_roc:.4f}")

