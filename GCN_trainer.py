import torch
import pandas as pd
import numpy as np
import os
import networkx as nx
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool
def read_file(directory_path, label):
    file_names = os.listdir(directory_path)
    data_list=[]
    i=0
    for file_name in file_names:
        if i>=1000:
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
        data_list.append(Data(x=node_features, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long)))
        i=i+1
    return data_list

positive_file = read_file('data/mlf/bpseq',1)
negative_file = read_file('data/neg/bpseq',0)

positive_file.extend(negative_file)

data_loader = DataLoader(positive_file, batch_size=5, shuffle=True)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)  # 图级分类输出


    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 全局池化
        x = global_mean_pool(x, batch)

        # 分类
        return F.log_softmax(self.fc(x), dim=1)

# Step 4: Train the model
def train_model(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch in data_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
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

    with torch.no_grad():
        for batch in data_loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)  # 预测类别
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total


# Initialize the model, optimizer, and loss function
model = GCN(input_dim=4, hidden_dim=64, output_dim=2)  # 二分类任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 训练流程
for epoch in range(100):  # 20个训练轮次
    train_loss = train_model(model, data_loader, optimizer, criterion)
    accuracy = evaluate_model(model, data_loader)
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")
