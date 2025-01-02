from torch.nn import functional as F
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
import torch.optim as optim
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score
DEVICE_NUM = 'cuda:0'

BASES = ['A', 'U', 'C', 'G']
BASE_TO_INDEX = {base: idx for idx, base in enumerate(BASES)}
SEQUENCE_LENGTH = 41
NUM_CLASSES = 2
import plotnine as p9
num_nodes = 41


class GraphCNN(nn.Module):
    def __init__(self, conv1_layer, conv2_layer, hidden_layer):
        super(GraphCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_layer, kernel_size=(3, 3), padding=1)  # 输入通道为1，输出通道为16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(conv1_layer, conv2_layer, kernel_size=(3, 3), padding=1)  # 输出通道为32
        self.fc1 = nn.Linear(conv2_layer * (num_nodes // 4) * (num_nodes // 4), hidden_layer)  # 全连接层
        self.fc2 = nn.Linear(hidden_layer, 4)
        self.fc3 = nn.Linear(41, 4)
        self.fc4 = nn.Linear(8, 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=4, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.2)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, batch):
        adj_x = x[:, :, :num_nodes]
        seq_x = x[:, :, num_nodes:]
        adj_x = adj_x.reshape([batch, 1, num_nodes, num_nodes])
        adj_x = self.pool(F.relu(self.conv1(adj_x)))
        adj_x = self.pool(F.relu(self.conv2(adj_x)))
        adj_x = adj_x.view(batch, -1)  # 展平
        # adj_x = self.dropout(adj_x)
        adj_x = F.relu(self.fc1(adj_x))
        adj_x = self.dropout1(adj_x)
        adj_x = F.relu(self.fc2(adj_x))
        # adj_x = self.dropout(adj_x)
        seq_x = self.transformer(seq_x)  # [batch_size, seq_len, hidden_dim]
        seq_x = seq_x.mean(dim=2)  # Pool over the sequence dimension
        # seq_x = self.dropout2(seq_x)
        seq_x = F.relu(self.fc3(seq_x))
        # seq_x = self.dropout(seq_x)
        x = torch.cat((adj_x, seq_x), dim=1)
        return self.Sigmoid(self.fc4(x))


def read_file(directory_path, label):
    file_names = os.listdir(directory_path)
    data_list = []
    num_file = 0

    for file_name in file_names:
        if num_file >= 1000:
            break
        # Read file
        file_path = os.path.join(directory_path, file_name)
        df = pd.read_csv(file_path, skiprows=1, header=None, sep='\t')
        df[0] = df[0] - 1
        df[2] = df[2] - 1

        # Create adjacency matrix
        num_nodes = df.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes))

        # Fill adjacency matrix
        for idx, row in df.iterrows():
            u, v = row[0], row[2]
            if v != -1:
                adj_matrix[u][v] = 1
                adj_matrix[v][u] = 1
            if u < num_nodes - 1:
                adj_matrix[u][u + 1] = 1
                adj_matrix[u + 1][u] = 1
            adj_matrix[u][u] = 1

        # Create sequence features
        sequence = "".join(df[1].values.reshape(-1).tolist())
        one_hot = torch.zeros((SEQUENCE_LENGTH, len(BASES)))
        for i, base in enumerate(sequence):
            if base in BASE_TO_INDEX:
                one_hot[i, BASE_TO_INDEX[base]] = 1.0

        # Combine features
        adj_matrix = np.hstack([adj_matrix, one_hot.numpy()])

        # Convert to tensors and move to device
        x = torch.tensor(adj_matrix, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor([[label]], dtype=torch.float)

        # Create Data object
        data = Data(x=x, y=y)
        data_list.append(data)
        num_file += 1
    return data_list


# 验证函数
def evaluate_model(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        pred_y=torch.tensor([], dtype=torch.float)
        tru_y = torch.tensor([], dtype=torch.float)
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.batch_size).cpu()
            pred_y=torch.cat((out,pred_y), dim=0)
            tru_y = torch.cat((batch.y.cpu(), tru_y), dim=0)
    return pred_y.numpy(),tru_y.numpy()


def set_seed(seed):
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  # 设置 NumPy 随机种子
    random.seed(seed)


positive_file = read_file('fortest/mlftest/bpseq', 1, )
negative_file = read_file('fortest/negtest/bpseq', 0, )
set_seed(42)
device = torch.device(DEVICE_NUM)
positive_file.extend(negative_file)
random.shuffle(positive_file)

feature_list = positive_file

test_dataset = DataLoader(feature_list, batch_size=100)
model = torch.load('model.pth')
pred_y,tru_y = evaluate_model(model, test_dataset, DEVICE_NUM)

precision, recall, thresholds = precision_recall_curve(tru_y, pred_y)
f1 = 2 * (precision * recall) / (precision + recall)

# 创建数据框
pr_df = pd.DataFrame({
    'Threshold': list(thresholds) + [1.0],  # 补全最后一个阈值
    'Precision': precision,
    'Recall': recall,
    'F1': f1
})
pr_df = pr_df.melt(id_vars=['Threshold'], value_vars=['Precision', 'Recall', 'F1'], var_name='Metric', value_name='Value')
print(1)
# 绘制包含 F1、精确率 和 召回率 三条曲线的图
plot = p9.ggplot(pr_df, p9.aes(x='Threshold',y='Value',group='Metric',color='Metric'))  \
    + p9.geom_smooth()\
    + p9.theme_bw() \
    + p9.theme(
    figure_size=(4,4),
    axis_text=p9.element_text(size=12,family='Arial'),
    axis_title=p9.element_text(size=12,family='Arial'),
    panel_grid_minor=p9.element_blank(),
    title=p9.element_text(size=12,family='Arial'),
    strip_background=p9.element_rect(alpha=0),
    strip_text=p9.element_text(size=12,family='Arial'),
    legend_position='bottom'
    )

print(plot)
plot.save('f1.pdf')

# 计算ROC曲线
fpr, tpr, _ = roc_curve(tru_y, pred_y)
auc_score = roc_auc_score(tru_y, pred_y)
roc_df = pd.DataFrame({ 'x': fpr, 'y': tpr })
roc_df['Type']='ROC'
# 计算PR曲线
precision, recall, _ = precision_recall_curve(tru_y, pred_y)
pr_df = pd.DataFrame({ 'x': precision, 'y': recall})
pr_df['Type']='PR'
df = pd.concat([pr_df, roc_df], ignore_index=True)
plot = p9.ggplot(df, p9.aes(x='x',y='y',group='Type',color='Type'))  \
    + p9.geom_smooth()\
    + p9.theme_bw() \
    + p9.theme(
    figure_size=(4,4),
    axis_text=p9.element_text(size=12,family='Arial'),
    axis_title=p9.element_text(size=12,family='Arial'),
    panel_grid_minor=p9.element_blank(),
    title=p9.element_text(size=12,family='Arial'),
    strip_background=p9.element_rect(alpha=0),
    strip_text=p9.element_text(size=12,family='Arial'),
    legend_position='bottom'
    )
plot.save('ROC_PR.pdf')

distribution_df = pd.DataFrame({ 'Prediction': pred_y.reshape(-1,), 'Class': tru_y.reshape(-1,) })
distribution_df['Class']=distribution_df['Class'].astype(int).astype(str)
plot = p9.ggplot(distribution_df, p9.aes(x='Prediction',color='Class',fill='Class'))  \
    + p9.geom_density(alpha=0.5)\
    + p9.theme_bw() \
    + p9.theme(
    figure_size=(4,4),
    axis_text=p9.element_text(size=12,family='Arial'),
    axis_title=p9.element_text(size=12,family='Arial'),
    panel_grid_minor=p9.element_blank(),
    title=p9.element_text(size=12,family='Arial'),
    strip_background=p9.element_rect(alpha=0),
    strip_text=p9.element_text(size=12,family='Arial'),
    legend_position='bottom'
    )
plot.save('distribution.pdf')

