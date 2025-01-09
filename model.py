from torch.nn import functional as F
import torch
import torch.nn as nn

class GraphCNN(nn.Module):
    def __init__(self, conv1_layer, conv2_layer, hidden_layer,num_nodes):
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
        self.num_nodes=num_nodes

    def forward(self, x, batch):
        adj_x = x[:, :, :self.num_nodes]
        seq_x = x[:, :, self.num_nodes:]
        adj_x = adj_x.reshape([batch, 1, self.num_nodes, self.num_nodes])
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
