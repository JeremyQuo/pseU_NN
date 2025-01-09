import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import functional as F
import random


def set_seeds(seed=42):
    """
    Set seeds for reproducibility.
    Args:
        seed (int): seed value
    """
    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


torch.manual_seed(42)
print(torch.rand(3))

DEVICE_NUM = 'cuda:0'

BASES = ['A', 'U', 'C', 'G']
BASE_TO_INDEX = {base: idx for idx, base in enumerate(BASES)}
SEQUENCE_LENGTH = 41
NUM_CLASSES = 2
from sklearn.metrics import roc_auc_score


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





# Step 4: Train the model
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.batch_size)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print(total_loss)

    return total_loss / len(data_loader)


# 验证函数
def evaluate_model(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.batch_size)
            auc_roc = roc_auc_score(batch.y.cpu(), out.cpu())
    return auc_roc


positive_file = read_file('pseU_NN/data/mlf/bpseq', 1, )
negative_file = read_file('pseU_NN/data/neg/bpseq', 0, )

device = torch.device(DEVICE_NUM)
positive_file.extend(negative_file)
random.shuffle(positive_file)

feature_list = positive_file
k = 5  # K 的值
roc_auc_list = []
for i in range(k):
    # set_seeds(42)
    unit_number = len(feature_list) // k
    test_list = feature_list[i * unit_number:(i + 1) * unit_number]
    train_list = feature_list[0:i * unit_number]
    train_list.extend(feature_list[(i + 1) * unit_number:])

    model = GraphCNN(32, 64, 128).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # optimizer=optim.SGD(model.parameters(),lr=0.001)
    train_dataset = DataLoader(train_list, batch_size=50)
    test_dataset = DataLoader(test_list, batch_size=len(test_list))
    best_roc_auc = 0
    for epoch in range(200):

        train_loss = train_model(model, train_dataset, optimizer, criterion, DEVICE_NUM)
        auc_roc = evaluate_model(model, test_dataset, DEVICE_NUM)
        if auc_roc > best_roc_auc:
            best_roc_auc = auc_roc
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f},AUC_ROC: {auc_roc:.4f}")
    if best_roc_auc >= 0.8:
        torch.save(model, 'pseuNN1_3/model.pth')
    roc_auc_list.append(best_roc_auc)

print(np.round(np.mean(roc_auc_list), 3))
meanauc = np.round(np.mean(roc_auc_list), 3) * 100

from sklearn.model_selection import KFold


def check_fold_distribution(feature_list, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(feature_list)):
        train_list = [feature_list[i] for i in train_idx]
        test_list = [feature_list[i] for i in test_idx]
        print(f"Fold {fold}:")
        print(f"Train size: {len(train_list)}, Test size: {len(test_list)}")
        # 检查标签分布
        train_labels = [x[1] for x in train_list]
        test_labels = [x[1] for x in test_list]
        print(f"Train label distribution: {np.bincount(train_labels)}")
        print(f"Test label distribution: {np.bincount(test_labels)}\n")


def cross_validation(feature_list, k=5, epochs=300, batch_size=600, lr=0.001):
    roc_auc_list = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(feature_list)):
        # 数据划分
        train_list = [feature_list[i] for i in train_idx]
        test_list = [feature_list[i] for i in test_idx]

        # 数据加载器
        train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_list, batch_size=len(test_list), shuffle=False)

        # 模型初始化
        model = GraphCNN().to(device)
        criterion = nn.BCELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 早停
        best_roc_auc = 0
        patience = 50
        no_improve = 0

        for epoch in range(epochs):
            train_loss = train_model(model, train_loader, optimizer, criterion, DEVICE_NUM)
            auc_roc = evaluate_model(model, test_loader, DEVICE_NUM)

            if auc_roc > best_roc_auc:
                best_roc_auc = auc_roc
                no_improve = 0
            else:
                no_improve += 1

            print(f"Fold {fold + 1}, Epoch {epoch + 1}, Loss: {train_loss:.4f}, AUC_ROC: {auc_roc:.4f}")

            # 早停判断
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        if best_roc_auc > 0.87:
            torch.save(model, 'model.pth')
        roc_auc_list.append(best_roc_auc)
        print(f"Fold {fold + 1} Best AUC_ROC: {best_roc_auc:.4f}")

    mean_auc = np.mean(roc_auc_list)
    std_auc = np.std(roc_auc_list)
    print(f"\nMean AUC_ROC: {mean_auc:.3f} ± {std_auc:.3f}")
    return roc_auc_list


check_fold_distribution(feature_list, k=5)

results = cross_validation(
    feature_list=positive_file,
    k=5,
    epochs=300,
    batch_size=600,
    lr=0.001
)

bestmodel = torch.load('best.pth')

modelinfo = bestmodel.state_dict()['fc3.weight']
bestmodel['fc3.weight']

# Hyperparameters
config = {
    'learning_rate': 0.001,  # Slightly conservative for better stability
    'weight_decay': 1e-4,  # Standard L2 regularization
    'batch_size': 64,  # Better for generalization than 64
    'num_epochs': 800,  # Your original setting
    'early_stopping_patience': 100,
    'scheduler_patience': 50,
    'scheduler_factor': 0.5,
    'min_lr': 1e-4
}


# Early Stopping Implementation
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# Modified main training loop
unit_number = len(feature_list) // k
roc_auc_list = []

for i in range(k):  # k-fold cross validation
    # Split data
    test_list = feature_list[i * unit_number:(i + 1) * unit_number]
    train_list = feature_list[0:i * unit_number]
    train_list.extend(feature_list[(i + 1) * unit_number:])

    # Initialize model and training components
    model = GraphCNN().to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        # weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['scheduler_factor'],  # Standard reduction factor (0.1 = reduce lr by 10x)
        patience=config['scheduler_patience'],  # Wait 3 epochs before reducing lr
        min_lr=config['min_lr'],  # Minimum learning rate threshold
        verbose=True
    )

    # Data loaders
    train_dataset = DataLoader(
        train_list,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    test_dataset = DataLoader(
        test_list,
        batch_size=len(test_list),
        shuffle=True,
        pin_memory=True
    )

    # Training tracking
    best_roc_auc = 0
    early_stopper = EarlyStopping(patience=config['early_stopping_patience'])

    # Training loop
    for epoch in range(config['num_epochs']):
        train_loss = train_model(model, train_dataset, optimizer, criterion, DEVICE_NUM)
        auc_roc = evaluate_model(model, test_dataset, DEVICE_NUM)

        # Update learning rate
        scheduler.step(auc_roc)

        # Save best model
        if auc_roc > best_roc_auc:
            best_roc_auc = auc_roc
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc_roc': best_roc_auc,
            }, f'trainedmodel/best_model_fold_{i}.pth')

        print(f"Fold {i + 1}, Epoch {epoch + 1}, Loss: {train_loss:.4f}, AUC_ROC: {auc_roc:.4f}")

        # Early stopping check
        early_stopper(auc_roc)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    roc_auc_list.append(best_roc_auc)

# Print final results
print("\nCross-validation Results:")
for i, auc in enumerate(roc_auc_list):
    print(f"Fold {i + 1}: AUC-ROC = {auc:.4f}")
print(f"Average AUC-ROC: {sum(roc_auc_list) / len(roc_auc_list):.4f}")
print(f"Standard Deviation: {np.std(roc_auc_list):.4f}")

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'auc_roc': best_roc_auc,
}, f'best_model_fold_{i}.pth')

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,
}, f'trainedmodel/cnn12_30_{meanauc}.pth')
