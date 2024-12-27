import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

def read_fasta_to_dic(filename):
    """
    function used to parser small fasta
    still effective for genome level file
    """

    with open(filename, "r") as f:
        seq_l=[]
        for n, line in enumerate(f.readlines()):
            if n%3==1:
                seq_l.append(line[:-1])

    return seq_l

# Define constants
BASES = ['A', 'U', 'C', 'G']
BASE_TO_INDEX = {base: idx for idx, base in enumerate(BASES)}
SEQUENCE_LENGTH = 41
NUM_CLASSES = 2



# Step 1: Define Dataset
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # List of sequences (strings of 'A', 'T', 'C', 'G')
        self.labels = labels  # List of labels (0 or 1)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Convert sequence to one-hot encoding
        one_hot = torch.zeros((SEQUENCE_LENGTH, len(BASES)))
        for i, base in enumerate(sequence):
            if base in BASE_TO_INDEX:
                one_hot[i, BASE_TO_INDEX[base]] = 1.0

        return one_hot, torch.tensor(label, dtype=torch.long)


# Step 2: Define Transformer-based Model
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_classes, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)



    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = self.embedding(x)  # [batch_size, seq_len, hidden_dim]
        x = self.transformer(x)  # [batch_size, seq_len, hidden_dim]
        x = x.mean(dim=1)  # Pool over the sequence dimension
        x = self.fc(x)  # [batch_size, num_classes]
        return F.log_softmax(x, dim=1)


# Step 3: Define Training and Evaluation Functions
def train_model(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(data_loader), accuracy


# Step 4: Example Usage
if __name__ == "__main__":
    positive_seq = read_fasta_to_dic('data/mlf/res')
    negative_seq = read_fasta_to_dic('data/mlf/res')

    positive_labels = [1] * len(positive_seq)
    negative_labels = [0] * len(negative_seq)
    positive_seq.extend(negative_seq)
    positive_labels.extend(negative_labels)

    # Hyperparameters
    batch_size = 100
    num_epochs = 10
    learning_rate = 0.001
    hidden_dim = 64
    num_heads = 4

    # Create Dataset and DataLoader
    dataset = SequenceDataset(positive_seq, positive_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = TransformerClassifier(input_dim=len(BASES), num_heads=num_heads, hidden_dim=hidden_dim,
                                  num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, data_loader, optimizer, criterion)
        val_loss, val_acc = evaluate_model(model, data_loader, criterion)
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
