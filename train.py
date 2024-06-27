import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

train_name = "first"
save_path = Path(f"models/{train_name}")
save_path.mkdir(exist_ok=True, parents=True)


def pad_or_trim(data, max_length):
    if data.shape[0] < max_length:
        padding = np.zeros((max_length - data.shape[0], data.shape[1]))
        data = np.vstack((data, padding))
    elif data.shape[0] > max_length:
        data = data[:max_length, :]
    return data


class NpyDataset(Dataset):
    def __init__(self, npy_files, labels, transform=None):
        self.npy_files = npy_files
        self.labels = labels
        self.transform = transform

        self.data = []
        max_length = 0
        for file in npy_files:
            data = np.load(file)
            max_length = max(len(data), max_length)

        with open(save_path / "config.txt", "w") as f:
            f.write(f"{max_length}\n")

        for file in npy_files:
            data = np.load(file)
            data = pad_or_trim(data, max_length)
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        tensor_data = torch.from_numpy(data).float()
        return tensor_data, label


class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return torch.softmax(out, dim=1)


npy_files_all = []
labels_all = []
data_path = Path("data")
for label in data_path.glob("*"):
    npy_files = list(label.glob("*.npy"))
    labels = [int(label.name)] * len(npy_files)
    npy_files_all.extend(npy_files)
    labels_all.extend(labels)

dataset = NpyDataset(npy_files_all, labels_all)
train_size = int(len(dataset) * 0.8)
train_dataset, val_dataset = random_split(
    dataset, [train_size, len(dataset) - train_size]
)

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0")

input_size = 8  # 输入特征数
hidden_size = 64  # LSTM隐藏层大小
num_layers = 2  # LSTM层数
num_classes = 2  # 输出类别数
model = LSTMNet(input_size, hidden_size, num_layers, num_classes)
model.to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in train_dataloader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.6f}"
    )

    if (epoch + 1) % 5 == 0:
        model_path = save_path / f"lstm_model_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")
        model.eval()
        total_loss = 0.0
        total_correct = 0
        for batch in val_dataloader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()

        print(
            f"Val loss at Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_dataloader):.6f}, Accuracy: {total_correct / len(val_dataset):.6f}"
        )
        model.train()
print("Training finished.")
