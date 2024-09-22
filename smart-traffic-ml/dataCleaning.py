import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight

# 加载数据
data = pd.read_csv('train_motion_data.csv')
test_data = pd.read_csv('test_motion_data.csv')

# 提取标签列并进行编码，将字符串转化为数字
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(data['Class'])  # 转换训练集标签
y_test = label_encoder.transform(test_data['Class'])  # 转换测试集标签

# 标准化特征
features = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])  # 标准化训练数据
test_data[features] = scaler.transform(test_data[features])  # 标准化测试数据

# 计算类别权重
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# 创建滑动窗口数据函数，提取序列特征
def create_sequences(data, y, window_size):
    sequences = []
    labels = []
    for i in range(len(data) - window_size):
        sequence = data.iloc[i:i + window_size][features].values
        window_labels = y[i:i + window_size]
        label = np.bincount(window_labels).argmax()  # 使用多数标签
        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# 提取滑动窗口序列
window_size = 25
X_train, y_train = create_sequences(data, y_train, window_size)
X_test, y_test = create_sequences(test_data, y_test, window_size)

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 定义 GRU 模型
class CNN_LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(CNN_LSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.batchnorm = nn.BatchNorm1d(64)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 调整维度以适应卷积层
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # 调整回 LSTM 所需的维度
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        final_output = lstm_out[:, -1, :]
        output = self.fc(final_output)
        return output

# 定义超参数
input_size = len(features)
hidden_size =12
output_size = len(np.unique(y_train))
num_layers = 2
num_epochs = 10

# 交叉验证
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kfold.split(X_train_tensor)):
    print(f'Fold {fold + 1}')

    # 将数据划分为训练集和验证集
    X_train_fold, X_val_fold = X_train_tensor[train_index], X_train_tensor[val_index]
    y_train_fold, y_val_fold = y_train_tensor[train_index], y_train_tensor[val_index]

    # 创建数据加载器
    train_dataset_fold = TensorDataset(X_train_fold, y_train_fold)
    val_dataset_fold = TensorDataset(X_val_fold, y_val_fold)

    train_loader_fold = DataLoader(train_dataset_fold, batch_size=32, shuffle=True)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=32, shuffle=False)

    # 定义模型、优化器和损失函数
    model = CNN_LSTMModel(input_size, hidden_size, output_size, num_layers, dropout=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 训练模型
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader_fold:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证集上的损失计算
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader_fold:
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader_fold)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Fold {fold + 1}, Training Loss: {total_loss / len(train_loader_fold):.4f}, Validation Loss: {val_loss:.4f}')

        # 早停判断
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            break

    # 评估该折模型的性能
    def evaluate_model(model, data_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        return 100 * correct / total

    train_accuracy = evaluate_model(model, train_loader_fold)
    val_accuracy = evaluate_model(model, val_loader_fold)
    print(f'Fold {fold + 1}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')
torch.save(model, 'model.pth')
