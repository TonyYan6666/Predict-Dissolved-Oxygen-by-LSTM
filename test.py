import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


# 数据预处理部分
def preprocess_data(training_data, evaluation_data):
    # 合并训练集和验证集的特征和标签
    features = ['Streamflow', 'Temperature']
    label = ['Oxygen']

    # 从训练集和验证集中提取特征和标签
    train_features = training_data[features]
    train_label = training_data[label]
    eval_features = evaluation_data[features]
    eval_label = evaluation_data[label]

    # 对数转换
    train_features_transformed = train_features.apply(lambda x: np.log10(x + 0.01))
    train_label_transformed = train_label.apply(lambda x: np.log10(x + 0.01))
    eval_features_transformed = eval_features.apply(lambda x: np.log10(x + 0.01))
    eval_label_transformed = eval_label.apply(lambda x: np.log10(x + 0.01))

    # 标准化（使用训练集的均值和标准差进行标准化）
    train_features_standardized = (
                                              train_features_transformed - train_features_transformed.mean()) / train_features_transformed.std()
    train_label_standardized = (
                                           train_label_transformed - train_label_transformed.mean()) / train_label_transformed.std()

    eval_features_standardized = (
                                             eval_features_transformed - train_features_transformed.mean()) / train_features_transformed.std()
    eval_label_standardized = (eval_label_transformed - train_label_transformed.mean()) / train_label_transformed.std()

    # 创建掩码，标记值为0的位置
    train_feature_mask = (train_features == 0).astype(int)
    train_label_mask = (train_label == 0).astype(int)
    eval_feature_mask = (eval_features == 0).astype(int)
    eval_label_mask = (eval_label == 0).astype(int)

    return (train_features_standardized, train_label_standardized, train_feature_mask, train_label_mask), \
        (eval_features_standardized, eval_label_standardized, eval_feature_mask, eval_label_mask)


# PyTorch Dataset类
class TimeSeriesDataset(Dataset):
    def __init__(self, features, labels, feature_mask, label_mask):
        # 将数据转换为张量
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)
        self.feature_mask = torch.tensor(feature_mask.values, dtype=torch.float32)
        self.label_mask = torch.tensor(label_mask.values, dtype=torch.float32)

    def __len__(self):
        # 返回数据集的大小
        return len(self.features)

    def __getitem__(self, idx):
        # 根据索引返回特征、标签和掩码
        return self.features[idx], self.labels[idx], self.feature_mask[idx], self.label_mask[idx]


# 定义一个简单的LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 全连接层
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# 自定义损失函数以支持掩码
def masked_mse_loss(output, target, mask):
    mask = mask.to(torch.bool)  # 将掩码转换为布尔类型
    loss = ((output - target) ** 2)
    loss = loss * mask  # 只计算非掩码区域的损失
    return loss.sum() / mask.sum()  # 返回掩码区域的平均损失


# 读取数据
file_path = 'data_final.csv'  # 请替换为您的文件路径
data = pd.read_csv(file_path)

# 将 'Date Time' 列转换为日期时间格式
data['Date Time'] = pd.to_datetime(data['Date Time'])

# 找到80%分位数的日期作为划分点
split_date = data['Date Time'].quantile(0.8)

# 按日期划分数据集为训练集和验证集
training_data = data[data['Date Time'] <= split_date]
evaluation_data = data[data['Date Time'] > split_date]

# 数据预处理
(train_features_standardized, train_label_standardized, train_feature_mask, train_label_mask), \
    (eval_features_standardized, eval_label_standardized, eval_feature_mask, eval_label_mask) = preprocess_data(
    training_data, evaluation_data)

# 创建 PyTorch 数据集
train_dataset = TimeSeriesDataset(train_features_standardized, train_label_standardized, train_feature_mask,
                                  train_label_mask)
eval_dataset = TimeSeriesDataset(eval_features_standardized, eval_label_standardized, eval_feature_mask,
                                 eval_label_mask)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

# 设置参数
input_size = len(['Streamflow', 'Temperature'])  # 输入特征数量
hidden_size = 64  # 隐藏层大小
output_size = 1  # 输出大小（氧气浓度）
num_layers = 1  # LSTM层数

# 实例化模型
model = LSTMModel(input_size, hidden_size, output_size, num_layers)

# 定义损失函数和优化器
criterion = masked_mse_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for features, labels, feature_mask, label_mask in train_loader:
        # 将数据移动到GPU（如果可用）
        features, labels, label_mask = features.to(device), labels.to(device), label_mask.to(device)

        # 前向传播
        outputs = model(features.unsqueeze(1))  # 将数据形状调整为 (batch_size, seq_len, input_size)
        loss = criterion(outputs, labels, label_mask)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    for features, labels, feature_mask, label_mask in eval_loader:
        features, labels, label_mask = features.to(device), labels.to(device), label_mask.to(device)

        outputs = model(features.unsqueeze(1))  # 将数据形状调整为 (batch_size, seq_len, input_size)
        loss = criterion(outputs, labels, label_mask)
        print(f'Evaluation Loss: {loss.item():.4f}')
