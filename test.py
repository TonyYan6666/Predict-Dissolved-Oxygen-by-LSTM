import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# 读取数据
file_path = 'data_final.csv'
data = pd.read_csv(file_path)

# 移除 Oxygen、Temperature 或 Streamflow 为 0 的行
data_cleaned = data[(data['Oxygen'] != 0) & (data['Temperature'] != 0) & (data['Streamflow'] != 0)]

# 解析 'Date Time' 为 datetime 格式
data_cleaned['Datetime'] = pd.to_datetime(data_cleaned['Date Time'])

# 按 'Datetime' 排序，然后按时间顺序进行80-20数据集分割
data_cleaned = data_cleaned.sort_values(by='Datetime')
split_index = int(len(data_cleaned) * 0.8)
train_data = data_cleaned.iloc[:split_index]
eval_data = data_cleaned.iloc[split_index:]

# 在每个数据集内按 'Site Number' 和 'Datetime' 排序，确保每个站点的数据是连续的
train_data = train_data.sort_values(by=['Site Number', 'Datetime'])
eval_data = eval_data.sort_values(by=['Site Number', 'Datetime'])

# 选择必要的列（Streamflow, Temperature 用作特征，Oxygen 用作预测目标）并归一化
features = ['Streamflow', 'Temperature']  # 输入特征
target = ['Oxygen']  # 预测目标

# 归一化数据（不包括 'Datetime' 和 'Site Number' 列）
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data[features + target])
scaled_eval_data = scaler.transform(eval_data[features + target])

# 创建数据集函数
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), :-1])  # 使用所有输入特征
        Y.append(dataset[i + look_back, -1])  # 目标是氧气浓度
    return np.array(X), np.array(Y)

# 定义时间步长（例如，10 个时间步长）
look_back = 10
X_train, Y_train = create_dataset(scaled_train_data, look_back)
X_eval, Y_eval = create_dataset(scaled_eval_data, look_back)

# 转换数据为 PyTorch 的张量
X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).float()
X_eval = torch.from_numpy(X_eval).float()
Y_eval = torch.from_numpy(Y_eval).float()

# 创建数据加载器
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
eval_loader = DataLoader(TensorDataset(X_eval, Y_eval), batch_size=32, shuffle=False)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 设置模型参数
input_size = len(features)  # 输入特征数量
hidden_size = 50
num_layers = 2
output_size = 1

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for X_batch, Y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证模型
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in eval_loader:
            outputs = model(X_batch)
            eval_loss += criterion(outputs, Y_batch.unsqueeze(1)).item()
    eval_loss /= len(eval_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Eval Loss: {eval_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), 'lstm_model.pth')
