import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ========== 数据读取和预处理 ==========

# 读取数据
file_path = 'data_final.csv'
data = pd.read_csv(file_path)

# 移除 Oxygen、Temperature 或 Streamflow 为 0 的行
data_cleaned = data[(data['Oxygen'] != 0) & (data['Temperature'] != 0) & (data['Streamflow'] != 0)]

# 解析 'Date Time' 为 datetime 格式
data_cleaned['Datetime'] = pd.to_datetime(data_cleaned['Date Time'])

# 按 'Datetime' 排序，然后按时间顺序进行80-20数据集分割
data_cleaned = data_cleaned.sort_values(by='Datetime').reset_index(drop=True)
split_index = int(len(data_cleaned) * 0.8)
train_data = data_cleaned.iloc[:split_index].reset_index(drop=True)
eval_data = data_cleaned.iloc[split_index:].reset_index(drop=True)

# 在每个数据集内按 'Site Number' 和 'Datetime' 排序，确保每个站点的数据是连续的
train_data = train_data.sort_values(by=['Site Number', 'Datetime']).reset_index(drop=True)
eval_data = eval_data.sort_values(by=['Site Number', 'Datetime']).reset_index(drop=True)

# 选择必要的列（Streamflow, Temperature 用作特征，Oxygen 用作预测目标）
features = ['Streamflow', 'Temperature']  # 输入特征
target = ['Oxygen']  # 预测目标

# 使用 StandardScaler 归一化数据（不包括 'Datetime' 和 'Site Number' 列）
scaler = StandardScaler()
scaler.fit(train_data[features + target])

scaled_train_data = train_data.copy()
scaled_eval_data = eval_data.copy()
scaled_train_data[features + target] = scaler.transform(train_data[features + target])
scaled_eval_data[features + target] = scaler.transform(eval_data[features + target])

# 定义根据站点创建数据集的函数
def create_dataset_by_site(data, look_back=1):
    X, Y = [], []
    site_numbers = data['Site Number'].unique()
    for site in site_numbers:
        site_data = data[data['Site Number'] == site].reset_index(drop=True)
        data_array = site_data[features + target].values
        for i in range(len(data_array) - look_back):
            X.append(data_array[i:i + look_back, :-1])
            Y.append(data_array[i + look_back, -1])
    return np.array(X), np.array(Y)

# 定义时间步长（例如，10 个时间步长）
look_back = 10

# 创建训练和验证数据集
X_train, Y_train = create_dataset_by_site(scaled_train_data, look_back)
X_eval, Y_eval = create_dataset_by_site(scaled_eval_data, look_back)

# 转换数据为 PyTorch 的张量
X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).float()
X_eval = torch.from_numpy(X_eval).float()
Y_eval = torch.from_numpy(Y_eval).float()

# 创建数据加载器，不打乱数据顺序
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=False, drop_last=True)
eval_loader = DataLoader(TensorDataset(X_eval, Y_eval), batch_size=batch_size, shuffle=False, drop_last=True)

# ========== 模型定义 ==========

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 设置模型参数
input_size = len(features)  # 输入特征数量
hidden_size = 50
num_layers = 2
output_size = 1

# 使用 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# ========== 模型训练 ==========

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    train_loss = 0  # 初始化训练损失

    for X_batch, Y_batch in train_loader:
        # 将数据移动到设备（CPU 或 GPU）
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), Y_batch)
        loss.backward()
        optimizer.step()

        # 计算训练损失的总和
        train_loss += loss.item()

    # 计算训练集的平均损失
    train_loss /= len(train_loader)

    # 验证模型
    model.eval()
    eval_loss = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for X_batch, Y_batch in eval_loader:
            # 将数据移动到设备（CPU 或 GPU）
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), Y_batch)
            eval_loss += loss.item()

            # 收集预测值和真实值
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(Y_batch.cpu().numpy())

    # 计算验证集的平均损失
    eval_loss /= len(eval_loader)

    # 计算额外的评估指标
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    # 调整学习率
    scheduler.step(eval_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

# 保存模型
torch.save(model.state_dict(), 'lstm_model.pth')

# ========== 模型评估 ==========

# 在验证集上进行预测并反标准化
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for X_batch, Y_batch in eval_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        predictions.extend(outputs.squeeze().cpu().numpy())
        actuals.extend(Y_batch.cpu().numpy())

# 将预测值和真实值转换为 NumPy 数组
predictions = np.array(predictions)
actuals = np.array(actuals)

# 为了反标准化，需要将预测值和对应的特征结合起来
# 使用每个序列的最后一个时间步的特征
X_eval_last = X_eval[:len(predictions), -1, :]  # 确保长度匹配

def inverse_transform(scaler, X, y):
    # X: 原始特征，形状为 (n_samples, n_features)
    # y: 目标值，形状为 (n_samples,)
    combined = np.hstack((X, y.reshape(-1, 1)))
    inversed = scaler.inverse_transform(combined)
    return inversed[:, -1]  # 返回反标准化后的目标值

# 反标准化预测值和真实值
predicted_oxygen = inverse_transform(scaler, X_eval_last.numpy(), predictions)
actual_oxygen = inverse_transform(scaler, X_eval_last.numpy(), actuals)

# 计算反标准化后的评估指标
mae_real = mean_absolute_error(actual_oxygen, predicted_oxygen)
r2_real = r2_score(actual_oxygen, predicted_oxygen)
print(f'After inverse transform - MAE: {mae_real:.4f}, R2: {r2_real:.4f}')

# ========== 可视化结果（可选） ==========

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(actual_oxygen, label='Actual Oxygen')
plt.plot(predicted_oxygen, label='Predicted Oxygen')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Oxygen Concentration')
plt.title('Actual vs Predicted Oxygen Concentration')
plt.show()