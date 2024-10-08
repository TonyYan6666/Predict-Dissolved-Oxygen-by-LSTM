import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from scipy.stats import boxcox
import torch.nn.functional as F

# ========== 数据读取和预处理 ==========

# 读取数据
file_path = 'data_final.csv'  # 确保路径正确
data = pd.read_csv(file_path)

# 将 Oxygen、Temperature 或 Streamflow 为 0 的值替换为 NaN 以便插值
data.replace({'Oxygen': {0: np.nan}, 'Temperature': {0: np.nan}, 'Streamflow': {0: np.nan}}, inplace=True)

# 进行线性插值填充缺失值
data_interpolated = data.interpolate(method='slinear', limit_direction='forward', axis=0)

# 如果插值后仍有缺失值，使用前向和后向填充以确保数据完整
data_interpolated.fillna(method='bfill', inplace=True)
data_interpolated.fillna(method='ffill', inplace=True)

# 解析 'Date Time' 为 datetime 格式
data_interpolated['Datetime'] = pd.to_datetime(data_interpolated['Date Time'])

# 按 'Datetime' 排序，然后按时间顺序进行数据集分割
data_interpolated = data_interpolated.sort_values(by='Datetime').reset_index(drop=True)

# 80%数据集分割
split_index = int(len(data_interpolated) * 0.8)
train_data = data_interpolated.iloc[:split_index].reset_index(drop=True)
test_data = data_interpolated.iloc[split_index:].reset_index(drop=True)

# 再次分割 train_data，70%为训练集，10%为验证集
val_split_index = int(len(train_data) * 0.875)  # 70% training and 10% validation
train_data, val_data = train_data.iloc[:val_split_index].reset_index(drop=True), train_data.iloc[val_split_index:].reset_index(drop=True)

# 确保数据按 'Site Number' 和 'Datetime' 排序
train_data = train_data.sort_values(by=['Site Number', 'Datetime']).reset_index(drop=True)
val_data = val_data.sort_values(by=['Site Number', 'Datetime']).reset_index(drop=True)
test_data = test_data.sort_values(by=['Site Number', 'Datetime']).reset_index(drop=True)

# 选择必要的列（Streamflow, Temperature 用作特征，Oxygen 用作预测目标）
features = ['Streamflow', 'Temperature']
target = ['Oxygen']

# ========== Box-Cox Transformation ==========

# Apply Box-Cox transformation to Streamflow (must be positive)
streamflow_min = data_interpolated['Streamflow'].min()
if streamflow_min <= 0:
    # Add a small constant to make Streamflow positive
    data_interpolated['Streamflow'] += abs(streamflow_min) + 1e-6

# Apply Box-Cox transformation to Streamflow
data_interpolated['Streamflow'], streamflow_lambda = boxcox(data_interpolated['Streamflow'])
print(f"Streamflow Box-Cox transformation lambda: {streamflow_lambda}")

# ========== 数据标准化 ==========
# 使用 StandardScaler 归一化数据（不包括 'Datetime' 和 'Site Number' 列）
scaler = StandardScaler()
scaler.fit(data_interpolated[features + target])

# 创建经过缩放的数据
scaled_train_data = train_data.copy()
scaled_val_data = val_data.copy()
scaled_test_data = test_data.copy()

# 对 Streamflow, Temperature 和 Oxygen 进行标准化
scaled_train_data[features + target] = scaler.transform(train_data[features + target])
scaled_val_data[features + target] = scaler.transform(val_data[features + target])
scaled_test_data[features + target] = scaler.transform(test_data[features + target])

# 定义根据站点创建数据集的函数
def create_dataset_by_site(data, look_back=12):
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

# 创建训练、验证和测试数据集
X_train, Y_train = create_dataset_by_site(scaled_train_data, look_back)
X_val, Y_val = create_dataset_by_site(scaled_val_data, look_back)
X_test, Y_test = create_dataset_by_site(scaled_test_data, look_back)

# 转换数据为 PyTorch 的张量
X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).float()
X_val = torch.from_numpy(X_val).float()
Y_val = torch.from_numpy(Y_val).float()
X_test = torch.from_numpy(X_test).float()
Y_test = torch.from_numpy(Y_test).float()

# 创建数据加载器，不打乱数据顺序
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False, drop_last=True)

# ========== 替换为 CudnnLstmModel ==========

# 定义 CudnnLstmModel
class CudnnLstmModel(nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.8, warmUpDay=None):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = nn.Linear(nx, hiddenSize)
        self.lstm = nn.LSTM(input_size=hiddenSize, hidden_size=hiddenSize, num_layers=1, dropout=dr, batch_first=True)
        self.linearOut = nn.Linear(hiddenSize, ny)
        self.name = "CudnnLstmModel"

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0)
        out = self.linearOut(outLSTM[:, -1, :])
        return out

# 设置模型参数
input_size = len(features)
hidden_size = 50
output_size = 1

# 使用 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 CudnnLstmModel
model = CudnnLstmModel(nx=input_size, ny=output_size, hiddenSize=hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# ========== 模型训练 ==========

num_epochs = 100
best_val_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), Y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # 验证模型
    model.eval()
    val_loss = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), Y_batch)
            val_loss += loss.item()

            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(Y_batch.cpu().numpy())

    val_loss /= len(val_loader)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    # 检查是否为最好的模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()

    scheduler.step(val_loss)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

# 加载最佳模型状态
model.load_state_dict(best_model_state)
torch.save(model.state_dict(), 'best_cudnn_lstm_model.pth')

# ========== 模型测试 ==========

model.eval()
test_predictions = []
test_actuals = []

with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        test_predictions.extend(outputs.squeeze().cpu().numpy())
        test_actuals.extend(Y_batch.cpu().numpy())

# 转换为 NumPy 数组
test_predictions = np.array(test_predictions)
test_actuals = np.array(test_actuals)

# 计算额外的评估指标函数
def calculate_metrics(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    nse = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_true_mean) ** 2)
    pbias = 100 * np.sum(np.abs(y_pred - y_true)) / np.sum(y_true)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    pcorr = np.corrcoef(y_pred, y_true)[0, 1]
    return nse, pbias, rmse, pcorr

# 计算标准化数据下的评估指标
nse, pbias, rmse, pcorr = calculate_metrics(test_actuals, test_predictions)
mae_real = mean_absolute_error(test_actuals, test_predictions)
r2_real = r2_score(test_actuals, test_predictions)

# 输出评估结果
print(f'MAE: {mae_real:.4f}, R2: {r2_real:.4f}')
print(f'NSE: {nse:.4f}, Pbias: {pbias:.4f}, RMSE: {rmse:.4f}, Pcorr: {pcorr:.4f}')

# ========== 可视化结果（标准化数据） ==========

plt.figure(figsize=(12, 6))
plt.plot(test_actuals, label='Actual (Standardized)')
plt.plot(test_predictions, label='Predicted (Standardized)')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Oxygen Level (Standardized)')
plt.title('Actual vs Predicted Oxygen Levels')
plt.show()