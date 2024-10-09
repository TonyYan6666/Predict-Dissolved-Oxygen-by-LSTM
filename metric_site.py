import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from scipy.stats import boxcox
import copy

# ========== Define CudnnLstmModel ==========

class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, warmUpDay=None):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = torch.nn.LSTM(input_size=hiddenSize, hidden_size=hiddenSize, num_layers=1, dropout=dr, batch_first=True)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0)
        out = self.linearOut(outLSTM[:, -1, :])
        return out

# ========== Dataset Creation ==========

def create_dataset_by_site(data, look_back=10):
    X, Y = [], []
    site_numbers = data['Site Number'].unique()
    for site in site_numbers:
        site_data = data[data['Site Number'] == site].reset_index(drop=True)
        data_array = site_data[['Streamflow', 'Temperature', 'Oxygen']].values
        for i in range(len(data_array) - look_back):
            X.append(data_array[i:i + look_back, :-1])
            Y.append(data_array[i + look_back, -1])
    return np.array(X), np.array(Y)

# ========== Model Loading ==========

def load_model(file_path, input_size, hidden_size, output_size, device):
    model = CudnnLstmModel(nx=input_size, ny=output_size, hiddenSize=hidden_size)
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.to(device)
    return model

# ========== Metrics Calculation ==========

def calculate_metrics(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    nse = 1 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_true_mean) ** 2)
    pbias = 100 * np.sum(np.abs(y_pred - y_true)) / np.sum(y_true)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    pcorr = np.corrcoef(y_pred, y_true)[0, 1]
    return nse, pbias, rmse, pcorr

# ========== Evaluation for Each Site ==========

def evaluate_model_on_sites(model, scaled_test_data, device, look_back=10):
    site_numbers_test = scaled_test_data['Site Number'].unique()

    # Dictionary to store metrics
    site_metrics = {'Site Number': [], 'NSE': [], 'Pbias': [], 'RMSE': [], 'Pcorr': []}

    for site in site_numbers_test:
        # Filter data by site
        site_test_data = scaled_test_data[scaled_test_data['Site Number'] == site].reset_index(drop=True)

        # Check if there is enough data for this site
        if len(site_test_data) < look_back:
            print(f"Skipping site {site} due to insufficient data")
            continue

        print(f"Evaluating site {site}")

        # Create the dataset for this site
        X_site_test, Y_site_test = create_dataset_by_site(site_test_data, look_back)

        if len(X_site_test) == 0:
            print(f"No data to evaluate for site {site}")
            continue

        X_site_test = torch.from_numpy(X_site_test).float().to(device)
        Y_site_test = torch.from_numpy(Y_site_test).float().to(device)

        # Make predictions
        model.eval()
        site_test_predictions = []
        site_test_actuals = []

        with torch.no_grad():
            outputs = model(X_site_test)
            site_test_predictions.extend(outputs.squeeze().cpu().numpy())
            site_test_actuals.extend(Y_site_test.cpu().numpy())

        # Convert to numpy arrays
        site_test_predictions = np.array(site_test_predictions)
        site_test_actuals = np.array(site_test_actuals)

        # Compute metrics
        if len(site_test_actuals) > 0 and len(site_test_predictions) > 0:
            nse, pbias, rmse, pcorr = calculate_metrics(site_test_actuals, site_test_predictions)

            # Store metrics for this site
            site_metrics['Site Number'].append(site)
            site_metrics['NSE'].append(nse)
            site_metrics['Pbias'].append(pbias)
            site_metrics['RMSE'].append(rmse)
            site_metrics['Pcorr'].append(pcorr)
        else:
            print(f"No valid predictions for site {site}")

    return pd.DataFrame(site_metrics)


# ========== Main Script ==========

if __name__ == "__main__":
    # Load test data
    file_path = 'data_final.csv'
    data = pd.read_csv(file_path)

    # ========== Data Preprocessing ==========

    # Replace zeros with NaNs for interpolation
    data.replace({'Oxygen': {0: np.nan}, 'Temperature': {0: np.nan}, 'Streamflow': {0: np.nan}}, inplace=True)

    # Linear interpolation
    data_interpolated = data.interpolate(method='slinear', limit_direction='forward', axis=0)
    data_interpolated.fillna(method='bfill', inplace=True)
    data_interpolated.fillna(method='ffill', inplace=True)

    # Parse 'Date Time' to datetime format and sort
    data_interpolated['Datetime'] = pd.to_datetime(data_interpolated['Date Time'])
    data_interpolated = data_interpolated.sort_values(by='Datetime').reset_index(drop=True)

    # 80% split for training and test data
    split_index = int(len(data_interpolated) * 0.8)
    test_data = data_interpolated.iloc[split_index:].reset_index(drop=True)

    # Select features (Streamflow, Temperature) and target (Oxygen)
    features = ['Streamflow', 'Temperature']
    target = ['Oxygen']

    # Apply Box-Cox transformation to Streamflow
    streamflow_min = data_interpolated['Streamflow'].min()
    if streamflow_min <= 0:
        data_interpolated['Streamflow'] += abs(streamflow_min) + 1e-6
    data_interpolated['Streamflow'], streamflow_lambda = boxcox(data_interpolated['Streamflow'])

    # Scale data using StandardScaler
    scaler = StandardScaler()
    scaler.fit(data_interpolated[features + target])

    scaled_test_data = test_data.copy()
    scaled_test_data[features + target] = scaler.transform(test_data[features + target])

    # ========== Model Loading and Evaluation ==========

    # Ensure the device is set correctly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained model
    model_path = 'best_cudnn_lstm_model.pth'
    input_size = 2  # Streamflow and Temperature
    hidden_size = 50
    output_size = 1

    model = load_model(model_path, input_size, hidden_size, output_size, device)

    # Evaluate the model on the test set and get metrics per site
    metrics_df = evaluate_model_on_sites(model, scaled_test_data, device)

    # Output the metrics for each site
    print(metrics_df)
    metrics_df.to_csv('site_metrics.csv', index=False)

    data_metrics_site = copy.deepcopy(metrics_df)

    # Convert 'Site Number' to string to ensure proper labeling on the x-axis
    data_metrics_site['Site Number'] = data_metrics_site['Site Number'].astype(str)

    # Metrics to plot (excluding Pbias)
    metrics_without_pbias = ['NSE', 'RMSE', 'Pcorr']

    # Create bar plots for the remaining three metrics
    plt.figure(figsize=(45, 7))

    for i, metric in enumerate(metrics_without_pbias, 1):
        plt.subplot(1, 3, i)
        plt.bar(data_metrics_site['Site Number'], data_metrics_site[metric])
        plt.title(f'{metric} by Site Number')
        plt.xlabel('Site Number')
        plt.ylabel(metric)

        # 设置显示的标签间隔，减少横坐标的拥挤
        plt.xticks(ticks=range(0, len(data_metrics_site['Site Number']), 2),
                   labels=data_metrics_site['Site Number'][::2], rotation=45, ha='right')

    plt.tight_layout()  # 自动调整子图之间的间距
    plt.savefig("site_metric.png")
    plt.show()

