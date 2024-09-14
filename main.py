import pandas as pd

# Load dataset
file_path = 'data_final.csv'  # dataset path
data = pd.read_csv(file_path)

# Transform 'Date Time' into the format of data
data['Date Time'] = pd.to_datetime(data['Date Time'])

# Find the date with (80-20)
split_date = data['Date Time'].quantile(0.8)

# Split data into 80-20 (training and evaluation)
training_data = data[data['Date Time'] <= split_date]
evaluation_data = data[data['Date Time'] > split_date]

# 输出划分结果
print("训练集：")
print(training_data.head())
print("\n验证集：")
print(evaluation_data.head())
