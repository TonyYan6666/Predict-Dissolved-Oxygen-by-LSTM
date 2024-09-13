import pandas as pd
import os

# 设置数据文件夹路径
data_folder = 'data'

# 获取所有子文件夹
subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]

# 处理每个站点
for folder in subfolders:
    site_number = os.path.basename(folder).split('.')[0]

    # 读取每个CSV文件
    oxygen_path = os.path.join(folder, f'{site_number}.dissolvedoxygendo.Daily_Mean.csv')
    streamflow_path = os.path.join(folder, f'{site_number}.streamflow.Daily_Mean.csv')
    temperature_path = os.path.join(folder, f'{site_number}.watertemperature.Daily_Mean.csv')

    oxygen_df = pd.read_csv(oxygen_path, parse_dates=['Date Time'])
    streamflow_df = pd.read_csv(streamflow_path, parse_dates=['Date Time'])
    temperature_df = pd.read_csv(temperature_path, parse_dates=['Date Time'])

    # 找到最小重合时间范围
    start_time = max(oxygen_df['Date Time'].min(), streamflow_df['Date Time'].min(), temperature_df['Date Time'].min())
    end_time = min(oxygen_df['Date Time'].max(), streamflow_df['Date Time'].max(), temperature_df['Date Time'].max())

    # 只保留在最小重合时间范围内的数据
    oxygen_df = oxygen_df[(oxygen_df['Date Time'] >= start_time) & (oxygen_df['Date Time'] <= end_time)]
    streamflow_df = streamflow_df[(streamflow_df['Date Time'] >= start_time) & (streamflow_df['Date Time'] <= end_time)]
    temperature_df = temperature_df[(temperature_df['Date Time'] >= start_time) & (temperature_df['Date Time'] <= end_time)]

    # 只保留‘Date Time’、‘VALUE’和‘quality’列，并重命名
    oxygen_df = oxygen_df[['Date Time', 'VALUE', 'QUALITY']].rename(columns={'VALUE': 'Oxygen', 'QUALITY': 'oxygen_quality'})
    streamflow_df = streamflow_df[['Date Time', 'VALUE']].rename(columns={'VALUE': 'Streamflow'})
    temperature_df = temperature_df[['Date Time', 'VALUE']].rename(columns={'VALUE': 'Temperature'})

    # 使用最近的时间戳合并
    merged_df = pd.merge_asof(oxygen_df.sort_values('Date Time'),
                              streamflow_df.sort_values('Date Time'),
                              on='Date Time',
                              direction='nearest')

    merged_df = pd.merge_asof(merged_df.sort_values('Date Time'),
                              temperature_df.sort_values('Date Time'),
                              on='Date Time',
                              direction='nearest')

    # 保存合并的数据到新的CSV文件
    merged_csv_path = os.path.join(folder, f'{site_number}_merged.csv')
    merged_df.to_csv(merged_csv_path, index=False)

    print(f"Merged CSV saved for site {site_number} at '{merged_csv_path}'")


# 设置数据文件夹路径
data_folder = 'data'

# 获取所有子文件夹
subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]

# 存储所有合并的数据
all_data = []

# 遍历每个站点的文件夹
for folder in subfolders:
    site_number = os.path.basename(folder).split('.')[0]
    merged_csv_path = os.path.join(folder, f'{site_number}_merged.csv')

    # 读取合并后的CSV文件
    if os.path.exists(merged_csv_path):
        df = pd.read_csv(merged_csv_path)
        df['Site Number'] = site_number  # 添加站点编号列
        all_data.append(df)

# 将所有站点的数据合并为一个DataFrame
final_df = pd.concat(all_data, ignore_index=True)

# 保存到最终的CSV文件
final_csv_path = 'data_final.csv'
final_df.to_csv(final_csv_path, index=False)

print(f"All merged data saved to '{final_csv_path}'")