import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置文件夹路径
data_folder = 'data'

# 获取所有子文件夹
subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]

# 存储结果
results = {}

# 遍历每个子文件夹
for folder in subfolders:
    cite_number = os.path.basename(folder).split('.')[0]

    # 读取每个CSV文件
    oxygen_path = os.path.join(folder, f'{cite_number}.dissolvedoxygendo.Raw_Data.csv')
    streamflow_path = os.path.join(folder, f'{cite_number}.streamflow.Raw_Data.csv')
    temperature_path = os.path.join(folder, f'{cite_number}.watertemperature.Raw_Data.csv')

    oxygen_df = pd.read_csv(oxygen_path, parse_dates=['Date Time'])
    streamflow_df = pd.read_csv(streamflow_path, parse_dates=['Date Time'])
    temperature_df = pd.read_csv(temperature_path, parse_dates=['Date Time'])

    # 设置日期时间为索引
    oxygen_df.set_index('Date Time', inplace=True)
    streamflow_df.set_index('Date Time', inplace=True)
    temperature_df.set_index('Date Time', inplace=True)

    # 找到时间重合的数据
    merged_df = oxygen_df.join(streamflow_df, lsuffix='_oxygen', rsuffix='_streamflow', how='inner')
    merged_df = merged_df.join(temperature_df, rsuffix='_temp', how='inner')

    # 统计每年重合时间的数据点数量
    yearly_counts = merged_df.groupby(merged_df.index.year).size()

    # 存储结果
    results[cite_number] = yearly_counts

# 将结果转换为 DataFrame
results_df = pd.DataFrame(results).fillna(0).astype(int).T

# 按照 site number 排序
results_df.sort_index(inplace=True)

# 绘制热图
plt.figure(figsize=(18, 14))  # 调整图形大小
sns.heatmap(results_df, cmap='YlGnBu', annot=True, fmt='d', cbar_kws={'label': 'Number of Data Points'}, linewidths=0.7)
plt.title('Heatmap of Data Points per Site and Year')
plt.xlabel('Year')
plt.ylabel('Site Number')
plt.tight_layout()

# 保存图像到文件
plt.savefig('heatmap_data_points.png')

print("Heatmap saved as 'heatmap_data_points.png'")