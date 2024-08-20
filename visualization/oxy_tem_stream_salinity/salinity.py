import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置文件夹路径
data_folder = 'data'

# 获取所有子文件夹
subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]

# 初始化图
fig, axes = plt.subplots(len(subfolders) + 1, 1, figsize=(16, (len(subfolders) + 1) * 4), sharex=True)

# 存储结果
results = {}

# 定义颜色
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# 遍历每个子文件夹
for idx, folder in enumerate(subfolders):
    cite_number = os.path.basename(folder).split('.')[0]

    # 读取每个CSV文件
    oxygen_path = os.path.join(folder, f'{cite_number}.dissolvedoxygendo.Raw_Data.csv')
    streamflow_path = os.path.join(folder, f'{cite_number}.streamflow.Raw_Data.csv')
    temperature_path = os.path.join(folder, f'{cite_number}.watertemperature.Raw_Data.csv')
    turbidity_path = os.path.join(folder, f'{cite_number}.salinityaselectricalconductivityec.Raw_Data.csv')

    oxygen_df = pd.read_csv(oxygen_path, parse_dates=['Date Time'])
    streamflow_df = pd.read_csv(streamflow_path, parse_dates=['Date Time'])
    temperature_df = pd.read_csv(temperature_path, parse_dates=['Date Time'])
    salinity_df = pd.read_csv(turbidity_path, parse_dates=['Date Time'])

    # 设置日期时间为索引
    oxygen_df.set_index('Date Time', inplace=True)
    streamflow_df.set_index('Date Time', inplace=True)
    temperature_df.set_index('Date Time', inplace=True)
    salinity_df.set_index('Date Time', inplace=True)

    # 找到时间重合的数据
    merged_df = oxygen_df.join(streamflow_df, lsuffix='_oxygen', rsuffix='_streamflow', how='inner')
    merged_df = merged_df.join(temperature_df, rsuffix='_temp', how='inner')
    merged_df = merged_df.join(salinity_df, rsuffix='_salinity', how='inner')

    # 统计每年重合时间的数据点数量
    yearly_counts = merged_df.groupby(merged_df.index.year).size()

    # 存储结果
    results[cite_number] = yearly_counts

    # 绘制折线图
    color = colors[idx % len(colors)]
    axes[idx].plot(yearly_counts.index, yearly_counts.values, label=f'Cite {cite_number}', marker='o', color=color)
    axes[idx].set_title(f'Cite {cite_number}')
    axes[idx].set_ylabel('Number of Data Points')
    axes[idx].grid(True)
    axes[idx].legend()
    axes[idx].set_xticks(yearly_counts.index)
    axes[idx].set_xticklabels(yearly_counts.index, rotation=45)

    # 每个点上标注数据值和年份
    for x, y in zip(yearly_counts.index, yearly_counts.values):
        axes[idx].text(x, y, f'{y} ({x})', ha='center', va='bottom')

# 统计所有cite每年数据点总和
total_counts = pd.DataFrame(results).sum(axis=1)

# 在总图中显示年份和总和
axes[-1].bar(total_counts.index, total_counts.values, color='gray')
axes[-1].set_title('Total Overlapping Data Points')
axes[-1].set_ylabel('Number of Data Points')
axes[-1].grid(True)
axes[-1].set_xticks(total_counts.index)
axes[-1].set_xticklabels(total_counts.index, rotation=45)
axes[-1].set_xlabel('Year')

# 显示年份和数据个数
for year, count in total_counts.items():
    axes[-1].text(year, count, f'{year}: {count}', ha='center', va='bottom')

plt.tight_layout()

# 保存图像到文件
plt.savefig('overlapping_data_points_plot.png')

print("Plot saved as 'overlapping_data_points_plot.png'")

# 保存结果到文件
results_df = pd.DataFrame(results)
results_df['Total'] = results_df.sum(axis=1)  # 添加总数列
results_df.to_csv('overlapping_data_points.csv')

print("Results saved to 'overlapping_data_points.csv'")