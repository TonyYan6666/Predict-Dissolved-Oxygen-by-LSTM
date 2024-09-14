import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

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
    train_features_standardized = (train_features_transformed - train_features_transformed.mean()) / train_features_transformed.std()
    train_label_standardized = (train_label_transformed - train_label_transformed.mean()) / train_label_transformed.std()

    eval_features_standardized = (eval_features_transformed - train_features_transformed.mean()) / train_features_transformed.std()
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