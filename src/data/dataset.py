"""
データセットの定義
"""
import torch
from torch.utils.data import Dataset

class SalesDataset(Dataset):
    """売上予測用のデータセット"""
    def __init__(self, X, y):
        """
        Args:
            X (numpy.ndarray): 入力特徴量
            y (numpy.ndarray): 目標値（売上）
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        """データセットの長さを返す"""
        return len(self.X)

    def __getitem__(self, idx):
        """指定されたインデックスのデータを返す"""
        return self.X[idx], self.y[idx]
