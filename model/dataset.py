import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MTGDataset(Dataset):
    def __init__(self, csv_file, target_len=1366):
        self.data = pd.read_csv(csv_file)
        self.target_len = target_len

        # Метаданные (не являются тегами)
        self.meta_cols = ['TRACK_ID', 'PATH', 'DURATION']
        # Все остальные колонки считаем тегами
        self.label_cols = [col for col in self.data.columns if col not in self.meta_cols]
        self.num_classes = len(self.label_cols)

        print(f"Dataset loaded: {len(self.data)} tracks, {self.num_classes} classes.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        npy_path = row['PATH']

        try:
            # Загружаем спектрограмму
            spec = np.load(npy_path, allow_pickle=False)

            # Приводим к (Freq, Time), если перевернуто
            if spec.shape[0] > spec.shape[1]:
                spec = spec.T

            # Фиксация длины (Crop или Pad)
            cur_len = spec.shape[1]
            if cur_len > self.target_len:
                start = (cur_len - self.target_len) // 2
                spec = spec[:, start: start + self.target_len]
            elif cur_len < self.target_len:
                pad_amt = self.target_len - cur_len
                spec = np.pad(spec, ((0, 0), (0, pad_amt)), 'constant')

        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            spec = np.zeros((96, self.target_len))

        # Возвращаем (1, 96, 1366) -> PyTorch сам добавит Batch dim позже
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)
        labels = torch.tensor(row[self.label_cols].values.astype('float32'))

        return spec_tensor, labels