import argparse
import os
from torch.utils.data import DataLoader, random_split

from dataset import MTGDataset
from model import CNN
from solver import Solver


def main(config):
    # 1. Загрузка датасета
    full_dataset = MTGDataset(config.csv_path)

    # Сплит 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # 2. Лоадеры
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # 3. Модель (передаем количество классов из датасета)
    model = CNN(num_class=full_dataset.num_classes)

    # 4. Запуск обучения
    solver = Solver(train_loader, val_loader, model, config)
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Укажите здесь путь к вашему CSV
    parser.add_argument('--csv_path', type=str, default='../datasets/MTG/dataset_metadata.csv')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_step', type=int, default=10)

    config = parser.parse_args()
    main(config)