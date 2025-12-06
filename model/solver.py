import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics


class Solver(object):
    def __init__(self, train_loader, val_loader, model, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=4
        )

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

    @property
    def model_save_path(self):
        return self.config.model_save_path

    def train(self):
        print(f"Start training on {self.device}...")
        best_roc_auc = 0.0
        start_t = time.time()

        for epoch in range(self.config.n_epochs):
            # --- TRAINING ---
            self.model.train()
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                out = self.model(x)
                loss = self.criterion(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % self.config.log_step == 0:
                    elapsed = datetime.timedelta(seconds=time.time() - start_t)
                    print(
                        f"Epoch [{epoch + 1}/{self.config.n_epochs}], Step [{i + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}, Time: {elapsed}")

            # --- VALIDATION ---
            print("Validating...")
            val_loss, roc_auc, pr_auc = self._validation()

            # Обновляем LR
            self.scheduler.step(roc_auc)

            print(f"Epoch [{epoch + 1}] Val Loss: {val_loss:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")

            # Сохранение лучшей модели
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                torch.save(self.model.state_dict(), os.path.join(self.model_save_path, 'best_model.pth'))
                print(f"Saved Best Model (AUC: {best_roc_auc:.4f})")

    def _validation(self):
        self.model.eval()
        val_loss = 0.0
        y_true, y_scores = [], []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)
                val_loss += loss.item()

                # Важно: добавляем Sigmoid для метрик, так как в модели его нет
                probs = torch.sigmoid(out)
                y_true.append(y.cpu().numpy())
                y_scores.append(probs.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_scores = np.concatenate(y_scores)

        try:
            roc_auc = metrics.roc_auc_score(y_true, y_scores, average='macro')
            pr_auc = metrics.average_precision_score(y_true, y_scores, average='macro')
        except:
            roc_auc, pr_auc = 0.0, 0.0

        return val_loss / len(self.val_loader), roc_auc, pr_auc