import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_class=56):
        super(CNN, self).__init__()

        # Initial Batch Norm
        self.bn_init = nn.BatchNorm2d(1)

        # Layer 1
        self.conv_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.mp_1 = nn.MaxPool2d((2, 4))

        # Layer 2
        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.mp_2 = nn.MaxPool2d((2, 4))

        # Layer 3
        self.conv_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)
        self.mp_3 = nn.MaxPool2d((2, 4))

        # Layer 4
        self.conv_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.mp_4 = nn.MaxPool2d((3, 5))

        # Layer 5
        self.conv_5 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(64)

        # Robust Pooling
        self.mp_final = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.dense = nn.Linear(64, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.bn_init(x)

        x = self.mp_1(nn.ELU()(self.bn_1(self.conv_1(x))))
        x = self.mp_2(nn.ELU()(self.bn_2(self.conv_2(x))))
        x = self.mp_3(nn.ELU()(self.bn_3(self.conv_3(x))))
        x = self.mp_4(nn.ELU()(self.bn_4(self.conv_4(x))))
        x = nn.ELU()(self.bn_5(self.conv_5(x)))

        x = self.mp_final(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logit = self.dense(x)

        return logit