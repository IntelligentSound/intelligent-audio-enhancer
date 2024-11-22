import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EnhancedGenreClassifier(nn.Module):
    def __init__(self, num_classes, n_features, fixed_length, dropout=0.5):
        super(EnhancedGenreClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d((2, 1), stride=(2, 1))
        self.dropout = nn.Dropout(dropout)

        self.conv_output_size = self._get_conv_output(n_features, fixed_length)
        self.fc1 = nn.Linear(self.conv_output_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_conv_output(self, n_features, fixed_length):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_features, fixed_length)
            x = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            return int(np.prod(x.size()))

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

