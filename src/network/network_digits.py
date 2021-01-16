"""
Multi-EPL: Accurate Multi-Source Domain Adaptation

Authors:
- Seongmin Lee (ligi214@snu.ac.kr)
- Hyunsik Jeon (jeon185@gmail.com)
- U Kang (ukang@snu.ac.kr)

File: src/network/network_digits.py
- Contains source code for the feature extractor and label classifier networks for Digits-Five
"""

import torch.nn as nn
import torch.nn.functional as F


# Feature extractor model for Digits-Five
class GeneratorDigits(nn.Module):
    def __init__(self):
        super(GeneratorDigits, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        # x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x


# Label classifier model for Digits-Five
class LabelClassifierDigits(nn.Module):
    def __init__(self, input_size=2048):
        super(LabelClassifierDigits, self).__init__()
        self.fc3 = nn.Linear(input_size, 10)

    def forward(self, x):
        x = self.fc3(x)
        return x
