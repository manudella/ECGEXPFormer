import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_residual=False, dropout=0.0):
        super(ConvBlock, self).__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

        if use_residual and in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.residual = None

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))

        if self.use_residual:
            if self.residual is not None:
                identity = self.residual(identity)
            x = x + identity

        return x

# Add data augmentation with stride

class ECGCNNForClassification(nn.Module):
    def __init__(
        self,
        input_channels=2,  # Correctly set to 2 channels
        num_classes=2,
        block_config=[(32, 3, 2, 0.2), (64, 3, 2, 0.3), (128, 3, 2, 0.4)],
        use_residual=True,
        global_pooling='avg'
    ):
        super(ECGCNNForClassification, self).__init__()
        self.blocks = nn.ModuleList()

        in_channels = input_channels  # Ensure starting with 2 channels
        for out_channels, kernel_size, stride, dropout in block_config:
            self.blocks.append(
                ConvBlock(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, use_residual=use_residual, dropout=dropout)
            )
            in_channels = out_channels

        if global_pooling == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif global_pooling == 'max':
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError("global_pooling must be either 'avg' or 'max'")

        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # Ensure input has the correct shape [batch_size, channels, height, width]
        if x.dim() == 3:  # If input has no channel dimension, add it
            x = x.unsqueeze(1)  # Add channel dimension 
            
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten for the FC layer
        x = self.fc(x)
        return x


