import torch.nn as nn
import torch

class LeNet5(nn.Module):
    def __init__(self, n_classes, greyscale=False):
        super(LeNet5, self).__init__()
        
        self.greyscale = greyscale
        if self.greyscale:
            in_channels = 1
        else:
            in_channels = 3

        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels, out_channels=6*in_channels, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6*in_channels, out_channels=16*in_channels, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16*in_channels, out_channels=120*in_channels, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120*in_channels, out_features=84*in_channels),
            nn.Tanh(),
            nn.Linear(in_features=84*in_channels, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = nn.functional.softmax(logits, dim=1)
        return logits, probs