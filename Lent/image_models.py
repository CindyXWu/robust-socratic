import torch.nn as nn
import torch
import torchvision.models as models

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
        return logits

class ResNet50_CIFAR10(models.ResNet):
    def __init__(self):
        super(ResNet50_CIFAR10, self).__init__(block=models.resnet.Bottleneck, layers=[3, 4, 6, 3], num_classes=10)

        # Modify the first convolution layer to accept 3-channel input with 32 x 32 dimensions
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x