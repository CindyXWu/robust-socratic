import torch.nn as nn
import torch
import torchvision.models as models
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGet

class LeNet5(nn.Module):
    """Changed input channels to 3 and added batchnorm.
    list(model.children()) gives 2 layers (feature extractor and classifier, which can be called with model.feature_extractor)
    """
    def __init__(self, n_classes, greyscale=False):
        super(LeNet5, self).__init__()
        
        self.greyscale = greyscale
        if self.greyscale:
            in_channels = 1
        else:
            in_channels = 3

        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels, out_channels=6*in_channels, kernel_size=5, stride=1),
            nn.BatchNorm2d(6*in_channels),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6*in_channels, out_channels=16*in_channels, kernel_size=5, stride=1),
            nn.BatchNorm2d(16*in_channels),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16*in_channels, out_channels=120*in_channels, kernel_size=5, stride=1),
            nn.BatchNorm2d(120*in_channels),
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

class ResNet18_CIFAR10(models.ResNet):
    """10 layers in total.
    Layers 4-7 inclusive are blocks. Each block has a two sub-block of BasicBlock() structure:
    Sequential(
    (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    )
    To access e.g. conv1, do list(model.children())[layer][0].conv1
    OR model.layer1[0].conv1 (note for this second method, conv layers numbered 1-3)
    """
    def __init__(self):
        super(ResNet18_CIFAR10, self).__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=10)
        # Modify the first convolution layer to accept 3-channel input with 32 x 32 dimensions
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1= self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def get_submodules(model):
    """ Get names of all submodules in model as dict."""
    submodules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            submodules[name] = f"conv_{name.split('.')[-1]}"
        elif isinstance(module, nn.BatchNorm2d):
            submodules[name] = f"bn_{name.split('.')[-1]}"
        elif isinstance(module, nn.Linear):
            submodules[name] = f"fc_{name.split('.')[-1]}"
        else:
            submodules[name] = name
    print(submodules)

def show_model(model):
    print("List of model layers:")
    for layer in range(len(list(model.children()))):
        print("LAYER {} +++++++++++++++++++++++++++++++".format(layer))
        print(list(model.children())[layer])

if __name__ == "__main__":
    resnet = ResNet18_CIFAR10()