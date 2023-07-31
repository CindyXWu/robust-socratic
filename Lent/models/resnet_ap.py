import torch.nn as nn
import torch
import torchvision.models as models
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGet

from resnet import wide_resnet_constructor
from common import get_submodules


class CustomResNet18(models.ResNet):
    """This model accepts images of any size using adaptive average pooling."""
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__(
            block=models.resnet.BasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=num_classes
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * models.resnet.BasicBlock.expansion, num_classes)

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
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class CustomResNet50(models.ResNet):
    """This model accepts images of any size using adaptive average pooling."""
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__(
            block=models.resnet.Bottleneck,
            layers=[3, 4, 6, 3],
            num_classes=num_classes
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048 * models.resnet.Bottleneck.expansion, num_classes)

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
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



def show_model(model):
    print("List of model layers:")
    for idx, module in enumerate(model.children()):
        print(f"Layer {idx}: {module}")


if __name__ == "__main__":
    resnet_ap = CustomResNet18(8)
    resnet50_ap = CustomResNet50(8)
    resnet = wide_resnet_constructor(3, 100)
    lenet = LeNet5(10)
    get_submodules(lenet)

# class ResNet18_CIFAR(models.ResNet):
#     """10 layers in total.
#     Layers 4-7 inclusive are blocks. Each block has a two sub-block of BasicBlock() structure:
#     Sequential(
#     (0): BasicBlock(
#         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (1): BasicBlock(
#         (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     """
#     def __init__(self, num_classes=10):
#         super(ResNet18_CIFAR, self).__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
#         # Modify the first convolution layer to accept 3-channel input with 32 x 32 dimensions
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

#     def attention_map(self, x, layer):
#         layer_index = None
#         # Check modules
#         # for module in self.modules():
#         #     print("==================================")
#         #     print(module)
#         for i, (name, _) in enumerate(self.named_modules()):
#             # For some reason first module is the entire model so skip it
#             if i == 0:
#                 continue
#             if name == layer:
#                 layer_index = i
#                 break
#         if layer_index is None:
#             raise ValueError("Layer not found.")
#         submodel = nn.Sequential(*list(self.modules())[1:layer_index+1])
#         return submodel(x)  

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x1= self.relu(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x