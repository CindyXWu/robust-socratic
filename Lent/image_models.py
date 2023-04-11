import torch.nn as nn
import torch
import torchvision.models as models
import functools
from typing import Callable, Protocol
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGet
from torch import Tensor


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SplitAggregate(nn.Module):
    """
    Allows for specifying a "split" in a sequential model where the input passes through multiple paths and is
    aggregated at the output (by default: added).
    Useful for specifying a residual connection in a model.
    """
    def __init__(self, path1: nn.Module, path2: nn.Module, aggregate_func: Callable[[Tensor, Tensor], Tensor] = torch.add) -> None:
        super().__init__()
        self.path1 = path1
        self.path2 = path2
        self.aggregate_func = aggregate_func
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aggregate_func(
            self.path1(x),
            self.path2(x),
        )


class NormalizationConstructorType(Protocol):
    def __call__(self, num_features: int) -> nn.Module:
        ...

class LeNet5(nn.Module):
    """Changed input channels to 3 and added batchnorm."""
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

    def attention_map(self, x, layer):
        layer_index = None
        for i, (name, _) in enumerate(self.named_modules()):
            # For some reason first 2 modles is the entire model so skip it
            if i <=1:
                continue
            if name == layer:
                layer_index = i
                break
        if layer_index is None:
            raise ValueError("Layer not found.")
        submodel = nn.Sequential(*list(self.modules())[2:layer_index+1])
        return submodel(x)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = nn.functional.softmax(logits, dim=1)
        return logits

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

def wide_resnet_constructor(
        blocks_per_stage: int,
        num_classes: int,
        width_factor: int = 1,
        activation_constructor: Callable[[], nn.Module] = functools.partial(nn.ReLU, inplace=True),
        normalization_constructor: NormalizationConstructorType = nn.BatchNorm2d,
    ) -> nn.Sequential:
    """
    Construct a Wide ResNet model.
    Follows the architecture described in Table 1 of the paper:
        Wide Residual Networks
        https://arxiv.org/pdf/1605.07146.pdf
    This is a Wide-ResNet with the block structure following the variant (sometimes known as ResNetV2) described in:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/abs/1603.05027
    Args:
        blocks_per_stage: Number of blocks per stage.
        width_factor: Width factor.
    Returns:
        The constructed model.
    """
    assert blocks_per_stage >= 1, f"blocks_per_stage must be >= 1, got {blocks_per_stage}"

    def block_constructor(in_channels: int, out_channels: int) -> nn.Module:
        return SplitAggregate(
            # Skip connection
            Identity(),
            # Conv. block
            nn.Sequential(
                normalization_constructor(in_channels),
                activation_constructor(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                normalization_constructor(out_channels),
                activation_constructor(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )
        )
    
    def downsample_block_constructor(in_channels: int, out_channels: int) -> nn.Module:
        return SplitAggregate(
            # Skip connection with 1x1 conv downsampling
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            # Conv. block
            nn.Sequential(
                normalization_constructor(in_channels),
                activation_constructor(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                normalization_constructor(out_channels),
                activation_constructor(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )
        )

    model = nn.Sequential(
        # The output width of the first conv. layer being 16 * width_factor is a slight
        # deviation from the paper repository 
        # https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py
        nn.Conv2d(3, 16 * width_factor, kernel_size=3, stride=1, padding=1, bias=False),
        normalization_constructor(16 * width_factor),
        activation_constructor(),
        # Stage 1
        block_constructor(16 * width_factor, 16 * width_factor),
        *(block_constructor(16 * width_factor, 16 * width_factor) for _ in range(blocks_per_stage - 1)),
        # Stage 2
        downsample_block_constructor(16 * width_factor, 32 * width_factor),
        *(block_constructor(32 * width_factor, 32 * width_factor) for _ in range(blocks_per_stage - 1)),
        # Stage 3
        downsample_block_constructor(32 * width_factor, 64 * width_factor),
        *(block_constructor(64 * width_factor, 64 * width_factor) for _ in range(blocks_per_stage - 1)),
        # Output
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64 * width_factor, num_classes),
    )

    # Initialise
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    return model

# Use this to find the names of the layers in the model for feature extractor in feature_match.py
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