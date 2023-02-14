import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import torchvision.models as models
from torch import nn, optim
import os
from tqdm import tqdm

from image_models import *
from plotting import *
from jacobian_srinivas import *
from contrastive import *
from feature_match import *

# Setup ========================================================================
# Suppress warnings "divide by zero" produced by NaN gradients
import warnings
warnings.filterwarnings("ignore")

output_dir = "Image Experiments/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Hyperparams ========================================================================
lr = 0.01
dropout = 0
temps = [1, 5]
alphas = [0.25, 0.5, 0.75]
# Use a long distillation training schedule
epochs = 1000
BATCH_SIZE = 64

def load_cifar_10(dims):
    """Load CIFAR-10 dataset and return dataloaders.
    :param dims: tuple, dimensions of the images
    """
    transform = transforms.Compose([transforms.Resize((dims[0],dims[1])),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,  
                                0.406], [0.229, 0.224, 0.225])])
    trainset = datasets.CIFAR10(root='./data/'+str(dims[0]), download=True, train=True, transform=transform)
    testset = datasets.CIFAR10(root='./data/'+str(dims[0]), download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
    return trainset, testset, trainloader, testloader

def evaluate(model, dataset, max_ex=0):
    """Evaluate model accuracy on dataset."""
    acc = 0
    for i, (features, labels) in enumerate(dataset):
        # Batch size in length, varying from 0 to 1
        scores = sigmoid(model(features))
        # Predictive class is closest from sigmoid output
        pred = torch.round(scores, decimals=0)
        # Save to pred 
        acc += torch.sum(torch.eq(pred, labels)).item()
        if max_ex != 0 and i >= max_ex:
            break
    # Return average accuracy as a percentage
    # Fraction of data points correctly classified
    return (acc*100 / ((i+1)*BATCH_SIZE))

def weight_reset(m):
    """Reset weights of model at start of training."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

# Get data
train_set, test_set, train_loader, test_loader = load_cifar_10((32, 32))

# Define loss functions
bceloss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()
kldivloss = nn.KLDivLoss(reduction='batchmean')

def train_distill(loss, teacher, student, lr, epochs, repeats, title):
    """Train student model with distillation loss."""
    optimizer = optim.SGD(student.parameters(), lr=lr)
    for _ in range(repeats):
        it = 0
        train_acc = []
        test_acc = []
        train_loss = [0]  # loss at iteration 0
        it_per_epoch = len(train_loader)
        for _ in range(epochs):
            student = student.to(device)
            student.apply(weight_reset)
            # Student
            for inputs, labels in tqdm(train_loader):
                inputs = inputs.to(device)
                print(inputs.shape)
                print(labels.shape)
                inputs.requires_grad = True
                labels = torch.tensor(labels).to(device)
                # Student outputs
                scores = student(inputs)
                # Teacher outputs
                targets = teacher(inputs)
                ## Trying to figure out what sort of object scores is
                # print(torch.is_tensor(scores[0]))
                # print(scores)
                loss = jacobian_loss(scores, targets, inputs, 1, student, teacher, 0.8, bceloss_fn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                if it % 100 == 0:
                    train_acc.append(evaluate(student, train_loader, max_ex=100))
                    test_acc.append(evaluate(student, test_loader))
                    plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
                    plot_acc(train_acc, test_acc, it, base_name=output_dir + "acc_"+title, title=title)
                    print('Iteration: %i, %.2f%%' % (it, test_acc[-1]))
                it += 1

        # Perform last book keeping
        train_acc.append(evaluate(student, train_loader, max_ex=100))
        test_acc.append(evaluate(student, test_loader))
        plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
        plot_acc(train_acc, test_acc, it, base_name=output_dir + "acc_"+title, title=title)

class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    lenet = LeNet5(10)
    # ResNet50 setup
    # Modified so it can be used with CIFAR-10
    resnet = ResNet50_CIFAR10()
    for param in resnet.parameters():
        param.requires_grad = False
    resnet = resnet.to(device)

    train_distill(jacobian_loss, resnet, lenet, lr, epochs, 1, "lenet_jac")