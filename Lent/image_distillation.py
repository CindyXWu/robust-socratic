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
epochs = 1
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    return trainset, testset, trainloader, testloader

def evaluate(model, dataset, max_ex=0):
    """Evaluate model accuracy on dataset."""
    acc = 0
    for i, (features, labels) in enumerate(dataset):
        labels = labels.to(device)
        features = features.to(device)
        # Batch size in length, varying from 0 to 1
        scores = nn.functional.softmax(model(features.to(device)), dim=1)
        _, pred = torch.max(scores, 1)
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
                loss.backward()
                optimizer.zero_grad()
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

def jacobian_loss(scores, targets, inputs, T, student, teacher, alpha, loss_fn):
    """Eq 10 adapted for input-output Jacobian matrix, not vector of output wrt largest pixel in attention map.

    See function below for adapation with attention maps.
    No hard targets used, purely distillation loss.
    Args:
        scores: torch tensor, output of the student model
        targets: torch tensor, output of the teacher model
        inputs: torch tensor, input to the model
        T: float, temperature
        s_jac: torch tensor, Jacobian matrix of the student model
        t_jac: torch tensor, Jacobian matrix of the teacher model
        alpha: float, weight of the jacobian penalty
        loss_fn: base loss function to be used for the input-output distillation loss - MSE, BCE, KLDiv
    """
    soft_pred = nn.functional.softmax(scores/T, dim=1)
    soft_targets = nn.functional.softmax(targets/T, dim=1)
    # Change these two lines of code depending on which Jacobian you want to use
    s_jac = get_approx_jacobian(student, inputs)
    t_jac = get_approx_jacobian(teacher, inputs)
    diff = s_jac/torch.norm(s_jac, 2) - t_jac/torch.norm(t_jac, 2)
    jacobian_loss = torch.norm(diff, 2)**2
    loss = (1-alpha) * T**2 * loss_fn(soft_pred, soft_targets) + alpha * jacobian_loss
    return loss

def get_approx_jacobian(model, x):
    """Rather than computing Jacobian for all output classes, compute for most probable class.
    
    Required due to computational constraints for Jacobian computation.
    Args:
        model: torch model
        x: torch tensor, input to the model
    Returns:
        jacobian: 1D torch tensor, vector of derivative of most probable class wrt input
    """
    batch_size = x.size(0)
    # numel returns the number of elements in the tensor
    input_dim = x.numel() // batch_size
    x = x.requires_grad_()
    y = model(x)
    jacobian = torch.zeros(batch_size, input_dim, device=x.device)
    output_dim = y.numel() // batch_size
    grad_output = torch.zeros(batch_size, output_dim, device=x.device)
    # Index of most likely class
    i = torch.argmax(y, dim=1)
    grad_output[:, i] = 1
    grad_input, = torch.autograd.grad(y, x, grad_output, retain_graph=True)
    jacobian = grad_input.view(batch_size, input_dim)
    return jacobian
    
if __name__ == "__main__":
    lenet = LeNet5(10)
    # ResNet50 setup
    # Modified so it can be used with CIFAR-10
    resnet = ResNet50_CIFAR10()
    for param in resnet.parameters():
        param.requires_grad = False
    resnet = resnet.to(device)

    train_distill(jacobian_loss, resnet, lenet, lr, epochs, 1, "lenet_jac")