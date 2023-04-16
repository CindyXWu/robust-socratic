import torch
from torch import nn, optim
import os
import wandb
from tqdm import tqdm

from image_models import *
from plotting import *
from jacobian import *
from contrastive import *
from feature_match import *
from utils_ekdeep import *
from info_dicts import *
from shapes_3D import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate losses
kl_loss = nn.KLDivLoss(reduction='mean', log_target=True)
ce_loss = nn.CrossEntropyLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')

output_dir = "Image_Experiments/"
# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

@torch.no_grad()
def evaluate(model, dataset, batch_size, max_ex=0):
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
    return (acc*100 / ((i+1)*batch_size))

def weight_reset(model):
    """Reset weights of model at start of training."""
    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        # Initialise
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

def train_teacher(model, train_loader, test_loader, lr, final_lr, epochs, project, base_path):
    """Fine tune a pre-trained teacher model for specific downstream task, or train from scratch."""
    optimizer = optim.SGD(model.parameters(), lr=lr)
    it = 0
    scheduler = LR_Scheduler(optimizer, epochs, base_lr=lr, final_lr=final_lr, iter_per_epoch=len(train_loader))

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        train_acc = []
        test_acc = []
        train_loss = []
        model.train()
        
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores = model(inputs)

            optimizer.zero_grad()
            loss = ce_loss(scores, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = scheduler.get_lr()
            train_loss.append(loss.detach().cpu().numpy())
            
            if it % 100 == 0:
                batch_size = inputs.shape[0]
                train_acc.append(evaluate(model, train_loader, batch_size, max_ex=100))
                test_acc.append(evaluate(model, test_loader, batch_size))
                print('Iteration: %i, %.2f%%' % (it, test_acc[-1]), "Epoch: ", epoch)
                print("Project {}, LR {}".format(project, lr))
                wandb.log({"Test Accuracy": test_acc[-1], "Loss": train_loss[-1], "LR": lr})
            it += 1

        # Checkpoint model at end of every epoch
        # Have two models: working copy (this one) and final fetched model
        # Save optimizer in case re-run, save test accuracy to compare models
        save_path = base_path+"_working"
        torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_hist': train_loss,
                'test_acc': test_acc,
                },
                save_path)
    
    save_path = base_path+"_final"
    torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_hist': train_loss,
            'test_acc': test_acc,
            },
            save_path)

def base_distill_loss(scores, targets, temp):
    scores = scores/temp
    targets = F.softmax(targets/temp).argmax(dim=1)
    return ce_loss(scores, targets)

def train_distill(teacher, student, train_loader, test_loader, base_dataset, lr, final_lr, temp, epochs, loss_num, run_name, alpha=None, tau=None, s_layer=None, t_layer=None):
    """Train student model with distillation loss.
    Includes LR scheduling. Change loss function as required. 
    Args:
        tau: contrastive loss temperature
        temp: base distillation loss temperature
        alpha: contrastive and Jacobian loss weight
        s_layer, t_layer: layer to extract features from for contrastive loss
        loss_num: index of loss dictionary (in info_dicts.py) describing which loss fn to use
        base_dataset: tells us which dataset out of CIFAR10, CIFAR100, Dominoes and Shapes to use
    """
    optimizer = optim.SGD(student.parameters(), lr=lr)
    scheduler = LR_Scheduler(optimizer, epochs, base_lr=lr, final_lr=final_lr, iter_per_epoch=len(train_loader))
    it = 0
    output_dim = len(train_loader.dataset.classes)
    # weight_reset(student)
    sample = next(iter(train_loader))
    batch_size, c, w, h = sample[0].shape
    input_dim = c*w*h
    teacher_test_acc = evaluate(teacher, test_loader, batch_size)

    # Format: {'Mechanism name as string': dataloader}
    dataloaders = {}
    if base_dataset in ['CIFAR10', 'CIFAR100']:
        dict_name = cifar_exp_dict
    elif base_dataset == 'Dominoes':
        dict_name = dominoes_exp_dict
    for key in dict_name:
        dataloaders[dict_name[key]] = create_dataloader(base_dataset=base_dataset, EXP_NUM=key, batch_size=batch_size, mode='test')

    for epoch in range(epochs):
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            inputs.requires_grad = True
            labels = labels.to(device)
            scores = student(inputs)
            targets = teacher(inputs) 
            student_preds = scores.argmax(dim=1)
            teacher_preds = targets.argmax(dim=1)

            # for param in student.parameters():
            #     assert param.requires_grad
            match loss_num:
                case 0: # Base distillation loss
                    loss = base_distill_loss(scores, targets, temp)
                case 1: # Jacobian loss   
                    output_dim = scores.shape[1]
                    loss = jacobian_loss(scores, targets, inputs, T=1, alpha=alpha, batch_size=batch_size, loss_fn=mse_loss, input_dim=input_dim, output_dim=output_dim)
                case 2: # Contrastive distillation
                    s_map = feature_extractor(student, inputs, s_layer).view(batch_size, -1)
                    t_map = feature_extractor(teacher, inputs, t_layer).view(batch_size, -1).detach()
                    # Initialise contrastive loss, temp=0.1 (as recommended in paper)
                    contrastive_loss = CRDLoss(s_map.shape[1], t_map.shape[1], T=tau)
                    loss = alpha*contrastive_loss(s_map, t_map, labels)+(1-alpha)*base_distill_loss(scores, targets, temp)
                # case 3: # Feature map loss - currently only for self-distillation
                #     layer = 'feature_extractor.10'
                # Format below only works if model has a feature extractor method
                #     s_map = student.attention_map(inputs, layer)
                #     t_map = teacher.attention_map(inputs, layer).detach() 
                    # loss = feature_map_diff(scores, targets, s_map, t_map, T=1, alpha=0.2, loss_fn=mse_loss, aggregate_chan=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = scheduler.get_lr()
            train_loss = loss.detach().cpu().numpy()

            if it == 0:
                # Check that model is training correctly
                for param in student.parameters():
                    assert param.grad is not None
            if it % 100 == 0:
                batch_size = inputs.shape[0]
                train_acc = evaluate(student, train_loader, batch_size, max_ex=100)
                test_acc = evaluate(student, test_loader, batch_size)
                for key in dataloaders:
                    wandb.log({key: evaluate(student, dataloaders[key], batch_size)})
                error = teacher_test_acc - test_acc
                KL_diff = kl_loss(F.log_softmax(scores/temp, dim=1), F.log_softmax(targets/temp, dim=1))
                top1_diff = torch.eq(student_preds, teacher_preds).float().mean()
                print('Iteration: %i, %.2f%%' % (it, test_acc), "Epoch: ", epoch, "Loss: ", train_loss)
                print("Project {}, LR {}, temp {}".format(run_name, lr, temp))
                wandb.log({
                    "T-S KL": KL_diff, 
                    "T-S Top 1 Fidelity": top1_diff, 
                    "S Train": train_acc, 
                    "S Test": test_acc, 
                    "T-S Test Difference": error, 
                    "S Loss": train_loss, 
                    "S LR": lr, }
                    )
            it += 1
    
def create_dataloader(base_dataset, EXP_NUM, batch_size, spurious_corr=1.0, mode='train'):
    """Set train and test loaders based on dataset and experiment. Used both for training and evaluation of counterfactuals."""
    if base_dataset in ["CIFAR10", "CIFAR100"]:
        randomize_cue = False
        match EXP_NUM:
            case 0:
                cue_type = 'nocue'
            case 1:
                cue_type = 'box'
            case 2: 
                cue_type = 'box'
                randomize_cue = True
        train_loader = get_box_dataloader(load_type='train', base_dataset=base_dataset, cue_type=cue_type, cue_proportion=spurious_corr, randomize_cue=randomize_cue, batch_size=batch_size)
        test_loader = get_box_dataloader(load_type ='test', base_dataset=base_dataset, cue_type=cue_type, cue_proportion=spurious_corr, randomize_cue=randomize_cue, batch_size=batch_size)

    if base_dataset == "Dominoes":
        randomize_cues = [False, False]
        randomize_img = False
        match EXP_NUM:
            case 0:
                cue_proportions = [0.0, 0.0]
            case 1:
                cue_proportions = [1.0, 0.0]
                randomize_img = True
            case 2:
                cue_proportions = [0.0, 1.0]
                randomize_img = True
            case 3:
                randomize_img = True
                cue_proportions = [1.0, 1.0]
            case 4:
                cue_proportions = [1.0, 0.0]
            case 5:
                cue_proportions = [0.0, 1.0]
            case 6:
                cue_proportions = [1.0, 1.0]
        train_loader = get_box_dataloader(load_type='train', base_dataset='Dominoes', batch_size=batch_size, randomize_img=randomize_img, cue_proportions=cue_proportions, randomize_cues=randomize_cues)
        test_loader = get_box_dataloader(load_type='test', base_dataset='Dominoes', batch_size=batch_size, randomize_img=randomize_img, cue_proportions=cue_proportions, randomize_cues=randomize_cues)

    if base_dataset == 'Shapes':
        randomise = False
        match EXP_NUM:
            case 0:
                mechanisms = []
            case 1:
                mechanisms = [0]
                randomise = True
            case 2:
                mechanisms = [3]
                randomise = True
            case 3:
                randomise = True
                mechanisms = [0, 3]
            case 4:
                mechanisms = [0]
            case 5:
                mechanisms = [3]
            case 6:
                mechanisms = [0, 3]
        train_loader = dataloader_3D_shapes('train', batch_size=batch_size, randomise=randomise, mechanisms=mechanisms)
        test_loader = dataloader_3D_shapes('test', batch_size=batch_size, randomise=randomise, mechanisms=mechanisms)

    if mode == 'test':
        return test_loader
    return train_loader, test_loader