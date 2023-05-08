import torch
from torch import nn, optim
import wandb
from tqdm import tqdm
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Tuple
from models.image_models import *
from plotting import *
from losses.jacobian import *
from losses.contrastive import *
from losses.feature_match import *
from datasets.utils_ekdeep import *
from datasets.shapes_3D import *
from info_dicts import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate losses
kl_loss = nn.KLDivLoss(reduction='batchmean')
ce_loss = nn.CrossEntropyLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')

@torch.no_grad()
def evaluate(model, dataset, batch_size, max_ex=0, title=None):
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
    if title:
        plot_images(dataset, num_images=batch_size, title=title)
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
            inputs, labels = inputs.to(device), labels.to(device)
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
    scores = F.log_softmax(scores/temp)
    targets = F.softmax(targets/temp)
    return kl_loss(scores, targets)


def train_distill(
    teacher: Module, 
    student: Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    base_dataset: str, 
    lr: float, 
    final_lr: float, 
    temp: float, 
    epochs: int, 
    loss_num: int, 
    run_name: str, 
    alpha: float = None, 
    tau: float = None, 
    s_layer: float = None, 
    t_layer: float = None
    ) -> None:
    """
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
    teacher.eval()

    dataloaders = get_counterfactual_dataloaders(base_dataset, batch_size)

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

            # if it == 0: check_grads(student)
            if it % 100 == 0:
                batch_size = inputs.shape[0]
                train_acc = evaluate(student, train_loader, batch_size, max_ex=100)
                test_acc = evaluate(student, test_loader, batch_size)
                for name in dataloaders:
                    title = 'Dominoes_'+name
                    # Currently not plotting datasets
                    wandb.log({name: evaluate(student, dataloaders[name], batch_size, title=None)})
                error = teacher_test_acc - test_acc
                KL_diff = kl_loss(F.log_softmax(scores, dim=1), F.softmax(targets, dim=1))
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
        # Visualise 3d at end of each epoch
        if loss_num == 2:
            visualise_features_3d(s_map, t_map, title=run_name+"_"+str(it))


def check_grads(model):
    for param in model.parameters():
        assert param.grad is not None


def get_counterfactual_dataloaders(base_dataset: str, batch_size: int) -> dict[str, DataLoader]:
    """Get dataloaders for counterfactual evaluation. Key of dictionary tells us which settings for counterfactual evals are used."""
    dataloaders = {}
    for i, key in enumerate(counterfactual_dict_all):
        dataloaders[key], _ = create_dataloader(base_dataset=base_dataset, EXP_NUM=i, batch_size=batch_size, counterfactual=True)
    return dataloaders


def create_dataloader(base_dataset: Dataset, EXP_NUM: int, batch_size: int = 64, counterfactual: bool = False) -> Tuple[DataLoader, DataLoader]:
    """Set train and test loaders based on dataset and experiment.
    Used only for training and testing, not counterfactual evals.
    For generating dominoes: box is cue 1, MNIST is cue 2.
    For CIFAR100 box: box is cue 1.
    """
    if counterfactual:
        key = list(counterfactual_dict_all.keys())[EXP_NUM]
        mech_1_frac, mech_2_frac, randomize_mech_1, randomize_mech_2, randomize_img = counterfactual_dict_all[key]
    else:
        key = list(exp_dict_all.keys())[EXP_NUM]
        mech_1_frac, mech_2_frac, randomize_mech_1, randomize_mech_2, randomize_img = exp_dict_all[key]

    if base_dataset in ["CIFAR10", "CIFAR100"]:
        cue_type='box' if mech_1_frac != 0 else 'nocue'
        train_loader = get_box_dataloader(load_type='train', base_dataset=base_dataset, cue_type=cue_type, cue_proportion=mech_1_frac, randomize_cue=randomize_mech_1, randomize_img = randomize_img, batch_size=batch_size)
        test_loader = get_box_dataloader(load_type ='test', base_dataset=base_dataset, cue_type=cue_type, cue_proportion=mech_1_frac, randomize_cue=randomize_mech_1, randomize_img = randomize_img, batch_size=batch_size)

    elif base_dataset == "Dominoes":
        randomize_cues = [randomize_mech_1, randomize_mech_2]
        train_loader = get_box_dataloader(load_type='train', base_dataset='Dominoes', batch_size=batch_size, randomize_img=randomize_img, box_frac=mech_1_frac, mnist_frac=mech_2_frac, randomize_cues=randomize_cues)
        test_loader = get_box_dataloader(load_type='test', base_dataset='Dominoes', batch_size=batch_size, randomize_img=randomize_img, box_frac=mech_1_frac, mnist_frac=mech_2_frac, randomize_cues=randomize_cues)

    elif base_dataset == 'Shapes':
        train_loader = dataloader_3D_shapes('train', batch_size=batch_size, randomise=randomize_img, floor_frac=mech_1_frac, scale_frac=mech_2_frac)
        test_loader = dataloader_3D_shapes('test', batch_size=batch_size, randomise=randomize_img, floor_frac=mech_1_frac, scale_frac=mech_2_frac)
 
    return train_loader, test_loader


if __name__ == "__main__":
    scores = torch.randn(10, 10)
    targets = torch.randn(10, 10)
    loss = base_distill_loss(scores, targets, 1)
    print(loss)