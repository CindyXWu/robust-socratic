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

def train_teacher(model, train_loader, test_loader, lr, final_lr, epochs, project, teach_num, exp_num, dataset, save=True):
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
        save_path = output_dir+"teacher_"+teacher_dict[teach_num]+"_"+dataset+"_"+exp_dict[exp_num]+"_working"
        if dataset:
            save_path += "_"+dataset
        torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_hist': train_loss,
                'test_acc': test_acc,
                },
                save_path)
    
    if save:
        save_path = output_dir+"teacher_"+teacher_dict[teach_num]+"_"+dataset+"_"+exp_dict[exp_num]+"_final"
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

def train_distill(teacher, student, train_loader, test_loader, plain_test_loader, box_test_loader, ranbox_test_loader, lr, final_lr, temp, epochs, loss_num, run_name, alpha=None):
    """Train student model with distillation loss.
    Includes LR scheduling. Change loss function as required. 
    N.B. I need to refactor this at some point.
    Student model should be kaiming initialised and teacher model should be pre-trained.
    """
    optimizer = optim.SGD(student.parameters(), lr=lr)
    scheduler = LR_Scheduler(optimizer, epochs, base_lr=lr, final_lr=final_lr, iter_per_epoch=len(train_loader))
    it = 0
    train_acc = []
    test_acc = []
    train_loss = []  # loss at iteration 0
    output_dim = num_classes = len(train_loader.dataset.classes)
    batch_size = train_loader.batch_size
    # Initialise contrastive loss, temp=0.1 (as recommended in paper)
    contrastive_loss = CRDLoss(num_classes, num_classes, T=0.1)
    teacher_test_acc = evaluate(teacher, test_loader, batch_size)
    input_dim = 32*32*3
    it_per_epoch = len(train_loader)
    # weight_reset(student)

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
                # case 3: # Feature map loss - currently only for self-distillation
                #     layer = 'feature_extractor.10'
                # Format below only works if model has a feature extractor method
                #     s_map = student.attention_map(inputs, layer)
                #     t_map = teacher.attention_map(inputs, layer).detach() 
                    # loss = feature_map_diff(scores, targets, s_map, t_map, T=1, alpha=0.2, loss_fn=mse_loss, aggregate_chan=False)
                case 2: # Contrastive distillation
                    layer = {"11.path2.5": "final_features"}
                    s_map = feature_extractor(student, inputs, layer).view(batch_size, -1)
                    t_map = feature_extractor(teacher, inputs, layer).view(batch_size, -1).detach()
                    alpha = 0.5
                    loss = alpha*contrastive_loss(s_map, t_map, labels)+(1-alpha)*base_distill_loss(scores, targets, temp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            lr = scheduler.get_lr()
            train_loss.append(loss.detach().cpu().numpy())

            if it == 0:
                # Check that model is training correctly
                for param in student.parameters():
                    assert param.grad is not None
            if it % 100 == 0:
                batch_size = inputs.shape[0]
                train_acc.append(evaluate(student, train_loader, batch_size, max_ex=100))
                test_acc.append(evaluate(student, test_loader, batch_size))
                plain_acc = evaluate(student, plain_test_loader, batch_size)
                box_acc = evaluate(student, box_test_loader, batch_size)
                randbox_acc = evaluate(student, ranbox_test_loader, batch_size)
                error = teacher_test_acc - test_acc[-1]
                KL_diff = kl_loss(F.log_softmax(scores/temp, dim=1), F.log_softmax(targets/temp, dim=1))
                top1_diff = torch.eq(student_preds, teacher_preds).float().mean()
                print('Iteration: %i, %.2f%%' % (it, test_acc[-1]), "Epoch: ", epoch, "Loss: ", train_loss[-1])
                print("Project {}, LR {}, temp {}".format(run_name, lr, temp))
                wandb.log({
                    "T-S Test Difference": error, 
                    "T-S KL": KL_diff, 
                    "T-S Top 1 Difference": top1_diff, 
                    "S Train": train_acc[-1], 
                    "S Test": test_acc[-1], 
                    "S P Test": plain_acc, 
                    "S B Test": box_acc, 
                    "S RB Test": randbox_acc, 
                    "S Loss": train_loss[-1], 
                    "S LR": lr, },
                    step = it/it_per_epoch
                    )
            it += 1
    
