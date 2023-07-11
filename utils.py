import torch
from torch import nn
from tqdm import tqdm
from config import config

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def save_model_checkpoint(model, optimizer, epoch, val_acc, path, scheduler=None):
    """
     function for saving the model, and optimizer states and the epoch
     :param model: the model to save
     :param optimizer: the optimizer to save
     :param epoch: the epoch at which we save model state
     :param val_acc: validation accuracy at which the model is being saved
     :param path: the path on the drive where to save the checkpoint
    """
    data = {}
    data["model_state"] = model.to('cpu').state_dict()
    data["optimizer_state"] = optimizer.state_dict()
    data["epoch"] = epoch
    data["val_acc"] = val_acc
    if scheduler is not None:
        data["scheduler_state"] = scheduler.state_dict()
        
    torch.save(data, path)

def load_model_checkpoint(model, optimizer, path, scheduler=None):
    """
     function for loading the checkpoint for model state and optimizer
     :param model: the model to load state to
     :param optimzier: the optimizer to load state to
     :param path: the path in the drive where to load checkpoint from
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint["epoch"]
    val_acc = checkpoint["val_acc"]
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        return model, optimizer, epoch, scheduler
    return model, optimizer, epoch, val_acc



def calculate_loss(logits, true_labels):
    """
     function for calculating the loss between predicted labels and true labels
     the function uses cross entropy loss creterio
     :param lagits: are the predictions from the model
     :param true_labels: are the expected label
     
     return: calculated loss
    """
    loss = nn.CrossEntropyLoss()
    return loss(logits, true_labels)


def calculate_accuracy(logits, true_labels):
    """
     function for calculating the accuracy for each batch
     :param lagits: are the predictions from the model
     :param true_labels: are the expected label
    """
    predicted_classes = logits.argmax(dim=1)
    true_classes = true_labels.argmax(dim=1)
    correct_predictions = (predicted_classes == true_classes).sum().item()
    
    return correct_predictions


def training(model, optimizer, train_data_loader, scaler):
    """
     function for training the model
     :param model: the model to be trained
     :param optimizer: optimization algorithm to use for updating the model parameters
     :param train_data_loader: the loader containing data to be used during training
     :param sclaler: float precision alogrithm
    """
    model = model.to(device)
    model.train()  # putting model in the training mode
    num_correct = 0
    training_loss = 0
    
    # progress bar for batch
    batch_bar = tqdm(total=len(train_data_loader), dynamic_ncols=True, leave=False, position=0, ncols=5, 
                     desc="Train", unit=" images")
    for index, (images, true_labels) in enumerate(train_data_loader):
        images, true_labels = images.to(device, non_blocking=True), true_labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss = calculate_loss(logits, true_labels)
       
        training_loss += float(loss.item())
        num_correct += int(calculate_accuracy(logits, true_labels))
        
        batch_bar.set_postfix(
            accuracy="{}%".format(round((100*num_correct) / (config['batch_size'] * (index + 1)), 4)),
            num_correct=num_correct,
            loss= "{}".format(round(training_loss / (index + 1), 4)),
            learning_rate=f"{optimizer.param_groups[0]['lr']}"
        )
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        batch_bar.update()
    batch_bar.close()
    acc = (100 * num_correct) / (len(train_data_loader) * config['batch_size'])
    training_loss = training_loss / len(train_data_loader) 
    
    return acc, training_loss 


def validating(model, valid_data_loader):
    """
     function for training the model
     :param model: the model to be validated
     :param valid_data_loader: the loader containing data to be used during validation
    """
    model = model.to(device, non_blocking=True)
    model.eval() # putting the model into evaluation mode
    validation_loss, num_correct = 0, 0
    batch_bar = tqdm(total=len(valid_data_loader), desc="Valid", position=0, dynamic_ncols=True, 
                     ncols=7, leave=False, unit=' images')
    for index, (images, true_labels) in enumerate(valid_data_loader):
        images, true_labels = images.to(device, non_blocking=True), true_labels.to(device)
        with torch.inference_mode():
            logits = model(images)
            loss = loss = calculate_loss(logits, true_labels)
        
        validation_loss += float(loss.item())
        num_correct += int(calculate_accuracy(logits, true_labels))
        batch_bar.set_postfix(
            accuracy="{}%".format(round((100*num_correct) / (config['batch_size'] * (index + 1)), 4)),
            num_correct=num_correct,
            loss= "{}".format(round(validation_loss / (index + 1), 4))
        )
        
        batch_bar.update()
    batch_bar.close()
    valid_acc = (100 * num_correct) / (config['batch_size'] * len(valid_data_loader))
    validation_loss =  validation_loss / len(valid_data_loader)
    
    return valid_acc, validation_loss 


def test_model(model, test_data_loader):
    """
     function for training the model
     :param model: the model to be tested
     :param test_data_loader: the loader containing data to be used during testing 
    """
    model = model.to(device, non_blocking=True)
    model.eval()
    test_results = []
    
    batch_bar = tqdm(total=len(test_data_loader), position=0, desc="Test", dynamic_ncols=True, leave=False, ncols=7)
    for images in test_data_loader:
        images = model(images, non_blocking=True)
        with torch.inference_mode():
            logits = model(images)
        num_correct = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        test_results.extend(num_correct)
        batch_bar.update()
    batch_bar.close()
    
    return test_results