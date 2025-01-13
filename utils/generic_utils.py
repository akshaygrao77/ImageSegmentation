from sklearn.metrics import accuracy_score,confusion_matrix
import torch
import numpy as np

# Additional helper functions for metrics
def pixel_accuracy(predictions, targets):
    """Calculate Pixel Accuracy"""
    predictions = predictions.view(-1).cpu().numpy().astype(np.int32)
    targets = targets.view(-1).cpu().numpy().astype(np.int32)
    return accuracy_score(targets, predictions)

def dice_coefficient(predictions, targets, num_classes):
    """Calculate Dice Coefficient"""
    dice_scores = []
    for class_id in range(num_classes):
        pred_class = (predictions == class_id).float()
        target_class = (targets == class_id).float()
        intersection = torch.sum(pred_class * target_class)
        union = torch.sum(pred_class) + torch.sum(target_class)
        dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice_score)
    return torch.tensor(dice_scores).mean()

def get_model_from_path(model,chkpath):
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load checkpoint
    checkpoint = torch.load(chkpath, map_location=map_location)
    
    # Load model state dict (strip `module.` for DataParallel models)
    state_dict = checkpoint['model_state_dict']
    
    # If using DataParallel, remove the 'module.' prefix
    if 'module.' in next(iter(state_dict)):
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    epoch = checkpoint['epoch']  # Return the epoch if needed
    
    print(f"Model loaded from {chkpath}")
    
    return model, epoch

def get_loraconfig_from_path(chkpath):
    # Load the checkpoint
    checkpoint = torch.load(chkpath)
    if('lora_config' in checkpoint and checkpoint['lora_config'] is not None):
        return  checkpoint['lora_config']
    
    return None

def get_optimizers_from_path(optimizer, lr_scheduler, chkpath):
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load checkpoint
    checkpoint = torch.load(chkpath, map_location=map_location)
    
    # Load optimizer and scheduler states
    if(optimizer is not None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer state tensors to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(map_location)
    if(lr_scheduler is not None):
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    print(f"Optimizer and LR scheduler loaded from {chkpath}")
    
    return optimizer, lr_scheduler
