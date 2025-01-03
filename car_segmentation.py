import os
import torch
from torch import nn
import numpy as np
import json
from structures.dataset_structure import COCOSegmentationDataset
from utils.data_preprocessor_utils import *
from utils.visualize_utils import *
from torch.utils.data import DataLoader

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AdamW, get_scheduler
from utils.generic_utils import *
from tqdm import tqdm
import evaluate
import wandb
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from structures.heirarchical_seg_model import Hierarchical_SegModel,Fusion_SegModel

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))  # Calculate intersection
        union = torch.sum(probs + targets_one_hot, dim=(2, 3))  # Calculate union
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)  # Dice score
        return 1 - dice_score.mean()  # Dice loss

class IOULoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IOULoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))  # Calculate intersection
        union = torch.sum(probs + targets_one_hot, dim=(2, 3))  # Calculate union
        iou_score = (intersection + self.smooth) / (union + self.smooth)  # Dice score
        return 1 - iou_score.mean()  # Dice loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        # Apply softmax to logits to get predicted probabilities (logits -> probabilities)
        probs = F.softmax(logits, dim=1)
        
        # Select the probabilities corresponding to the true class
        target_probs = probs.gather(1, targets.unsqueeze(1))  # Shape: (B, 1)
        
        # Compute Cross-Entropy Loss (for the true class)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Compute Focal Loss
        focal_loss = self.alpha * (1 - target_probs) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def combined_loss(logits, targets,loss_type, alpha=0.5):
    """
    Combined Cross-Entropy and Dice Loss with proper upsampling of logits.
    """
    # upsampled_logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    # Downsample targets to match the input image size
    # Important: Downsampling targets for loss calculation seems to be better compared to upsampling logits bcoz upsampling logits can introduce artifacts
    targets = F.interpolate(targets.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest").squeeze(1).long()
    
    # Cross-Entropy Loss
    ce_loss = F.cross_entropy(logits, targets, reduction='mean')
    m_loss = 1/(1-alpha)
    if(loss_type == 'dice'):
        # Dice Loss
        m_loss = DiceLoss()(logits, targets)
    elif(loss_type == 'focal'):
        m_loss = FocalLoss()(logits, targets)
    elif(loss_type == 'iou'):
        m_loss = IOULoss()(logits, targets)
    elif(loss_type == 'di_foc'):
        return (1-alpha) * DiceLoss()(logits, targets) + alpha * FocalLoss()(logits, targets)
    elif(loss_type == 'di_iou'):
        return (1-alpha) * DiceLoss()(logits, targets) + alpha * IOULoss()(logits, targets)
    # Combined loss
    return (1-alpha) * ce_loss + alpha * m_loss

def get_segformermodel(num_labels,model_name):
    # nvidia/segformer-b5-finetuned-cityscapes-1024-1024
    model = SegformerForSemanticSegmentation.from_pretrained(model_name,num_labels=num_labels+1,ignore_mismatched_sizes=True)

    return model

def evaluate_model(model,num_labels,val_dataloader):
    metric = evaluate.load("mean_iou")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluation after each epoch
    model.eval()
    total_loss = 0
    avg_pixel_acc = 0
    avg_dice_coeff = 0
    with torch.no_grad():
        for idx,batch in enumerate(tqdm(val_dataloader,desc=f"Evaluating")):
            images, masks = batch
            images = images.to(device)
            masks = masks.squeeze(1).to(device)

            outputs = model(images, labels=masks)
            loss, logits = outputs.loss.mean(), outputs.logits
            total_loss += loss.item()

            with torch.no_grad():
                upsampled_logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)

                # note that the metric expects predictions + labels as numpy arrays
                metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=masks.detach().cpu().numpy())

                # Calculate Pixel Accuracy, Dice Coefficient, Mean Accuracy
                pixel_acc = pixel_accuracy(predicted, masks)
                dice_coeff = dice_coefficient(predicted, masks, num_labels+1)
                avg_pixel_acc += pixel_acc
                avg_dice_coeff += dice_coeff

            # let's print loss and metrics every 100 batches
            if idx % 10 == 0:
                # currently using _compute instead of compute
                # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
                metrics = metric._compute(
                        predictions=predicted.cpu(),
                        references=masks.cpu(),
                        num_labels=num_labels+1,
                        ignore_index=None,
                        reduce_labels=False
                    )

    avg_val_loss = total_loss / len(val_dataloader)
    print(f"Validation loss: {avg_val_loss} mean_iou :{metrics['mean_iou']}, mean_accuracy :{metrics['mean_accuracy']} val_pixel_accuracy: {avg_pixel_acc / len(val_dataloader)} val_dice_coeff : {avg_dice_coeff / len(val_dataloader)}")

    return avg_val_loss,metrics["mean_iou"],metrics["mean_accuracy"],avg_pixel_acc / len(val_dataloader), avg_dice_coeff / len(val_dataloader)

def train_model(model,optimizer,lr_scheduler,num_labels,num_epochs,train_dataloader,val_dataloader,model_path,wand_project_name=None,start_epoch=0,loss_type=None,alpha=0.5):
    is_log_wandb = not(wand_project_name is None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        avg_pixel_acc = 0
        avg_dice_coeff = 0
        metric = evaluate.load("mean_iou")
        for idx, batch in enumerate(progress_bar):
            images, masks = batch
            images = images.to(device)
            masks = masks.squeeze(1).to(device)

            assert masks.max() <= num_labels, f"Mask contains invalid class index: {masks.max()}"
            assert masks.min() >= 0, "Mask contains negative class indices"
            assert images.size()[2:] == masks.size()[1:], "Size mismatch between mask and images"

            # Forward pass
            outputs = model(images, labels=masks)
            loss, logits = outputs.loss.mean(), outputs.logits

            if(loss_type is not None):
                loss = combined_loss(logits, masks,loss_type,alpha)

            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Evaluation metrics
            with torch.no_grad():
                upsampled_logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                predicted = upsampled_logits.argmax(dim=1)
                # print(f"Logits shape after interpolation: {upsampled_logits.shape}, Predicted shape: {predicted.shape}")

                # Ensure predicted is the same shape as masks (batch_size, height, width)
                assert predicted.shape == masks.shape, "Predicted shape doesn't match masks shape"

                # Add predictions and ground truth to metric (for Mean IoU)
                metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=masks.detach().cpu().numpy())
            
            total_loss += loss.item()
            # Log metrics every 10 batches
            if idx % 10 == 0:
                metrics = metric._compute(
                    predictions=predicted.cpu(),
                    references=masks.cpu(),
                    num_labels=num_labels + 1,
                    ignore_index=None,
                    reduce_labels=False
                )
                del predicted, masks
                torch.cuda.empty_cache()

            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "mean_iou": metrics["mean_iou"],
                "mean_accuracy": metrics["mean_accuracy"]
            })

        # Save model and optimizer state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }, model_path + "_ep_" + str(epoch) + ".pt")

        # Log validation performance
        current_lr = optimizer.param_groups[0]['lr']
        scheduler_type = str(lr_scheduler)
        train_loss, train_iou, train_acc,train_pixel_accuracy,train_dice_coeff = evaluate_model(model, num_labels, train_dataloader)
        val_loss, val_iou, val_acc,val_pixel_accuracy,val_dice_coeff = evaluate_model(model, num_labels, val_dataloader)

        if is_log_wandb:
            wandb.log({
                "current_epoch": epoch,
                'learning_rate': current_lr,
                'scheduler_type': scheduler_type,
                "val_iou": val_iou,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "mean_iou": train_iou,
                "mean_accuracy": train_acc,
                "pixel_accuracy": train_pixel_accuracy,
                "dice_coeff": train_dice_coeff,
                "val_pixel_accuracy": val_pixel_accuracy,
                "val_dice_coeff": val_dice_coeff,
                "loss": train_loss
            })
    
    return 

if __name__ == '__main__':
    os.environ["TMPDIR"] = "./tmp"
    wand_project_name = None
    wand_project_name="Car_Damage_Segmentation"
    # dice, focal , None , di_foc , iou , di_iou
    loss_type = None
    alpha = 0.5

    # None, 'hierarchical' , 'fusion'
    model_type = 'fusion'

    # Car_damages_dataset, Car_parts_dataset
    dataset = "Car_damages_dataset"

    coco_path = get_cocopath(dataset)
    pretrained_model_name = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    # pretrained_model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
    datadir = "./data/car-parts-and-car-damages/"

    car_dir = os.path.join(datadir,dataset)
    car_imgs = os.path.join(car_dir,"split_dataset")
    car_anns = os.path.join(car_dir,"split_annotations")

    # Important: BS below 16 causes performance degradation
    batch_size = 16
    num_epochs = 100

    # Get the colormapping from labelID of segmentation classes to color
    car_id_to_color = get_colormapping(os.path.join(car_dir,coco_path),car_dir+"/meta.json")

    train_car_dataset = get_dataset(car_imgs,car_anns,is_train=True)
    val_car_dataset = get_dataset(car_imgs,car_anns)

    tr_cd_dataloader = DataLoader(train_car_dataset, batch_size=batch_size, shuffle=True,num_workers=8,pin_memory=True)
    val_cd_dataloader = DataLoader(val_car_dataset, batch_size=batch_size,num_workers=8,pin_memory=True)

    start_net_path = None
    start_net_path = "./checkpoints/Car_damages_dataset/fusi/default/nvidia_segformer-b3-finetuned-cityscapes-1024-1024_ep_19.pt"

    continue_run_id = None
    continue_run_id = "28z0hr4f"
    
    superseg_model_name = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    super_segmodel_path = "./checkpoints/Car_parts_dataset/nvidia_segformer-b3-finetuned-cityscapes-1024-1024_ep_90.pt"

    start_epoch = 0
    if(model_type is None):
        model = get_segformermodel(len(car_id_to_color),pretrained_model_name)
    elif(model_type == 'hierarchical' or model_type == 'fusion'):
        superseg_ds = "Car_parts_dataset"
        superseg_dir = os.path.join(datadir,superseg_ds)
        superseg_id_to_color = get_colormapping(os.path.join(superseg_dir,get_cocopath(superseg_ds)),superseg_dir+"/meta.json")
        super_segmodel = get_segformermodel(len(superseg_id_to_color),superseg_model_name)
        super_segmodel,_ = get_model_from_path(super_segmodel,super_segmodel_path)
        if(model_type=='hierarchical'):
            model = Hierarchical_SegModel(super_segmodel,len(superseg_id_to_color)+1,len(car_id_to_color)+1,pretrained_model_name)
        elif(model_type == 'fusion'):
            model = Fusion_SegModel(super_segmodel,len(superseg_id_to_color)+1,len(car_id_to_color)+1,pretrained_model_name)
    
    if(start_net_path is not None):
        model,start_epoch = get_model_from_path(model,start_net_path)
    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    # Set up the learning rate scheduler
    num_training_steps = num_epochs * len(tr_cd_dataloader)
    print("num_training_steps ",num_training_steps,num_epochs * len(tr_cd_dataloader),car_id_to_color)
    # lr_scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=200,
    #     num_training_steps=num_training_steps,
    # )

    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=200,
        num_training_steps=num_training_steps,
        num_cycles=10
    )
    
    optimizer,lr_scheduler = get_optimizers_from_path(optimizer, lr_scheduler, start_net_path)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    if device_str == 'cuda':
        if(torch.cuda.device_count() > 1):
            print("Parallelizing model")
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.to(device)
    print(model)
    model_save_dir = os.path.join(os.path.join("./checkpoints/",dataset+("" if model_type is None else "/"+model_type[:4])),"default" if loss_type is None else (loss_type+"_"+str(alpha)))
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir,pretrained_model_name.replace("/","_"))
    is_log_wandb = not(wand_project_name is None)
    if(is_log_wandb):
        wandb_config = dict()
        wandb_config["optimizer"] = optimizer
        wandb_config["final_model_save_path"] = model_save_path
        wandb_config["num_epochs"] = num_epochs
        wandb_config["batch_size"] = batch_size
        wandb_config["model_name"] = pretrained_model_name
        wandb_config["dataset"] = dataset
        wandb_config["start_net_path"] = start_net_path
        wandb_config["loss_type"] = loss_type
        wandb_config["alpha"] = alpha
        wandb_config["model_type"]=model_type
        wandb_run_name = ("" if model_type is None else model_type[:4]+"_")+("DMG" if "damage" in dataset else "PRT") +"_"+ pretrained_model_name[pretrained_model_name.find("segformer")+len("segformer")+1:pretrained_model_name.find("finetun")-1]+"_"+pretrained_model_name[pretrained_model_name.find("finetun")+len("finetuned")+1:][:4]+ "_"+("def" if loss_type is None else loss_type+"_"+str(alpha))

        if(continue_run_id is None):
            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                config=wandb_config,
            )
        else:
            wandb.init(
                project=f"{wand_project_name}",
                name=f"{wandb_run_name}",
                config=wandb_config,
                id=continue_run_id,  # ID of the previous run
                resume="allow"     # Use "must" to enforce resumption or "allow" to create a new run if not found
            )

    train_model(model,optimizer,lr_scheduler,len(car_id_to_color),num_epochs,tr_cd_dataloader,val_cd_dataloader,model_save_path,wand_project_name,start_epoch,loss_type,alpha)


