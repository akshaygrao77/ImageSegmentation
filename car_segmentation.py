import os
import torch
from torch import nn
import numpy as np
import json
from structures.dataset_structure import COCOSegmentationDataset
from structures.heirarchical_seg_model import Fusion_SegModel, Hierarchical_SegModel, MOE_Fusion_SegModel, modify_segformer_output_channels
from utils.data_preprocessor_utils import *
from utils.visualize_utils import *
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AdamW, get_scheduler
from utils.generic_utils import *
from tqdm import tqdm
import evaluate
import wandb
import torch.nn.functional as F
# from sklearn.metrics import accuracy_score, confusion_matrix

import torch.nn.functional as F
from accelerate import Accelerator

import torchmetrics.functional as F_metrics
from torchmetrics.functional import dice, accuracy, jaccard_index

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
    def __init__(self, alpha=1, gamma=2, reduction='mean',class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, logits, targets):
        # Apply softmax to logits to get predicted probabilities (logits -> probabilities)
        probs = F.softmax(logits, dim=1)
        targets = targets.long()

        # Select the probabilities corresponding to the true class
        target_probs = probs.gather(1, targets.unsqueeze(1))  # Shape: (B, 1)

        # Compute Cross-Entropy Loss (for the true class)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            ce_loss = ce_loss * weights

        # Compute Focal Loss
        focal_loss = self.alpha * (1 - target_probs) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def combined_loss(logits, targets, loss_type, dataset, alpha=0.5):
    """
    Combined Cross-Entropy and Dice Loss with proper upsampling of logits.
    """
    # upsampled_logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    # Downsample targets to match the input image size
    # Important: Downsampling targets for loss calculation seems to be better compared to upsampling logits bcoz upsampling logits can introduce artifacts
    targets = F.interpolate(targets.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest").squeeze(1).long()
    
    if dataset == "Car_damages_dataset":
        # Median balanced weights pre-computed
        median_weights = torch.tensor([0.007330882283506196,0.28334594392635665,0.13777006322875054,0.8670423260239092,5.213402101030479,10.336641529852535,1.0,4.878550231779302,1.6600393131311797], device=logits.device)
    elif dataset == "CarDNN_Kaggle_merged_Car_damages_dataset":
        # Median balanced weights pre-computed
        median_weights = torch.tensor([0.00217733043830974,0.03197547010896361,0.012607789107358536,2.0573060536004006,12.370288485786435,24.526640228491583,0.03335967671433147,11.575756596173864,1.0], device=logits.device)
        # Inverse log normalized weights pre-computed
        inverse_normalized_weights = torch.tensor([0.7249, 0.8299, 0.7903, 1.0704, 1.2231, 1.2935, 0.8318, 1.2167, 1.0193], device=logits.device)
    elif dataset == "Car_parts_dataset":
        # Median balanced weights pre-computed
        median_weights = torch.tensor([0.04713219181108924,0.46780447957804006,0.8226677237979122,2.295045034695851,1.6657497540963886,0.3523892849398128,0.6480218689411196,0.7877552602216747,0.3301445520518123,1.5692173388105741,4.9711122707088915,3.11386416386265,1.6210466382167426,0.49036456494308545,1.1161060908411526,3.5601432517995577,0.7916959768684058,6.8852797356351045,2.837301590521953,0.6831727317740973,0.9057742712114433,1.854989695026368], device=logits.device)
    ce_weights = None
    if "wt_" in loss_type:
        if "wt_i_" in loss_type:
            ce_weights = inverse_normalized_weights
            loss_type = loss_type.replace("wt_i_","")
        else:
            ce_weights = median_weights
            loss_type = loss_type.replace("wt_","")
    
    ce_loss = F.cross_entropy(logits, targets, reduction='mean',weight=ce_weights)
    m_loss = 1 / (1 - alpha)
    if loss_type == 'dice':
        # Dice Loss
        m_loss = DiceLoss()(logits, targets)
    elif loss_type == 'focal':
        m_loss = FocalLoss(class_weights=ce_weights)(logits, targets)
    elif loss_type == 'iou':
        m_loss = IOULoss()(logits, targets)
    elif loss_type == 'di_foc':
        return (1 - alpha) * DiceLoss()(logits, targets) + alpha * FocalLoss(class_weights=ce_weights)(logits, targets)
    elif loss_type == 'di_iou':
        return (1 - alpha) * DiceLoss()(logits, targets) + alpha * IOULoss()(logits, targets)
    # Combined loss
    return (1 - alpha) * ce_loss + alpha * m_loss

def get_segformermodel(num_labels, model_name):
    # nvidia/segformer-b5-finetuned-cityscapes-1024-1024
    model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=num_labels + 1,
                                                           ignore_mismatched_sizes=True)

    return model

def gather_if_needed(tensor, accelerator):
    if accelerator is not None:
        return accelerator.gather_for_metrics(tensor)
    return tensor

def evaluate_model(model, num_labels, val_dataloader, accelerator=None, is_return_metric_obj=False):
    device = accelerator.device if accelerator is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    total_loss = 0.0
    total_correct_pixels = 0
    total_pixels = 0
    avg_val_loss = 0.
    mean_iou = 0.
    mean_accuracy = 0.
    overall_pixel_acc = 0.
    mean_dice = 0.
    metrics_dict={}

    total_confusion_matrix = None
    if accelerator is None or accelerator.is_main_process:
         total_confusion_matrix = torch.zeros(num_labels+1, num_labels+1, dtype=torch.long, device=device)

    for idx, batch in enumerate(tqdm(val_dataloader, desc="Evaluating", disable=(accelerator is not None and not accelerator.is_main_process))):
        with torch.no_grad():
            images, masks = batch
            masks = masks.squeeze(1)

            if accelerator is None:
                images = images.to(device)
                masks = masks.to(device)

            outputs = model(images, labels=masks)
            loss, logits = outputs.loss.mean(), outputs.logits
            if accelerator is not None:
                loss = accelerator.gather(loss).mean()
            total_loss += loss.item()

            upsampled_logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            predicted = gather_if_needed(predicted, accelerator)
            masks = gather_if_needed(masks, accelerator)

            if accelerator is None or accelerator.is_main_process:
                predicted = predicted.view(-1)
                masks = masks.view(-1)

                total_correct_pixels += (predicted == masks).sum().item()
                total_pixels += predicted.numel()
                # Compute confusion matrix for the current batch and add to total
                batch_confusion_matrix = F_metrics.confusion_matrix(
                    predicted,
                    masks,
                    task="multiclass",
                    num_classes=num_labels+1
                ).to(device)

                total_confusion_matrix += batch_confusion_matrix
    
    # Make sure to wait for all processes to finish
    if accelerator is not None:
        accelerator.wait_for_everyone()

    if accelerator is None or accelerator.is_main_process:
        avg_val_loss = total_loss / len(val_dataloader)
        # Calculate metrics from the total confusion matrix
        if total_confusion_matrix is not None and total_confusion_matrix.sum() > 0:
            # Calculate True Positives (TP), False Positives (FP), False Negatives (FN) per class
            # TP: diagonal elements of the confusion matrix
            tp = total_confusion_matrix.diag()
            # FP: sum of the column (predicted class) minus TP
            fp = total_confusion_matrix.sum(dim=0) - tp
            # FN: sum of the row (actual class) minus TP
            fn = total_confusion_matrix.sum(dim=1) - tp

            # Add a small epsilon to avoid division by zero for classes not present
            epsilon = 1e-6

            # Calculate IoU (Jaccard Index) per class: TP / (TP + FP + FN)
            # Handle potential NaNs or Infs if a class has no ground truth or predictions (TP+FP+FN=0)
            per_class_iou = tp.float() / (tp + fp + fn + epsilon).float()
            # Replace NaNs with 0 if a class was entirely absent
            per_class_iou[torch.isnan(per_class_iou)] = 0.0
             # You might choose to average only classes that were actually present if needed
            mean_iou = per_class_iou.mean().item()


            # Calculate Accuracy per class (optional, often overall or mean IoU is key for segmentation)
            # TP + TN / Total. TN is tricky from just TP, FP, FN directly for multi-class without total N.
            # Overall pixel accuracy is simpler and more standard
            overall_pixel_acc = total_correct_pixels / total_pixels if total_pixels > 0 else 0.0
            # You can also get overall accuracy from confusion matrix: total_confusion_matrix.diag().sum() / total_confusion_matrix.sum()
            # Let's use the accumulated pixel counts as it's already done.

            # Calculate Dice score per class: 2 * TP / (2 * TP + FP + FN)
            per_class_dice = (2. * tp.float()) / (2. * tp + fp + fn + epsilon).float()
            # Replace NaNs with 0
            per_class_dice[torch.isnan(per_class_dice)] = 0.0
            # Macro Dice is the mean of per-class Dice scores
            mean_dice = per_class_dice.mean().item()

            # Mean Accuracy (average of per-class accuracies)
            # Calculating true per-class accuracy requires TN per class, which is sum of all non-TPs outside row/column
            # A common approximation or what torchmetrics does for average="none" accuracy is per_class_correct / per_class_total
            # where per_class_correct is TP and per_class_total is FN + TP (actual positives).
            # Let's stick to a common definition, possibly what MulticlassAccuracy(average="none") does,
            # which might be TP[i] / (total samples with true label i). Let's recalculate based on true counts.
            true_class_counts = total_confusion_matrix.sum(dim=1) # Sum of rows
            per_class_acc_from_counts = tp.float() / (true_class_counts.float() + epsilon)
            per_class_acc_from_counts[torch.isnan(per_class_acc_from_counts)] = 0.0
            mean_accuracy = per_class_acc_from_counts.mean().item()

        overall_pixel_acc = total_correct_pixels / total_pixels if total_pixels > 0 else 0.0

        print(f"\n loss: {avg_val_loss:.4f}, mean_iou: {mean_iou:.4f}, "
              f"mean_accuracy: {mean_accuracy:.4f}, pixel_accuracy: {overall_pixel_acc:.4f}, "
              f"dice_coeff: {mean_dice:.4f}")

        for cls_id, iou in enumerate(per_class_iou):
            print(f"Class {cls_id} IoU: {iou.item():.4f} Dice:{per_class_dice[cls_id].item():.4f} Acc:{per_class_acc_from_counts[cls_id].item():.4f}")

        metrics_dict = {
            "mean_iou": mean_iou,
            "mean_accuracy": mean_accuracy,
            "per_category_iou": per_class_iou.tolist()
        }

    if accelerator is not None:
        accelerator.wait_for_everyone()

    if is_return_metric_obj:
        return avg_val_loss, mean_iou, mean_accuracy, overall_pixel_acc, mean_dice, metrics_dict
    return avg_val_loss, mean_iou, mean_accuracy, overall_pixel_acc, mean_dice

def train_model(model, optimizer, lr_scheduler, num_labels, num_epochs, train_dataloader, val_dataloader,
                model_path, dataset, accelerator=None, wand_project_name=None, start_epoch=0, loss_type=None, alpha=0.5,
                lora_config=None,best_perf_metric=0):
    device = accelerator.device if accelerator is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_log_wandb = wand_project_name is not None
    if accelerator is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

    for epoch in range(start_epoch, num_epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}",
                            disable=(accelerator is not None and not accelerator.is_main_process))

        for idx, batch in enumerate(progress_bar):
            images, masks = batch
            masks = masks.squeeze(1)

            if accelerator is None:
                images = images.to(device)
                masks = masks.to(device)

            assert masks.max() <= num_labels, f"Mask contains invalid class index: {masks.max()}"
            assert masks.min() >= 0, "Mask contains negative class indices"
            assert images.size()[2:] == masks.size()[1:], "Size mismatch between mask and images"

            if(accelerator is not None):
                with accelerator.accumulate(model):
                    outputs = model(images, labels=masks)
                    loss, logits = outputs.loss.mean(), outputs.logits
                    if loss_type is not None:
                        loss = combined_loss(logits, masks, loss_type, dataset, alpha)

                    accelerator.backward(loss)
                    optimizer.step()
                    if lr_scheduler is not None and accelerator.sync_gradients:
                        lr_scheduler.step()
                    optimizer.zero_grad()
            else:
                outputs = model(images, labels=masks)
                loss, logits = outputs.loss.mean(), outputs.logits
                if loss_type is not None:
                    loss = combined_loss(logits, masks, loss_type, dataset, alpha)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if idx % 10 == 0 and (accelerator is None or accelerator.is_main_process):
                with torch.no_grad():
                    upsampled_logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    predicted = upsampled_logits.argmax(dim=1)

                    # Ensure predicted is the same shape as masks (batch_size, height, width)
                    assert predicted.shape == masks.shape, "Predicted shape doesn't match masks shape"
            
                preds_flat = predicted.view(-1).to(device)
                targets_flat = masks.view(-1).to(device)
                # Compute metrics for this batch only
                batch_iou = jaccard_index(preds_flat, targets_flat, num_classes=num_labels + 1, average="macro",task="multiclass")
                batch_acc = accuracy(preds_flat, targets_flat, num_classes=num_labels + 1, average="macro", task="multiclass")
                batch_dice = dice(preds_flat, targets_flat, num_classes=num_labels + 1, average="macro")
                progress_bar.set_postfix({
                    "loss": loss.item(),
                    "mean_iou": batch_iou.item(),
                    "mean_accuracy": batch_acc.item(),
                    "mean_dice":batch_dice.item()
                })
        
        train_loss, mean_iou, mean_acc, train_pixel_acc, mean_dice = evaluate_model(model, num_labels, train_dataloader, accelerator)
        val_loss, val_iou, val_acc, val_pixel_acc, val_dice = evaluate_model(model, num_labels, val_dataloader, accelerator)
        
        if(best_perf_metric < val_iou*val_dice):
            best_perf_metric = val_iou*val_dice
            if accelerator is None or accelerator.is_main_process:
                unwrapped_model = model if accelerator is None else accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler' : lr_scheduler.state_dict(),
                    'lora_config' : lora_config,
                    'best_perf_metric' : best_perf_metric,
                }, model_path + "_best.pt")

        if accelerator is None or accelerator.is_main_process:
            tmp_path = model_path + "_tmp"
            unwrapped_model = model if accelerator is None else accelerator.unwrap_model(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'lora_config': lora_config,
                'best_perf_metric': best_perf_metric,
            }, tmp_path)
            # This intermediate step prevents corrupted overwrites when process is interrupted in between while writing
            os.replace(tmp_path, model_path + "_backup.pt")  # atomic on most OSes

        current_lr = optimizer.param_groups[0]['lr']
        scheduler_type = str(lr_scheduler)
        if is_log_wandb and (accelerator is None or accelerator.is_main_process):
            wandb.log({
                "current_epoch": epoch,
                "learning_rate": current_lr,
                "scheduler_type": scheduler_type,
                "val_iou": val_iou,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "mean_iou": mean_iou,
                "mean_accuracy": mean_acc,
                "pixel_accuracy": train_pixel_acc,
                "dice_coeff": mean_dice,
                "val_pixel_accuracy": val_pixel_acc,
                "val_dice_coeff": val_dice,
                "loss": train_loss,
                "best_perf_metric": best_perf_metric,
            })
        
    return

if __name__ == '__main__':
    os.environ["TMPDIR"] = "./tmp"
    wand_project_name = None
    wand_project_name = "new_Car_Damage_Segmentation"
    # Any loss involving focal or cce loss can receive 'wt_' in its loss function to consider the median balancing weights as class weights
    # dice, focal , None , di_foc , iou , di_iou
    loss_type = 'wt_focal'
    alpha = 0.9

    # None, 'hierarchical' , 'fusion' , 'extend_tune' , 'ex_fusion' , 'moe_fusion
    model_type = 'fusion'

    # Wrap SegFormer with LoRA
    lora_config = None
    # lora_config = LoraConfig(
    #     task_type="TOKEN_CLASSIFICATION",  # Better aligned with segmentation tasks
    #     r=8,  # Low-rank adaptation dimension
    #     lora_alpha=16,  # Scaling factor
    #     lora_dropout=0.1,  # Dropout for LoRA layers
    #     target_modules=["query", "value"],  # Target attention layers
    #     bias="none"  # No bias added
    # )

    # Car_damages_dataset, Car_parts_dataset , CarDNN_Kaggle_merged_Car_damages_dataset
    dataset = "CarDNN_Kaggle_merged_Car_damages_dataset"

    coco_path = get_cocopath(dataset)
    # pretrained_model_name = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    # pretrained_model_name = "nvidia/segformer-b3-finetuned-ade-512-512"
    # pretrained_model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    pretrained_model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
    datadir = "./data/car-parts-and-car-damages/"
    tmp_dir = os.path.join(datadir, dataset)

    if (dataset != "CarDNN_Kaggle_merged_Car_damages_dataset"):
        car_dirs = [tmp_dir]
    elif (dataset == "CarDNN_Kaggle_merged_Car_damages_dataset"):
        car_dirs = [os.path.join(datadir, "Car_damages_dataset"),
                    os.path.join("./data/CarDD_release/", "CarDD_COCO/")]

    car_imgs = []
    for car_dir in car_dirs:
        car_imgs.append(os.path.join(car_dir, "split_dataset"))
    car_anns = (os.path.join(tmp_dir, "split_annotations"))

    accelerator = None
    accelerator = Accelerator(mixed_precision='fp16')
    # accelerator = Accelerator(gradient_accumulation_steps=2)

    # Important: BS below 16 causes performance degradation
    batch_size = 24
    num_epochs = 80

    if(accelerator is not None):
        print(f"accelerator.num_processes:{accelerator.num_processes}, accelerator.state:   {accelerator.state}")
        batch_size = batch_size //accelerator.num_processes

    # Get the colormapping from labelID of segmentation classes to color
    car_id_to_color = get_colormapping(os.path.join(tmp_dir, coco_path), tmp_dir + "/meta.json")

    train_car_dataset = get_dataset(car_imgs, car_anns, is_train=True, dataset=dataset)
    val_car_dataset = get_dataset(car_imgs, car_anns, dataset=dataset)

    tr_cd_dataloader = DataLoader(train_car_dataset, batch_size=batch_size, shuffle=True, num_workers=6,
                                  pin_memory=True)
    val_cd_dataloader = DataLoader(val_car_dataset, batch_size=batch_size, num_workers=6, pin_memory=True)

    start_net_path = None
    # start_net_path = "./checkpoints/Car_parts_dataset/nvidia_segformer-b3-finetuned-cityscapes-1024-1024_ep_90/new_checkpoints/high_aug_tnorm_/CarDNN_Kaggle_merged_Car_damages_dataset/fusion/wt_dice_0.9/nvidia_segformer-b5-finetuned-ade-640-640_backup.pt"

    continue_run_id = None
    # continue_run_id = "1mx6hjpu"

    superseg_model_name = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    # superseg_model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
    super_segmodel_path = "./checkpoints/Car_parts_dataset/nvidia_segformer-b3-finetuned-cityscapes-1024-1024_ep_90.pt"

    if (start_net_path is not None):
        lora_config = get_loraconfig_from_path(start_net_path)

    start_epoch = 0
    if (model_type is None):
        model = get_segformermodel(len(car_id_to_color), pretrained_model_name)
        save_prefix = "./"
    elif (model_type is not None):
        superseg_ds = "Car_parts_dataset"
        superseg_dir = os.path.join(datadir, superseg_ds)
        superseg_id_to_color = get_colormapping(os.path.join(superseg_dir, get_cocopath(superseg_ds)),
                                                superseg_dir + "/meta.json")
        super_segmodel = get_segformermodel(len(superseg_id_to_color), superseg_model_name)
        super_segmodel, _, _ = get_model_from_path(super_segmodel, super_segmodel_path)
        save_prefix = super_segmodel_path[:super_segmodel_path.find('.pt')] + "/"
        if (model_type == 'hierarchical'):
            model = Hierarchical_SegModel(super_segmodel, len(superseg_id_to_color) + 1,
                                          len(car_id_to_color) + 1, pretrained_model_name)
        elif (model_type == 'fusion'):
            model = Fusion_SegModel(super_segmodel, len(superseg_id_to_color) + 1,
                                    len(car_id_to_color) + 1, pretrained_model_name)
        elif (model_type == 'moe_fusion'):
            model = MOE_Fusion_SegModel(super_segmodel, len(superseg_id_to_color) + 1,
                                        len(car_id_to_color) + 1, pretrained_model_name)
        elif (model_type == 'extend_tune'):
            model = modify_segformer_output_channels(super_segmodel, len(car_id_to_color) + 1)
            if (lora_config is not None):
                model = get_peft_model(model, lora_config)
                model_type = 'lr' + model_type
        elif (model_type == 'ex_fusion'):
            model = Fusion_SegModel(super_segmodel, len(superseg_id_to_color) + 1,
                                    len(car_id_to_color) + 1, pretrained_model_name)
            superseg_model_name = pretrained_model_name
            super_segmodel_path = "./checkpoints/Car_parts_dataset/nvidia_segformer-b3-finetuned-cityscapes-1024-1024_ep_90.pt"
            super_segmodel = get_segformermodel(len(superseg_id_to_color), superseg_model_name)
            model.model = get_model_from_path(super_segmodel, super_segmodel_path)[0]
            model.model = modify_segformer_output_channels(model.model, len(car_id_to_color) + 1)
            if (lora_config is not None):
                model.model = get_peft_model(model.model, lora_config)
                model_type = 'lr' + model_type

    best_perf_metric = 0
    if (start_net_path is not None):
        model, start_epoch, best_perf_metric = get_model_from_path(model, start_net_path)
    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    # Set up the learning rate scheduler
    num_training_steps = num_epochs * len(tr_cd_dataloader)
    if (accelerator is None or accelerator.is_main_process):
        print("num_training_steps ", num_training_steps, num_epochs * len(tr_cd_dataloader), car_id_to_color)
    # lr_scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=200,
    #     num_training_steps=num_training_steps,
    # )

    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps), # 10% of training time increase LR linearly
        num_training_steps=num_training_steps,
        num_cycles = (num_epochs//4) # In the remaining 90% time of training, hop every 4 epoch
    )
    if(start_net_path is not None):
        optimizer,lr_scheduler = get_optimizers_from_path(optimizer, lr_scheduler, start_net_path)

    torch.cuda.empty_cache()
    if(accelerator is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_str == 'cuda':
            if(torch.cuda.device_count() > 1):
                print("Parallelizing model")
                model = torch.nn.DataParallel(model).cuda()
            else:
                model = model.to(device)
    if (accelerator is None or accelerator.is_main_process):
        print(model)
    model_save_dir = os.path.join(
        os.path.join(save_prefix + "new_checkpoints/high_aug_tnorm_/",
                     dataset + ("" if model_type is None else "/" + model_type[:7])),
        "default" if loss_type is None else (loss_type + "_" + str(alpha)))
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, pretrained_model_name.replace("/", "_"))
    is_log_wandb = not (wand_project_name is None)
    if (is_log_wandb and (accelerator is None or accelerator.is_main_process)):
        wandb_config = dict()
        wandb_config["optimizer"] = optimizer
        if(accelerator is not None):
            wandb_config["accelerator"] = accelerator.state
        wandb_config["final_model_save_path"] = model_save_path
        wandb_config["num_epochs"] = num_epochs
        wandb_config["batch_size"] = batch_size
        wandb_config["model_name"] = pretrained_model_name
        wandb_config["dataset"] = dataset
        wandb_config["start_net_path"] = start_net_path
        wandb_config["loss_type"] = loss_type
        wandb_config["alpha"] = alpha
        wandb_config["model_type"] = model_type
        wandb_config["lora_config"] = lora_config
        wandb_config["super_segmodel_path"] = '' if model_type is None else super_segmodel_path
        wandb_run_name = "high_aug_tnorm_" + ("" if model_type is None else model_type[:7] + "_") + (
            "DMG" if "damage" in dataset else "PRT") + "_" + \
                         pretrained_model_name[
                         pretrained_model_name.find("segformer") + len("segformer") + 1:pretrained_model_name.find(
                             "finetun") - 1] + "_" + \
                         pretrained_model_name[pretrained_model_name.find("finetun") + len("finetuned") + 1:][:4] + \
                         "_" + ("def" if loss_type is None else loss_type + "_" + str(alpha))

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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    if (accelerator is None or accelerator.is_main_process):
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")

    train_model(model, optimizer, lr_scheduler, len(car_id_to_color), num_epochs, tr_cd_dataloader,
                val_cd_dataloader, model_save_path, dataset, accelerator, wand_project_name, start_epoch,
                loss_type, alpha, lora_config, best_perf_metric)