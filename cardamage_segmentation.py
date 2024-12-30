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
from tqdm import tqdm
import evaluate
import wandb
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,confusion_matrix

def get_segformermodel(num_labels,model_name):
    # nvidia/segformer-b5-finetuned-cityscapes-1024-1024
    model = SegformerForSemanticSegmentation.from_pretrained(model_name,num_labels=num_labels+1,ignore_mismatched_sizes=True)

    # Modify the classifier layer to match the number of classes in your dataset
    # Assuming the model's `decode_head` has a classifier head for segmentation.
    # model.decode_head.classifier = torch.nn.Conv2d(768, num_labels, kernel_size=1)

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


def train_model(model,optimizer,lr_scheduler,num_labels,num_epochs,train_dataloader,val_dataloader,model_path,wand_project_name=None,start_epoch=0):
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

                # Calculate Pixel Accuracy, Dice Coefficient, Mean Accuracy
                pixel_acc = pixel_accuracy(predicted, masks)
                dice_coeff = dice_coefficient(predicted, masks, num_labels+1)
                avg_pixel_acc += pixel_acc
                avg_dice_coeff += dice_coeff
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
                "mean_accuracy": metrics["mean_accuracy"],
                "pixel_accuracy": pixel_acc,
                "dice_coeff": dice_coeff,
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
        val_loss, val_iou, val_acc,val_pixel_accuracy,val_dice_coeff = evaluate_model(model, num_labels, val_dataloader)

        if is_log_wandb:
            wandb.log({
                "current_epoch": epoch,
                'learning_rate': current_lr,
                'scheduler_type': scheduler_type,
                "val_iou": val_iou,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "mean_iou": metrics["mean_iou"],
                "mean_accuracy": metrics["mean_accuracy"],
                "pixel_accuracy": avg_pixel_acc/len(train_dataloader),
                "dice_coeff": avg_dice_coeff/len(train_dataloader),
                "val_pixel_accuracy": val_pixel_accuracy,
                "val_dice_coeff": val_dice_coeff,
                "loss": total_loss/len(train_dataloader)
            })
    
    return 

def get_model_from_path(model,optimizer, lr_scheduler, model_path):
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=map_location)
    
    # Load model state dict (strip `module.` for DataParallel models)
    state_dict = checkpoint['model_state_dict']
    
    # If using DataParallel, remove the 'module.' prefix
    if 'module.' in next(iter(state_dict)):
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    # Load optimizer and scheduler states
    if(optimizer is not None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if(lr_scheduler is not None):
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    epoch = checkpoint['epoch']  # Return the epoch if needed
    
    print(f"Checkpoint loaded from {model_path}")
    
    return model, optimizer, lr_scheduler, epoch

if __name__ == '__main__':
    os.environ["TMPDIR"] = "./tmp"
    wand_project_name = None
    wand_project_name="Car_Damage_Segmentation"

    pretrained_model_name = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    datadir = "./data/car-parts-and-car-damages/"
    cardamages_dir = os.path.join(datadir,"Car_damages_dataset")
    cardamages_imgs = os.path.join(cardamages_dir,"split_dataset")
    cardamages_anns = os.path.join(cardamages_dir,"split_annotations")

    batch_size = 12
    num_epochs = 100

    # Get the colormapping from labelID of segmentation classes to color
    cardamage_id_to_color = get_colormapping(cardamages_dir+"/coco_damage_annotations.json",cardamages_dir+"/meta.json")

    train_cardamage_dataset = get_dataset(cardamages_imgs,cardamages_anns,is_train=True)
    val_cardamage_dataset = get_dataset(cardamages_imgs,cardamages_anns)

    tr_cd_dataloader = DataLoader(train_cardamage_dataset, batch_size=batch_size, shuffle=True,num_workers=6,pin_memory=True)
    val_cd_dataloader = DataLoader(val_cardamage_dataset, batch_size=batch_size,num_workers=6,pin_memory=True)

    start_net_path = None
    # start_net_path = "./checkpoints/nvidia_segformer-b3-finetuned-cityscapes-1024-1024_ep_17.pt"
    
    start_epoch = 0        
    model = get_segformermodel(len(cardamage_id_to_color),pretrained_model_name)

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)

    # Set up the learning rate scheduler
    num_training_steps = num_epochs * len(train_cardamage_dataset)
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

    if(start_net_path is not None):
        model = get_segformermodel(len(cardamage_id_to_color),pretrained_model_name)
        model,optimizer,lr_scheduler,start_epoch = get_model_from_path(model,optimizer,lr_scheduler,start_net_path)
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    if device_str == 'cuda':
        if(torch.cuda.device_count() > 1):
            print("Parallelizing model")
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.to(device)
    print(model)
    model_save_path = os.path.join("./checkpoints",pretrained_model_name.replace("/","_"))
    is_log_wandb = not(wand_project_name is None)
    if(is_log_wandb):
        wandb_config = dict()
        wandb_config["optimizer"] = optimizer
        wandb_config["final_model_save_path"] = model_save_path
        wandb_config["num_epochs"] = num_epochs
        wandb_config["batch_size"] = batch_size
        wandb_config["model_name"] = pretrained_model_name
        wandb_config["dataset"] = cardamages_dir
        wandb_config["start_net_path"] = start_net_path

        wandb.init(
            project=f"{wand_project_name}",
            config=wandb_config,
        )

    train_model(model,optimizer,lr_scheduler,len(cardamage_id_to_color),num_epochs,tr_cd_dataloader,val_cd_dataloader,model_save_path,wand_project_name,start_epoch)


