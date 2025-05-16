import os
import torch
from torch import nn
import numpy as np
import json
from structures.dataset_structure import COCOSegmentationDataset
from structures.heirarchical_seg_model import Fusion_SegModel, Hierarchical_SegModel, MOE_Fusion_SegModel, BaseSegModel
from utils.data_preprocessor_utils import *
from utils.visualize_utils import *
from torch.utils.data import DataLoader

from peft import LoraConfig
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AdamW, get_scheduler
from utils.generic_utils import *
from tqdm import tqdm
import evaluate
import wandb
import torch.nn.functional as F
from car_segmentation import evaluate_model,generate_model_based_on_model_type,combined_loss

import torch.nn.functional as F
from accelerate import Accelerator

import torchmetrics.functional as F_metrics
from torchmetrics.functional import dice, accuracy, jaccard_index
from typing import Dict, List, Tuple

class CosineFeatureLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dimensions, keep batch
        s_flat = student_feat.flatten(1)
        t_flat = teacher_feat.flatten(1)
        cosine_sim = F.cosine_similarity(s_flat, t_flat, dim=1, eps=self.eps)  # shape: (batch,)
        return 1 - cosine_sim.mean()

run_gtl = run_kdl = run_fl = 1.0
def distillation_losses(gt_loss,distill_config,teacher_logits,student_logits,feature_maps,student_projections):
    global run_gtl, run_kdl, run_fl
    T = distill_config.get("temperature", 1.0)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)
    
    alpha_ce = distill_config.get("alpha_ce", 0.5)
    alpha_kd = distill_config.get("alpha_kd", 0.5)
    alpha_feat = distill_config.get("alpha_feat", 0.)

    feat_loss = 0.0
    if alpha_feat > 0:
        for t_layer, s_layer in distill_config.get("feature_layers", []):
            t_feat = feature_maps["teacher"][t_layer]
            s_feat = feature_maps["student"][s_layer]
            # Unpack tuple outputs
            if isinstance(t_feat, (list, tuple)):
                t_feat = t_feat[0]
            if isinstance(s_feat, (list, tuple)):
                s_feat = s_feat[0]
            # Apply projection if shape mismatch
            if t_feat.shape != s_feat.shape:
                tmp = s_layer.replace(".","___")
                if tmp in student_projections:
                    proj = student_projections[tmp]
                else:
                    raise ValueError("Key :{tmp} was not found in student_projections")
                if proj is not None:
                    # Align dtype: cast s_feat to proj dtype, apply projection, then cast back
                    orig_dtype = s_feat.dtype
                    target_dtype = next(proj.parameters()).dtype
                    s_feat_cast = s_feat.to(target_dtype)
                    s_feat = proj(s_feat_cast).to(orig_dtype)
            feat_loss += distill_config.get("feature_loss_fn", F.mse_loss)(s_feat, t_feat)

    run_gtl = 0.99 * run_gtl + 0.01 * gt_loss.detach()
    if alpha_feat > 0:
        run_fl = 0.99 * run_fl + 0.01 * feat_loss.detach()
    run_kdl = 0.99 * run_kdl + 0.01 * kd_loss.detach()
    return alpha_ce * (gt_loss/run_gtl) + alpha_kd * (kd_loss/run_kdl) + alpha_feat * (feat_loss/run_fl)

def get_feature_shapes(model, feature_layers, input_shape=(2, 3, 224, 224)):
    feature_shapes = {}
    dummy_input = torch.randn(*input_shape).to(next(model.parameters()).device)
    handles = []

    for name, module in model.named_modules():
        if name in feature_layers:
            def save_shape(_, __, output, name=name):
                if isinstance(output, (list, tuple)):
                    output = output[0]
                feature_shapes[name] = output.shape
            handles.append(module.register_forward_hook(save_shape))

    model.eval()
    with torch.no_grad():
        model(dummy_input,labels=None)

    for h in handles:
        h.remove()
    return feature_shapes

def create_projection_heads(feature_pairs, feature_shapes):
    projection_heads = nn.ModuleDict()
    for t_layer, s_layer in feature_pairs:
        t_shape = feature_shapes['teacher'][t_layer]
        s_shape = feature_shapes['student'][s_layer]
        tmp = s_layer.replace(".","___")
        if t_shape != s_shape:
            projection_heads["module___"+tmp] = nn.Linear(s_shape[-1],t_shape[-1])

    return projection_heads

def train_with_distillation(teacher_model,student_model,teacher_path, distill_config, optimizer, lr_scheduler, num_labels, num_epochs, train_dataloader, val_dataloader,
                model_path, dataset, accelerator=None, wand_project_name=None, start_epoch=0, loss_type=None, alpha=0.5,best_perf_metric=0):
    device = accelerator.device if accelerator is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_log_wandb = wand_project_name is not None
    if accelerator is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        teacher_model, student_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            teacher_model, student_model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
    
    teacher_model.eval()

    feature_maps = {"teacher": {}, "student": {}}
    hooks = []

    for t_layer, s_layer in distill_config.get("feature_layers", []):
        t1 = dict(teacher_model.named_modules())
        t2 = dict(student_model.named_modules())
        if t_layer not in t1:
            print(f"t_layer:{t_layer} doesn't exist in teacher_model:{t1.keys()}")
        th = t1[t_layer]
        if s_layer not in t2:
            print(f"s_layer:{s_layer} doesn't exist in student_model:{t2.keys()}")
        sh = t2[s_layer]
        hooks.append(
            th.register_forward_hook(
                lambda m, i, o, key=t_layer: feature_maps["teacher"].__setitem__(key, o)
            )
        )
        hooks.append(
            sh.register_forward_hook(
                lambda m, i, o, key=s_layer: feature_maps["student"].__setitem__(key, o)
            )
        )

    for epoch in range(start_epoch, num_epochs):
        if(epoch>50):
            break
        student_model.train()
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
                with accelerator.accumulate(student_model):
                    with torch.no_grad():
                        teacher_outputs = teacher_model(images,labels=masks)
                        teacher_logits = teacher_outputs.logits
                    outputs = student_model(images, labels=masks)
                    gt_loss, logits = outputs.loss.mean(), outputs.logits
                    if loss_type is not None:
                        # print(f"logits:{logits.size()} masks:{masks.size()} device:{device}")
                        gt_loss = combined_loss(logits, masks, loss_type, dataset, alpha)
                    loss =  distillation_losses(gt_loss, distill_config,teacher_logits,logits,feature_maps,accelerator.unwrap_model(student_model).projection_heads)

                    accelerator.backward(loss)
                    optimizer.step()
                    if lr_scheduler is not None and accelerator.sync_gradients:
                        lr_scheduler.step()
                    optimizer.zero_grad()
            else:
                with torch.no_grad():
                    teacher_outputs = teacher_model(images,labels=masks)
                    teacher_logits = teacher_outputs.logits
                outputs = student_model(images, labels=masks)
                gt_loss, logits = outputs.loss.mean(), outputs.logits
                if loss_type is not None:
                    gt_loss = combined_loss(logits, masks, loss_type, dataset, alpha)
                loss =  distillation_losses(gt_loss, distill_config,teacher_logits,logits,feature_maps,student_model.projection_heads)

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
        
        train_loss, mean_iou, mean_acc, train_pixel_acc, mean_dice = evaluate_model(student_model, num_labels, train_dataloader, accelerator)
        val_loss, val_iou, val_acc, val_pixel_acc, val_dice = evaluate_model(student_model, num_labels, val_dataloader, accelerator)
        
        if(best_perf_metric < val_iou*val_dice):
            best_perf_metric = val_iou*val_dice
            if accelerator is None or accelerator.is_main_process:
                unwrapped_model = student_model if accelerator is None else accelerator.unwrap_model(student_model)
                torch.save({
                    'epoch': epoch,
                    'teacher_path':teacher_path,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler' : lr_scheduler.state_dict(),
                    'best_perf_metric' : best_perf_metric,
                }, model_path + "_best.pt")

        if accelerator is None or accelerator.is_main_process:
            tmp_path = model_path + "_tmp"
            unwrapped_model = student_model if accelerator is None else accelerator.unwrap_model(student_model)
            torch.save({
                'epoch': epoch,
                'teacher_path':teacher_path,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
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

    for h in hooks:
        h.remove()    
    return

def get_matching_layers(teacher: nn.Module, student: nn.Module, capture_list: List[str], start_prefix:str = "") -> List[Tuple[str,str]]:
    """
    Recursively matches layers between teacher and student models.
    When a module class name matches one in capture_list, the last layer is matched and recursion stops.
    Returns a List[Tuple(str,str)]: {teacher_full_name: student_full_name}
    """
    matches = []

    def recurse(t_mod: nn.Module, s_mod: nn.Module, prefix: str = ""):
        ismatchfound = False
        # Handle sequences like nn.Sequential or ModuleList
        if isinstance(t_mod, (nn.Sequential, nn.ModuleList)) and isinstance(s_mod, (nn.Sequential, nn.ModuleList)):
            if len(t_mod) > 0 and len(s_mod) > 0 and t_mod[-1].__class__.__name__ in capture_list and s_mod[-1].__class__.__name__ in capture_list:
                t_last_name = f"{prefix}.{len(t_mod) - 1}"
                s_last_name = f"{prefix}.{len(s_mod) - 1}"
                matches.append((t_last_name,s_last_name))
                ismatchfound = True
        elif s_mod.__class__.__name__ in capture_list and t_mod.__class__.__name__ in capture_list:
            matches.append((prefix,prefix))
            ismatchfound = True
        if ismatchfound:
            return  

        # Get child modules as OrderedDict
        t_children = dict(t_mod.named_children())
        s_children = dict(s_mod.named_children())

        # Traverse common child keys
        for name in t_children:
            if name not in s_children:
                continue  # mismatch — skip

            t_child = t_children[name]
            s_child = s_children[name]
            full_name = f"{prefix}.{name}" if prefix else name

            # Recurse deeper
            recurse(t_child, s_child, full_name)

    recurse(teacher, student, start_prefix)
    return matches

def get_feature_map_capture_list_for_model_type(model:nn.Module):
    if isinstance(model,Fusion_SegModel):
        return ["SegformerOverlapPatchEmbeddings","SegformerLayer","SegformerMLP"]
    elif isinstance(model,BaseSegModel):
        return ["SegformerOverlapPatchEmbeddings","SegformerLayer","SegformerMLP"]

if __name__ == '__main__':
    os.environ["TMPDIR"] = "./tmp"
    wand_project_name = None
    wand_project_name = "distillation_Car_Damage_Segmentation"
    # Any loss involving focal or cce loss can receive 'wt_' in its loss function to consider the median balancing weights as class weights
    # dice, focal , None , di_foc , iou , di_iou , all_dynamic
    loss_type = "wt_all_dynamic"
    alpha = 0.5

    # Car_damages_dataset, Car_parts_dataset , CarDNN_Kaggle_merged_Car_damages_dataset , Car_DD_dataset , roboflow_vehicle_damage , roboflow_dmg_merged_carDD
    dataset = "Car_DD_dataset"

    continue_run_id = None
    # continue_run_id = "29puyrds"

    # ****************** Teacher model **********************
    # None, 'hierarchical' , 'fusion' , 'extend_tune' , 'ex_fusion' , 'moe_fusion'
    teacher_model_type = 'fusion'
    teacher_pretrained_model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
    # Path will be overridden if student model was saved and loaded
    # teacher_path = "./new_checkpoints/high_aug_tnorm_/Car_DD_dataset/wt_all_dynamic_0.5/nvidia_segformer-b5-finetuned-ade-640-640_best.pt"
    teacher_path = "./new_checkpoints/high_aug_tnorm_/Car_parts_dataset/wt_all_dynamic_0.5/nvidia_segformer-b5-finetuned-ade-640-640_best/new_checkpoints/high_aug_tnorm_/Car_DD_dataset/fusion/wt_all_dynamic_0.5/nvidia_segformer-b5-finetuned-ade-640-640_best.pt"
    teacher_superseg_ds = "Car_parts_dataset"
    # teacher_superseg_model_name = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
    teacher_superseg_model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
    teacher_super_segmodel_path = None
    # Below is second best supersegformer
    # teacher_super_segmodel_path = "./checkpoints/Car_parts_dataset/nvidia_segformer-b3-finetuned-cityscapes-1024-1024_ep_90.pt"
    # Below is best supersegformer
    teacher_super_segmodel_path = "./new_checkpoints/high_aug_tnorm_/Car_parts_dataset/wt_all_dynamic_0.5/nvidia_segformer-b5-finetuned-ade-640-640_best.pt"
    # teacher_super_segmodel_path = "./new_checkpoints/high_aug_tnorm_/Car_DD_dataset/wt_all_dynamic_0.5/nvidia_segformer-b5-finetuned-ade-640-640_best.pt"

    # ******************************************************

    # ==================== Student model ====================
    # Student and teacher model architecture should be same even if offcourse the number of layers and parameters are different
    student_model_type = teacher_model_type
    student_pretrained_model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    student_path = None
    # student_path = ""
    student_superseg_ds = "Car_parts_dataset"
    student_superseg_model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    student_super_segmodel_path = None
    student_super_segmodel_path = "./new_checkpoints/high_aug_tnorm_/Car_parts_dataset/wt_all_dynamic_0.5/nvidia_segformer-b0-finetuned-ade-512-512_best.pt"
    # =======================================================

    coco_path = get_cocopath(dataset)
    datadir = "./data/car-parts-and-car-damages/"
    tmp_dir = os.path.join(datadir, dataset)

    if (dataset != "CarDNN_Kaggle_merged_Car_damages_dataset"):
        car_dirs = [tmp_dir]
    elif (dataset == "CarDNN_Kaggle_merged_Car_damages_dataset"):
        car_dirs = [os.path.join(datadir, "Car_damages_dataset"),
                    os.path.join("./data/CarDD_release/", "CarDD_COCO/")]
    if (dataset == "Car_DD_dataset"):
        car_dirs = [os.path.join("./data/CarDD_release/", "CarDD_COCO/")]
    elif (dataset == "roboflow_vehicle_damage"):
        car_dirs = ["./data/roboflow_vehicle_damage/"]
    elif (dataset == "roboflow_dmg_merged_carDD"):
        car_dirs = [os.path.join("./data/CarDD_release/", "CarDD_COCO/"),"./data/roboflow_vehicle_damage/"]

    car_imgs = []
    for car_dir in car_dirs:
        car_imgs.append(os.path.join(car_dir, "split_dataset"))
    car_anns = (os.path.join(tmp_dir, "split_annotations"))

    accelerator = None
    accelerator = Accelerator(mixed_precision='fp16')

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

    start_epoch = -1
    teacher_model,teacher_save_prefix = generate_model_based_on_model_type(teacher_model_type, car_id_to_color, teacher_pretrained_model_name, datadir, teacher_superseg_model_name, teacher_super_segmodel_path, teacher_superseg_ds)
    student_model,student_save_prefix = generate_model_based_on_model_type(student_model_type, car_id_to_color, student_pretrained_model_name, datadir, student_superseg_model_name, student_super_segmodel_path, student_superseg_ds)
    if (accelerator is None or accelerator.is_main_process):
        print(f"teacher_save_prefix:{teacher_save_prefix} student_save_prefix:{student_save_prefix}")

    # Example distillation configuration:
    distillation_config = {
        "temperature": 4.0,
        "alpha_ce": 1,
        "alpha_kd": 10,
        "alpha_feat": 0.,
        "feature_loss_fn": CosineFeatureLoss(),
        "attn_loss_fn": lambda s, t: F.kl_div(
            F.log_softmax(s, dim=-1),
            F.softmax(t, dim=-1),
            reduction="batchmean"
        ),
    }

    if distillation_config.get("alpha_feat",0) > 0:
        match_feature_layers = get_matching_layers(teacher_model, student_model, capture_list=get_feature_map_capture_list_for_model_type(student_model),start_prefix="")
        feature_shapes = {
                "teacher": get_feature_shapes(teacher_model, [t for t, _ in match_feature_layers]),
                "student": get_feature_shapes(student_model, [s for _, s in match_feature_layers])
        }
        if (accelerator is None or accelerator.is_main_process):
            print(f"feature_shapes : {feature_shapes}")

        student_model.projection_heads = create_projection_heads(match_feature_layers, feature_shapes)
        if (accelerator is None or accelerator.is_main_process):
            print(f"student_model.projection_heads :{student_model.projection_heads}")
    else:
        student_model.projection_heads = None

    match_feature_layers = get_matching_layers(teacher_model, student_model, capture_list=get_feature_map_capture_list_for_model_type(student_model),start_prefix="module")
    if (accelerator is None or accelerator.is_main_process):
        print(f"match_feature_layers:::{match_feature_layers}")

    distillation_config["feature_layers"] = match_feature_layers

    best_perf_metric = 0
    if (student_path is not None):
        ttmp = get_teacher_path_from_path(student_path)
        teacher_path = ttmp if not None else teacher_path
        student_model, start_epoch, best_perf_metric = get_model_from_path(student_model, student_path)
    if (teacher_path is not None):
        teacher_model, _, _ = get_model_from_path(teacher_model, teacher_path)
    
    num_training_steps = num_epochs * len(tr_cd_dataloader)
    if (accelerator is None or accelerator.is_main_process):
        print("num_training_steps ", num_training_steps, num_epochs * len(tr_cd_dataloader), car_id_to_color)
    
    # Define optimizer and learning rate scheduler
    optimizer = AdamW(student_model.parameters(), lr=1e-4, weight_decay=0.05)

    # Set up the learning rate scheduler
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps), # 10% of training time increase LR linearly
        num_training_steps=num_training_steps,
        num_cycles = (num_epochs//4) # In the remaining 90% time of training, hop every 4 epoch
    )

    if(student_path is not None):
        optimizer,lr_scheduler = get_optimizers_from_path(optimizer, lr_scheduler, student_path)

    torch.cuda.empty_cache()
    if(accelerator is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_str == 'cuda':
            if(torch.cuda.device_count() > 1):
                print("Parallelizing model")
                student_model = torch.nn.DataParallel(student_model).cuda()
                teacher_model = torch.nn.DataParallel(teacher_model).cuda()
            else:
                student_model = student_model.to(device)
                teacher_model = teacher_model.to(device)
    if (accelerator is None or accelerator.is_main_process):
        print(f"student_model:{student_model}")
        print(f"student_model.named_modules:{dict(student_model.named_modules()).keys()}")
        print(f"teacher_model:{teacher_model}")
        print(f"teacher_model.named_modules:{dict(teacher_model.named_modules()).keys()}")
    
    stmp = teacher_path[:teacher_path.find('.pt')] + "/"
    dist_name = f"tmp_{distillation_config.get('temperature',0)}_pure_{distillation_config.get('alpha_ce',0)}_kl_{distillation_config.get('alpha_kd',0)}_flos_{distillation_config.get('alpha_feat',0)}"
    model_save_dir = os.path.join(
        os.path.join(student_save_prefix+"/"+ stmp + dist_name + "/",
                     dataset + ("" if student_model_type is None else "/" + student_model_type[:7])),
        "default" if loss_type is None else (loss_type + "_" + str(alpha)))
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, student_pretrained_model_name.replace("/", "_"))

    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    if (accelerator is None or accelerator.is_main_process):
        print(f"model_save_path: {model_save_path}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")

    is_log_wandb = not (wand_project_name is None)
    if (is_log_wandb and (accelerator is None or accelerator.is_main_process)):
        if "segformer" in student_pretrained_model_name:
            ttmp = student_pretrained_model_name.find("segformer") + len("segformer") + 1
        elif "mask2former" in student_pretrained_model_name:
            ttmp = student_pretrained_model_name.find("mask2former") + len("mask2former") + 1
        wandb_config = dict()
        wandb_config["optimizer"] = optimizer
        if(accelerator is not None):
            wandb_config["accelerator"] = accelerator.state
        wandb_config["total_params"] = total_params
        wandb_config["trainable_params"] = trainable_params
        wandb_config["non_trainable_params"] = non_trainable_params
        wandb_config["final_model_save_path"] = model_save_path
        wandb_config["num_epochs"] = num_epochs
        wandb_config["batch_size"] = batch_size
        wandb_config["teacher_model_name"] = teacher_pretrained_model_name
        wandb_config["student_model_name"] = student_pretrained_model_name
        wandb_config["dataset"] = dataset
        wandb_config["teacher_path"] = teacher_path
        wandb_config["student_path"] = student_path
        wandb_config["loss_type"] = loss_type
        wandb_config["alpha"] = alpha
        wandb_config["student_model_type"] = student_model_type
        wandb_config["teacher_model_type"] = teacher_model_type
        wandb_config["distillation_config"] = distillation_config
        wandb_config["dist_name"] = dist_name
        wandb_config["teacher_super_segmodel_path"] = '' if teacher_model_type is None else teacher_super_segmodel_path
        wandb_config["student_super_segmodel_path"] = '' if student_model_type is None else student_super_segmodel_path
        wandb_run_name = dist_name + ("" if student_model_type is None else student_model_type[:7] + "_") + (
            "DMG" if "damage" in dataset else "PRT") + "_" + \
                         student_pretrained_model_name[
                         ttmp:student_pretrained_model_name.find(
                             "finetun") - 1] + "_" + \
                         student_pretrained_model_name[student_pretrained_model_name.find("finetun") + len("finetuned") + 1:][:4] + \
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

    train_with_distillation(teacher_model, student_model,teacher_path, distillation_config, optimizer, lr_scheduler, len(car_id_to_color), num_epochs, tr_cd_dataloader,
                val_cd_dataloader, model_save_path, dataset, accelerator, wand_project_name, start_epoch+1,
                loss_type, alpha, best_perf_metric)