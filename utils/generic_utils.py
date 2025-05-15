import torch
from structures.heirarchical_seg_model import *

def modify_output_channels(model, new_num_labels, model_name):
    internal_model = model
    # Check if the model is BaseSegModel
    if isinstance(model,BaseSegModel):
        internal_model = model.base_model
    internal_model.config.num_labels = new_num_labels
    # SegFormer head
    if 'segformer' in model_name.lower():
        # Ensure the correct configuration for number of labels
        internal_model.config.num_labels = new_num_labels

        # Update the classifier layer in the decode head
        internal_model.decode_head.classifier = torch.nn.Conv2d(
            in_channels=internal_model.decode_head.classifier.in_channels,
            out_channels=new_num_labels,
            kernel_size=(1, 1)
        )
    return model

def get_model_from_path(model,chkpath):
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load checkpoint
    checkpoint = torch.load(chkpath, map_location=map_location)
    
    # Load model state dict (strip `module.` for DataParallel models)
    state_dict = checkpoint['model_state_dict']
    
    # If using DataParallel, remove the 'module.' prefix except when using mask2former
    if not (hasattr(model,"base_model") and isinstance(model.base_model,Mask2FormerForUniversalSegmentation)) and 'module.' in next(iter(state_dict)):
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    # if not isinstance(model,BaseSegModel):
    #     print("***********************************************************************************************")
    #     print(state_dict.keys())
    #     print("###################################################################################################")
    #     print(model)
    #     print(model.state_dict().keys())
    # 3) build mapping from unwrapped to wrapped keys
    curr_sd = model.state_dict()
    unwrap_to_wrap = {}
    for wrapped_key in curr_sd.keys():
        if "base_model." in wrapped_key:
            unwrapped = wrapped_key.replace("base_model.", "")
            unwrap_to_wrap[unwrapped] = wrapped_key

    # 4) remap saved keys
    cnt = 0
    saved_model_dict_keys = list(state_dict.keys())
    for key in saved_model_dict_keys:
        if key in unwrap_to_wrap:
            cnt += 1
            state_dict[ unwrap_to_wrap[key] ] = state_dict[key]
            del state_dict[key]
    # if not isinstance(model,BaseSegModel):
    print(f"================================================================================================== {cnt}")
    #     print(state_dict.keys())

    model.load_state_dict(state_dict)

    epoch = checkpoint['epoch']  # Return the epoch if needed
    best_perf_metric = checkpoint['best_perf_metric'] if 'best_perf_metric' in checkpoint else 0.0
    
    print(f"Model loaded from {chkpath} at epoch:{epoch}")
    
    return model, epoch, best_perf_metric

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

def get_teacher_path_from_path(chkpath):
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load checkpoint
    checkpoint = torch.load(chkpath, map_location=map_location)
    
    teacher_path = None

    if "teacher_path"  in checkpoint:
        teacher_path = checkpoint['teacher_path']
    
    return teacher_path
