from torch import nn
import torch
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

def modify_segformer_input_channels(segmodel, new_input_channels):
    """
    Modify the SegFormer model to handle a different number of input channels.
    
    Args:
        segmodel (SegformerForSemanticSegmentation): The name of the pretrained model to load.
        new_input_channels (int): The number of input channels for the modified model.
    
    Returns:
        segmodel: The modified SegFormer model.
    """
    # Step 2: Extract the original input embedding layer weights
    old_proj_layer = segmodel.segformer.encoder.patch_embeddings[0].proj
    old_weight = old_proj_layer.weight  # Shape: (out_channels, in_channels, kernel_height, kernel_width)

    # Step 3: Adjust the input embedding layer to handle the new number of input channels
    # Repeat the original weights for the new input channels and truncate if necessary
    scale_factor = new_input_channels // old_proj_layer.in_channels
    new_weight = torch.cat([old_weight] * scale_factor, dim=1)[:, :new_input_channels, :, :]

    # Step 4: Replace the input embedding layer with an updated one
    segmodel.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(
        in_channels=new_input_channels,             # New number of input channels
        out_channels=old_proj_layer.out_channels,   # Same number of output channels as the original
        kernel_size=old_proj_layer.kernel_size,     # Same kernel size as the original
        stride=old_proj_layer.stride,               # Same stride as the original
        padding=old_proj_layer.padding              # Same padding as the original
    )
    # Step 5: Assign the new weights to the updated input embedding layer
    segmodel.segformer.encoder.patch_embeddings[0].proj.weight = torch.nn.Parameter(new_weight)

    return segmodel

class Hierarchical_SegModel(nn.Module):
    def __init__(self,supersegmodel, input_channel,num_labels,model_name, seed=2022,num_out_channels=3,intermediate_channels=512):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.input_channel = input_channel
        self.mask_reducer = nn.Sequential(
            nn.Conv2d(input_channel, intermediate_channels, 1),  # Reduce dimensions
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, num_out_channels, 1)  # Final output
        )

        self.relu = nn.ReLU()

        self.supersegmodel = supersegmodel
        for param in self.supersegmodel.parameters():
            param.requires_grad = False  # Freeze parameters
        
        self.supersegmodel.eval()

        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name,num_labels=num_labels+1,ignore_mismatched_sizes=True)
        # The modified segformer input size needs to be mask output channels + 3(here, 3 is original input image channel size)
        self.model = modify_segformer_input_channels(self.model,num_out_channels+3)

    def get_mask_from_supermodel(self,inp,masks):
        with torch.no_grad():  # Ensure superclass model is frozen
            outputs = self.supersegmodel(inp,masks)
            upsampled_logits = F.interpolate(outputs.logits, size=inp.shape[-2:], mode="bilinear", align_corners=False)
        
        return upsampled_logits


    def forward(self, inp, labels):
        with torch.no_grad():
            superseg_masks = self.get_mask_from_supermodel(inp,None)
        superseg_masks = self.mask_reducer(superseg_masks)
        # Concatenate the input image with the superclass segmentation masks
        combined_input = torch.cat([inp, superseg_masks], dim=1)  # Shape: (B, C+M, H, W)

        # Pass the combined input through the SegFormer model
        output = self.model(combined_input,labels)

        return output
