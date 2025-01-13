from torch import nn
import torch
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from typing import Optional

def modify_segformer_output_channels(segmodel, new_output_channels):
    """
    Modify the SegFormer model to handle a different number of output channels.
    
    Args:
        segmodel (SegformerForSemanticSegmentation): The pretrained SegFormer model object.
        new_output_channels (int): The number of output channels for the modified model.
    
    Returns:
        segmodel: The modified SegFormer model.
    """
    # Ensure the correct configuration for number of labels
    segmodel.config.num_labels = new_output_channels

    # Update the classifier layer in the decode head
    segmodel.decode_head.classifier = torch.nn.Conv2d(
        in_channels=segmodel.decode_head.classifier.in_channels,
        out_channels=new_output_channels,
        kernel_size=(1, 1)
    )

    return segmodel


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

        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name,num_labels=num_labels,ignore_mismatched_sizes=True)
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

class FusionSegOutput(ModelOutput):
    """
    Custom output class to mimic the SegFormer output structure.
    """
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None

class Fusion_SegModel(nn.Module):
    def __init__(self,supersegmodel,num_labels_superseg,num_labels,model_name, seed=2022,intermediate_channels=512):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(num_labels_superseg+num_labels, intermediate_channels, 3,padding='same'),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, intermediate_channels//2, 3,padding='same'),
            nn.BatchNorm2d(intermediate_channels//2),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels//2, num_labels, 3,padding='same')  # Final output
        )

        self.supersegmodel = supersegmodel
        for param in self.supersegmodel.parameters():
            param.requires_grad = False  # Freeze parameters
        
        self.supersegmodel.eval()

        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name,num_labels=num_labels,ignore_mismatched_sizes=True)

    def get_mask_from_supermodel(self,inp):
        with torch.no_grad():  # Ensure superclass model is frozen
            outputs = self.supersegmodel(inp)
        
        return outputs.logits


    def forward(self, inp, labels):
        with torch.no_grad():
            superseg_masks = self.get_mask_from_supermodel(inp)
        # Pass the input through the SegFormer model
        output = self.model(inp,labels)
        # Concatenate the output masks with the superclass segmentation masks
        combined_masks = torch.cat([output.logits, superseg_masks], dim=1)  # Shape: (B, C+M, H, W)

        output_logits = self.fusion_layer(combined_masks)

        labels = F.interpolate(labels.unsqueeze(1).float(), size=output_logits.shape[-2:], mode="nearest").squeeze(1).long()
    
        # Cross-Entropy Loss
        ce_loss = F.cross_entropy(output_logits, labels, reduction='mean')

        return FusionSegOutput(loss=ce_loss, logits=output_logits)

class GatingNetwork(nn.Module):
    def __init__(self,num_labels_superseg):
        super().__init__()
        # Image processing branch
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
        )
        
        # Logits processing branch
        self.logits_branch = nn.Sequential(
            nn.Conv2d(num_labels_superseg, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=3, padding='same'),
            nn.ReLU(),
        )

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(32 + 32, 128, kernel_size=3, padding='same'),  # Fuse features
            nn.ReLU(),
            nn.Conv2d(128, 2, kernel_size=3, padding='same'),  # Two output channels
        )
    
    def forward(self, image, logits):
        image_features = self.image_branch(image)  # Features from image
        logits_features = self.logits_branch(logits)  # Features from logits
        combined_features = torch.cat([image_features, logits_features], dim=1)  # Concatenate
        gate_weights = self.fusion_layer(combined_features)  # Output gate weights
        return gate_weights


class MOE_Fusion_SegModel(nn.Module):
    def __init__(self, supersegmodel, num_labels_superseg, num_labels, model_name, seed=2022, intermediate_channels=512):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)

        # Define fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(num_labels_superseg + num_labels_superseg, intermediate_channels, 3, padding='same'),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, intermediate_channels // 2, 3, padding='same'),
            nn.BatchNorm2d(intermediate_channels // 2),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels // 2, num_labels, 3, padding='same')  # Final output
        )

        self.supersegmodel = supersegmodel
        for param in self.supersegmodel.parameters():
            param.requires_grad = False  # Freeze parameters

        self.supersegmodel.eval()

        # SegFormer model
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)

        # Adjust output layer to match supersegmodel's number of labels
        self.adjust_segformer_output = nn.Conv2d(num_labels, num_labels_superseg, kernel_size=1)

        # Gate mechanism to decide the contribution of each model
        self.gate_network = GatingNetwork(num_labels_superseg)

    def get_mask_from_supermodel(self, inp):
        with torch.no_grad():
            outputs = self.supersegmodel(inp)
        return outputs.logits

    def forward(self, inp, labels):
        # Get logits from the super segmentation model
        with torch.no_grad():
            superseg_logits = self.get_mask_from_supermodel(inp)  # Shape: (B, num_labels_superseg, H, W)
        
        # Get logits from the SegFormer model
        segformer_output = self.model(inp, labels)  # Shape: (B, num_labels, H, W)
        
        # Adjust SegFormer logits to match the supersegmodel's number of classes
        adjusted_segformer_logits = self.adjust_segformer_output(segformer_output.logits)  # Shape: (B, num_labels_superseg, H, W)

        # Upsample superseg_logits to match the SegFormer output size
        upsampled_superseg_logits = F.interpolate(superseg_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        # Gating mechanism: compute gate weights using the image and upsampled supersegmodel logits
        gate_weights = self.gate_network(inp,upsampled_superseg_logits)  # Shape: (B, 2, H, W)

        # Normalize gate weights (optional: if needed, to ensure sum of weights is 1)
        gate_weights = torch.softmax(gate_weights, dim=1)  # Shape: (B, 2, H, W)
        
        # Resize gate weights to match the size of superseg_logits and segformer logits (H, W)
        # Resize gate weights to match logits' spatial dimensions
        gate_weights_resized = F.interpolate(gate_weights, size=superseg_logits.shape[-2:], mode="bilinear", align_corners=False)

        # Combine logits using gating weights
        combined_logits = (
            gate_weights_resized[:, 0:1] * superseg_logits +  # Contribution from the super segmentation model
            gate_weights_resized[:, 1:2] * adjusted_segformer_logits  # Contribution from the SegFormer model
        )  # Shape: (B, num_labels_superseg, H, W)

        # Optional: Refine combined logits using a fusion layer
        refined_logits = self.fusion_layer(torch.cat([combined_logits, adjusted_segformer_logits], dim=1))

        # Resample labels to match output size
        labels = F.interpolate(labels.unsqueeze(1).float(), size=refined_logits.shape[-2:], mode="nearest").squeeze(1).long()
        
        # Compute Cross-Entropy Loss
        ce_loss = F.cross_entropy(refined_logits, labels, reduction='mean')

        return FusionSegOutput(loss=ce_loss, logits=refined_logits)
