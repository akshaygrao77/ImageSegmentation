from torch import nn
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from typing import NamedTuple, Optional
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation,Mask2FormerForUniversalSegmentation
from peft import LoraConfig, get_peft_model

class SegmentationOutput(NamedTuple):
    loss: Optional[torch.Tensor]
    logits: torch.Tensor

class BaseSegModel(nn.Module):
    """
    Unified wrapper for SegFormer and Mask2Former:
      - adapts input channels
      - sets num_labels
      - provides uniform forward returning full-res logits and optional loss
    """
    def __init__(
        self,
        num_labels: int,
        model_name: str,
        in_channels: int = 3,
        ignore_mismatched_sizes: bool = True
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name.lower()
        self.num_labels = num_labels
        self.in_channels = in_channels
        if "segformer" in model_name:
            self.base_model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=num_labels,
                                                           ignore_mismatched_sizes=ignore_mismatched_sizes)
        elif "mask2former" in model_name:
            self.base_model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name,num_labels=num_labels,
                                                                ignore_mismatched_sizes=ignore_mismatched_sizes)
        else:
            raise ValueError(f"Wrong model_name:{model_name}. It needs to have be either segformer or mask2former.")
    
    def initialize_peft_model(self,lora_config):
        self.base_model = get_peft_model(self.base_model, lora_config)

    def modify_input_channels(self):
        # Adapt input embeddings for SegFormer
        if 'segformer' in self.model_name.lower():
            # Step 2: Extract the original input embedding layer weights
            old_proj_layer = self.base_model.segformer.encoder.patch_embeddings[0].proj
            old_weight = old_proj_layer.weight  # Shape: (out_channels, in_channels, kernel_height, kernel_width)

            # Step 3: Adjust the input embedding layer to handle the new number of input channels
            # Repeat the original weights for the new input channels and truncate if necessary
            scale_factor = self.in_channels // old_proj_layer.in_channels
            new_weight = torch.cat([old_weight] * scale_factor, dim=1)[:, :self.in_channels, :, :]

            # Step 4: Replace the input embedding layer with an updated one
            self.base_model.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(
                in_channels=self.in_channels,             # New number of input channels
                out_channels=old_proj_layer.out_channels,   # Same number of output channels as the original
                kernel_size=old_proj_layer.kernel_size,     # Same kernel size as the original
                stride=old_proj_layer.stride,               # Same stride as the original
                padding=old_proj_layer.padding              # Same padding as the original
            )
            # Step 5: Assign the new weights to the updated input embedding layer
            self.base_model.segformer.encoder.patch_embeddings[0].proj.weight = torch.nn.Parameter(new_weight)
        # Adapt input embeddings for Mask2Former
        elif 'mask2former' in self.model_name.lower():
            old = self.base_model.model.pixel_level_module.backbone.patch_embed.proj
            old_w = old.weight
            repeat = self.in_channels // old.in_channels + 1
            new_w = old_w.repeat(1, repeat, 1, 1)[:, :self.in_channels]
            self.base_model.model.pixel_level_module.backbone.patch_embed.proj = nn.Conv2d(
                self.in_channels, old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding
            )
            self.base_model.model.pixel_level_module.backbone.patch_embed.proj.weight = nn.Parameter(new_w)
        else:
            raise ValueError("Model name must contain 'segformer' or 'mask2former'")
    
    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> SegmentationOutput:
        if 'segformer' in self.model_name.lower():
            outputs = self.base_model(images, labels=labels)
        elif 'mask2former' in self.model_name.lower():
            if labels is None:
                outputs = self.base_model(images, labels=labels)
            else:
                batch_mask_labels = []
                batch_class_labels = []
                for lbl in labels:
                    lbl = lbl.to(images.device)
                    unique_cls = torch.unique(lbl)
                    masks = []
                    cls = []
                    for c in unique_cls:
                        masks.append((lbl == c).to(torch.float))
                        cls.append(torch.tensor(c, device=images.device, dtype=torch.long))
                    batch_mask_labels.append(torch.stack(masks))
                    batch_class_labels.append(torch.stack(cls))
                outputs = self.base_model(
                    pixel_values=images,
                    mask_labels=batch_mask_labels,
                    class_labels=batch_class_labels
                )
        loss = outputs.loss.mean() if getattr(outputs, 'loss', None) is not None else None
        logits = self._postprocess_logits(outputs, images)
        return SegmentationOutput(loss=loss, logits=logits)

    def _postprocess_logits(
        self,
        outputs,
        pixel_values: torch.Tensor
    ) -> torch.Tensor:
        if 'segformer' in self.model_name.lower():
            logits = outputs.logits
            # Only segformer needs upsampling
            return F.interpolate(
                logits,
                size=pixel_values.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        elif 'mask2former' in self.model_name.lower():
            # Mask2Former: combine masks and class logits
            pred = outputs.pred_masks
            cls = outputs.class_queries_logits.softmax(dim=-1)
            return torch.einsum('bqhw,bqc->bchw', pred, cls)
        else:
            raise ValueError(f"Unknown model type '{self.model_name}' in _postprocess_logits")
    
class Hierarchical_SegModel(nn.Module):
    def __init__(self,supersegmodel, input_channel,num_labels,model_name, seed=2022,num_out_channels=3,intermediate_channels=512):
        super().__init__()
        torch.manual_seed(seed)
        self.model_name = model_name
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

        self.model = BaseSegModel(model_name=model_name,num_labels=num_labels,in_channels=num_out_channels+3) 
        # The modified segmentation model input size needs to be mask output channels + 3(here, 3 is original input image channel size)
        self.model.modify_input_channels()

    def initialize_peft_model(self,lora_config):
        self.model.initialize_peft_model(lora_config)

    def forward(self, inp, labels=None):
        with torch.no_grad():
            superseg_masks = self.supersegmodel(inp,None).logits
        superseg_masks = self.mask_reducer(superseg_masks)
        # Concatenate the input image with the superclass segmentation masks
        combined_input = torch.cat([inp, superseg_masks], dim=1)  # Shape: (B, C+M, H, W)

        # Pass the combined input through the segmentation model
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

        self.model = BaseSegModel(model_name=model_name,num_labels=num_labels)

    def initialize_peft_model(self,lora_config):
        self.model.initialize_peft_model(lora_config)

    def forward(self, inp, labels):
        with torch.no_grad():
            superseg_masks = self.supersegmodel(inp).logits
        # Pass the input through the segmentation model
        output = self.model(inp,labels)
        # Concatenate the output masks with the superclass segmentation masks
        # print(f"superseg_masks:{superseg_masks.size()} output.logits:{output.logits.size()}")
        combined_masks = torch.cat([output.logits, superseg_masks], dim=1)  # Shape: (B, C+M, H, W)

        output_logits = self.fusion_layer(combined_masks)

        # The base model returns logits same size as image. So no changes needed
        # labels = F.interpolate(labels.unsqueeze(1).float(), size=output_logits.shape[-2:], mode="nearest").squeeze(1).long()
    
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

        # segmentation model
        self.model = BaseSegModel(model_name=model_name,num_labels=num_labels)

        # Adjust output layer to match supersegmodel's number of labels
        self.adjust_segformer_output = nn.Conv2d(num_labels, num_labels_superseg, kernel_size=1)

        # Gate mechanism to decide the contribution of each model
        self.gate_network = GatingNetwork(num_labels_superseg)

    def initialize_peft_model(self,lora_config):
        self.model.initialize_peft_model(lora_config)

    def forward(self, inp, labels):
        # Get logits from the super segmentation model
        with torch.no_grad():
            superseg_logits = self.supersegmodel(inp).logits  # Shape: (B, num_labels_superseg, H, W)
        
        # Get logits from the segmentation model
        segformer_output = self.model(inp, labels)  # Shape: (B, num_labels, H, W)
        
        # Adjust segmentation logits to match the supersegmodel's number of classes
        adjusted_segformer_logits = self.adjust_segformer_output(segformer_output.logits)  # Shape: (B, num_labels_superseg, H, W)

        # Gating mechanism: compute gate weights using the image and supersegmodel logits
        gate_weights = self.gate_network(inp,superseg_logits)  # Shape: (B, 2, H, W)

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

        # Resample labels to match output size. Now since segmentation model logits output same sized output, there is no need for this.
        # labels = F.interpolate(labels.unsqueeze(1).float(), size=refined_logits.shape[-2:], mode="nearest").squeeze(1).long()
        
        # Compute Cross-Entropy Loss
        ce_loss = F.cross_entropy(refined_logits, labels, reduction='mean')

        return FusionSegOutput(loss=ce_loss, logits=refined_logits)
