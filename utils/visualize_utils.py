import numpy as np
import cv2

def unnormalize_image(image_array, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Unnormalize an image tensor based on the given mean and std.

    Args:
        image_tensor (numpy array): The normalized image tensor.
        mean (list or tuple): The mean values used during normalization.
        std (list or tuple): The standard deviation values used during normalization.

    Returns:
        numpy array: The unnormalized image tensor.
    """
     # Ensure the array is in (H, W, C) format
    mean = np.array(mean).reshape(1, 1, -1)  # Reshape mean to (1, 1, C)
    std = np.array(std).reshape(1, 1, -1)    # Reshape std to (1, 1, C)

    # Reverse normalization
    unnormalized_image = image_array * std + mean
    unnormalized_image = (unnormalized_image * 255).clip(0, 255).astype(np.uint8)
    
    return unnormalized_image


def hex_to_rgb(hex_code):
    """Convert a hex color code to an RGB tuple."""
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def overlay_colored_mask(image, mask, id_to_color, alpha=0.5):
    """
    Overlay a segmentation mask on an image using colors defined in meta.json.

    Args:
        image (numpy array): The image tensor (H, W, 3) in float32 format (normalized).
        mask (numpy array): The segmentation mask tensor (H, W) with class labels.
        id_to_color (dict): Stores mapping between category ID and color for it.
        alpha (float): Transparency for the mask overlay (0.0 to 1.0).

    Returns:
        overlayed_image (numpy array): The image with the mask overlay.
    """

    # Unnormalize the image
    image = unnormalize_image(image)

    # Ensure the image and mask dimensions match
    assert image.shape[:2] == mask.shape, "Image and mask dimensions do not match."

    # Initialize a blank color mask (H, W, 3)
    color_mask = np.zeros_like(image, dtype=np.uint8)

    # Fill the color mask based on the segmentation mask and id_to_color mapping
    for cls, color in id_to_color.items():
        rgb_color = hex_to_rgb(color)  # Convert hex color to RGB
        color_mask[mask == cls] = rgb_color  # Assign color to corresponding pixels

    # Step 3: Create a blended mask where mask != 0
    blended_image = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    # Step 4: Combine blended and original image based on mask
    overlayed_image = np.where(mask[..., None] != 0, blended_image, image)

    return overlayed_image
