"""
Image preprocessing for StripUnetMCSA inference
"""
import numpy as np
import torch
from PIL import Image
from config import IMAGENET_MEAN, IMAGENET_STD


def preprocess_image(image):
    """
    Preprocess image for model inference
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        tuple: (preprocessed_tensor, original_size)
    """
    # Convert to PIL Image if numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Store original size for reconstruction
    original_size = image.size  # (width, height)
    
    # Convert to numpy array
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize using ImageNet statistics
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)
    img_array = (img_array - mean) / std
    
    # Convert to torch tensor (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
    
    return img_tensor, original_size


def denormalize_image(tensor):
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor: normalized torch tensor (C, H, W)
    
    Returns:
        numpy array (H, W, C) in range [0, 255]
    """
    # Convert to numpy
    img = tensor.permute(1, 2, 0).cpu().numpy()
    
    # Denormalize
    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)
    img = img * std + mean
    
    # Clip and convert to uint8
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    
    return img


def prepare_for_tiling(image_tensor, tile_size=512):
    """
    Prepare image tensor for tiling
    Ensures dimensions are compatible with tile size
    
    Args:
        image_tensor: torch tensor (C, H, W)
        tile_size: size of each tile
    
    Returns:
        tuple: (padded_tensor, padding_info)
    """
    c, h, w = image_tensor.shape
    
    # Calculate padding needed
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    
    # Pad if necessary
    if pad_h > 0 or pad_w > 0:
        # Pad on right and bottom
        image_tensor = torch.nn.functional.pad(
            image_tensor,
            (0, pad_w, 0, pad_h),
            mode='reflect'
        )
    
    padding_info = {
        'original_h': h,
        'original_w': w,
        'padded_h': h + pad_h,
        'padded_w': w + pad_w,
        'pad_h': pad_h,
        'pad_w': pad_w
    }
    
    return image_tensor, padding_info
