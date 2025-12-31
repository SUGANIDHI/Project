"""
Postprocessing for segmentation masks
Stitches tiles and applies thresholding
"""
import torch
import numpy as np
from scipy import ndimage
from config import THRESHOLD


def stitch_tiles(predictions, tile_positions, original_shape):
    """
    Stitch tile predictions into full image prediction
    
    Args:
        predictions: list of prediction tensors
        tile_positions: list of position dictionaries
        original_shape: tuple (H, W)
    
    Returns:
        torch tensor: stitched prediction (H, W)
    """
    from tiling import reconstruct_from_tiles
    return reconstruct_from_tiles(predictions, tile_positions, original_shape)


def apply_threshold(prediction, threshold=THRESHOLD):
    """
    Apply threshold to convert probabilities to binary mask
    
    Args:
        prediction: torch tensor with values in [0, 1]
        threshold: threshold value
    
    Returns:
        torch tensor: binary mask (0 or 1)
    """
    binary_mask = (prediction >= threshold).float()
    return binary_mask


def remove_small_objects(binary_mask, min_size=100):
    """
    Remove small connected components (noise removal)
    
    Args:
        binary_mask: torch tensor or numpy array
        min_size: minimum object size in pixels
    
    Returns:
        cleaned mask
    """
    # Convert to numpy if needed
    if isinstance(binary_mask, torch.Tensor):
        mask_np = binary_mask.cpu().numpy()
        was_tensor = True
    else:
        mask_np = binary_mask
        was_tensor = False
    
    # Label connected components
    labeled, num_features = ndimage.label(mask_np)
    
    # Remove small components
    sizes = ndimage.sum(mask_np, labeled, range(num_features + 1))
    mask_cleaned = np.zeros_like(mask_np)
    
    for i in range(1, num_features + 1):
        if sizes[i] >= min_size:
            mask_cleaned[labeled == i] = 1
    
    # Convert back to tensor if needed
    if was_tensor:
        mask_cleaned = torch.from_numpy(mask_cleaned).float()
    
    return mask_cleaned


def fill_holes(binary_mask):
    """
    Fill holes in binary mask
    
    Args:
        binary_mask: torch tensor or numpy array
    
    Returns:
        mask with holes filled
    """
    # Convert to numpy if needed
    if isinstance(binary_mask, torch.Tensor):
        mask_np = binary_mask.cpu().numpy()
        was_tensor = True
    else:
        mask_np = binary_mask
        was_tensor = False
    
    # Fill holes
    mask_filled = ndimage.binary_fill_holes(mask_np).astype(float)
    
    # Convert back to tensor if needed
    if was_tensor:
        mask_filled = torch.from_numpy(mask_filled).float()
    
    return mask_filled


def postprocess_prediction(prediction, original_shape, threshold=THRESHOLD, 
                          remove_noise=True, min_object_size=100):
    """
    Complete postprocessing pipeline
    
    Args:
        prediction: prediction tensor (H, W) with probabilities
        original_shape: tuple (H, W) for final output
        threshold: threshold for binarization
        remove_noise: whether to remove small objects
        min_object_size: minimum size for objects
    
    Returns:
        numpy array: final binary mask (H, W) as uint8
    """
    # Apply threshold
    binary_mask = apply_threshold(prediction, threshold)
    
    # Remove small objects if requested
    if remove_noise:
        binary_mask = remove_small_objects(binary_mask, min_size=min_object_size)
    
    # Fill holes
    binary_mask = fill_holes(binary_mask)
    
    # Ensure correct size
    h, w = original_shape
    current_h, current_w = binary_mask.shape
    
    if current_h != h or current_w != w:
        # Crop to original size
        binary_mask = binary_mask[:h, :w]
    
    # Convert to numpy uint8
    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.cpu().numpy()
    
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    
    return mask_uint8


def create_overlay(image, mask, alpha=0.5, mask_color=(0, 255, 0)):
    """
    Create overlay visualization
    
    Args:
        image: original image as numpy array (H, W, 3)
        mask: binary mask as numpy array (H, W)
        alpha: transparency of overlay
        mask_color: RGB color for mask
    
    Returns:
        numpy array: overlay image (H, W, 3)
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    for i in range(3):
        colored_mask[:, :, i] = (mask > 0) * mask_color[i]
    
    # Blend
    overlay = image.copy()
    overlay = (1 - alpha) * overlay + alpha * colored_mask
    overlay = overlay.astype(np.uint8)
    
    return overlay
