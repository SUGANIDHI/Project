"""
StripUnet Tiling Strategy Implementation
Splits large images into tiles for efficient processing
"""
import torch
from config import TILE_SIZE


def create_tiles(image_tensor, tile_size=TILE_SIZE):
    """
    Split image into tiles using StripUnet strategy
    
    StripUnet uses a specific tiling pattern:
    - Creates 5 overlapping tiles of 512x512
    - Ensures complete coverage of the image
    - Maintains spatial consistency
    
    Args:
        image_tensor: torch tensor (C, H, W)
        tile_size: size of each tile (default 512)
    
    Returns:
        tuple: (tiles_list, tile_positions)
    """
    c, h, w = image_tensor.shape
    
    tiles = []
    positions = []
    
    # Calculate stride to create 5 tiles with overlap
    # This is a simplified version - adjust based on actual StripUnet paper
    
    if h <= tile_size and w <= tile_size:
        # Image fits in one tile
        # Pad if necessary
        pad_h = max(0, tile_size - h)
        pad_w = max(0, tile_size - w)
        
        if pad_h > 0 or pad_w > 0:
            tile = torch.nn.functional.pad(
                image_tensor,
                (0, pad_w, 0, pad_h),
                mode='reflect'
            )
        else:
            tile = image_tensor
        
        tiles.append(tile)
        positions.append({'row': 0, 'col': 0, 'h': h, 'w': w})
    
    else:
        # Create tiling grid
        # For StripUnet: 5 tiles configuration
        # Adjust this based on your specific needs
        
        # Calculate number of tiles in each dimension
        n_tiles_h = max(1, (h + tile_size - 1) // tile_size)
        n_tiles_w = max(1, (w + tile_size - 1) // tile_size)
        
        # Calculate overlap
        if n_tiles_h > 1:
            stride_h = (h - tile_size) // (n_tiles_h - 1)
        else:
            stride_h = 0
            
        if n_tiles_w > 1:
            stride_w = (w - tile_size) // (n_tiles_w - 1)
        else:
            stride_w = 0
        
        # Create tiles
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate position
                start_h = min(i * stride_h if i > 0 else 0, h - tile_size)
                start_w = min(j * stride_w if j > 0 else 0, w - tile_size)
                
                end_h = min(start_h + tile_size, h)
                end_w = min(start_w + tile_size, w)
                
                # Extract tile
                tile = image_tensor[:, start_h:end_h, start_w:end_w]
                
                # Pad if necessary
                actual_h, actual_w = tile.shape[1], tile.shape[2]
                if actual_h < tile_size or actual_w < tile_size:
                    pad_h = tile_size - actual_h
                    pad_w = tile_size - actual_w
                    tile = torch.nn.functional.pad(
                        tile,
                        (0, pad_w, 0, pad_h),
                        mode='reflect'
                    )
                
                tiles.append(tile)
                positions.append({
                    'row': start_h,
                    'col': start_w,
                    'h': actual_h,
                    'w': actual_w,
                    'tile_idx': len(tiles) - 1
                })
    
    return tiles, positions


def reconstruct_from_tiles(tile_predictions, tile_positions, original_shape):
    """
    Reconstruct full prediction from tile predictions
    
    Args:
        tile_predictions: list of prediction tensors (1, H, W) or (H, W)
        tile_positions: list of position dictionaries
        original_shape: tuple (H, W) of original image
    
    Returns:
        torch tensor: reconstructed prediction (H, W)
    """
    h, w = original_shape
    
    # Initialize output and weight map for averaging overlaps
    output = torch.zeros((h, w), dtype=torch.float32)
    weight_map = torch.zeros((h, w), dtype=torch.float32)
    
    for pred, pos in zip(tile_predictions, tile_positions):
        # Remove batch dimension if present
        if pred.dim() == 3:
            pred = pred.squeeze(0)
        
        # Extract valid region (remove padding)
        tile_h = pos['h']
        tile_w = pos['w']
        valid_pred = pred[:tile_h, :tile_w]
        
        # Get position
        start_h = pos['row']
        start_w = pos['col']
        
        # Add to output with weighted averaging
        output[start_h:start_h+tile_h, start_w:start_w+tile_w] += valid_pred
        weight_map[start_h:start_h+tile_h, start_w:start_w+tile_w] += 1.0
    
    # Average overlapping regions
    output = output / (weight_map + 1e-8)
    
    return output


def get_tiling_info(image_shape, tile_size=TILE_SIZE):
    """
    Get information about tiling for a given image shape
    
    Args:
        image_shape: tuple (H, W)
        tile_size: size of each tile
    
    Returns:
        dict: tiling information
    """
    h, w = image_shape
    
    if h <= tile_size and w <= tile_size:
        n_tiles = 1
    else:
        n_tiles_h = max(1, (h + tile_size - 1) // tile_size)
        n_tiles_w = max(1, (w + tile_size - 1) // tile_size)
        n_tiles = n_tiles_h * n_tiles_w
    
    return {
        'num_tiles': n_tiles,
        'tile_size': tile_size,
        'image_height': h,
        'image_width': w
    }
