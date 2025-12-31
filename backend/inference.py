"""
Model Inference Pipeline for StripUnetMCSA
"""
import torch
import torch.nn.functional as F
from config import DEVICE
from model_loader import get_model


def predict_tile(model, tile_tensor):
    """
    Run inference on a single tile
    
    Args:
        model: StripUnetMCSA model
        tile_tensor: torch tensor (C, H, W)
    
    Returns:
        torch tensor: prediction (1, H, W) with sigmoid applied
    """
    # Add batch dimension
    tile_batch = tile_tensor.unsqueeze(0).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        prediction = model(tile_batch)
        
        # Apply sigmoid to get probabilities
        prediction = torch.sigmoid(prediction)
    
    # Remove batch dimension
    prediction = prediction.squeeze(0)
    
    return prediction


def predict_tiles_batch(model, tiles_list, batch_size=4):
    """
    Run inference on multiple tiles with batching for efficiency
    
    Args:
        model: StripUnetMCSA model
        tiles_list: list of tile tensors
        batch_size: number of tiles to process at once
    
    Returns:
        list of prediction tensors
    """
    predictions = []
    
    # Process in batches
    for i in range(0, len(tiles_list), batch_size):
        batch_tiles = tiles_list[i:i + batch_size]
        
        # Stack into batch
        batch_tensor = torch.stack(batch_tiles).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            batch_predictions = model(batch_tensor)
            
            # Apply sigmoid
            batch_predictions = torch.sigmoid(batch_predictions)
        
        # Collect predictions
        for j in range(batch_predictions.shape[0]):
            predictions.append(batch_predictions[j].cpu())
    
    return predictions


def run_inference(image_tensor, tiles, tile_positions):
    """
    Complete inference pipeline on tiled image
    
    Args:
        image_tensor: original image tensor (C, H, W)
        tiles: list of tile tensors
        tile_positions: list of tile position dictionaries
    
    Returns:
        list of prediction tensors (one per tile)
    """
    # Get model
    model = get_model()
    
    # Run inference on all tiles
    predictions = predict_tiles_batch(model, tiles, batch_size=4)
    
    return predictions


def inference_single_image(image_tensor):
    """
    Simple inference for single image (no tiling)
    Used for small images that fit in one tile
    
    Args:
        image_tensor: torch tensor (C, H, W)
    
    Returns:
        torch tensor: prediction (1, H, W)
    """
    model = get_model()
    
    # Add batch dimension
    image_batch = image_tensor.unsqueeze(0).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        prediction = model(image_batch)
        prediction = torch.sigmoid(prediction)
    
    # Remove batch dimension and move to CPU
    prediction = prediction.squeeze(0).cpu()
    
    return prediction
