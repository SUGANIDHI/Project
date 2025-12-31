"""
Visualization utilities for Streamlit frontend
"""
import numpy as np
from PIL import Image
import io


def prepare_image_display(image, max_width=800):
    """
    Prepare image for display in Streamlit
    
    Args:
        image: PIL Image or numpy array
        max_width: maximum display width
    
    Returns:
        PIL Image: resized image for display
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Resize if too large
    if image.width > max_width:
        aspect = image.height / image.width
        new_width = max_width
        new_height = int(max_width * aspect)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return image


def create_side_by_side(image1, image2, labels=None):
    """
    Create side-by-side comparison image
    
    Args:
        image1: first PIL Image
        image2: second PIL Image
        labels: tuple of (label1, label2)
    
    Returns:
        PIL Image: combined image
    """
    # Ensure same height
    if image1.height != image2.height:
        target_height = max(image1.height, image2.height)
        
        if image1.height < target_height:
            aspect = image1.width / image1.height
            image1 = image1.resize((int(target_height * aspect), target_height), Image.LANCZOS)
        
        if image2.height < target_height:
            aspect = image2.width / image2.height
            image2 = image2.resize((int(target_height * aspect), target_height), Image.LANCZOS)
    
    # Create combined image
    total_width = image1.width + image2.width
    combined = Image.new('RGB', (total_width, image1.height))
    combined.paste(image1, (0, 0))
    combined.paste(image2, (image1.width, 0))
    
    return combined


def colorize_mask(mask, color=(0, 255, 0)):
    """
    Convert binary mask to colored RGB image
    
    Args:
        mask: PIL Image (grayscale) or numpy array
        color: RGB tuple for mask color
    
    Returns:
        PIL Image: colored mask
    """
    if isinstance(mask, Image.Image):
        mask_array = np.array(mask)
    else:
        mask_array = mask
    
    # Ensure 2D
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    
    # Create RGB
    colored = np.zeros((*mask_array.shape, 3), dtype=np.uint8)
    for i in range(3):
        colored[:, :, i] = (mask_array > 0) * color[i]
    
    return Image.fromarray(colored)


def create_overlay_visualization(original, mask, alpha=0.4, mask_color=(0, 255, 0)):
    """
    Create overlay of mask on original image
    
    Args:
        original: PIL Image (original)
        mask: PIL Image (binary mask)
        alpha: transparency (0-1)
        mask_color: RGB color for mask
    
    Returns:
        PIL Image: overlay
    """
    # Convert to arrays
    if isinstance(original, Image.Image):
        orig_array = np.array(original.convert('RGB'))
    else:
        orig_array = original
    
    if isinstance(mask, Image.Image):
        mask_array = np.array(mask.convert('L'))
    else:
        mask_array = mask
    
    # Ensure 2D mask
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    
    # Create colored mask
    colored_mask = np.zeros_like(orig_array)
    for i in range(3):
        colored_mask[:, :, i] = (mask_array > 0) * mask_color[i]
    
    # Blend
    overlay = (1 - alpha) * orig_array + alpha * colored_mask
    overlay = overlay.astype(np.uint8)
    
    return Image.fromarray(overlay)


def get_download_link_data(image, filename="output.png"):
    """
    Convert image to downloadable bytes
    
    Args:
        image: PIL Image
        filename: suggested filename
    
    Returns:
        bytes: image as bytes
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def display_statistics(mask):
    """
    Calculate statistics from mask
    
    Args:
        mask: PIL Image or numpy array
    
    Returns:
        dict: statistics
    """
    if isinstance(mask, Image.Image):
        mask_array = np.array(mask)
    else:
        mask_array = mask
    
    # Ensure 2D
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]
    
    # Calculate stats
    total_pixels = mask_array.size
    road_pixels = np.sum(mask_array > 0)
    background_pixels = total_pixels - road_pixels
    
    road_percentage = (road_pixels / total_pixels) * 100
    
    return {
        'total_pixels': total_pixels,
        'road_pixels': int(road_pixels),
        'background_pixels': int(background_pixels),
        'road_percentage': road_percentage
    }


def format_statistics(stats):
    """
    Format statistics for display
    
    Args:
        stats: dict from display_statistics
    
    Returns:
        str: formatted string
    """
    return f"""
**Segmentation Statistics:**
- Total Pixels: {stats['total_pixels']:,}
- Road Pixels: {stats['road_pixels']:,}
- Background Pixels: {stats['background_pixels']:,}
- Road Coverage: {stats['road_percentage']:.2f}%
    """.strip()
