"""
Configuration settings for StripUnetMCSA backend
"""
import os
import torch

# Model Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_f1_0.778.pt")
TILE_SIZE = 512
NUM_TILES = 5

# Inference Configuration
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image Preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Output Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
OVERLAYS_DIR = os.path.join(OUTPUT_DIR, "overlays")

# Model Architecture
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1
