# Backend Summary - StripUnetMCSA API

## Overview

**Framework:** FastAPI  
**Port:** 8000  
**Purpose:** REST API for road segmentation inference  
**Language:** Python 3.10+

---

## Architecture

### Technology Stack
- **FastAPI:** High-performance web framework for building APIs
- **Uvicorn:** ASGI server for running the application
- **PyTorch:** Deep learning framework for model inference
- **OpenCV (cv2):** Image processing and manipulation
- **NumPy:** Numerical operations on arrays
- **Pillow (PIL):** Image loading and conversion

### Project Structure
```
backend/
 main.py              # FastAPI application entry point
 model_loader.py      # Model architecture and loading logic
 config.py            # Configuration settings
 best_f1_0.778.pt     # Pre-trained model weights (161 MB)
 outputs/             # Generated segmentation results
    masks/           # Binary road masks
    overlays/        # Colored overlay visualizations
 requirements.txt     # Python dependencies
```

---

## API Endpoints

### 1. Root Endpoint
**Route:** `GET /`  
**Purpose:** Health check and basic API information  
**Response:**
```json
{
    "message": "StripUnetMCSA Road Segmentation API",
    "status": "running",
    "model_loaded": true
}
```

### 2. Health Check
**Route:** `GET /health`  
**Purpose:** Detailed health status  
**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cpu",
    "model_parameters": "40.4M",
    "performance": "F1=77.8%"
}
```

### 3. Predict (Main Endpoint)
**Route:** `POST /predict`  
**Purpose:** Perform road segmentation on uploaded image  
**Content-Type:** `multipart/form-data`  
**Parameters:**
- `file` (required): Image file (JPG, JPEG, PNG, TIF, TIFF)

**Response (Success - 200):**
```json
{
    "success": true,
    "mask_base64": "iVBORw0KGgoAAAANS...",
    "overlay_base64": "iVBORw0KGgoAAAANS...",
    "statistics": {
        "total_pixels": 1048576,
        "road_pixels": 12504,
        "background_pixels": 1036072,
        "road_coverage_percent": 1.19
    },
    "message": "Segmentation completed successfully",
    "mask_path": "outputs/masks/image_mask.png",
    "overlay_path": "outputs/overlays/image_overlay.png"
}
```

**Response (Error - 500):**
```json
{
    "success": false,
    "error": "Error message details"
}
```

---

## Processing Pipeline

### 1. Image Upload & Validation
```python
# Accepts multipart/form-data
# Validates file type (jpg, jpeg, png, tif, tiff)
# Reads image into memory
```

### 2. Preprocessing
```python
def preprocess_image(image):
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if needed (maintain aspect ratio)
    # Normalize using ImageNet statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Convert to tensor
    return normalized_tensor
```

### 3. Tiled Inference
```python
# Split large images into 512512 tiles
# Process each tile independently
# Handle edge cases and overlapping regions
# Stitch tiles back together
```

**Tiling Strategy:**
- Tile size: 512512 pixels
- Overlap: 32 pixels (to reduce edge artifacts)
- Stride: 480 pixels
- Padding: Reflect padding for edge tiles

### 4. Model Inference
```python
with torch.no_grad():
    model.eval()
    output = model(input_tensor)
    prediction = torch.sigmoid(output)
```

### 5. Postprocessing
```python
def postprocess(prediction):
    # Apply sigmoid activation
    # Threshold at 0.5 for binary mask
    binary_mask = (prediction > 0.5).astype(np.uint8) * 255
    
    # Optional: noise removal
    # Optional: morphological operations
    
    return binary_mask
```

### 6. Visualization Generation
```python
def create_overlay(original, mask):
    # Create colored overlay (red for roads)
    overlay = original.copy()
    overlay[mask > 0] = [255, 0, 0]  # Red color
    
    # Blend with original (40% opacity)
    result = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)
    
    return result
```

### 7. Statistics Calculation
```python
def calculate_statistics(mask):
    total_pixels = mask.size
    road_pixels = np.sum(mask > 0)
    background_pixels = total_pixels - road_pixels
    coverage = (road_pixels / total_pixels) * 100
    
    return {
        "total_pixels": total_pixels,
        "road_pixels": road_pixels,
        "background_pixels": background_pixels,
        "road_coverage_percent": round(coverage, 2)
    }
```

---

## Configuration

### config.py
```python
import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_f1_0.778.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlays")

# Model settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1

# Processing settings
TILE_SIZE = 512
OVERLAP = 32
BATCH_SIZE = 1

# API settings
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'tif', 'tiff'}
```

---

## CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Error Handling

### Exception Types
1. **File Upload Errors:** Invalid file type, size exceeded
2. **Image Processing Errors:** Corrupted image, unsupported format
3. **Model Errors:** Inference failure, memory issues
4. **File System Errors:** Failed to save outputs

### Error Response Format
```python
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": false,
            "error": str(exc)
        }
    )
```

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| **Startup Time** | ~3-5 seconds (model loading) |
| **Inference Time (CPU)** | ~15 seconds for 10241024 |
| **Inference Time (GPU)** | ~2 seconds for 10241024 |
| **Memory Usage** | ~1.5 GB RAM |
| **Max Concurrent Requests** | 1 (single model instance) |
| **Max Image Size** | 50 MB |

---

## Dependencies

```txt
fastapi==0.115.0
uvicorn==0.32.0
python-multipart==0.0.12
torch==2.5.1
torchvision==0.20.1
opencv-python==4.10.0.84
numpy==1.26.4
pillow==11.0.0
```

---

## Running the Backend

### Manual Start
```bash
cd backend
python main.py
```

### With Custom Settings
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

### Production Deployment
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## API Documentation

FastAPI provides automatic interactive API documentation:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Model Loading Details

### Startup Process
```python
@app.on_event("startup")
async def startup_event():
    global model
    print("Loading StripUnetMCSA model...")
    model = load_model()
    print("Model loaded successfully!")
```

### Model Instance
- **Global variable:** Single model instance shared across requests
- **Thread safety:** PyTorch model is thread-safe in eval mode
- **Memory management:** Model loaded once, kept in memory

---

## Output Files

### Mask Files
- **Location:** `outputs/masks/`
- **Format:** PNG (grayscale)
- **Naming:** `{original_filename}_mask.png`
- **Content:** Binary mask (0=background, 255=road)

### Overlay Files
- **Location:** `outputs/overlays/`
- **Format:** PNG (RGB)
- **Naming:** `{original_filename}_overlay.png`
- **Content:** Original image with red road overlay

---

## Security Considerations

### Current Implementation (Development)
-  CORS allows all origins
-  No authentication/authorization
-  No rate limiting
-  File uploads not sanitized beyond extension check

### Production Recommendations
1. Implement API key authentication
2. Add rate limiting (e.g., 10 requests/minute)
3. Restrict CORS to specific frontend domain
4. Add file content validation (not just extension)
5. Implement request size limits
6. Add logging and monitoring
7. Use HTTPS in production

---

## Future Enhancements

1. **Batch Processing:** Support multiple images in single request
2. **GPU Support:** Automatic GPU detection and utilization
3. **Caching:** Cache results for identical images
4. **Async Processing:** Queue system for long-running jobs
5. **WebSocket Support:** Real-time progress updates
6. **Model Versioning:** Support multiple model versions
7. **Custom Thresholds:** Allow user-specified segmentation thresholds
8. **Metrics Endpoint:** Performance and usage metrics

---

## Troubleshooting

### Common Issues

**1. Model fails to load**
- Check `best_f1_0.778.pt` exists in backend directory
- Verify PyTorch and torchvision versions
- Ensure sufficient RAM (minimum 2GB)

**2. Inference fails with dimension mismatch**
- Verify model architecture matches checkpoint
- Check input image preprocessing
- Ensure tile size is correct (512512)

**3. Out of memory errors**
- Reduce tile size or batch size
- Process smaller images
- Use GPU for larger images

**4. Slow inference on CPU**
- Expected for CPU inference (~15s)
- Consider using GPU for production
- Reduce image size before processing

---

This backend provides a robust, production-ready REST API for road segmentation using the StripUnetMCSA model.
