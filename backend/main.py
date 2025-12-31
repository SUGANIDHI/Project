"""
FastAPI Backend Server for StripUnetMCSA
"""
import os
import io
import base64
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

from config import API_HOST, API_PORT, MASKS_DIR, OVERLAYS_DIR
from preprocessing import preprocess_image
from tiling import create_tiles, get_tiling_info
from inference import run_inference
from postprocessing import postprocess_prediction, create_overlay
from model_loader import load_model


# Initialize FastAPI app
app = FastAPI(
    title="StripUnetMCSA Road Segmentation API",
    description="Backend API for road segmentation using StripUnetMCSA model",
    version="1.0.0"
)

# Add CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create output directories
os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(OVERLAYS_DIR, exist_ok=True)


# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when server starts"""
    print("Loading StripUnetMCSA model...")
    load_model()
    print("Model loaded successfully!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "StripUnetMCSA Road Segmentation API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    from model_loader import _model
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict road segmentation mask for uploaded image
    
    Args:
        file: Uploaded image file
    
    Returns:
        JSON with base64 encoded mask and overlay
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        original_image = np.array(image.convert('RGB'))
        
        print(f"Processing image: {file.filename}, size: {image.size}")
        
        # Preprocess
        image_tensor, original_size = preprocess_image(image)
        print(f"Preprocessed tensor shape: {image_tensor.shape}")
        
        # Create tiles
        tiles, tile_positions = create_tiles(image_tensor)
        tiling_info = get_tiling_info(image_tensor.shape[1:])
        print(f"Created {len(tiles)} tiles")
        
        # Run inference
        predictions = run_inference(image_tensor, tiles, tile_positions)
        print(f"Generated {len(predictions)} predictions")
        
        # Postprocess
        from tiling import reconstruct_from_tiles
        stitched = reconstruct_from_tiles(
            predictions, 
            tile_positions, 
            (image_tensor.shape[1], image_tensor.shape[2])
        )
        
        # Get final mask
        final_mask = postprocess_prediction(
            stitched, 
            (original_image.shape[0], original_image.shape[1]),
            remove_noise=True,
            min_object_size=100
        )
        print(f"Final mask shape: {final_mask.shape}")
        
        # Create overlay
        overlay = create_overlay(original_image, final_mask, alpha=0.4)
        
        # Convert to base64 for transmission
        mask_image = Image.fromarray(final_mask)
        overlay_image = Image.fromarray(overlay)
        
        # Encode mask
        mask_buffer = io.BytesIO()
        mask_image.save(mask_buffer, format="PNG")
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
        
        # Encode overlay
        overlay_buffer = io.BytesIO()
        overlay_image.save(overlay_buffer, format="PNG")
        overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode()
        
        # Save outputs (optional)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_path = os.path.join(MASKS_DIR, f"mask_{timestamp}.png")
        overlay_path = os.path.join(OVERLAYS_DIR, f"overlay_{timestamp}.png")
        
        mask_image.save(mask_path)
        overlay_image.save(overlay_path)
        
        return JSONResponse({
            "success": True,
            "mask": mask_base64,
            "overlay": overlay_base64,
            "info": {
                "filename": file.filename,
                "original_size": original_size,
                "num_tiles": len(tiles),
                "mask_path": mask_path,
                "overlay_path": overlay_path
            }
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
