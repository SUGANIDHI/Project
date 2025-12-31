"""
API Client for communicating with FastAPI backend
"""
import requests
import base64
import io
from PIL import Image
import numpy as np


class BackendClient:
    """Client for StripUnetMCSA backend API"""
    
    def __init__(self, backend_url="http://localhost:8000"):
        """
        Initialize API client
        
        Args:
            backend_url: URL of the FastAPI backend
        """
        self.backend_url = backend_url
        self.predict_endpoint = f"{backend_url}/predict"
        self.health_endpoint = f"{backend_url}/health"
    
    def check_health(self):
        """
        Check if backend is healthy
        
        Returns:
            dict: health status or None if unreachable
        """
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"Health check failed: {e}")
            return None
    
    def predict(self, image):
        """
        Send image to backend for prediction
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            dict: response with mask and overlay or None if failed
        """
        try:
            # Convert to PIL Image if numpy array
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Send request
            files = {'file': ('image.png', img_byte_arr, 'image/png')}
            response = requests.post(self.predict_endpoint, files=files, timeout=60)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Prediction failed: {response.status_code}")
                print(response.text)
                return None
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def decode_base64_image(self, base64_string):
        """
        Decode base64 string to PIL Image
        
        Args:
            base64_string: base64 encoded image
        
        Returns:
            PIL Image
        """
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        return image
    
    def predict_and_decode(self, image):
        """
        Send prediction request and decode results
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            tuple: (mask_image, overlay_image, info) or (None, None, None) if failed
        """
        result = self.predict(image)
        
        if result is None or not result.get('success'):
            return None, None, None
        
        # Decode images
        mask = self.decode_base64_image(result['mask'])
        overlay = self.decode_base64_image(result['overlay'])
        info = result.get('info', {})
        
        return mask, overlay, info


# Convenience function
def get_client(backend_url="http://localhost:8000"):
    """
    Get API client instance
    
    Args:
        backend_url: URL of backend server
    
    Returns:
        BackendClient instance
    """
    return BackendClient(backend_url)
