# Frontend Summary - StripUnetMCSA Web Application

## Overview

**Framework:** Streamlit  
**Port:** 8501  
**Purpose:** Interactive web interface for road segmentation  
**Language:** Python 3.10+

---

## Architecture

### Technology Stack
- **Streamlit:** Rapid web app development framework
- **Requests:** HTTP client for backend API communication
- **Pillow (PIL):** Image handling and display
- **NumPy:** Array operations for image processing
- **Base64:** Encoding for image transmission

### Project Structure
```
frontend/
├── app.py               # Main Streamlit application
└── requirements.txt     # Python dependencies
```

---

## User Interface Layout

### Page Structure
```
┌─────────────────────────────────────────────────┐
│  🛣️ StripUnetMCSA Road Segmentation            │
│  Detect road networks from satellite imagery    │
├─────────────────────────────────────────────────┤
│ Sidebar              │ Main Content             │
│ ┌─────────────────┐ │ ┌────────────────────┐  │
│ │ Settings        │ │ │ Upload Area        │  │
│ │                 │ │ └────────────────────┘  │
│ │ • Backend URL   │ │ ┌────────────────────┐  │
│ │ • Overlay       │ │ │ Results            │  │
│ │ • Transparency  │ │ │ • Original         │  │
│ │ • Statistics    │ │ │ • Mask             │  │
│ │                 │ │ │ • Overlay          │  │
│ │ About           │ │ └────────────────────┘  │
│ └─────────────────┘ │ ┌────────────────────┐  │
│                      │ │ Statistics         │  │
│                      │ └────────────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## Components Breakdown

### 1. Header Section
```python
st.set_page_config(
    page_title="StripUnetMCSA Road Segmentation",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ StripUnetMCSA Road Segmentation")
st.markdown("Detect road networks from satellite/aerial imagery...")
```

**Features:**
- Custom page title and icon
- Wide layout for better visualization
- Professional branding

### 2. Sidebar Configuration

#### Backend Settings
```python
backend_url = st.sidebar.text_input(
    "Backend URL",
    value="http://localhost:8000",
    help="FastAPI backend server address"
)
```

#### Display Settings
```python
show_overlay = st.sidebar.checkbox(
    "Show Overlay",
    value=True,
    help="Display original image with road overlay"
)

overlay_alpha = st.sidebar.slider(
    "Overlay Transparency",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Adjust overlay transparency"
)
```

#### Analysis Options
```python
show_stats = st.sidebar.checkbox(
    "Show Statistics",
    value=True,
    help="Display segmentation statistics"
)
```

#### About Section
```python
with st.sidebar.expander("ℹ️ About"):
    st.markdown("""
    **Model:** StripUnetMCSA
    **Performance:** F1=77.8%
    **Architecture:** ResNet50 + Dual-Branch Decoder
    **Parameters:** 40.4M
    """)
```

### 3. Main Content Area

#### File Upload Widget
```python
uploaded_file = st.file_uploader(
    "Upload Image",
    type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
    help="Upload satellite/aerial imagery"
)
```

**Features:**
- Drag-and-drop support
- File type validation
- Clear visual feedback

#### Image Preview
```python
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
```

#### Action Button
```python
if st.button("🚀 Run Segmentation", type="primary"):
    # Trigger segmentation process
```

**Styling:**
- Primary button style (blue/prominent)
- Rocket emoji for visual appeal
- Disabled if no file uploaded

### 4. Results Display

#### Three-Column Layout
```python
col1, col2, col3 = st.columns(3)

with col1:
    st.image(original, caption="Original Image")

with col2:
    st.image(mask, caption="Road Mask")

with col3:
    st.image(overlay, caption="Overlay")
```

**Features:**
- Side-by-side comparison
- Synchronized sizing
- Clear labels

#### Download Buttons
```python
col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="📥 Download Mask",
        data=mask_bytes,
        file_name="road_mask.png",
        mime="image/png"
    )

with col2:
    st.download_button(
        label="📥 Download Overlay",
        data=overlay_bytes,
        file_name="road_overlay.png",
        mime="image/png"
    )
```

### 5. Statistics Dashboard

```python
if show_stats and result.get('statistics'):
    st.subheader("📊 Segmentation Statistics")
    
    metric_cols = st.columns(4)
    stats = result['statistics']
    
    with metric_cols[0]:
        st.metric(
            "Total Pixels",
            f"{stats['total_pixels']:,}"
        )
    
    with metric_cols[1]:
        st.metric(
            "Road Pixels",
            f"{stats['road_pixels']:,}",
            delta="Detected"
        )
    
    with metric_cols[2]:
        st.metric(
            "Background",
            f"{stats['background_pixels']:,}"
        )
    
    with metric_cols[3]:
        st.metric(
            "Road Coverage",
            f"{stats['road_coverage_percent']}%",
            delta=f"{stats['road_coverage_percent']}%"
        )
```

**Features:**
- Clear metric cards
- Formatted numbers (with commas)
- Visual hierarchy

---

## User Workflow

### Step-by-Step Process

1. **Open Application**
   - Navigate to http://localhost:8501
   - View welcome screen

2. **Check Backend Connection**
   - Automatic connection test
   - Visual indicator (✅ Connected / ❌ Not Connected)

3. **Configure Settings** (Optional)
   - Adjust overlay transparency
   - Toggle statistics display
   - Change backend URL if needed

4. **Upload Image**
   - Click "Browse files" or drag-and-drop
   - Supported formats: JPG, JPEG, PNG, TIF, TIFF
   - Preview appears immediately

5. **Run Segmentation**
   - Click "🚀 Run Segmentation" button
   - Processing indicator appears
   - Wait for results (~15 seconds on CPU)

6. **View Results**
   - Original image displayed
   - Binary mask shown
   - Colored overlay generated
   - Statistics calculated and displayed

7. **Download Results** (Optional)
   - Download mask as PNG
   - Download overlay as PNG
   - Save for further analysis

---

## Backend Communication

### API Integration

#### Health Check
```python
def check_backend():
    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        if response.status_code == 200:
            return True, "Connected to backend - Model loaded"
        else:
            return False, f"Backend error: {response.status_code}"
    except Exception as e:
        return False, f"Cannot connect: {str(e)}"
```

#### Prediction Request
```python
def predict_segmentation(image_file):
    files = {'file': image_file}
    response = requests.post(
        f"{backend_url}/predict",
        files=files,
        timeout=60
    )
    return response.json()
```

#### Response Handling
```python
if result['success']:
    # Decode base64 images
    mask = base64.b64decode(result['mask_base64'])
    overlay = base64.b64decode(result['overlay_base64'])
    
    # Display results
    show_results(image, mask, overlay, result['statistics'])
else:
    st.error(f"Prediction failed: {result.get('error')}")
```

---

## State Management

### Session State Usage
```python
# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

if 'segmentation_result' not in st.session_state:
    st.session_state.segmentation_result = None

# Store results
st.session_state.segmentation_result = result
```

**Benefits:**
- Persist data across reruns
- Avoid redundant API calls
- Maintain user context

---

## Visual Design

### Color Scheme
- **Primary:** Blue (#0068C9) - Streamlit default
- **Success:** Green (#00C851) - Connected status
- **Error:** Red (#FF4B4B) - Error messages
- **Road Overlay:** Red (#FF0000) - Road visualization

### Typography
- **Headers:** Streamlit default (Source Sans Pro)
- **Body:** Streamlit default
- **Metrics:** Bold, larger font size

### Layout Principles
- **Wide layout:** Maximum screen utilization
- **Columns:** Balanced 3-column grid for results
- **Spacing:** Adequate padding between elements
- **Responsive:** Adapts to different screen sizes

---

## Error Handling

### User-Facing Errors

```python
# No backend connection
if not backend_connected:
    st.error("❌ Cannot connect to backend server")
    st.info("Make sure backend is running on http://localhost:8000")

# Upload validation
if not uploaded_file:
    st.info("👆 Please upload an image to get started")

# API errors
if not result['success']:
    st.error(f"❌ {result.get('error', 'Unknown error')}")
    st.info("Please check backend logs for details")

# Processing timeout
except requests.Timeout:
    st.error("⏱️ Request timeout. Image may be too large.")
```

---

## Performance Optimizations

### Image Caching
```python
@st.cache_data
def load_image(uploaded_file):
    return Image.open(uploaded_file)
```

### Result Caching
```python
@st.cache_data
def decode_base64_image(base64_str):
    return base64.b64decode(base64_str)
```

### Connection Pooling
```python
session = requests.Session()
# Reuse connection for multiple requests
```

---

## Dependencies

```txt
streamlit==1.40.1
requests==2.32.3
pillow==11.0.0
numpy==1.26.4
```

---

## Running the Frontend

### Manual Start
```bash
cd frontend
streamlit run app.py
```

### Headless Mode (No Browser)
```bash
streamlit run app.py --server.headless=true
```

### Custom Port
```bash
streamlit run app.py --server.port=8502
```

### Production Deployment
```bash
streamlit run app.py --server.port=8501 --server.headless=true --server.enableCORS=false
```

---

## Configuration

### .streamlit/config.toml
```toml
[server]
port = 8501
headless = true
enableCORS = false

[theme]
primaryColor = "#0068C9"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

---

## User Experience Features

### 1. Real-time Feedback
- ✅ Connection status indicator
- ⏳ Processing spinner
- 📊 Progress updates
- ✅ Success/error notifications

### 2. Intuitive Controls
- 🎨 Visual sliders for transparency
- ☑️ Toggle switches for options
- 🖱️ Clear action buttons
- 💾 Easy download options

### 3. Helpful Guidance
- ℹ️ Tooltips on hover
- 📝 Placeholder instructions
- ⚠️ Clear error messages
- 📖 About section with model info

### 4. Responsive Design
- 📱 Mobile-friendly layout
- 🖥️ Desktop optimized
- 🔄 Auto-refresh on changes
- ⚡ Fast load times

---

## Accessibility

### Current Features
- Clear labels for all inputs
- Descriptive button text
- High contrast colors
- Keyboard navigation support

### Future Enhancements
- ARIA labels for screen readers
- Keyboard shortcuts
- Dark mode support
- Font size controls

---

## Future Enhancements

1. **Batch Processing**
   - Upload multiple images
   - Bulk download results
   - Progress tracking

2. **Advanced Visualization**
   - Side-by-side slider comparison
   - Zoom and pan controls
   - Heatmap overlays

3. **Analysis Tools**
   - Road length estimation
   - Connectivity analysis
   - Export to GeoJSON

4. **Customization**
   - Custom color schemes for overlay
   - Adjustable threshold
   - Different mask styles

5. **History**
   - View previous segmentations
   - Comparison mode
   - Export history

6. **Export Options**
   - Export as PDF report
   - GeoTIFF format
   - Vector shapefile

---

## Troubleshooting

### Common Issues

**1. Backend not connecting**
- Verify backend is running on port 8000
- Check firewall settings
- Ensure correct URL in settings

**2. Upload fails**
- Check file format (must be jpg/png/tif)
- Verify file size (<50MB)
- Try converting image format

**3. Results not displaying**
- Check browser console for errors
- Refresh the page
- Clear browser cache

**4. Slow performance**
- Large images take longer to process
- CPU inference is slower than GPU
- Consider resizing very large images

---

This frontend provides an intuitive, professional web interface for interacting with the StripUnetMCSA road segmentation model.
