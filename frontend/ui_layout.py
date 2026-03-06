"""
UI Layout components for Streamlit frontend
"""
import streamlit as st


def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="StripUnetMCSA Road Extraction",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def apply_custom_css():
    """Apply custom CSS styling - Force light/white theme"""
    st.markdown("""
        <style>
        /* Force white background throughout */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], 
        [data-testid="stToolbar"], [data-testid="stDecoration"], 
        [data-testid="stStatusWidget"], .main, .block-container {
            background-color: #ffffff !important;
            color: #1f1f1f !important;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
            background-color: #f8f9fa !important;
            color: #1f1f1f !important;
        }
        
        /* Text and headings */
        h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
            color: #1f1f1f !important;
        }
        
        /* Input fields and text areas */
        .stTextInput input, .stTextArea textarea, .stSelectbox select {
            background-color: #ffffff !important;
            color: #1f1f1f !important;
            border-color: #d1d5db !important;
        }
        
        /* Metrics and expanders */
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        .streamlit-expanderHeader {
            color: #1f1f1f !important;
        }
        
        [data-testid="stExpander"] {
            background-color: #f8f9fa !important;
            border-color: #e0e0e0 !important;
        }
        
        .main {
            padding: 2rem;
            background-color: #ffffff !important;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            border: none;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .upload-section {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 1rem;
            margin: 1rem 0;
            border: 1px solid #e0e0e0;
        }
        .result-section {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        h1 {
            color: #1f77b4;
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        h2 {
            color: #2c3e50;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 0.5rem;
        }
        .info-box {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #2196f3;
            margin: 1rem 0;
        }
        
        /* File uploader styling - Force white dropzone */
        [data-testid="stFileUploader"], 
        [data-testid="stFileUploadDropzone"],
        [data-testid="stFileUploadDropzone"] > div,
        section[role="presentation"],
        [data-testid="stFileUploader"] section,
        .st-emotion-cache-1erivf3,
        .st-emotion-cache-1gulkj5 {
            background-color: #ffffff !important;
            border-color: #4CAF50 !important;
        }
        
        [data-testid="stFileUploadDropzone"] section,
        section[role="presentation"] {
            background-color: #ffffff !important;
            border: 2px dashed #4CAF50 !important;
            border-radius: 0.5rem !important;
        }
        
        /* Drag and drop text and icons */
        [data-testid="stFileUploadDropzone"] span,
        [data-testid="stFileUploadDropzone"] small,
        [data-testid="stFileUploadDropzone"] p,
        section[role="presentation"] span,
        section[role="presentation"] small {
            color: #1f1f1f !important;
        }
        
        /* Upload icon/svg */
        [data-testid="stFileUploadDropzone"] svg,
        section[role="presentation"] svg {
            fill: #1f1f1f !important;
            color: #1f1f1f !important;
            stroke: #1f1f1f !important;
        }
        
        /* Browse files button - green style */
        [data-testid="stFileUploadDropzone"] button,
        section[role="presentation"] button,
        .st-emotion-cache-13lcgu3 {
            background-color: #4CAF50 !important;
            color: white !important;
            border: none !important;
            border-radius: 0.25rem !important;
        }
        
        [data-testid="stFileUploadDropzone"] button:hover,
        section[role="presentation"] button:hover {
            background-color: #45a049 !important;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render app header"""
    st.markdown("#  StripUnetMCSA Road Extraction")
    st.markdown("""
        <div class="info-box">
        <b>Advanced Road Extraction using StripUnetMCSA</b><br>
        Upload a satellite or aerial image to detect and segment roads using our state-of-the-art deep learning model.
        </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.header(" Settings")
        
        # Backend settings
        st.subheader("Backend Configuration")
        backend_url = st.text_input(
            "Backend URL",
            value="http://localhost:8000",
            help="URL of the FastAPI backend server"
        )
        
        # Display settings
        st.subheader("Display Settings")
        show_overlay = st.checkbox("Show Overlay", value=True)
        overlay_alpha = st.slider(
            "Overlay Transparency",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.1,
            help="Adjust overlay transparency"
        )
        
        # Statistics
        st.subheader("Analysis Options")
        show_stats = st.checkbox("Show Statistics", value=True)
        
        return {
            'backend_url': backend_url,
            'show_overlay': show_overlay,
            'overlay_alpha': overlay_alpha,
            'show_stats': show_stats
        }


def render_upload_section():
    """Render image upload section"""
    st.subheader(" Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Upload a satellite or aerial image for road segmentation"
    )
    
    return uploaded_file


def render_results_section():
    """Create placeholder for results"""
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.subheader(" Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Original Image**")
        original_placeholder = st.empty()
    
    with col2:
        st.markdown("**Segmentation Mask**")
        mask_placeholder = st.empty()
    
    with col3:
        st.markdown("**Overlay**")
        overlay_placeholder = st.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return original_placeholder, mask_placeholder, overlay_placeholder


def render_action_buttons():
    """Render action buttons"""
    # Center the Run Segmentation button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button(" Run Segmentation", type="primary", use_container_width=True)
    
    download_mask = st.empty()
    download_overlay = st.empty()
    
    return predict_button, download_mask, download_overlay


def show_error(message):
    """Display error message"""
    st.error(f" {message}")


def show_success(message):
    """Display success message"""
    st.success(f"{message}")


def show_info(message):
    """Display info message"""
    st.info(f" {message}")


def show_warning(message):
    """Display warning message"""
    st.warning(f" {message}")
