"""
UI Layout components for Streamlit frontend
"""
import streamlit as st


def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="StripUnetMCSA Road Segmentation",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
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
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 1rem;
            margin: 1rem 0;
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
        </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render app header"""
    st.markdown("# 🛣️ StripUnetMCSA Road Segmentation")
    st.markdown("""
        <div class="info-box">
        <b>Advanced Road Segmentation using StripUnetMCSA</b><br>
        Upload a satellite or aerial image to detect and segment roads using our state-of-the-art deep learning model.
        </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
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
        
        # About
        st.subheader("About")
        st.markdown("""
        **StripUnetMCSA** is a deep learning model for road segmentation featuring:
        - Multi-Context Spatial Attention (MCSA)
        - Dual-Branch Decoder
        - Efficient tiling strategy
        
        **Model Performance:** F1 = 0.778
        """)
        
        return {
            'backend_url': backend_url,
            'show_overlay': show_overlay,
            'overlay_alpha': overlay_alpha,
            'show_stats': show_stats
        }


def render_upload_section():
    """Render image upload section"""
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Upload a satellite or aerial image for road segmentation"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return uploaded_file


def render_results_section():
    """Create placeholder for results"""
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.subheader("📊 Results")
    
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
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        predict_button = st.button("🚀 Run Segmentation", type="primary")
    
    with col2:
        download_mask = st.empty()
    
    with col3:
        download_overlay = st.empty()
    
    return predict_button, download_mask, download_overlay


def show_error(message):
    """Display error message"""
    st.error(f"❌ {message}")


def show_success(message):
    """Display success message"""
    st.success(f"✅ {message}")


def show_info(message):
    """Display info message"""
    st.info(f"ℹ️ {message}")


def show_warning(message):
    """Display warning message"""
    st.warning(f"⚠️ {message}")
