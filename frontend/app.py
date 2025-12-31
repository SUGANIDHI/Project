"""
StripUnetMCSA Road Segmentation - Streamlit Frontend
Main application entry point
"""
import streamlit as st
from PIL import Image
import numpy as np
import time

# Import custom modules
from ui_layout import (
    setup_page_config, apply_custom_css, render_header, 
    render_sidebar, render_upload_section, render_action_buttons,
    show_error, show_success, show_info, show_warning
)
from api_client import get_client
from visualization import (
    prepare_image_display, display_statistics, 
    format_statistics, get_download_link_data
)


def main():
    """Main application function"""
    
    # Setup page
    setup_page_config()
    apply_custom_css()
    
    # Render header
    render_header()
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Initialize API client
    client = get_client(settings['backend_url'])
    
    # Check backend health
    health = client.check_health()
    if health is None:
        show_warning(f"⚠️ Cannot connect to backend at {settings['backend_url']}. Please start the backend server.")
        st.markdown("""
        **To start the backend:**
        ```bash
        cd backend
        python main.py
        ```
        """)
        st.stop()
    else:
        if health.get('model_loaded'):
            show_success(f"✅ Connected to backend - Model loaded")
        else:
            show_warning("⚠️ Backend connected but model not loaded")
    
    # Upload section
    uploaded_file = render_upload_section()
    
    # Initialize session state for results
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    
    # Display uploaded image preview
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.original_image = image
        
        st.markdown("### 👁️ Preview")
        preview = prepare_image_display(image, max_width=600)
        st.image(preview, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        
        # Action buttons
        predict_button, download_mask_placeholder, download_overlay_placeholder = render_action_buttons()
        
        # Run prediction
        if predict_button:
            with st.spinner("🔄 Running segmentation... This may take a moment."):
                # Send to backend
                mask, overlay, info = client.predict_and_decode(image)
                
                if mask is None:
                    show_error("Prediction failed. Please check backend logs.")
                else:
                    # Store results
                    st.session_state.results = {
                        'mask': mask,
                        'overlay': overlay,
                        'info': info
                    }
                    
                    show_success("Segmentation completed successfully!")
        
        # Display results if available
        if st.session_state.results is not None:
            st.markdown("---")
            st.markdown("## 📊 Segmentation Results")
            
            results = st.session_state.results
            mask = results['mask']
            overlay = results['overlay']
            info = results['info']
            
            # Display images in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🖼️ Original Image**")
                original_display = prepare_image_display(st.session_state.original_image)
                st.image(original_display, use_container_width=True)
            
            with col2:
                st.markdown("**🎯 Segmentation Mask**")
                mask_display = prepare_image_display(mask)
                st.image(mask_display, use_container_width=True)
            
            with col3:
                st.markdown("**🎨 Overlay**")
                overlay_display = prepare_image_display(overlay)
                st.image(overlay_display, use_container_width=True)
            
            # Statistics
            if settings['show_stats']:
                st.markdown("---")
                st.markdown("### 📈 Analysis")
                
                stats = display_statistics(mask)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Pixels", f"{stats['total_pixels']:,}")
                
                with col2:
                    st.metric("Road Pixels", f"{stats['road_pixels']:,}")
                
                with col3:
                    st.metric("Background", f"{stats['background_pixels']:,}")
                
                with col4:
                    st.metric("Road Coverage", f"{stats['road_percentage']:.2f}%")
            
            # Processing info
            with st.expander("ℹ️ Processing Information"):
                st.json(info)
            
            # Download buttons
            st.markdown("---")
            st.markdown("### 💾 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                mask_bytes = get_download_link_data(mask)
                st.download_button(
                    label="⬇️ Download Mask",
                    data=mask_bytes,
                    file_name="segmentation_mask.png",
                    mime="image/png"
                )
            
            with col2:
                overlay_bytes = get_download_link_data(overlay)
                st.download_button(
                    label="⬇️ Download Overlay",
                    data=overlay_bytes,
                    file_name="segmentation_overlay.png",
                    mime="image/png"
                )
    
    else:
        # No image uploaded yet
        st.info("👆 Please upload an image to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
        <small>StripUnetMCSA Road Segmentation System | Powered by Deep Learning | F1 Score: 0.778</small>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
