"""
Road Network Graph Analysis Page
Extracts navigable graph structures from road segmentation masks
"""
import streamlit as st
import json
from PIL import Image
import numpy as np

# Import API client
import sys
sys.path.append('..')
from api_client import get_client

# Page config
st.set_page_config(
    page_title="Road Network Graph - StripUnetMCSA",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .stat-value {
        font-size: 2em;
        font-weight: bold;
    }
    .stat-label {
        font-size: 0.9em;
        opacity: 0.9;
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title(" Road Network Graph Extraction")
st.markdown("### Convert road segmentation masks into navigable graph structures")
st.markdown("---")

# Sidebar settings
st.sidebar.header(" Settings")
backend_url = st.sidebar.text_input("Backend URL", value="http://localhost:8000")

# Check backend connection
client = get_client(backend_url)
health = client.check_health()

if health:
    st.sidebar.success(" Backend Connected")
else:
    st.sidebar.error(" Backend Offline")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About This Feature
This tool extracts a **graph structure** from road segmentation:

- **Nodes**: Intersections & endpoints
- **Edges**: Road segments with lengths
- **GeoJSON**: Export for mapping tools
- **Statistics**: Network analysis metrics

*No model retraining required!*
""")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("###  Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a satellite/aerial image",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Upload an image to extract road network graph"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Extract button
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            extract_button = st.button(" Extract Road Network", type="primary", use_container_width=True)

# Process extraction
if uploaded_file and 'extract_button' in dir() and extract_button:
    if not health:
        st.error(" Cannot connect to backend. Please ensure the backend server is running.")
    else:
        with st.spinner(" Extracting road network graph... This may take a moment."):
            result = client.extract_graph(image)
        
        if result and result.get('success'):
            st.success(" Graph extraction complete!")
            
            # Display results
            st.markdown("---")
            st.markdown("##  Results")
            
            # Statistics cards
            stats = result.get('statistics', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label=" Total Nodes",
                    value=stats.get('num_nodes', 0),
                    help="Intersections + Endpoints"
                )
            
            with col2:
                st.metric(
                    label=" Intersections",
                    value=stats.get('num_intersections', 0),
                    help="Points where 3+ roads meet"
                )
            
            with col3:
                st.metric(
                    label=" Endpoints",
                    value=stats.get('num_endpoints', 0),
                    help="Dead ends and road terminations"
                )
            
            with col4:
                st.metric(
                    label=" Total Edges",
                    value=stats.get('num_edges', 0),
                    help="Road segments connecting nodes"
                )
            
            # Second row of stats
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                st.metric(
                    label=" Total Length (px)",
                    value=f"{stats.get('total_length_pixels', 0):,.0f}",
                    help="Total road length in pixels"
                )
            
            with col6:
                st.metric(
                    label=" Components",
                    value=stats.get('num_connected_components', 0),
                    help="Disconnected road networks"
                )
            
            with col7:
                st.metric(
                    label=" Avg Degree",
                    value=f"{stats.get('average_degree', 0):.2f}",
                    help="Average connections per node"
                )
            
            with col8:
                st.metric(
                    label=" Dead Ends",
                    value=stats.get('dead_ends', 0),
                    help="Number of road terminations"
                )
            
            st.markdown("---")
            
            # Visualization comparison
            st.markdown("###  Visualization")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("**Original Image**")
                st.image(image, use_container_width=True)
            
            with viz_col2:
                st.markdown("**Road Network Graph**")
                st.markdown(" Skeleton |  Intersections |  Endpoints")
                # Decode skeleton overlay
                skeleton_base64 = result.get('skeleton_overlay')
                if skeleton_base64:
                    skeleton_img = client.decode_base64_image(skeleton_base64)
                    st.image(skeleton_img, use_container_width=True)
            
            st.markdown("---")
            
            # GeoJSON download
            st.markdown("###  Export Data")
            
            geojson_data = result.get('geojson', {})
            geojson_str = json.dumps(geojson_data, indent=2)
            
            col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
            
            with col_dl2:
                st.download_button(
                    label=" Download GeoJSON",
                    data=geojson_str,
                    file_name="road_network.geojson",
                    mime="application/geo+json",
                    use_container_width=True
                )
            
            # Show GeoJSON preview
            with st.expander(" Preview GeoJSON (first 50 features)"):
                preview_geojson = geojson_data.copy()
                if 'features' in preview_geojson and len(preview_geojson['features']) > 50:
                    preview_geojson['features'] = preview_geojson['features'][:50]
                    preview_geojson['_note'] = f"Showing 50 of {len(geojson_data['features'])} features"
                st.json(preview_geojson)
            
            # Statistics JSON download
            with st.expander(" Full Statistics"):
                st.json(stats)
                st.download_button(
                    label=" Download Statistics JSON",
                    data=json.dumps(stats, indent=2),
                    file_name="road_network_stats.json",
                    mime="application/json"
                )
        
        else:
            st.error(" Graph extraction failed. Please check the backend logs.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <strong>Road Network Graph Extraction</strong> | Powered by StripUnetMCSA<br>
    <small>Converts binary masks to navigable graph structures using skeletonization + NetworkX</small>
</div>
""", unsafe_allow_html=True)
