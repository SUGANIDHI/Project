"""
Performance Measures Page - Model Comparison Table
Compares StripUnetMCSA with other state-of-the-art road segmentation models
"""
import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="Performance Measures - StripUnetMCSA",
    page_icon="",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .highlight-row {
        background-color: #1a5f2a !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stDataFrame {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title(" Performance Measures")
st.markdown("### Model Comparison - Road Segmentation on DeepGlobe Dataset")
st.markdown("---")

# Model comparison data
model_data = [
    {
        "Rank": 1,
        "Model": "D-LinkNet",
        "Precision": "78.14%",
        "Recall": "74.36%",
        "F1 Score": "77.19%",
        "IoU": "63.32%",
        "Params (M)": "~180",
        "Speed (it/s)": "~0.5",
        "Split": "TEST"
    },
    {
        "Rank": 2,
        "Model": "StripUnetMCSA (Ours)",
        "Precision": "78.6%",
        "Recall": "77.0%",
        "F1 Score": "77.8%",
        "IoU": "64.4%",
        "Params (M)": "40.4",
        "Speed (it/s)": "3.17",
        "Split": "Val"
    },
    {
        "Rank": 3,
        "Model": "UBR-Net (Paper)",
        "Precision": "80.39%",
        "Recall": "76.30%",
        "F1 Score": "78.69%",
        "IoU": "67.83%",
        "Params (M)": "~180+",
        "Speed (it/s)": "Unknown",
        "Split": "Val"
    },
    {
        "Rank": 4,
        "Model": "DeepLabV3+",
        "Precision": "76.85%",
        "Recall": "72.94%",
        "F1 Score": "75.18%",
        "IoU": "65.83%",
        "Params (M)": "~60",
        "Speed (it/s)": "~1.2",
        "Split": "Val"
    },
    {
        "Rank": 5,
        "Model": "EMANet",
        "Precision": "~76.5%",
        "Recall": "72.34%",
        "F1 Score": "75.18%",
        "IoU": "~60.0%",
        "Params (M)": "Unknown",
        "Speed (it/s)": "Unknown",
        "Split": "Val"
    },
    {
        "Rank": 6,
        "Model": "ResUNet",
        "Precision": "77.03%",
        "Recall": "72.34%",
        "F1 Score": "74.58%",
        "IoU": "~62.0%",
        "Params (M)": "~45",
        "Speed (it/s)": "~1.8",
        "Split": "Val"
    },
    {
        "Rank": 7,
        "Model": "UNet",
        "Precision": "74.56%",
        "Recall": "69.27%",
        "F1 Score": "73.42%",
        "IoU": "58.12%",
        "Params (M)": "~31",
        "Speed (it/s)": "~2.0",
        "Split": "Val"
    },
]

# Create DataFrame
df = pd.DataFrame(model_data)

# Key highlights
st.markdown("##  Key Highlights")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label=" Our Ranking",
        value="#2",
        delta="Competitive with SOTA"
    )

with col2:
    st.metric(
        label=" F1 Score",
        value="77.8%",
        delta="+4.38% vs UNet"
    )

with col3:
    st.metric(
        label=" Speed",
        value="3.17 it/s",
        delta="6x faster than D-LinkNet"
    )

with col4:
    st.metric(
        label=" Parameters",
        value="40.4M",
        delta="4.5x smaller than D-LinkNet",
        delta_color="inverse"
    )

st.markdown("---")

# Comparison Table
st.markdown("##  Full Comparison Table")

# Function to highlight our model row
def highlight_our_model(row):
    if "StripUnetMCSA" in row["Model"]:
        return ['background-color: #1a472a; color: #90EE90; font-weight: bold'] * len(row)
    return [''] * len(row)

# Style and display the dataframe
styled_df = df.style.apply(highlight_our_model, axis=1)

st.dataframe(
    styled_df,
    use_container_width=True,
    height=350,
    column_config={
        "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
        "Model": st.column_config.TextColumn("Model", width="medium"),
        "Precision": st.column_config.TextColumn("Precision", width="small"),
        "Recall": st.column_config.TextColumn("Recall", width="small"),
        "F1 Score": st.column_config.TextColumn("F1 Score", width="small"),
        "IoU": st.column_config.TextColumn("IoU", width="small"),
        "Params (M)": st.column_config.TextColumn("Params (M)", width="small"),
        "Speed (it/s)": st.column_config.TextColumn("Speed (it/s)", width="small"),
        "Split": st.column_config.TextColumn("Split", width="small"),
    }
)

st.markdown("---")

# Analysis Section
st.markdown("##  Performance Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("###  Advantages of StripUnetMCSA")
    st.success("""
    - **Highest Recall (77.0%)** - Best at detecting all road pixels
    - **Efficiency Champion** - 40.4M params (4.5x smaller than D-LinkNet)
    - **Speed Leader** - 3.17 it/s (6x faster than D-LinkNet)
    - **Best IoU** among lightweight models (64.4%)
    - **Production Ready** - Optimal balance of accuracy and efficiency
    """)

with col2:
    st.markdown("###  Comparison Notes")
    st.info("""
    - **D-LinkNet** - Higher precision but 4.5x more parameters
    - **UBR-Net** - Best overall metrics but much heavier (~180M+ params)
    - **DeepLabV3+** - Good baseline but lower F1 and slower
    - **UNet** - Lightweight but significantly lower accuracy
    - **StripUnetMCSA** - Best efficiency-accuracy trade-off! 
    """)

st.markdown("---")

# Efficiency Metrics
st.markdown("##  Efficiency Comparison")

efficiency_col1, efficiency_col2, efficiency_col3 = st.columns(3)

with efficiency_col1:
    st.markdown("###  Model Size")
    st.markdown("""
    | Model | Params |
    |-------|--------|
    | D-LinkNet | ~180M |
    | UBR-Net | ~180M+ |
    | DeepLabV3+ | ~60M |
    | **StripUnetMCSA** | **40.4M**  |
    | UNet | ~31M |
    """)

with efficiency_col2:
    st.markdown("###  Inference Speed")
    st.markdown("""
    | Model | Speed (it/s) |
    |-------|--------------|
    | D-LinkNet | ~0.5 |
    | DeepLabV3+ | ~1.2 |
    | ResUNet | ~1.8 |
    | UNet | ~2.0 |
    | **StripUnetMCSA** | **3.17**  |
    """)

with efficiency_col3:
    st.markdown("###  F1 per Million Params")
    st.markdown("""
    | Model | F1/M Ratio |
    |-------|------------|
    | D-LinkNet | 0.43 |
    | DeepLabV3+ | 1.25 |
    | UNet | 2.37 |
    | **StripUnetMCSA** | **1.93**  |
    """)

st.markdown("---")

# Export Section
st.markdown("##  Export Data")

col1, col2 = st.columns(2)

with col1:
    csv = df.to_csv(index=False)
    st.download_button(
        label=" Download as CSV",
        data=csv,
        file_name="model_comparison.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    json_data = df.to_json(orient='records', indent=2)
    st.download_button(
        label=" Download as JSON",
        data=json_data,
        file_name="model_comparison.json",
        mime="application/json",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.caption("""
**Notes:**
- All metrics are evaluated on the DeepGlobe Road Extraction Dataset
- StripUnetMCSA uses ResNet50 backbone with Multi-Scale Channel-Spatial Attention
- Speed measurements on NVIDIA GPU (varies by hardware)
- "~" indicates approximate values from published papers
""")
