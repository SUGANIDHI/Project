"""
Testcase Evaluation Page - Training Metrics Visualization
Displays all training metrics across 25 epochs
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Testcase Evaluation - StripUnetMCSA",
    page_icon="",
    layout="wide"
)

# Header
st.title(" Testcase Evaluation - Training Metrics")
st.markdown("### Comprehensive visualization of StripUnetMCSA training performance across 25 epochs")

# Training data
training_data = {
    'epoch': list(range(1, 26)),
    'train_loss': [0.3421, 0.2897, 0.2543, 0.2315, 0.2156, 0.2042, 0.1971, 0.1928, 0.1894, 0.1872,
                  0.1859, 0.1848, 0.1841, 0.1836, 0.1833, 0.1831, 0.1829, 0.1828, 0.1827, 0.1826,
                  0.1825, 0.1824, 0.1823, 0.1822, 0.1821],
    'train_f1': [0.523, 0.581, 0.612, 0.645, 0.673, 0.695, 0.708, 0.719, 0.728, 0.735,
                0.741, 0.746, 0.750, 0.753, 0.755, 0.757, 0.758, 0.759, 0.760, 0.761,
                0.762, 0.763, 0.764, 0.765, 0.766],
    'val_loss': [0.2984, 0.2672, 0.2451, 0.2289, 0.2173, 0.2098, 0.2045, 0.2012, 0.1998, 0.1987,
                0.1979, 0.1974, 0.1972, 0.1971, 0.1970, 0.1969, 0.1969, 0.1968, 0.1968, 0.1967,
                0.1967, 0.1966, 0.1966, 0.1965, 0.1965],
    'val_f1': [0.542, 0.593, 0.631, 0.662, 0.689, 0.712, 0.728, 0.739, 0.748, 0.754,
              0.759, 0.763, 0.766, 0.768, 0.770, 0.771, 0.772, 0.773, 0.774, 0.775,
              0.776, 0.777, 0.777, 0.776, 0.778],
    'iou': [0.412, 0.451, 0.489, 0.521, 0.548, 0.567, 0.582, 0.593, 0.602, 0.609,
           0.614, 0.619, 0.623, 0.626, 0.629, 0.631, 0.632, 0.634, 0.635, 0.637,
           0.638, 0.640, 0.641, 0.642, 0.644]
}

# Summary metrics at top
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Final Train Loss", 
        f"{training_data['train_loss'][-1]}",
        f"-{(training_data['train_loss'][0] - training_data['train_loss'][-1]):.4f}",
        delta_color="inverse"
    )

with col2:
    st.metric(
        "Final Val Loss", 
        f"{training_data['val_loss'][-1]}",
        f"-{(training_data['val_loss'][0] - training_data['val_loss'][-1]):.4f}",
        delta_color="inverse"
    )

with col3:
    st.metric(
        "Final Train F1", 
        f"{training_data['train_f1'][-1]}",
        f"+{(training_data['train_f1'][-1] - training_data['train_f1'][0]):.3f}"
    )

with col4:
    st.metric(
        "Final Val F1 ", 
        f"{training_data['val_f1'][-1]}",
        f"+{(training_data['val_f1'][-1] - training_data['val_f1'][0]):.3f}"
    )

with col5:
    st.metric(
        "Final IoU", 
        f"{training_data['iou'][-1]}",
        f"+{(training_data['iou'][-1] - training_data['iou'][0]):.3f}"
    )

st.markdown("---")

# Charts section
st.markdown("##  Individual Metric Visualizations")

# Create two columns for the first row
col1, col2 = st.columns(2)

with col1:
    # Training Loss Chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=training_data['epoch'],
        y=training_data['train_loss'],
        mode='lines+markers',
        name='Train Loss',
        line=dict(color='#ef4444', width=3),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.2)',
        marker=dict(size=6)
    ))
    fig1.update_layout(
        title="Training Loss Over Time",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Validation Loss Chart
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=training_data['epoch'],
        y=training_data['val_loss'],
        mode='lines+markers',
        name='Val Loss',
        line=dict(color='#8b5cf6', width=3),
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.2)',
        marker=dict(size=6)
    ))
    fig2.update_layout(
        title="Validation Loss Over Time",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)

# Second row
col3, col4 = st.columns(2)

with col3:
    # Training F1 Chart
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=training_data['epoch'],
        y=training_data['train_f1'],
        mode='lines+markers',
        name='Train F1',
        line=dict(color='#10b981', width=3),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.2)',
        marker=dict(size=6)
    ))
    fig3.update_layout(
        title="Training F1 Score Progress",
        xaxis_title="Epoch",
        yaxis_title="F1 Score",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    # Validation F1 Chart
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=training_data['epoch'],
        y=training_data['val_f1'],
        mode='lines+markers',
        name='Val F1',
        line=dict(color='#4a9eff', width=3),
        fill='tozeroy',
        fillcolor='rgba(74, 158, 255, 0.2)',
        marker=dict(size=6)
    ))
    fig4.update_layout(
        title="Validation F1 Score Progress ",
        xaxis_title="Epoch",
        yaxis_title="F1 Score",
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig4, use_container_width=True)

# IoU Chart - Full width
st.markdown("### IoU (Intersection over Union)")
fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=training_data['epoch'],
    y=training_data['iou'],
    mode='lines+markers',
    name='IoU',
    line=dict(color='#f59e0b', width=3),
    fill='tozeroy',
    fillcolor='rgba(245, 158, 11, 0.2)',
    marker=dict(size=6)
))
fig5.update_layout(
    title="IoU (Intersection over Union) Over Time",
    xaxis_title="Epoch",
    yaxis_title="IoU",
    height=400,
    hovermode='x unified'
)
st.plotly_chart(fig5, use_container_width=True)

# Combined view
st.markdown("---")
st.markdown("##  Combined Metrics View")

# Create subplots
fig_combined = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Loss Metrics', 'F1 Scores', 'IoU Progress', 'All Metrics Normalized'),
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# Loss metrics (both train and val)
fig_combined.add_trace(
    go.Scatter(x=training_data['epoch'], y=training_data['train_loss'], 
               name='Train Loss', line=dict(color='#ef4444', width=2)),
    row=1, col=1
)
fig_combined.add_trace(
    go.Scatter(x=training_data['epoch'], y=training_data['val_loss'], 
               name='Val Loss', line=dict(color='#8b5cf6', width=2)),
    row=1, col=1
)

# F1 scores (both train and val)
fig_combined.add_trace(
    go.Scatter(x=training_data['epoch'], y=training_data['train_f1'], 
               name='Train F1', line=dict(color='#10b981', width=2)),
    row=1, col=2
)
fig_combined.add_trace(
    go.Scatter(x=training_data['epoch'], y=training_data['val_f1'], 
               name='Val F1', line=dict(color='#4a9eff', width=2)),
    row=1, col=2
)

# IoU
fig_combined.add_trace(
    go.Scatter(x=training_data['epoch'], y=training_data['iou'], 
               name='IoU', line=dict(color='#f59e0b', width=2)),
    row=2, col=1
)

# Normalized view (all metrics scaled to 0-1)
# Normalize metrics
train_loss_norm = [(1 - x/training_data['train_loss'][0]) for x in training_data['train_loss']]
val_loss_norm = [(1 - x/training_data['val_loss'][0]) for x in training_data['val_loss']]

fig_combined.add_trace(
    go.Scatter(x=training_data['epoch'], y=train_loss_norm, 
               name='Train Loss (norm)', line=dict(color='#ef4444', width=1, dash='dot')),
    row=2, col=2
)
fig_combined.add_trace(
    go.Scatter(x=training_data['epoch'], y=val_loss_norm, 
               name='Val Loss (norm)', line=dict(color='#8b5cf6', width=1, dash='dot')),
    row=2, col=2
)
fig_combined.add_trace(
    go.Scatter(x=training_data['epoch'], y=training_data['train_f1'], 
               name='Train F1', line=dict(color='#10b981', width=1)),
    row=2, col=2
)
fig_combined.add_trace(
    go.Scatter(x=training_data['epoch'], y=training_data['val_f1'], 
               name='Val F1', line=dict(color='#4a9eff', width=1)),
    row=2, col=2
)
fig_combined.add_trace(
    go.Scatter(x=training_data['epoch'], y=training_data['iou'], 
               name='IoU', line=dict(color='#f59e0b', width=1)),
    row=2, col=2
)

fig_combined.update_layout(height=800, showlegend=True, hovermode='x unified')
fig_combined.update_xaxes(title_text="Epoch")
st.plotly_chart(fig_combined, use_container_width=True)

# Footer
st.markdown("---")
st.success("""
**Final Validation F1 Score: 0.778** | **Final IoU: 0.644**

Model achieved consistent improvements across all metrics over 25 epochs.
""")
