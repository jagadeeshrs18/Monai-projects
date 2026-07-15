# Advanced Retinal Vessel Segmentation Streamlit App
# SegResNet Model with Interactive Controls

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from monai.networks.nets import SegResNet
import json
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
import os

# Page configuration
st.set_page_config(
    page_title="Retinal Vessel Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 2rem 0;
        letter-spacing: -1px;
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #64748b;
        margin-top: -1rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
    }
    
    .metric-card h1 {
        margin: 0.5rem 0 0 0;
        font-size: 2.5rem;
        font-weight: 800;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .info-box {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .success-box {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #22c55e;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #f59e0b;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
    }
    
    .badge-info {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Image containers */
    .image-container {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.02);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 3rem 0;
        margin-top: 3rem;
        border-top: 2px solid #e2e8f0;
    }
    
    .footer strong {
        color: #1e293b;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# Model path - Auto-load from this location
MODEL_PATH = r"E:\monai\retinal-model\best_segresnet_model.pth"

# Model loading function
@st.cache_resource
def load_model(model_path):
    """Load the trained SegResNet model with proper PyTorch 2.6+ compatibility"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Add safe globals for numpy types (PyTorch 2.6+ requirement)
        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
        ])
        
        # Load checkpoint with weights_only=True (secure method)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
    except Exception as e:
        # Fallback: try loading without weights_only restriction
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    
    # Build model
    model = SegResNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        init_filters=config['init_filters'],
        blocks_down=config['blocks_down'],
        blocks_up=config['blocks_up'],
        dropout_prob=config['dropout_prob'],
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint, device

# Auto-load model on startup
@st.cache_resource
def autoload_model():
    """Automatically load model if it exists"""
    if os.path.exists(MODEL_PATH):
        try:
            model, checkpoint, device = load_model(MODEL_PATH)
            return model, checkpoint, device, True, None
        except Exception as e:
            return None, None, None, False, str(e)
    return None, None, None, False, "Model file not found"

# Image preprocessing
def preprocess_image(image, target_size=(512, 512)):
    """Preprocess image for model input"""
    # Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # CLAHE enhancement on green channel
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    
    return image_tensor, image

# Prediction function
def predict_vessels(model, image_tensor, device, threshold=0.5):
    """Run inference on image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        prediction = torch.sigmoid(output[0, 0]).cpu().numpy()
    
    binary_prediction = (prediction > threshold).astype(np.uint8)
    return prediction, binary_prediction

# Metrics computation
def compute_metrics(prediction, binary_pred):
    """Compute various metrics from prediction"""
    vessel_percentage = (np.sum(binary_pred) / binary_pred.size) * 100
    avg_confidence = np.mean(prediction[binary_pred == 1]) if np.any(binary_pred) else 0
    
    return {
        'vessel_percentage': vessel_percentage,
        'avg_confidence': avg_confidence,
        'total_pixels': binary_pred.size,
        'vessel_pixels': np.sum(binary_pred)
    }

# Visualization functions
def create_overlay(original, prediction, binary_pred, alpha=0.5):
    """Create overlay visualization"""
    overlay = original.copy()
    
    # Green for vessels
    vessel_mask = np.zeros_like(original)
    vessel_mask[:, :, 1] = binary_pred * 255
    
    overlay = cv2.addWeighted(overlay, 1-alpha, vessel_mask, alpha, 0)
    return overlay

def create_heatmap(prediction):
    """Create heatmap visualization"""
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(prediction, cmap='hot', vmin=0, vmax=1)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

def create_comparison_plot(original, mask, overlay):
    """Create side-by-side comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Detected Vessels', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

# ============================================================
# AUTO-LOAD MODEL ON STARTUP
# ============================================================
if not st.session_state.model_loaded:
    with st.spinner("🔄 Loading model..."):
        model, checkpoint, device, success, error = autoload_model()
        if success:
            st.session_state.model = model
            st.session_state.checkpoint = checkpoint
            st.session_state.device = device
            st.session_state.model_loaded = True
        else:
            st.session_state.model_load_error = error

# ============================================================
# MAIN APP
# ============================================================

# Header with animation
st.markdown('<h1 class="main-header">🔬 Retinal Vessel Segmentation</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Medical Image Analysis | SegResNet Deep Learning</p>', unsafe_allow_html=True)

# Model status banner
if st.session_state.model_loaded:
    st.markdown("""
    <div class="success-box">
        <span class="status-badge badge-success">✓ Model Loaded</span>
        <strong>SegResNet is ready for analysis</strong>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="warning-box">
        <span class="status-badge badge-info">⚠ Model Not Found</span>
        Please check model path: <code>E:\monai\retinal-model\best_segresnet_model.pth</code>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    st.markdown("---")
    
    # Model status in sidebar
    if st.session_state.model_loaded:
        st.success("✅ Model Active")
        with st.expander("📊 Model Details"):
            st.write(f"**Architecture:** SegResNet")
            st.write(f"**Dice Score:** {st.session_state.checkpoint['best_dice']:.4f}")
            st.write(f"**Epochs Trained:** {st.session_state.checkpoint['epoch']}")
            st.write(f"**Parameters:** {sum(p.numel() for p in st.session_state.model.parameters()):,}")
            st.write(f"**Device:** {st.session_state.device}")
    else:
        st.error("❌ Model Not Loaded")
        if st.button("🔄 Retry Loading Model"):
            st.cache_resource.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Segmentation controls
    st.markdown("### 🎛️ Segmentation Settings")
    
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Higher = More strict vessel detection"
    )
    
    overlay_alpha = st.slider(
        "Overlay Opacity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Transparency of vessel overlay"
    )
    
    st.markdown("---")
    
    # Visualization options
    st.markdown("### 🎨 Visualization")
    
    colormap = st.selectbox(
        "Heatmap Style",
        ['hot', 'jet', 'viridis', 'plasma', 'inferno', 'magma'],
        index=0
    )
    
    show_grid = st.checkbox("Grid Analysis", value=False)
    
    if show_grid:
        grid_size = st.slider("Grid Size", 2, 8, 4)
    
    st.markdown("---")
    
    # Info section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        **Retinal Vessel Segmentation**
        
        Advanced deep learning for automated blood vessel detection in fundus images.
        
        **Features:**
        - ✨ Real-time segmentation
        - 📊 Detailed metrics
        - 🎨 Multiple visualizations
        - 💾 Export results
        
        **Technology:**
        - Model: SegResNet
        - Dataset: DRIVE
        - Framework: MONAI + PyTorch
        """)

# Main content
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-header">📤 Upload Image</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image",
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
        help="Supported formats: JPG, PNG, TIF"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        st.session_state.uploaded_image = image_np
        
        # Display with container
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption='📷 Uploaded Image', use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Image info
        st.markdown(f"""
        <div class="info-box">
            📏 <strong>Dimensions:</strong> {image_np.shape[1]} × {image_np.shape[0]} pixels<br>
            🎨 <strong>Channels:</strong> {image_np.shape[2]}<br>
            💾 <strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">🎯 Analysis Results</div>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_image is not None and st.session_state.model is not None:
        
        if st.button("🚀 RUN SEGMENTATION", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("⚙️ Preprocessing image...")
            progress_bar.progress(25)
            
            # Preprocess
            image_tensor, processed_img = preprocess_image(
                st.session_state.uploaded_image,
                target_size=(512, 512)
            )
            
            status_text.text("🧠 Running AI model...")
            progress_bar.progress(50)
            
            # Predict
            prediction, binary_pred = predict_vessels(
                st.session_state.model,
                image_tensor,
                st.session_state.device,
                threshold=threshold
            )
            
            status_text.text("📊 Computing metrics...")
            progress_bar.progress(75)
            
            st.session_state.prediction = prediction
            st.session_state.binary_pred = binary_pred
            st.session_state.processed_img = (processed_img * 255).astype(np.uint8)
            
            # Compute metrics
            metrics = compute_metrics(prediction, binary_pred)
            st.session_state.metrics = metrics
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Clear progress indicators
            import time
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
        
        # Display results if available
        if st.session_state.prediction is not None:
            # Create visualizations
            overlay = create_overlay(
                st.session_state.processed_img,
                st.session_state.prediction,
                st.session_state.binary_pred,
                alpha=overlay_alpha
            )
            
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(overlay, caption='🎨 Vessel Segmentation Result', use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Metrics display
            metrics = st.session_state.metrics
            
            st.markdown("### 📈 Key Metrics")
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Vessel Coverage</h3>
                    <h1>{metrics['vessel_percentage']:.2f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Confidence</h3>
                    <h1>{metrics['avg_confidence']:.3f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Vessel Pixels</h3>
                    <h1>{metrics['vessel_pixels']:,}</h1>
                </div>
                """, unsafe_allow_html=True)
    
    elif st.session_state.model is None:
        st.warning("⚠️ Model not loaded. Please check the model path.")
    elif st.session_state.uploaded_image is None:
        st.info("👆 Upload a retinal image to start analysis")

# Advanced Visualizations
if st.session_state.prediction is not None:
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Advanced Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Heatmap", "📈 Statistics", "🔬 Detailed View", "💾 Export"])
    
    with tab1:
        st.markdown("**Confidence Heatmap** - Pixel-wise prediction confidence")
        
        # Interactive heatmap with plotly
        fig = px.imshow(
            st.session_state.prediction,
            color_continuous_scale=colormap,
            aspect='equal',
            title='Vessel Detection Confidence Map'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Matplotlib version
        fig_mpl = create_heatmap(st.session_state.prediction)
        st.pyplot(fig_mpl)
    
    with tab2:
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("**Confidence Distribution**")
            
            vessel_conf = st.session_state.prediction[st.session_state.binary_pred == 1].flatten()
            bg_conf = st.session_state.prediction[st.session_state.binary_pred == 0].flatten()
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=vessel_conf, name='Vessel', opacity=0.7, 
                                           marker_color='#ef4444'))
            fig_hist.add_trace(go.Histogram(x=bg_conf, name='Background', opacity=0.7, 
                                           marker_color='#3b82f6'))
            fig_hist.update_layout(
                barmode='overlay',
                title='Prediction Confidence Distribution',
                xaxis_title='Confidence',
                yaxis_title='Frequency',
                height=400
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_s2:
            st.markdown("**Statistics Summary**")
            
            stats_data = {
                'Metric': [
                    'Total Pixels',
                    'Vessel Pixels',
                    'Background Pixels',
                    'Vessel Coverage',
                    'Mean Confidence (Vessel)',
                    'Mean Confidence (Background)',
                    'Max Confidence',
                    'Min Confidence'
                ],
                'Value': [
                    f"{st.session_state.metrics['total_pixels']:,}",
                    f"{st.session_state.metrics['vessel_pixels']:,}",
                    f"{st.session_state.metrics['total_pixels'] - st.session_state.metrics['vessel_pixels']:,}",
                    f"{st.session_state.metrics['vessel_percentage']:.2f}%",
                    f"{np.mean(vessel_conf):.4f}",
                    f"{np.mean(bg_conf):.4f}",
                    f"{np.max(st.session_state.prediction):.4f}",
                    f"{np.min(st.session_state.prediction):.4f}"
                ]
            }
            
            st.dataframe(stats_data, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("**Detailed Comparison View**")
        
        # Side-by-side comparison
        fig_comp = create_comparison_plot(
            st.session_state.processed_img,
            st.session_state.binary_pred * 255,
            create_overlay(
                st.session_state.processed_img,
                st.session_state.prediction,
                st.session_state.binary_pred,
                alpha=overlay_alpha
            )
        )
        st.pyplot(fig_comp)
        
        # Grid analysis
        if show_grid:
            st.markdown(f"**Grid Analysis** ({grid_size}×{grid_size})")
            
            h, w = st.session_state.binary_pred.shape
            grid_h, grid_w = h // grid_size, w // grid_size
            
            grid_stats = []
            for i in range(grid_size):
                for j in range(grid_size):
                    cell = st.session_state.binary_pred[
                        i*grid_h:(i+1)*grid_h,
                        j*grid_w:(j+1)*grid_w
                    ]
                    vessel_pct = (np.sum(cell) / cell.size) * 100
                    grid_stats.append(vessel_pct)
            
            grid_array = np.array(grid_stats).reshape(grid_size, grid_size)
            
            fig_grid = px.imshow(
                grid_array,
                labels=dict(x="Column", y="Row", color="Vessel %"),
                x=[f"C{i+1}" for i in range(grid_size)],
                y=[f"R{i+1}" for i in range(grid_size)],
                color_continuous_scale='RdYlGn',
                title=f'Vessel Density Grid Analysis ({grid_size}×{grid_size})'
            )
            fig_grid.update_traces(text=np.round(grid_array, 1), texttemplate="%{text}%")
            st.plotly_chart(fig_grid, use_container_width=True)
    
    with tab4:
        st.markdown("**Export Analysis Results**")
        
        col_e1, col_e2, col_e3 = st.columns(3)
        
        with col_e1:
            mask_img = Image.fromarray((st.session_state.binary_pred * 255).astype(np.uint8))
            buf = BytesIO()
            mask_img.save(buf, format='PNG')
            st.download_button(
                label="💾 Binary Mask",
                data=buf.getvalue(),
                file_name="vessel_mask.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col_e2:
            overlay_img = Image.fromarray(create_overlay(
                st.session_state.processed_img,
                st.session_state.prediction,
                st.session_state.binary_pred,
                alpha=overlay_alpha
            ))
            buf = BytesIO()
            overlay_img.save(buf, format='PNG')
            st.download_button(
                label="💾 Overlay Image",
                data=buf.getvalue(),
                file_name="vessel_overlay.png",
                mime="image/png",
                use_container_width=True
            )
        
        with col_e3:
            metrics_json = json.dumps(st.session_state.metrics, indent=4)
            st.download_button(
                label="💾 Metrics JSON",
                data=metrics_json,
                file_name="vessel_metrics.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Full report
        st.markdown("---")
        if st.button("📄 Generate Full Report", use_container_width=True):
            report = f"""
# Retinal Vessel Segmentation Report

## Image Information
- **Filename:** {uploaded_file.name if uploaded_file else 'N/A'}
- **Dimensions:** {st.session_state.processed_img.shape[1]} × {st.session_state.processed_img.shape[0]} pixels

## Segmentation Parameters
- **Threshold:** {threshold}
- **Overlay Alpha:** {overlay_alpha}

## Results
- **Vessel Coverage:** {st.session_state.metrics['vessel_percentage']:.2f}%
- **Total Vessel Pixels:** {st.session_state.metrics['vessel_pixels']:,}
- **Average Confidence:** {st.session_state.metrics['avg_confidence']:.4f}

## Model Information
- **Architecture:** SegResNet
- **Best Dice Score:** {st.session_state.checkpoint['best_dice']:.4f}
- **Training Epochs:** {st.session_state.checkpoint['epoch']}

## Statistical Analysis
- **Max Confidence:** {np.max(st.session_state.prediction):.4f}
- **Min Confidence:** {np.min(st.session_state.prediction):.4f}
- **Mean Confidence (Vessels):** {np.mean(vessel_conf):.4f}
- **Mean Confidence (Background):** {np.mean(bg_conf):.4f}

---
*Generated by Retinal Vessel Segmentation App*
"""
            st.download_button(
                label="📥 Download Report (Markdown)",
                data=report,
                file_name="segmentation_report.md",
                mime="text/markdown",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>Retinal Vessel Segmentation System</strong></p>
    <p style="font-size: 1rem; margin: 1rem 0;">
        <span style="color: #667eea;">●</span> Powered by SegResNet 
        <span style="color: #667eea;">●</span> Built with Streamlit 
        <span style="color: #667eea;">●</span> MONAI Framework
    </p>
    <p style="font-size: 0.9rem; color: #94a3b8;">⚕️ For research and educational purposes only</p>
    <p style="font-size: 0.85rem; color: #cbd5e1; margin-top: 1rem;">
        © 2024 Medical AI Lab | Deep Learning for Healthcare
    </p>
</div>
""", unsafe_allow_html=True)