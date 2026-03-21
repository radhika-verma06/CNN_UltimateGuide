import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import io
import os

from model import FashionCNN, preprocess_image, load_trained_model, load_pro_model, get_imagenet_labels, preprocess_pro_image
from visualization import plot_prediction_probs, render_feature_maps, get_premium_css, render_pipeline_horizontal
from interpretability import GradCAM, apply_heatmap, get_saliency_map
from components import get_hero_component, get_3d_cube_component, get_dense_network_component, get_parameter_counter_component, get_quiz_component

# Set page config
st.set_page_config(
    page_title="CNN Interpretability Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown(get_premium_css(), unsafe_allow_html=True)

# Load models
@st.cache_resource
def get_models():
    lab_path = "/Users/radhika/Desktop/cnn/fashion_mnist_cnn.pth"
    lab_model = load_trained_model(lab_path)
    pro_model = load_pro_model()
    imagenet_labels = get_imagenet_labels()
    return lab_model, pro_model, imagenet_labels

lab_model, pro_model, imagenet_labels = get_models()

# --- 1. CINEMATIC HERO ---
get_hero_component()

# --- SIDEBAR & MODE SELECTION ---
with st.sidebar:
    st.markdown("<h3 class='highlight-text' style='margin-bottom:20px;'>CONTROL PANEL</h3>", unsafe_allow_html=True)
    mode = st.selectbox("CHOOSE EXPERIENCE", ["🎓 Lab Mode (Fashion-MNIST)", "🚀 Pro Mode (ImageNet)"])
    st.divider()
    
    input_image = None
    if "Lab" in mode:
        current_model = lab_model
        classes = lab_model.classes
        st.markdown("<div style='font-size:0.9rem; color:#7070a8;'>Lab Mode is optimized for learning the fundamental math of a 2-layer CNN.</div>", unsafe_allow_html=True)
    else:
        current_model = pro_model
        classes = imagenet_labels
        st.markdown("<div style='font-size:0.9rem; color:#7070a8;'>Pro Mode explores 1,000+ real-world concepts using a state-of-the-art MobileNet model.</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("UPLOAD AN IMAGE", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Current Input", use_container_width=True)

if input_image is None:
    st.markdown("""
    <div style='text-align:center; padding: 60px; background: rgba(255,255,255,0.02); border-radius: 20px; border: 1px dashed rgba(255,255,255,0.1);'>
        <div style='font-size: 3rem;'>🖼</div>
        <h3 style='color: #7070a8;'>Waiting for Image Upload...</h3>
        <p style='color: #444466;'>Please select an image in the sidebar to begin the cinematic analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- 2. ARCHITECTURAL PIPELINE TRACK ---
st.markdown("<h3 style='margin-bottom:20px; font-family:Syne, sans-serif;'>Model Architecture Flow</h3>", unsafe_allow_html=True)
if "Lab" in mode:
    steps = [
        ("Input", "28x28", "🖼"),
        ("Conv1", "16 @ 26x26", "🔢"),
        ("Pool1", "16 @ 13x13", "📐"),
        ("Conv2", "32 @ 11x11", "🔢"),
        ("Pool2", "32 @ 5x5", "📐"),
        ("Flatten", "1600 Vector", "→"),
        ("Dense", "128 Neurons", "🧠"),
        ("Softmax", "10 Classes", "🎯")
    ]
else:
    steps = [
        ("Input", "224x224", "🖼"),
        ("Backbone", "MobileNet-V3", "🚀"),
        ("GlobalPool", "1x1", "📐"),
        ("Classifier", "1000 Neurons", "🎯")
    ]
render_pipeline_horizontal(steps)

# --- 3. INFERENCE DASHBOARD ---
st.markdown("<h2 style='margin-top:60px; margin-bottom:20px;'>1. Neural Inference Engine</h2>", unsafe_allow_html=True)
col_inf1, col_inf2 = st.columns([1, 1.5])
with col_inf1:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    if "Lab" in mode:
        img_tensor = preprocess_image(input_image)
    else:
        img_tensor = preprocess_pro_image(input_image)
        
    with torch.no_grad():
        outputs = current_model(img_tensor)
        probs = F.softmax(outputs, dim=1).squeeze().numpy()
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
    
    st.markdown(f"### <span class='highlight-text'>{classes[pred_idx]}</span>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:4rem; font-weight:800; font-family:Syne, sans-serif;'>{confidence:.1%}</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7070a8; letter-spacing:2px;'>CONFIDENCE SCORE</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_inf2:
    if "Pro" in mode:
        top10_idx = np.argsort(probs)[-10:][::-1]
        display_probs = probs[top10_idx]
        display_classes = [classes[i] for i in top10_idx]
    else:
        display_probs = probs
        display_classes = classes
    fig_probs = plot_prediction_probs(display_probs, display_classes)
    st.plotly_chart(fig_probs, use_container_width=True)

# --- 4. VISUAL EXPLANATION (GRAD-CAM) ---
st.markdown("<h2 style='margin-top:60px; margin-bottom:20px;'>2. Visual Attribution Heatmaps</h2>", unsafe_allow_html=True)
col_cam1, col_cam2 = st.columns(2)

target_layer = current_model.conv2 if "Lab" in mode else current_model.features[-1]
gcam = GradCAM(current_model, target_layer)
heatmap, _ = gcam.generate_heatmap(img_tensor)
img_rgb = input_image.convert('RGB')
input_np = np.array(img_rgb) / 255.0
overlay = apply_heatmap(input_np, heatmap)

with col_cam1:
    st.image(overlay, caption="Hotspots (Where the model looked)", use_container_width=True)
with col_cam2:
    st.image(heatmap, caption="Class Activation Sensitivity", use_container_width=True)

# --- 5. CINEMATIC LAYER DEEP-DIVE (Lab Mode) ---
if "Lab" in mode:
    st.divider()
    st.markdown("<h2 class='highlight-text' style='font-size:3rem;'>3. Layer-by-Layer Interactivity</h2>", unsafe_allow_html=True)
    
    # --- CONVOLUTION SECTION ---
    st.markdown("### 🔍 Spatial Feature Detection (Convolution)")
    col_math1, col_math2, col_math3 = st.columns([1.2, 1, 0.8])
    with col_math1:
        f_idx = st.slider("SELECT FILTER", 0, 15, 0)
        px = st.slider("X COORD", 0, 25, 12)
        py = st.slider("Y COORD", 0, 25, 12)
        patch = img_tensor[0, 0, py:py+3, px:px+3].detach().numpy()
        weights = current_model.conv1.weight[f_idx, 0].detach().numpy()
        bias = current_model.conv1.bias[f_idx].detach().numpy()
        
        st.markdown("<p style='font-family:\"Syne Mono\"; font-size:10px; color:#7070a8;'>INPUT PATCH (3x3)</p>", unsafe_allow_html=True)
        st.markdown("<div style='background:#0d0d1a; padding:15px; border-radius:10px; border:1px solid #1a1a3a;'>", unsafe_allow_html=True)
        for r in patch:
            cs = st.columns(3)
            for i,v in enumerate(r): cs[i].markdown(f"<p style='text-align:center;font-family:monospace; margin:0;'>{v:.2f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_math2:
        st.markdown("<p style='text-align:center; font-size:2rem; margin-top:60px;'>×</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-family:\"Syne Mono\"; font-size:10px; color:#7070a8; text-align:center;'>FILTER WEIGHTS (3x3)</p>", unsafe_allow_html=True)
        st.markdown("<div style='background:#0d0d1a; padding:15px; border-radius:10px; border:1px solid #ff6b35;'>", unsafe_allow_html=True)
        for r in weights:
            cs = st.columns(3)
            for i,v in enumerate(r):
                c = "#00d4ff" if v>0 else "#ec4899"
                cs[i].markdown(f"<p style='text-align:center;font-family:monospace;color:{c}; margin:0;'>{v:.1f}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_math3:
        st.markdown("<p style='text-align:center; font-size:2rem; margin-top:60px;'>=</p>", unsafe_allow_html=True)
        dot = np.sum(patch * weights) + bias
        relu = max(0, dot)
        st.markdown("<div class='metric-card' style='text-align:center;'>")
        st.markdown(f"<div style='font-size:1.5rem; font-family:\"Syne Mono\"; color:#ff6b35;'>{relu:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:0.7rem; color:#7070a8;'>{'ACTIVE' if relu>0 else 'INACTIVE'}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- FLATTEN SECTION ---
    st.markdown("---")
    st.markdown("### 🧊 Data Reshaping (Flattening)")
    col_flat1, col_flat2 = st.columns([1.5, 1])
    with col_flat1:
        st.markdown("""
        <div class='metric-card'>
            <p style='color:#7070a8; line-height:1.7;'>The 3D volume of feature maps (5x5x64) is expensive for full connectivity. 
            We <strong>unroll</strong> it into a 1D vector of 1,600 numbers to feed the decision-making layers.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_flat2:
        get_3d_cube_component()

    # --- DENSE SECTION ---
    st.markdown("---")
    st.markdown("### 🕸 Global Decision Making (Fully Connected)")
    col_dense1, col_dense2 = st.columns([1, 1.5])
    with col_dense1:
        get_dense_network_component()
    with col_dense2:
        st.markdown("""
        <div class='metric-card'>
            <p style='color:#d4d4f0; font-size:1.1rem;'>Every one of the <strong>1,600 inputs</strong> connects to every one of the <strong>128 neurons</strong>.</p>
            <p style='color:#7070a8; font-size:0.9rem; margin-top:10px;'>This layer accounts for <strong>84%</strong> of all parameters in our network—it is the brain that interprets the features found by the filters.</p>
        </div>
        """, unsafe_allow_html=True)

# --- 6. PARAMS & QUIZ (Educational Finish) ---
st.divider()
st.markdown("<h2 style='text-align:center; font-size:2.5rem; margin-bottom:40px;'>Parameters & Knowledge Check</h2>", unsafe_allow_html=True)

col_q1, col_q2 = st.columns(2)
with col_q1:
    get_parameter_counter_component(243786)
with col_q2:
    get_quiz_component()

# Footer
st.markdown("<div style='text-align:center; padding: 60px; color: #444466; font-family: \"Syne Mono\", monospace; font-size: 0.8rem; border-top: 1px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
st.markdown("CNN INTERPRETABILITY LAB // ENGINEERED FOR EXCELLENCE // UTS 42028 // RADHIKA VERMA", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
