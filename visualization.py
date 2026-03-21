import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import torch
from PIL import Image

def plot_prediction_probs(probs, classes):
    """
    Plot a horizontal bar chart of class probabilities.
    """
    fig = go.Figure(go.Bar(
        x=probs,
        y=classes,
        orientation='h',
        marker_color='#00d4ff',
        text=[f"{p:.2%}" for p in probs],
        textposition='auto',
    ))
    fig.update_layout(
        template='plotly_dark',
        title='Class Probabilities',
        xaxis_title='Probability',
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def render_feature_maps(activations, title="Feature Maps", max_cols=8):
    """
    activations: (1, C, H, W) tensor
    """
    acts = activations.squeeze(0).detach().cpu().numpy()
    num_channels = acts.shape[0]
    num_rows = (num_channels + max_cols - 1) // max_cols
    
    fig, axes = plt.subplots(num_rows, max_cols, figsize=(max_cols * 1.5, num_rows * 1.5))
    fig.patch.set_facecolor('#04040a')
    
    for i in range(num_rows * max_cols):
        ax = axes[i // max_cols, i % max_cols] if num_rows > 1 else axes[i % max_cols]
        if i < num_channels:
            ax.imshow(acts[i], cmap='magma')
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    return fig

def get_premium_css():
    """
    Returns the full cinematic CSS inspired by the original HTML.
    Includes custom fonts, noise overlay, and premium component styling.
    """
    return """
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Syne+Mono&family=Outfit:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #04040a;
            --surface: #0d0d1a;
            --surface2: #111120;
            --border: rgba(255,255,255,0.06);
            --c1: #00d4ff;   /* cyan */
            --c2: #a855f7;   /* purple */
            --c3: #ff6b35;   /* orange */
            --c4: #22c55e;   /* green */
            --font-display: 'Syne', sans-serif;
            --font-mono: 'Syne Mono', monospace;
            --font-body: 'Outfit', sans-serif;
        }

        /* Essential Reset for Streamlit */
        .main { background-color: var(--bg); }
        .stApp { background-color: var(--bg); color: #f4f4fc; }
        
        /* Noise Overlay */
        .stApp::before {
            content: '';
            position: fixed; inset:0; z-index:0; pointer-events:none;
            opacity: 0.015;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
        }

        /* Typography */
        h1, h2, h3 { font-family: var(--font-display) !important; font-weight: 800 !important; letter-spacing: -1px !important; }
        p, span, div { font-family: var(--font-body); }
        code { font-family: var(--font-mono) !important; color: var(--c1) !important; background: rgba(0,212,255,0.05) !important; }

        /* Custom Components */
        .metric-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            border-color: rgba(0,212,255,0.2);
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }
        
        .highlight-text {
            background: linear-gradient(135deg, var(--c1) 0%, var(--c2) 50%, var(--c3) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        /* Pipeline Track */
        .pipeline-node {
            padding: 15px;
            border: 1px solid var(--border);
            border-radius: 12px;
            text-align: center;
            background: var(--surface2);
            min-width: 120px;
        }
        .node-label { font-family: var(--font-mono); font-size: 0.7rem; color: var(--c1); text-transform: uppercase; }
        .node-val { font-family: var(--font-display); font-size: 1rem; font-weight: 700; }

        /* Hide Streamlit elements for a cleaner Look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """

def render_pipeline_horizontal(steps):
    """
    Renders a cinematic horizontal pipeline track.
    steps: List of (Label, Value, Icon)
    """
    cols = st.columns(len(steps))
    for i, (label, val, icon) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
                <div class='pipeline-node'>
                    <div style='font-size: 1.5rem; margin-bottom: 5px;'>{icon}</div>
                    <div class='node-label'>{label}</div>
                    <div class='node-val'>{val}</div>
                </div>
            """, unsafe_allow_html=True)
            if i < len(steps) - 1:
                pass # Arrows would need custom SVG or absolute positioning
