"""
Demo Recording Helper

Generates a sequence of frames for GIF creation.
Run alongside screen recording software (OBS, Windows Game Bar, etc.)

Usage:
    streamlit run app/record_demo.py

This will auto-play through 5 trials with smooth animations.
Use screen capture to record a 15-20 second GIF.
"""

import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
import torch

from src.config import set_seed, DEVICE, SEED, data_config, MODELS_DIR
from src.models import EEGNet
from src.preprocessing import load_processed_subject, preprocess_subject

# Page config
st.set_page_config(
    page_title="BCI Demo Recording",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal dark CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 100%);
    }
    #MainMenu, footer, header {visibility: hidden;}

    .big-title {
        font-family: 'Courier New', monospace;
        font-size: 2.5rem;
        color: #00d4ff;
        text-align: center;
        text-shadow: 0 0 20px #00d4ff;
        margin-bottom: 20px;
    }

    .prediction-box {
        font-family: 'Courier New', monospace;
        font-size: 3rem;
        text-align: center;
        padding: 30px;
        border-radius: 15px;
        margin: 20px;
        animation: pulse 1s infinite;
    }

    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 30px currentColor; }
        50% { box-shadow: 0 0 60px currentColor; }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = EEGNet(n_channels=22, n_timepoints=500, n_classes=4)
    checkpoint_path = MODELS_DIR / "eegnet_subject1.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(DEVICE)


@st.cache_data
def load_data():
    data = load_processed_subject(1)
    if data is None:
        try:
            data = preprocess_subject(1, use_moabb=True, save=True, verbose=False)
        except Exception:
            pass

    if data is None or data['test'][0].shape[0] == 0:
        # Generate synthetic data
        np.random.seed(42)
        n_trials, n_channels, n_timepoints = 50, 22, 500
        X = np.random.randn(n_trials, n_channels, n_timepoints).astype(np.float32)
        y = np.array([i % 4 for i in range(n_trials)]).astype(np.int64)
        data = {'test': (X, y)}

    return data['test']


def create_brain_activity(activity_level):
    """Create a simple brain activity visualization."""
    electrodes = {
        'C3': (-0.6, 0), 'Cz': (0, 0), 'C4': (0.6, 0),
        'FC3': (-0.4, 0.4), 'FCz': (0, 0.5), 'FC4': (0.4, 0.4),
        'CP3': (-0.4, -0.4), 'CPz': (0, -0.5), 'CP4': (0.4, -0.4),
    }

    fig = go.Figure()

    # Head circle
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=0.9*np.cos(theta), y=0.9*np.sin(theta),
        mode='lines', line=dict(color='#00d4ff', width=3),
        showlegend=False
    ))

    # Nose
    fig.add_trace(go.Scatter(
        x=[0], y=[1.0], mode='markers',
        marker=dict(size=15, color='#00d4ff', symbol='triangle-up'),
        showlegend=False
    ))

    # Electrodes with pulsing
    for name, (x, y) in electrodes.items():
        intensity = np.random.rand() * activity_level
        size = 30 + intensity * 30
        color = f'rgba(57, 255, 20, {0.3 + intensity * 0.7})'

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=size, color=color, line=dict(color='white', width=1)),
            text=[name], textposition='middle center',
            textfont=dict(size=10, color='white'),
            showlegend=False
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-1.2, 1.2]),
        yaxis=dict(visible=False, range=[-1.2, 1.2], scaleanchor='x'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350, margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig


def create_confidence_bars(probs, pred_class):
    """Create horizontal confidence bars."""
    classes = ['LEFT', 'RIGHT', 'FEET', 'TONGUE']
    colors = ['#ff006e', '#00d4ff', '#39ff14', '#ffaa00']

    fig = go.Figure()
    for i, (cls, prob, color) in enumerate(zip(classes, probs, colors)):
        opacity = 1.0 if i == pred_class else 0.4
        fig.add_trace(go.Bar(
            y=[cls], x=[prob], orientation='h',
            marker=dict(color=color, opacity=opacity),
            text=f'{prob*100:.0f}%', textposition='inside',
            textfont=dict(size=16, color='white'),
            showlegend=False
        ))

    fig.update_layout(
        xaxis=dict(range=[0, 1], visible=False),
        yaxis=dict(tickfont=dict(size=14, color='white')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=200, margin=dict(l=80, r=20, t=10, b=10)
    )
    return fig


def main():
    set_seed(SEED)

    st.markdown('<div class="big-title">🧠 NEURAL DECODER DEMO</div>', unsafe_allow_html=True)

    # Load resources
    model = load_model()
    X_test, y_test = load_data()

    # Session state
    if 'demo_trial' not in st.session_state:
        st.session_state.demo_trial = 0
    if 'demo_step' not in st.session_state:
        st.session_state.demo_step = 0
    if 'demo_running' not in st.session_state:
        st.session_state.demo_running = False
    if 'trials_shown' not in st.session_state:
        st.session_state.trials_shown = 0

    # Auto-start button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎬 START AUTO-DEMO (5 trials)", use_container_width=True):
            st.session_state.demo_running = True
            st.session_state.trials_shown = 0
            st.session_state.demo_step = 0
            st.session_state.demo_trial = np.random.randint(0, len(X_test))

    # Current trial
    trial_idx = st.session_state.demo_trial
    trial_data = X_test[trial_idx]
    true_label = y_test[trial_idx]

    # Get prediction
    with torch.no_grad():
        x = torch.FloatTensor(trial_data).unsqueeze(0).to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        # Add animation variation
        noise = np.sin(st.session_state.demo_step * 0.5) * 0.1
        probs = np.clip(probs + noise * np.random.randn(4), 0.05, 0.95)
        probs = probs / probs.sum()
        pred_class = np.argmax(probs)

    # Layout
    col_brain, col_bars = st.columns([1, 1])

    with col_brain:
        activity = 0.5 + 0.5 * np.sin(st.session_state.demo_step * 0.3)
        brain_fig = create_brain_activity(activity)
        st.plotly_chart(brain_fig, use_container_width=True, config={'displayModeBar': False})

    with col_bars:
        bar_fig = create_confidence_bars(probs, pred_class)
        st.plotly_chart(bar_fig, use_container_width=True, config={'displayModeBar': False})

    # Big prediction
    class_names = ['⬅ LEFT HAND', '➡ RIGHT HAND', '🦶 FEET', '👅 TONGUE']
    class_colors = ['#ff006e', '#00d4ff', '#39ff14', '#ffaa00']

    st.markdown(f"""
    <div class="prediction-box" style="
        background: {class_colors[pred_class]}22;
        border: 3px solid {class_colors[pred_class]};
        color: {class_colors[pred_class]};
    ">
        {class_names[pred_class]}
        <div style="font-size: 1.2rem; color: #888; margin-top: 10px;">
            Confidence: {probs[pred_class]*100:.0f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

    # True label indicator
    true_name = data_config.class_names[true_label].upper()
    correct = pred_class == true_label
    st.markdown(f"""
    <div style="text-align: center; margin-top: 20px; color: {'#39ff14' if correct else '#ff006e'};">
        TRUE: {true_name} {'✓ CORRECT' if correct else '✗'}
    </div>
    """, unsafe_allow_html=True)

    # Progress
    st.markdown(f"""
    <div style="text-align: center; color: #666; margin-top: 30px;">
        Trial {st.session_state.trials_shown + 1} / 5
    </div>
    """, unsafe_allow_html=True)

    # Auto-play logic
    if st.session_state.demo_running:
        st.session_state.demo_step += 1

        if st.session_state.demo_step >= 15:  # ~3 seconds per trial
            st.session_state.demo_step = 0
            st.session_state.trials_shown += 1
            st.session_state.demo_trial = np.random.randint(0, len(X_test))

            if st.session_state.trials_shown >= 5:
                st.session_state.demo_running = False
                st.balloons()

        time.sleep(0.2)
        st.rerun()


if __name__ == "__main__":
    main()
