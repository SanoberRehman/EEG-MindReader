"""
EEG Mind Reader - Live Decoder Demo

A futuristic Streamlit app that visualizes EEG motor imagery decoding in real-time.
Shows streaming brain signals, model predictions, and topographic activity maps.

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our modules
from src.config import (
    set_seed, DEVICE, SEED,
    data_config, MODELS_DIR, PROCESSED_DATA_DIR
)
from src.models import EEGNet, CNNLSTM, EEGTransformer
from src.preprocessing import load_processed_subject, preprocess_subject

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="EEG Mind Reader",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic dark theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0f0f23 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f0f23 100%);
        border-right: 1px solid #00d4ff33;
    }

    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Courier New', monospace !important;
        text-shadow: 0 0 10px #00d4ff44;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00ff88 !important;
        font-family: 'Courier New', monospace !important;
        font-size: 2rem !important;
        text-shadow: 0 0 15px #00ff8844;
    }

    [data-testid="stMetricLabel"] {
        color: #888 !important;
        font-family: 'Courier New', monospace !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff 0%, #00ff88 100%);
        color: #0a0a0a;
        border: none;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 5px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .stButton > button:hover {
        box-shadow: 0 0 20px #00d4ff88, 0 0 40px #00ff8844;
        transform: translateY(-2px);
    }

    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #1a1a2e;
        border: 1px solid #00d4ff44;
        color: #fff;
    }

    /* Info boxes */
    .prediction-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid #00d4ff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px #00d4ff22;
    }

    .class-label {
        font-family: 'Courier New', monospace;
        font-size: 1.5rem;
        color: #00ff88;
        text-shadow: 0 0 10px #00ff8844;
    }

    /* Neon glow animation */
    @keyframes neon-pulse {
        0%, 100% { box-shadow: 0 0 5px #00d4ff, 0 0 10px #00d4ff, 0 0 20px #00d4ff; }
        50% { box-shadow: 0 0 10px #00d4ff, 0 0 20px #00d4ff, 0 0 40px #00d4ff; }
    }

    .neon-border {
        animation: neon-pulse 2s infinite;
    }

    /* Status indicator */
    .status-online {
        color: #00ff88;
        text-shadow: 0 0 10px #00ff88;
    }

    .status-processing {
        color: #ffaa00;
        text-shadow: 0 0 10px #ffaa00;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource
def load_model(model_name: str, subject_id: int = 1):
    """Load a trained model from checkpoint."""
    n_channels = data_config.n_channels
    n_timepoints = 500  # 2 seconds at 250 Hz
    n_classes = data_config.n_classes

    if model_name == "EEGNet":
        model = EEGNet(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=n_classes)
    elif model_name == "CNN-LSTM":
        model = CNNLSTM(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=n_classes)
    elif model_name == "Transformer":
        model = EEGTransformer(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Try to load checkpoint
    checkpoint_path = MODELS_DIR / f"{model_name.lower().replace('-', '_')}_subject{subject_id}.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        st.sidebar.success(f"Loaded trained {model_name}")
    else:
        st.sidebar.warning(f"No checkpoint found, using random weights")

    model.eval()
    model = model.to(DEVICE)
    return model


@st.cache_data
def load_test_data(subject_id: int = 1):
    """Load preprocessed test data."""
    # Try to load cached data
    data = load_processed_subject(subject_id)

    if data is None:
        # Try to preprocess, fall back to synthetic
        try:
            st.info("Preprocessing data (first run only)...")
            data = preprocess_subject(subject_id, use_moabb=True, save=True, verbose=False)
        except Exception:
            pass

    if data is None or data['test'][0].shape[0] == 0:
        # Generate synthetic demo data
        st.warning("Using synthetic demo data")
        np.random.seed(42)
        n_trials, n_channels, n_timepoints = 50, 22, 500
        X = np.random.randn(n_trials, n_channels, n_timepoints).astype(np.float32)
        y = np.array([i % 4 for i in range(n_trials)]).astype(np.int64)
        return X, y

    X_test, y_test = data['test']
    return X_test, y_test


def predict_with_confidence(model, trial_data: np.ndarray) -> tuple:
    """Run model prediction and return class probabilities."""
    with torch.no_grad():
        x = torch.FloatTensor(trial_data).unsqueeze(0).to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
    return pred_class, probs


def create_eeg_plot(data: np.ndarray, time_idx: int, window_size: int = 250):
    """Create animated EEG signal plot."""
    n_channels = min(8, data.shape[0])  # Show top 8 channels
    channel_indices = [7, 8, 9, 10, 11, 6, 12, 15]  # Motor cortex focused
    channel_names = [data_config.channel_names[i] for i in channel_indices[:n_channels]]

    # Get visible window
    start_idx = max(0, time_idx - window_size)
    end_idx = time_idx

    if end_idx <= start_idx:
        end_idx = min(window_size, data.shape[1])
        start_idx = 0

    time_axis = np.arange(start_idx, end_idx) / 250  # Convert to seconds

    fig = go.Figure()

    # Add traces for each channel
    colors = ['#00d4ff', '#00ff88', '#ff6b6b', '#ffd93d', '#6bceff', '#c56bff', '#ff6bcd', '#6bffb8']

    for i, ch_idx in enumerate(channel_indices[:n_channels]):
        offset = i * 3  # Vertical offset for display
        signal = data[ch_idx, start_idx:end_idx] + offset

        fig.add_trace(go.Scatter(
            x=time_axis,
            y=signal,
            mode='lines',
            name=channel_names[i],
            line=dict(color=colors[i], width=1.5),
            hovertemplate=f'{channel_names[i]}: %{{y:.2f}}<extra></extra>'
        ))

    # Add current time marker
    current_time = time_idx / 250
    fig.add_vline(
        x=current_time,
        line_dash="dash",
        line_color="#ff6b6b",
        annotation_text="NOW",
        annotation_position="top"
    )

    fig.update_layout(
        title=dict(
            text="<b>LIVE EEG SIGNAL</b>",
            font=dict(color='#00d4ff', size=16, family='Courier New')
        ),
        xaxis=dict(
            title="Time (s)",
            color='#888',
            gridcolor='#333',
            range=[max(0, current_time - 1), current_time + 0.1]
        ),
        yaxis=dict(
            title="Channel",
            color='#888',
            gridcolor='#333',
            tickmode='array',
            tickvals=[i * 3 for i in range(n_channels)],
            ticktext=channel_names
        ),
        plot_bgcolor='rgba(10, 10, 20, 0.8)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family='Courier New', color='#888'),
        showlegend=False,
        height=400,
        margin=dict(l=60, r=20, t=50, b=40)
    )

    return fig


def create_confidence_bars(probs: np.ndarray, pred_class: int):
    """Create confidence bar chart."""
    class_names = data_config.class_names
    colors_full = ['rgba(255,107,107,1)', 'rgba(78,205,196,1)', 'rgba(69,183,209,1)', 'rgba(150,206,180,1)']
    colors_dim = ['rgba(255,107,107,0.4)', 'rgba(78,205,196,0.4)', 'rgba(69,183,209,0.4)', 'rgba(150,206,180,0.4)']

    # Highlight predicted class
    bar_colors = [colors_full[i] if i == pred_class else colors_dim[i] for i in range(len(probs))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=class_names,
        y=probs * 100,
        marker_color=bar_colors,
        marker_line=dict(color='#fff', width=2),
        text=[f'{p*100:.1f}%' for p in probs],
        textposition='outside',
        textfont=dict(color='#00ff88', size=14, family='Courier New')
    ))

    fig.update_layout(
        title=dict(
            text="<b>CLASS CONFIDENCE</b>",
            font=dict(color='#00d4ff', size=16, family='Courier New')
        ),
        xaxis=dict(
            color='#888',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="Confidence (%)",
            color='#888',
            range=[0, 110],
            gridcolor='#333'
        ),
        plot_bgcolor='rgba(10, 10, 20, 0.8)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family='Courier New', color='#888'),
        height=300,
        margin=dict(l=60, r=20, t=50, b=40)
    )

    return fig


def create_topomap(data: np.ndarray, time_idx: int):
    """Create topographic brain activity map."""
    # Electrode positions (approximate 10-20 positions normalized to [0,1])
    positions = {
        'Fz': (0.5, 0.85), 'FC3': (0.3, 0.7), 'FC1': (0.4, 0.7),
        'FCz': (0.5, 0.7), 'FC2': (0.6, 0.7), 'FC4': (0.7, 0.7),
        'C5': (0.15, 0.5), 'C3': (0.3, 0.5), 'C1': (0.4, 0.5),
        'Cz': (0.5, 0.5), 'C2': (0.6, 0.5), 'C4': (0.7, 0.5), 'C6': (0.85, 0.5),
        'CP3': (0.3, 0.3), 'CP1': (0.4, 0.3), 'CPz': (0.5, 0.3),
        'CP2': (0.6, 0.3), 'CP4': (0.7, 0.3),
        'P1': (0.4, 0.15), 'Pz': (0.5, 0.15), 'P2': (0.6, 0.15), 'POz': (0.5, 0.05)
    }

    # Get current values (average over small window)
    window = 25  # 100ms
    start = max(0, time_idx - window)
    end = min(data.shape[1], time_idx + 1)
    values = np.abs(data[:, start:end]).mean(axis=1)
    values = (values - values.min()) / (values.max() - values.min() + 1e-8)

    # Create scatter plot
    xs, ys, vals, texts = [], [], [], []
    for i, ch_name in enumerate(data_config.channel_names):
        if ch_name in positions:
            x, y = positions[ch_name]
            xs.append(x)
            ys.append(y)
            vals.append(values[i])
            texts.append(ch_name)

    fig = go.Figure()

    # Draw head outline
    theta = np.linspace(0, 2*np.pi, 100)
    head_x = 0.5 + 0.45 * np.cos(theta)
    head_y = 0.45 + 0.45 * np.sin(theta)

    fig.add_trace(go.Scatter(
        x=head_x, y=head_y,
        mode='lines',
        line=dict(color='#00d4ff', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Draw nose
    fig.add_trace(go.Scatter(
        x=[0.5, 0.5], y=[0.9, 0.95],
        mode='lines',
        line=dict(color='#00d4ff', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Draw ears
    for ear_x in [0.02, 0.98]:
        fig.add_trace(go.Scatter(
            x=[ear_x, ear_x], y=[0.4, 0.5],
            mode='lines',
            line=dict(color='#00d4ff', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Electrode scatter
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='markers+text',
        marker=dict(
            size=30,
            color=vals,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(
                title=dict(text='Activity', font=dict(color='#888')),
                tickfont=dict(color='#888')
            ),
            line=dict(color='#fff', width=1)
        ),
        text=texts,
        textposition='middle center',
        textfont=dict(size=8, color='#fff'),
        hovertemplate='%{text}: %{marker.color:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="<b>BRAIN ACTIVITY MAP</b>",
            font=dict(color='#00d4ff', size=16, family='Courier New')
        ),
        xaxis=dict(
            visible=False,
            range=[-0.1, 1.1]
        ),
        yaxis=dict(
            visible=False,
            range=[-0.1, 1.1],
            scaleanchor='x'
        ),
        plot_bgcolor='rgba(10, 10, 20, 0.8)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 3rem; margin-bottom: 0;">🧠 EEG MIND READER</h1>
        <p style="color: #00d4ff; font-family: 'Courier New'; font-size: 1.2rem;">
            Real-Time Motor Imagery Decoder
        </p>
        <p style="color: #666; font-family: 'Courier New'; font-size: 0.9rem;">
            Decoding imagined movements from brain signals using deep learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("## ⚙️ CONTROL PANEL")
    st.sidebar.markdown("---")

    # Model selection
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["EEGNet", "CNN-LSTM", "Transformer"],
        help="Choose the neural network architecture"
    )

    # Subject selection
    subject_id = st.sidebar.selectbox(
        "Test Subject",
        list(range(1, 10)),
        help="Select which subject's data to decode"
    )

    # Animation speed
    speed = st.sidebar.slider(
        "Playback Speed",
        min_value=1,
        max_value=10,
        value=5,
        help="Control animation speed"
    )

    st.sidebar.markdown("---")

    # Status indicator
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 10px; border: 1px solid #00d4ff33; border-radius: 5px;">
        <p style="margin: 0; color: #888; font-size: 0.8rem;">SYSTEM STATUS</p>
        <p class="status-online" style="margin: 0; font-size: 1.2rem;">● ONLINE</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="color: #666; font-size: 0.8rem;">
        <p><b>Classes:</b></p>
        <p>🖐️ Left Hand</p>
        <p>✋ Right Hand</p>
        <p>🦶 Feet</p>
        <p>👅 Tongue</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model and data
    with st.spinner("Loading neural network..."):
        model = load_model(model_name, subject_id)

    with st.spinner("Loading EEG data..."):
        X_test, y_test = load_test_data(subject_id)

    # Initialize session state
    if 'trial_idx' not in st.session_state:
        st.session_state.trial_idx = 0
    if 'time_idx' not in st.session_state:
        st.session_state.time_idx = 0
    if 'playing' not in st.session_state:
        st.session_state.playing = False

    # Control buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🎲 RANDOM TRIAL", use_container_width=True):
            st.session_state.trial_idx = np.random.randint(0, len(X_test))
            st.session_state.time_idx = 0
            st.session_state.playing = False

    with col2:
        if st.button("▶️ PLAY" if not st.session_state.playing else "⏸️ PAUSE", use_container_width=True):
            st.session_state.playing = not st.session_state.playing

    with col3:
        if st.button("⏮️ RESET", use_container_width=True):
            st.session_state.time_idx = 0
            st.session_state.playing = False

    with col4:
        if st.button("⏭️ NEXT TRIAL", use_container_width=True):
            st.session_state.trial_idx = (st.session_state.trial_idx + 1) % len(X_test)
            st.session_state.time_idx = 0
            st.session_state.playing = False

    # Get current trial
    trial_idx = st.session_state.trial_idx
    trial_data = X_test[trial_idx]
    true_label = y_test[trial_idx]
    true_class = data_config.class_names[true_label]

    # Progress bar
    progress = st.session_state.time_idx / trial_data.shape[1]
    st.progress(progress)

    # Main display area
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # EEG signal plot
        eeg_fig = create_eeg_plot(trial_data, st.session_state.time_idx)
        st.plotly_chart(eeg_fig, use_container_width=True)

    with col_right:
        # Brain topomap
        topo_fig = create_topomap(trial_data, st.session_state.time_idx)
        st.plotly_chart(topo_fig, use_container_width=True)

    # Run prediction
    pred_class, probs = predict_with_confidence(model, trial_data)
    pred_name = data_config.class_names[pred_class]

    # Prediction display
    col_pred, col_conf = st.columns([1, 2])

    with col_pred:
        st.markdown(f"""
        <div class="prediction-box" style="text-align: center;">
            <p style="color: #888; margin-bottom: 5px; font-size: 0.9rem;">PREDICTED CLASS</p>
            <p class="class-label" style="font-size: 2rem; margin: 0;">{pred_name.upper()}</p>
            <p style="color: #888; margin-top: 10px; font-size: 0.8rem;">Confidence: {probs[pred_class]*100:.1f}%</p>
            <hr style="border-color: #333; margin: 15px 0;">
            <p style="color: #888; margin-bottom: 5px; font-size: 0.9rem;">TRUE CLASS</p>
            <p style="color: {'#00ff88' if pred_class == true_label else '#ff6b6b'}; font-size: 1.5rem; margin: 0;">
                {true_class.upper()}
            </p>
            <p style="color: #888; margin-top: 5px; font-size: 0.8rem;">
                {'✓ CORRECT' if pred_class == true_label else '✗ INCORRECT'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_conf:
        # Confidence bars
        conf_fig = create_confidence_bars(probs, pred_class)
        st.plotly_chart(conf_fig, use_container_width=True)

    # Trial info
    st.markdown(f"""
    <div style="text-align: center; padding: 10px; color: #666; font-family: 'Courier New';">
        Trial {trial_idx + 1}/{len(X_test)} |
        Time: {st.session_state.time_idx / 250:.2f}s / {trial_data.shape[1] / 250:.2f}s |
        Model: {model_name} |
        Subject: {subject_id}
    </div>
    """, unsafe_allow_html=True)

    # Animation loop
    if st.session_state.playing:
        step = speed * 10  # Samples per frame
        st.session_state.time_idx += step

        if st.session_state.time_idx >= trial_data.shape[1]:
            st.session_state.time_idx = trial_data.shape[1] - 1
            st.session_state.playing = False

        time.sleep(0.05)  # ~20 FPS
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #444; font-size: 0.8rem; font-family: 'Courier New';">
        <p>EEG Motor Imagery Classification | Built with PyTorch & Streamlit</p>
        <p>Dataset: BCI Competition IV 2a | 22 channels | 250 Hz</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    set_seed(SEED)
    main()
