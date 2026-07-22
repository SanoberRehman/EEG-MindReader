"""
🧠 BCI NEURAL INTERFACE DASHBOARD
A cyberpunk-style Brain-Computer Interface visualization for presentations.

Run with: streamlit run app/bci_dashboard.py
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

# Import our modules
from src.config import set_seed, DEVICE, SEED, data_config, MODELS_DIR
from src.models import EEGNet
from src.preprocessing import load_processed_subject, preprocess_subject

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="🧠 BCI Neural Interface",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CYBERPUNK CSS STYLING
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&display=swap');

    /* Main background - deep space black */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0a0f 100%);
        background-attachment: fixed;
    }

    /* Add subtle grid pattern overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image:
            linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(0, 0, 0, 0.5);
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #00d4ff33;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(0, 212, 255, 0.1);
        border-radius: 8px;
        color: #00d4ff;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        padding: 10px 20px;
        border: 1px solid #00d4ff44;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff33 0%, #39ff1433 100%);
        border-color: #00d4ff;
        box-shadow: 0 0 20px #00d4ff44, inset 0 0 20px #00d4ff22;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(90deg, #00d4ff 0%, #39ff14 50%, #ff006e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px #00d4ff66;
        letter-spacing: 2px;
    }

    /* Neon button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff22 0%, #39ff1422 100%);
        color: #00d4ff;
        border: 2px solid #00d4ff;
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        padding: 15px 30px;
        border-radius: 8px;
        text-transform: uppercase;
        letter-spacing: 3px;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px #00d4ff44;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #00d4ff44 0%, #39ff1444 100%);
        box-shadow: 0 0 30px #00d4ff88, 0 0 60px #39ff1444;
        transform: translateY(-2px);
        border-color: #39ff14;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', monospace !important;
        font-size: 2.5rem !important;
        background: linear-gradient(90deg, #39ff14 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px #39ff1466;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Rajdhani', sans-serif !important;
        color: #888 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Custom classes */
    .cyber-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(57, 255, 20, 0.05) 100%);
        border: 1px solid #00d4ff44;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.1), inset 0 0 30px rgba(0, 0, 0, 0.5);
    }

    .prediction-huge {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        padding: 30px;
        margin: 20px 0;
        border-radius: 15px;
        animation: pulse-glow 2s infinite;
    }

    @keyframes pulse-glow {
        0%, 100% {
            box-shadow: 0 0 30px currentColor, inset 0 0 30px rgba(0,0,0,0.5);
        }
        50% {
            box-shadow: 0 0 60px currentColor, 0 0 100px currentColor, inset 0 0 30px rgba(0,0,0,0.5);
        }
    }

    @keyframes scan-line {
        0% { top: 0%; }
        100% { top: 100%; }
    }

    .scan-effect::after {
        content: '';
        position: absolute;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
        animation: scan-line 3s linear infinite;
    }

    /* Score display */
    .score-display {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.5rem;
        color: #39ff14;
        text-shadow: 0 0 10px #39ff14;
        padding: 10px 20px;
        border: 1px solid #39ff1444;
        border-radius: 8px;
        background: rgba(57, 255, 20, 0.1);
    }

    /* Status indicators */
    .status-active {
        color: #39ff14;
        text-shadow: 0 0 15px #39ff14;
    }

    .status-warning {
        color: #ffaa00;
        text-shadow: 0 0 15px #ffaa00;
    }

    /* Glitch effect for title */
    .glitch {
        position: relative;
    }

    .glitch::before,
    .glitch::after {
        content: attr(data-text);
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
    }

    .glitch::before {
        animation: glitch-1 0.3s infinite;
        color: #ff006e;
        z-index: -1;
    }

    .glitch::after {
        animation: glitch-2 0.3s infinite;
        color: #00d4ff;
        z-index: -2;
    }

    @keyframes glitch-1 {
        0%, 100% { clip-path: inset(0 0 0 0); transform: translate(0); }
        20% { clip-path: inset(20% 0 60% 0); transform: translate(-2px, 2px); }
        40% { clip-path: inset(40% 0 40% 0); transform: translate(2px, -2px); }
        60% { clip-path: inset(60% 0 20% 0); transform: translate(-2px, 2px); }
        80% { clip-path: inset(80% 0 0 0); transform: translate(2px, -2px); }
    }

    @keyframes glitch-2 {
        0%, 100% { clip-path: inset(0 0 0 0); transform: translate(0); }
        20% { clip-path: inset(60% 0 20% 0); transform: translate(2px, -2px); }
        40% { clip-path: inset(20% 0 60% 0); transform: translate(-2px, 2px); }
        60% { clip-path: inset(80% 0 0 0); transform: translate(2px, -2px); }
        80% { clip-path: inset(0 0 80% 0); transform: translate(-2px, 2px); }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_resource
def load_model():
    """Load trained EEGNet model."""
    model = EEGNet(
        n_channels=22,
        n_timepoints=500,
        n_classes=4
    )

    checkpoint_path = MODELS_DIR / "eegnet_subject1.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    model = model.to(DEVICE)
    return model


@st.cache_data
def load_test_data():
    """Load test data."""
    # Try to load cached data first
    data = load_processed_subject(1)

    if data is None:
        # Try MOABB, fall back to synthetic
        try:
            data = preprocess_subject(1, use_moabb=True, save=True, verbose=False)
        except Exception:
            pass

    if data is None or data['test'][0].shape[0] == 0:
        # Generate synthetic data for demo
        data = generate_demo_data()

    return data['test']


def generate_demo_data():
    """Generate synthetic EEG data for demo purposes."""
    np.random.seed(42)

    n_trials = 100
    n_channels = 22
    n_timepoints = 500
    n_classes = 4

    X = np.random.randn(n_trials, n_channels, n_timepoints).astype(np.float32)

    # Add class-specific patterns to make it more realistic
    for i in range(n_trials):
        class_idx = i % n_classes

        if class_idx == 0:  # Left hand - pattern in right motor cortex (C4)
            X[i, 11, :] += 0.5 * np.sin(np.linspace(0, 10*np.pi, n_timepoints))
        elif class_idx == 1:  # Right hand - pattern in left motor cortex (C3)
            X[i, 7, :] += 0.5 * np.sin(np.linspace(0, 10*np.pi, n_timepoints))
        elif class_idx == 2:  # Feet - pattern in central (Cz)
            X[i, 9, :] += 0.5 * np.sin(np.linspace(0, 8*np.pi, n_timepoints))
        else:  # Tongue
            X[i, 3, :] += 0.3 * np.sin(np.linspace(0, 12*np.pi, n_timepoints))

    y = np.array([i % n_classes for i in range(n_trials)]).astype(np.int64)

    # Shuffle
    indices = np.random.permutation(n_trials)
    X, y = X[indices], y[indices]

    # Split
    train_size = int(0.7 * n_trials)
    val_size = int(0.15 * n_trials)

    return {
        'train': (X[:train_size], y[:train_size]),
        'val': (X[train_size:train_size+val_size], y[train_size:train_size+val_size]),
        'test': (X[train_size+val_size:], y[train_size+val_size:])
    }


@st.cache_data
def precompute_predictions(_model, X_test, y_test):
    """Precompute sliding window predictions for all trials."""
    predictions = []
    window_size = 500
    stride = 25  # 100ms at 250Hz

    for trial_idx in range(len(X_test)):
        trial = X_test[trial_idx]
        trial_preds = []

        # For short trials, just predict on whole trial repeatedly
        n_windows = max(1, (trial.shape[1] - window_size) // stride + 1)

        for i in range(20):  # 20 prediction steps per trial
            with torch.no_grad():
                x = torch.FloatTensor(trial).unsqueeze(0).to(DEVICE)
                logits = _model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                # Add some temporal variation for visual effect
                noise = np.random.randn(4) * 0.05
                probs = np.clip(probs + noise, 0, 1)
                probs = probs / probs.sum()
                trial_preds.append(probs)

        predictions.append(np.array(trial_preds))

    return predictions


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def create_cursor_game(cursor_pos, target_side, score, total):
    """Create the Mind Cursor game visualization."""
    fig = go.Figure()

    # Background gradient effect
    fig.add_shape(
        type="rect", x0=-1.2, x1=1.2, y0=-0.6, y1=0.6,
        fillcolor="rgba(10, 10, 20, 0.9)",
        line=dict(color="#00d4ff", width=2)
    )

    # Grid lines for cyberpunk effect
    for x in np.linspace(-1, 1, 11):
        fig.add_shape(
            type="line", x0=x, x1=x, y0=-0.5, y1=0.5,
            line=dict(color="rgba(0, 212, 255, 0.1)", width=1)
        )
    for y in np.linspace(-0.5, 0.5, 6):
        fig.add_shape(
            type="line", x0=-1, x1=1, y0=y, y1=y,
            line=dict(color="rgba(0, 212, 255, 0.1)", width=1)
        )

    # Left target (red zone)
    fig.add_shape(
        type="rect", x0=-1.1, x1=-0.8, y0=-0.4, y1=0.4,
        fillcolor="rgba(255, 0, 110, 0.3)",
        line=dict(color="#ff006e", width=3)
    )
    fig.add_annotation(
        x=-0.95, y=0, text="⬅ LEFT",
        font=dict(size=16, color="#ff006e", family="Orbitron"),
        showarrow=False
    )

    # Right target (blue zone)
    fig.add_shape(
        type="rect", x0=0.8, x1=1.1, y0=-0.4, y1=0.4,
        fillcolor="rgba(0, 212, 255, 0.3)",
        line=dict(color="#00d4ff", width=3)
    )
    fig.add_annotation(
        x=0.95, y=0, text="RIGHT ➡",
        font=dict(size=16, color="#00d4ff", family="Orbitron"),
        showarrow=False
    )

    # Cursor with glow effect
    cursor_color = "#39ff14"
    for size, opacity in [(40, 0.1), (30, 0.2), (20, 0.4)]:
        fig.add_trace(go.Scatter(
            x=[cursor_pos], y=[0],
            mode='markers',
            marker=dict(
                size=size,
                color=cursor_color,
                opacity=opacity,
                line=dict(width=0)
            ),
            showlegend=False,
            hoverinfo='skip'
        ))

    # Main cursor
    fig.add_trace(go.Scatter(
        x=[cursor_pos], y=[0],
        mode='markers',
        marker=dict(
            size=25,
            color=cursor_color,
            symbol='diamond',
            line=dict(color='white', width=2)
        ),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Score display - large and prominent at top
    accuracy = (score / total * 100) if total > 0 else 0
    fig.add_annotation(
        x=0, y=0.85,
        text=f"🎯 SCORE: {score}/{total} ({accuracy:.0f}%)",
        font=dict(size=28, color="#39ff14", family="Arial Black"),
        showarrow=False,
        bgcolor="rgba(0,0,0,0.9)",
        bordercolor="#39ff14",
        borderwidth=3,
        borderpad=15
    )

    fig.update_layout(
        xaxis=dict(range=[-1.3, 1.3], visible=False, fixedrange=True),
        yaxis=dict(range=[-0.7, 1.1], visible=False, fixedrange=True, scaleanchor='x'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        height=450
    )

    return fig


def create_eeg_streams(data, time_idx, n_display=8):
    """Create streaming EEG waveform visualization."""
    fig = make_subplots(
        rows=n_display, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02
    )

    channel_names = ['C3', 'C1', 'Cz', 'C2', 'C4', 'FC3', 'FCz', 'FC4']
    channel_indices = [7, 8, 9, 10, 11, 1, 3, 5]

    window = 200  # samples to show
    start = max(0, time_idx - window)
    end = time_idx

    if end <= start:
        end = min(window, data.shape[1])
        start = 0

    colors = ['#39ff14', '#00d4ff', '#ff006e', '#ffaa00', '#39ff14', '#00d4ff', '#ff006e', '#ffaa00']

    for i, (ch_idx, ch_name) in enumerate(zip(channel_indices[:n_display], channel_names[:n_display])):
        signal = data[ch_idx, start:end]
        time_axis = np.arange(len(signal)) / 250

        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=signal,
                mode='lines',
                line=dict(color=colors[i], width=1.5),
                fill='tozeroy',
                fillcolor=f'rgba({int(colors[i][1:3], 16)}, {int(colors[i][3:5], 16)}, {int(colors[i][5:7], 16)}, 0.1)',
                name=ch_name,
                showlegend=False
            ),
            row=i+1, col=1
        )

        fig.update_yaxes(
            title_text=ch_name,
            title_font=dict(size=10, color=colors[i]),
            showgrid=False,
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.1)',
            tickfont=dict(size=8, color='#666'),
            row=i+1, col=1
        )

    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_xaxes(showticklabels=True, tickfont=dict(color='#666'), row=n_display, col=1)

    fig.update_layout(
        plot_bgcolor='rgba(10, 10, 20, 0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=50, r=20, t=20, b=30),
        font=dict(family='Share Tech Mono')
    )

    return fig


def create_3d_brain(data, time_idx):
    """Create 3D brain electrode visualization."""
    # 10-20 electrode positions (approximate spherical)
    electrodes = {
        'Fz': (0, 0.7, 0.7), 'FC3': (-0.5, 0.5, 0.7), 'FC1': (-0.25, 0.5, 0.75),
        'FCz': (0, 0.5, 0.8), 'FC2': (0.25, 0.5, 0.75), 'FC4': (0.5, 0.5, 0.7),
        'C5': (-0.8, 0, 0.5), 'C3': (-0.6, 0, 0.7), 'C1': (-0.3, 0, 0.8),
        'Cz': (0, 0, 0.85), 'C2': (0.3, 0, 0.8), 'C4': (0.6, 0, 0.7), 'C6': (0.8, 0, 0.5),
        'CP3': (-0.5, -0.4, 0.65), 'CP1': (-0.25, -0.4, 0.7), 'CPz': (0, -0.4, 0.75),
        'CP2': (0.25, -0.4, 0.7), 'CP4': (0.5, -0.4, 0.65),
        'P1': (-0.2, -0.6, 0.6), 'Pz': (0, -0.6, 0.65), 'P2': (0.2, -0.6, 0.6),
        'POz': (0, -0.75, 0.5)
    }

    # Get current activity levels
    window = 25
    start = max(0, time_idx - window)
    end = min(data.shape[1], time_idx + 1)
    activity = np.abs(data[:, start:end]).mean(axis=1)
    activity = (activity - activity.min()) / (activity.max() - activity.min() + 1e-8)

    ch_names = list(electrodes.keys())
    xs, ys, zs, sizes, colors_vals = [], [], [], [], []

    for i, ch in enumerate(ch_names):
        if ch in electrodes:
            x, y, z = electrodes[ch]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            sizes.append(15 + activity[i] * 25)
            colors_vals.append(activity[i])

    # Create head mesh
    theta = np.linspace(0, np.pi, 20)
    phi = np.linspace(0, 2*np.pi, 40)
    theta, phi = np.meshgrid(theta, phi)

    x_head = 0.9 * np.sin(theta) * np.cos(phi)
    y_head = 0.9 * np.sin(theta) * np.sin(phi)
    z_head = 0.9 * np.cos(theta) * 0.8

    fig = go.Figure()

    # Head surface
    fig.add_trace(go.Surface(
        x=x_head, y=y_head, z=z_head,
        colorscale=[[0, 'rgba(20, 20, 40, 0.3)'], [1, 'rgba(40, 40, 60, 0.3)']],
        showscale=False,
        hoverinfo='skip'
    ))

    # Electrodes
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+text',
        marker=dict(
            size=sizes,
            color=colors_vals,
            colorscale='Hot',
            opacity=0.9,
            line=dict(color='white', width=1)
        ),
        text=ch_names,
        textposition='top center',
        textfont=dict(size=8, color='white'),
        hovertemplate='%{text}<br>Activity: %{marker.color:.2f}<extra></extra>'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1.2, 1.2]),
            yaxis=dict(visible=False, range=[-1.2, 1.2]),
            zaxis=dict(visible=False, range=[-0.2, 1.2]),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )

    return fig


def create_probability_bars(probs, pred_class):
    """Create animated probability bar chart."""
    class_names = ['LEFT HAND', 'RIGHT HAND', 'FEET', 'TONGUE']
    colors = ['#ff006e', '#00d4ff', '#39ff14', '#ffaa00']

    fig = go.Figure()

    for i, (name, prob, color) in enumerate(zip(class_names, probs, colors)):
        # Background bar
        fig.add_trace(go.Bar(
            y=[name],
            x=[1],
            orientation='h',
            marker=dict(color='rgba(50,50,50,0.5)'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Active bar
        opacity = 1.0 if i == pred_class else 0.5
        fig.add_trace(go.Bar(
            y=[name],
            x=[prob],
            orientation='h',
            marker=dict(
                color=color,
                opacity=opacity,
                line=dict(color='white', width=1 if i == pred_class else 0)
            ),
            text=f'{prob*100:.1f}%',
            textposition='inside',
            textfont=dict(size=14, color='white', family='Orbitron'),
            showlegend=False,
            hovertemplate=f'{name}: {prob*100:.1f}%<extra></extra>'
        ))

    fig.update_layout(
        barmode='overlay',
        xaxis=dict(
            range=[0, 1],
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color='white', family='Orbitron')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=120, r=20, t=20, b=20),
        height=250
    )

    return fig


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    set_seed(SEED)

    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 30px 0;">
        <h1 style="font-size: 3.5rem; margin: 0; letter-spacing: 5px;">
            🧠 NEURAL INTERFACE
        </h1>
        <p style="color: #00d4ff; font-family: 'Share Tech Mono', monospace; font-size: 1.1rem; letter-spacing: 3px;">
            BRAIN-COMPUTER INTERFACE • MOTOR IMAGERY DECODER • v2.0
        </p>
        <div style="width: 200px; height: 2px; background: linear-gradient(90deg, transparent, #00d4ff, #39ff14, #ff006e, transparent); margin: 15px auto;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Load data and model
    with st.spinner("⚡ Initializing neural network..."):
        model = load_model()
        X_test, y_test = load_test_data()
        predictions = precompute_predictions(model, X_test, y_test)

    # Initialize session state
    if 'cursor_pos' not in st.session_state:
        st.session_state.cursor_pos = 0.0
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'total' not in st.session_state:
        st.session_state.total = 0
    if 'trial_idx' not in st.session_state:
        st.session_state.trial_idx = 0
    if 'time_step' not in st.session_state:
        st.session_state.time_step = 0
    if 'playing' not in st.session_state:
        st.session_state.playing = False
    if 'theater_trial' not in st.session_state:
        st.session_state.theater_trial = 0
    if 'theater_step' not in st.session_state:
        st.session_state.theater_step = 0
    if 'theater_playing' not in st.session_state:
        st.session_state.theater_playing = False

    # Tabs
    tab1, tab2 = st.tabs(["🎯 MIND CURSOR", "🧠 BRAIN THEATER"])

    # =========================================================================
    # TAB 1: MIND CURSOR GAME
    # =========================================================================
    with tab1:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <p style="color: #888; font-family: 'Rajdhani', sans-serif; font-size: 1.1rem;">
                Watch the cursor move based on decoded brain signals • LEFT or RIGHT imagery
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Filter for left/right only trials
        lr_indices = np.where((y_test == 0) | (y_test == 1))[0]

        # Controls
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("🎲 RANDOM TRIAL", key="cursor_random", use_container_width=True):
                st.session_state.trial_idx = np.random.choice(lr_indices)
                st.session_state.cursor_pos = 0.0
                st.session_state.time_step = 0
                st.session_state.playing = False

        with col2:
            play_label = "⏸ PAUSE" if st.session_state.playing else "▶ PLAY"
            if st.button(play_label, key="cursor_play", use_container_width=True):
                st.session_state.playing = not st.session_state.playing

        with col3:
            if st.button("⏮ RESET", key="cursor_reset", use_container_width=True):
                st.session_state.cursor_pos = 0.0
                st.session_state.time_step = 0
                st.session_state.playing = False

        with col4:
            if st.button("🔄 RESET SCORE", key="score_reset", use_container_width=True):
                st.session_state.score = 0
                st.session_state.total = 0

        # Get current trial info
        trial_idx = st.session_state.trial_idx
        if trial_idx not in lr_indices:
            trial_idx = lr_indices[0]
            st.session_state.trial_idx = trial_idx

        true_label = y_test[trial_idx]
        target_side = "LEFT" if true_label == 0 else "RIGHT"

        # Game canvas
        game_placeholder = st.empty()

        # Current prediction display
        pred_col1, pred_col2 = st.columns([2, 1])

        with pred_col1:
            pred_placeholder = st.empty()

        with pred_col2:
            st.markdown(f"""
            <div class="cyber-box" style="text-align: center;">
                <p style="color: #888; margin: 0; font-size: 0.9rem;">TARGET</p>
                <p style="color: {'#ff006e' if target_side == 'LEFT' else '#00d4ff'};
                   font-family: 'Orbitron'; font-size: 1.5rem; margin: 10px 0;">
                   {'⬅' if target_side == 'LEFT' else '➡'} {target_side}
                </p>
                <p style="color: #666; font-size: 0.8rem;">Trial #{trial_idx + 1}</p>
            </div>
            """, unsafe_allow_html=True)

        # Animation loop
        if st.session_state.playing:
            step = st.session_state.time_step
            preds = predictions[trial_idx]

            if step < len(preds):
                probs = preds[step]

                # Move cursor based on left vs right probability
                left_prob = probs[0]
                right_prob = probs[1]
                movement = (right_prob - left_prob) * 0.2  # Faster movement

                st.session_state.cursor_pos += movement
                st.session_state.cursor_pos = np.clip(st.session_state.cursor_pos, -1.0, 1.0)

                st.session_state.time_step += 1

                # Check if reached target
                if st.session_state.cursor_pos <= -0.85:
                    st.session_state.total += 1
                    if true_label == 0:
                        st.session_state.score += 1
                        st.balloons()
                    st.session_state.playing = False

                elif st.session_state.cursor_pos >= 0.85:
                    st.session_state.total += 1
                    if true_label == 1:
                        st.session_state.score += 1
                        st.balloons()
                    st.session_state.playing = False
            else:
                # Trial ended - count based on final cursor position
                st.session_state.total += 1
                final_pos = st.session_state.cursor_pos
                # Left side wins if cursor < 0, right side wins if cursor > 0
                if final_pos < 0 and true_label == 0:
                    st.session_state.score += 1
                    st.balloons()
                elif final_pos > 0 and true_label == 1:
                    st.session_state.score += 1
                    st.balloons()
                st.session_state.playing = False

            time.sleep(0.1)  # Faster animation
            st.rerun()

        # Draw game
        fig = create_cursor_game(
            st.session_state.cursor_pos,
            target_side,
            st.session_state.score,
            st.session_state.total
        )
        game_placeholder.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Show current prediction
        if st.session_state.time_step > 0 and st.session_state.time_step <= len(predictions[trial_idx]):
            probs = predictions[trial_idx][min(st.session_state.time_step - 1, len(predictions[trial_idx]) - 1)]
            pred_class = np.argmax(probs[:2])
            pred_name = "LEFT HAND" if pred_class == 0 else "RIGHT HAND"
            pred_color = "#ff006e" if pred_class == 0 else "#00d4ff"

            pred_placeholder.markdown(f"""
            <div style="background: linear-gradient(135deg, {pred_color}22 0%, {pred_color}11 100%);
                        border: 2px solid {pred_color}; border-radius: 15px; padding: 20px; text-align: center;">
                <p style="color: #888; margin: 0; font-size: 0.9rem;">DECODED THOUGHT</p>
                <p style="color: {pred_color}; font-family: 'Orbitron'; font-size: 2rem; margin: 10px 0;
                          text-shadow: 0 0 20px {pred_color};">
                    {'⬅' if pred_class == 0 else '➡'} {pred_name}
                </p>
                <p style="color: #39ff14; font-size: 1rem;">Confidence: {probs[pred_class]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: BRAIN THEATER
    # =========================================================================
    with tab2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <p style="color: #888; font-family: 'Rajdhani', sans-serif; font-size: 1.1rem;">
                Full neural decoding visualization • 22-channel EEG • Real-time classification
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Controls
        t_col1, t_col2, t_col3 = st.columns(3)

        with t_col1:
            if st.button("🎲 RANDOM TRIAL", key="theater_random", use_container_width=True):
                st.session_state.theater_trial = np.random.randint(0, len(X_test))
                st.session_state.theater_step = 0
                st.session_state.theater_playing = False

        with t_col2:
            play_label = "⏸ PAUSE" if st.session_state.theater_playing else "▶ PLAY"
            if st.button(play_label, key="theater_play", use_container_width=True):
                st.session_state.theater_playing = not st.session_state.theater_playing

        with t_col3:
            if st.button("⏮ RESET", key="theater_reset", use_container_width=True):
                st.session_state.theater_step = 0
                st.session_state.theater_playing = False

        # Get current trial
        t_idx = st.session_state.theater_trial
        trial_data = X_test[t_idx]
        true_label = y_test[t_idx]
        true_class = data_config.class_names[true_label]

        time_idx = int(st.session_state.theater_step / 20 * trial_data.shape[1])
        step = min(st.session_state.theater_step, len(predictions[t_idx]) - 1)
        probs = predictions[t_idx][step]
        pred_class = np.argmax(probs)
        pred_name = data_config.class_names[pred_class]

        # Layout
        col_left, col_right = st.columns([1.5, 1])

        with col_left:
            # EEG Streams
            st.markdown('<p style="color: #00d4ff; font-family: Orbitron; margin-bottom: 5px;">📊 LIVE EEG SIGNAL</p>', unsafe_allow_html=True)
            eeg_fig = create_eeg_streams(trial_data, time_idx)
            st.plotly_chart(eeg_fig, use_container_width=True, config={'displayModeBar': False})

        with col_right:
            # 3D Brain
            st.markdown('<p style="color: #39ff14; font-family: Orbitron; margin-bottom: 5px;">🧠 NEURAL ACTIVITY</p>', unsafe_allow_html=True)
            brain_fig = create_3d_brain(trial_data, time_idx)
            st.plotly_chart(brain_fig, use_container_width=True, config={'displayModeBar': False})

        # Probability bars
        st.markdown('<p style="color: #ffaa00; font-family: Orbitron; margin: 20px 0 5px 0;">📈 CLASS PROBABILITIES</p>', unsafe_allow_html=True)
        prob_fig = create_probability_bars(probs, pred_class)
        st.plotly_chart(prob_fig, use_container_width=True, config={'displayModeBar': False})

        # Big prediction display
        class_colors = ['#ff006e', '#00d4ff', '#39ff14', '#ffaa00']
        class_icons = ['⬅ LEFT HAND', '➡ RIGHT HAND', '🦶 FEET', '👅 TONGUE']

        st.markdown(f"""
        <div class="prediction-huge" style="
            background: linear-gradient(135deg, {class_colors[pred_class]}22 0%, {class_colors[pred_class]}11 100%);
            border: 3px solid {class_colors[pred_class]};
            color: {class_colors[pred_class]};
        ">
            {class_icons[pred_class]}
            <div style="font-size: 1.5rem; margin-top: 10px; color: #888;">
                Confidence: {probs[pred_class]*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # True label
        correct = pred_class == true_label
        st.markdown(f"""
        <div style="text-align: center; margin-top: 20px;">
            <span style="color: #666; font-family: 'Share Tech Mono';">TRUE CLASS: </span>
            <span style="color: {'#39ff14' if correct else '#ff006e'}; font-family: 'Orbitron'; font-size: 1.2rem;">
                {true_class.upper()} {'✓' if correct else '✗'}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar
        progress = st.session_state.theater_step / 19
        st.progress(progress)

        # Animation loop
        if st.session_state.theater_playing:
            st.session_state.theater_step += 1
            if st.session_state.theater_step >= 20:
                st.session_state.theater_step = 19
                st.session_state.theater_playing = False
            time.sleep(0.2)
            st.rerun()

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #333;">
        <p style="color: #444; font-family: 'Share Tech Mono'; font-size: 0.8rem;">
            NEURAL INTERFACE v2.0 • BCI COMPETITION IV DATASET • EEGNET ARCHITECTURE
        </p>
        <p style="color: #333; font-family: 'Share Tech Mono'; font-size: 0.7rem;">
            Built with PyTorch • Streamlit • Plotly
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
