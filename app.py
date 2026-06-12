import streamlit as st
import os

# Page config
st.set_page_config(
    page_title="SASRec Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main title styling */
    .main-title {
        background: linear-gradient(90deg, #A855F7 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #94A3B8;
        font-size: 1.25rem;
        margin-bottom: 2rem;
    }
    
    /* Card style */
    .card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        min-height: 180px;
    }
    
    .card:hover {
        transform: translateY(-5px);
        border-color: #A855F7;
        box-shadow: 0 10px 20px rgba(168, 85, 247, 0.15);
    }
    
    .card-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .card-title {
        color: #F8FAFC;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .card-desc {
        color: #94A3B8;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Status Badge styling */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.4rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .status-trained {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-untrained {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .pulse {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .pulse-green {
        background-color: #10B981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse-green-anim 2s infinite;
    }
    
    .pulse-red {
        background-color: #EF4444;
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
        animation: pulse-red-anim 2s infinite;
    }
    
    @keyframes pulse-green-anim {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }
    
    @keyframes pulse-red-anim {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 6px rgba(239, 68, 68, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {"NDCG@10": 0.0, "Recall@10": 0.0, "Loss": 0.0}

# Sidebar Content
with st.sidebar:
    # App Logo and Name
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_column_width=True)
    else:
        st.markdown("## SASRec Studio")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Status Badge
    st.markdown("### Model Status")
    if st.session_state.model_trained:
        st.markdown(
            '<div class="status-badge status-trained"><span class="pulse pulse-green"></span>Model Trained</div>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-badge status-untrained"><span class="pulse pulse-red"></span>Not Trained</div>', 
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    
    # Navigation Info
    st.markdown("### Available Sections")
    st.markdown(
        """
        - **Home**
        - **Data Preprocessing**
        - **Model Training**
        - **Predictions & Recommendations**
        - **Attention Visualizer**
        - **Evaluation & Metrics**
        """
    )
    
    st.markdown("---")
    st.markdown("*SASRec Studio v1.0.0*")

# Main Header Layout
col_header_logo, col_header_text = st.columns([1, 8])
with col_header_logo:
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=90)
    else:
        st.markdown("<h1>SASRec</h1>", unsafe_allow_html=True)
with col_header_text:
    st.markdown('<div class="main-title">SASRec Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Sequential Product Recommender Engine</div>', unsafe_allow_html=True)

# Project Description (in English)
st.markdown("### Project Overview")
st.markdown(
    """
    **SASRec Studio** is an interactive experimentation environment for sequential product recommendation. 
    The application is based on the **SASRec** (*Self-Attentive Sequential Recommendation*) architecture, which uses self-attention mechanisms
    to model user purchase histories and predict the next item of interest.
    
    Unlike traditional collaborative filtering or simplified Markov-chain-based approaches,
    SASRec is capable of dynamically capturing both short-term temporal dependencies and long-term
    user preferences to deliver highly relevant and contextualized recommendations.
    """
)

st.markdown("---")

# Global Stats Section
st.markdown("### Global Project Statistics")
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

# Dynamic dataset information loading
import pickle
processed_path = os.path.join("data", "processed.pkl")
dataset_name = "Not Preprocessed"
volume_str = "N/A"
delta_dataset = "Please run step 1"
delta_volume = "No processed.pkl file"

if os.path.exists(processed_path):
    try:
        with open(processed_path, 'rb') as f:
            proc_data = pickle.load(f)
        
        raw_name = proc_data.get('dataset_name', 'ratings_Electronics.csv')
        if "Datafiniti" in raw_name:
            dataset_name = "Amazon Consumer Reviews"
        else:
            dataset_name = "Amazon Electronics"
        delta_dataset = f"Source: {raw_name}"
        
        raw_stats = proc_data.get('raw_stats', {})
        if raw_stats:
            raw_rows = raw_stats.get('num_rows', 0)
            if raw_rows > 1000000:
                volume_str = f"{raw_rows / 1000000:.2f}M interactions"
            else:
                volume_str = f"{raw_rows:,} interactions"
        else:
            if "Datafiniti" in raw_name:
                volume_str = "5,000 interactions"
            else:
                volume_str = "7.82M interactions"
                
        if os.path.exists(raw_name):
            size_mb = os.path.getsize(raw_name) / (1024 * 1024)
            delta_volume = f"File size: {size_mb:.2f} MB"
        else:
            delta_volume = "Source file not found"
    except Exception:
        dataset_name = "Amazon Electronics"
        volume_str = "7.82M interactions"
        delta_dataset = "Source: ratings_Electronics.csv"
        delta_volume = "File size: 318 MB"

with stat_col1:
    st.metric(
        label="Selected Dataset",
        value=dataset_name,
        delta=delta_dataset
    )
with stat_col2:
    st.metric(
        label="Data Volume",
        value=volume_str,
        delta=delta_volume
    )
with stat_col3:
    st.metric(
        label="Model Architecture",
        value="SASRec (Self-Attention)",
        delta="Causal Transformer"
    )
with stat_col4:
    if st.session_state.model_trained:
        ndcg_val = f"{st.session_state.model_metrics['NDCG@10']:.4f}"
        delta_val = "Model ready"
    else:
        ndcg_val = "N/A"
        delta_val = "Awaiting training"
    st.metric(
        label="Target NDCG@10 (Test)",
        value=ndcg_val,
        delta=delta_val
    )

st.markdown("---")

# 4 Available Pages Grid (2x2)
st.markdown("### Available Pages in the Studio")

p_col1, p_col2 = st.columns(2)

with p_col1:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">1. Data Preprocessing</div>
            <div class="card-desc">
                Load raw product ratings, observe distributions, filter infrequent users and products (k-core filtering), and generate chronologically sorted sequences for training.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="card">
            <div class="card-title">3. Predictions & Recommendations</div>
            <div class="card-desc">
                Simulate real-time recommendations for any user profile. Enter custom sequences of viewed or purchased items and watch dynamic predictions update instantly.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with p_col2:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">2. Model Training</div>
            <div class="card-desc">
                Configure hyperparameters (number of self-attention heads, hidden dimensions, batch size, dropout rate). Track real-time training loss and manage model checkpoints.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="card">
            <div class="card-title">4. Attention Visualizer</div>
            <div class="card-desc">
                Inspect attention maps showing how past item interactions influence predictions. Navigate across layers and heads of the multi-head attention module.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
