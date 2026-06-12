import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go

from models.sasrec import SASRec

# Page config
st.set_page_config(
    page_title="Attention Visualizer — SASRec Studio",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .page-title {
        background: linear-gradient(90deg, #A855F7 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
    }
    .explanation-text {
        font-size: 0.95rem;
        color: #94A3B8;
        background-color: #1E293B;
        border-left: 4px solid #A855F7;
        padding: 0.8rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache raw CSV loading for user history details (invalidates cache if file changes on disk)
@st.cache_data
def load_raw_data_cached(raw_filename, mtime):
    if not raw_filename or not os.path.exists(raw_filename):
        return None
    try:
        from utils.preprocessing import load_dataset
        df = load_dataset(raw_filename)
        return df
    except Exception:
        return None

# Sidebar layout
with st.sidebar:
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_column_width=True)
    else:
        st.markdown("## SASRec Studio")
        
    st.markdown("---")
    st.markdown("### Model Status")
    if st.session_state.get('model_trained', False) or os.path.exists(os.path.join("data", "sasrec_model.pt")):
        st.success("Model Loaded")
    else:
        st.error("Model Unavailable")
        
    st.markdown("---")
    st.markdown("*SASRec Studio v1.0.0*")

# Main page title
st.markdown('<div class="page-title">Attention Visualizer (Self-Attention)</div>', unsafe_allow_html=True)

model_path = os.path.join("data", "sasrec_model.pt")
data_path = os.path.join("data", "processed.pkl")

# Check file availability
if not os.path.exists(model_path):
    st.warning("No trained model weights found. Please train the model first in the **2. Model Training** section.")
elif not os.path.exists(data_path):
    st.warning("Preprocessed data file `data/processed.pkl` not found.")
else:
    # Load processed mapping and sequences
    with open(data_path, 'rb') as f:
        dataset_data = pickle.load(f)
        
    train_seqs = dataset_data['train']
    val_seqs = dataset_data['val']
    item2id = dataset_data['item2id']
    id2item = dataset_data['id2item']
    item2name = dataset_data.get('item2name', {})
    num_items = len(item2id)
    
    # Load model configuration to avoid mismatch errors if hyperparameters were changed during training
    config_path = os.path.join("data", "model_config.pkl")
    hidden_size = 64
    num_heads = 2
    num_layers = 2
    max_len = 50
    if os.path.exists(config_path):
        try:
            with open(config_path, 'rb') as f_cfg:
                config = pickle.load(f_cfg)
                hidden_size = config.get('hidden_size', hidden_size)
                num_heads = config.get('num_heads', num_heads)
                num_layers = config.get('num_layers', num_layers)
                max_len = config.get('max_len', max_len)
        except Exception:
            pass
            
    # Load model cache
    @st.cache_resource
    def load_sasrec_model(num_items, model_mtime, hidden_size, num_heads, num_layers, max_len):
        model = SASRec(
            num_items=num_items,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            max_len=max_len,
            dropout=0.2
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
        
    try:
        model_mtime = os.path.getmtime(model_path)
        model = load_sasrec_model(num_items, model_mtime, hidden_size, num_heads, num_layers, max_len)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        
    # UI: Select User ID
    user_list = sorted(list(train_seqs.keys()))
    
    col_input1, col_input2, col_input3 = st.columns([2, 1, 1])
    with col_input1:
        selected_user = st.selectbox("Select user to analyze:", user_list)
    with col_input2:
        selected_layer = st.selectbox("Attention Layer:", list(range(1, num_layers + 1))) - 1
    with col_input3:
        selected_head = st.selectbox("Attention Head:", list(range(1, num_heads + 1))) - 1
        
    # Extract chronological sequence (history + validation if exists)
    user_seq = train_seqs[selected_user].copy()
    if val_seqs.get(selected_user) is not None:
        user_seq.append(val_seqs[selected_user])
        
    # Limit to last N items (max_len = 50)
    actual_len = len(user_seq)
    if actual_len > max_len:
        user_seq = user_seq[-max_len:]
        actual_len = max_len
        
    st.info(f"Analyzing interaction history: **{actual_len}** products viewed chronologically.")
    
    # Retrieve product names/IDs
    user_pids = [id2item[idx] for idx in user_seq if idx in id2item]
    display_names = [f"{i+1}. {item2name.get(pid, pid)[:15]} ({pid})" for i, pid in enumerate(user_pids)]
    
    # Padding input for model
    padded_seq = [0] * (max_len - actual_len) + user_seq
    input_tensor = torch.tensor([padded_seq], dtype=torch.long)
    
    # Retrieve multihead attention weights
    # List of length num_layers, each having shape (1, num_heads, max_len, max_len)
    with torch.no_grad():
        attn_weights = model.get_attention_weights(input_tensor)
        
    # Get the matrix of the selected layer and head
    # Shape: (max_len, max_len)
    matrix = attn_weights[selected_layer][0, selected_head].numpy()
    
    # Slice the padding. The actual sequence is at the end: from index (max_len - actual_len) to (max_len - 1)
    sliced_matrix = matrix[-actual_len:, -actual_len:]
    
    # Visualizations layout
    col_plot1, col_plot2 = st.columns([1, 1])
    
    with col_plot1:
        st.subheader("Self-Attention Weights Matrix")
        
        # Plotly Heatmap
        # x is keys (what is attended to), y is queries (what is attending)
        hover_text = []
        for q_idx, q_pid in enumerate(user_pids):
            row_hover = []
            for k_idx, k_pid in enumerate(user_pids):
                q_name = item2name.get(q_pid, q_pid)
                k_name = item2name.get(k_pid, k_pid)
                weight = sliced_matrix[q_idx, k_idx]
                row_hover.append(
                    f"Query: {q_pid}<br>{q_name}<br><br>"
                    f"Key: {k_pid}<br>{k_name}<br><br>"
                    f"Attention weight: {weight:.4f}"
                )
            hover_text.append(row_hover)

        fig_heat = go.Figure(data=go.Heatmap(
            z=sliced_matrix,
            x=display_names,
            y=display_names,
            text=hover_text,
            hoverinfo="text",
            colorscale='Viridis',
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Attention")
        ))
        
        fig_heat.update_layout(
            template="plotly_dark",
            xaxis_title="Keys (Past viewed products)",
            yaxis_title="Queries (Sequence products)",
            margin=dict(l=10, r=10, t=30, b=10),
            height=450
        )
        
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown(
            '<div class="explanation-text"><b>Explanation:</b> Items in bright yellow/green have the highest influence on recommendations. '
            'The lower-triangular structure is causal: each product can only attend to products viewed before it.</div>', 
            unsafe_allow_html=True
        )
        
    with col_plot2:
        st.subheader("Influence of past items on the final prediction")
        
        # The last row of the sliced matrix represents how much the model attends
        # to past items to predict the next item at the end of the sequence
        last_row = sliced_matrix[-1, :]
        
        # Build hover text for bar chart
        bar_hover = []
        for pid, weight in zip(user_pids, last_row):
            name = item2name.get(pid, pid)
            bar_hover.append(f"Product: {pid}<br>{name}<br>Attention weight: {weight:.4f}")

        # Plotly Bar Chart with color scale based on values
        fig_bar = go.Figure(data=go.Bar(
            x=display_names,
            y=last_row,
            hovertext=bar_hover,
            hoverinfo="text",
            marker=dict(
                color=last_row,
                colorscale='YlOrRd', # Red/Orange color scale for visual highlight
                cmin=0.0,
                cmax=1.0
            ),
            text=[f"{val:.3f}" for val in last_row],
            textposition='auto'
        ))
        
        fig_bar.update_layout(
            template="plotly_dark",
            xaxis_title="Historical products",
            yaxis_title="Attention weight",
            margin=dict(l=10, r=10, t=30, b=10),
            height=450
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown(
            '<div class="explanation-text"><b>Explanation:</b> Red items have the highest influence on the final product '
            'recommendation. The sum of attention weights equals 1.0 (Softmax).</div>', 
            unsafe_allow_html=True
        )
        
    # Compare all layers side-by-side
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Compare All Attention Layers", type="secondary"):
        st.markdown("### Layer Comparison (Current Attention Head)")
        cols_layers = st.columns(num_layers)
        
        for l_idx in range(num_layers):
            with cols_layers[l_idx]:
                st.markdown(f"**Layer {l_idx + 1}**")
                
                # Extract matrix
                l_matrix = attn_weights[l_idx][0, selected_head].numpy()
                l_sliced = l_matrix[-actual_len:, -actual_len:]
                
                l_hover_text = []
                for q_idx, q_pid in enumerate(user_pids):
                    l_row_hover = []
                    for k_idx, k_pid in enumerate(user_pids):
                        q_name = item2name.get(q_pid, q_pid)
                        k_name = item2name.get(k_pid, k_pid)
                        weight = l_sliced[q_idx, k_idx]
                        l_row_hover.append(
                            f"Query: {q_pid}<br>{q_name}<br><br>"
                            f"Key: {k_pid}<br>{k_name}<br><br>"
                            f"Attention weight: {weight:.4f}"
                        )
                    l_hover_text.append(l_row_hover)

                fig_comp = go.Figure(data=go.Heatmap(
                    z=l_sliced,
                    x=display_names,
                    y=display_names,
                    text=l_hover_text,
                    hoverinfo="text",
                    colorscale='Magma',
                    zmin=0.0,
                    zmax=1.0,
                    showscale=False
                ))
                
                fig_comp.update_layout(
                    template="plotly_dark",
                    margin=dict(l=5, r=5, t=10, b=5),
                    height=300
                )
                
                st.plotly_chart(fig_comp, use_container_width=True, key=f"comp_layer_{l_idx}")
