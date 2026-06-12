import streamlit as st
import os
import pickle
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.sasrec import SASRec
from utils.metrics import evaluate_model

# Page config
st.set_page_config(
    page_title="Model Training — SASRec Studio",
    layout="wide"
)

# Custom CSS for premium styling
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
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# PyTorch Dataset for SASRec Training
class SASRecTrainingDataset(Dataset):
    def __init__(self, train_seqs, num_items, max_len=50):
        self.user_ids = list(train_seqs.keys())
        self.train_seqs = train_seqs
        self.num_items = num_items
        self.max_len = max_len
        
    def __len__(self):
        return len(self.user_ids)
        
    def __getitem__(self, index):
        user_id = self.user_ids[index]
        seq = self.train_seqs[user_id]
        
        # Pad & Truncate sequence to max_len + 1 (for input/target shift)
        # s_1, s_2, ..., s_n
        seq_len = len(seq)
        
        # SASRec expects an input sequence of at most max_len
        # Extract subsequence
        if seq_len > self.max_len:
            sub_seq = seq[-self.max_len:]
        else:
            sub_seq = [0] * (self.max_len - seq_len) + seq
            
        # Temporal shift:
        # input_ids: seq[:-1]
        # pos_targets: seq[1:]
        input_ids = np.array(sub_seq[:-1], dtype=np.int64)
        pos_targets = np.array(sub_seq[1:], dtype=np.int64)
        
        # Negative sampling for each element of the sequence
        neg_targets = []
        for pos in pos_targets:
            if pos == 0:
                neg_targets.append(0)
            else:
                neg = np.random.randint(1, self.num_items + 1)
                while neg == pos:
                    neg = np.random.randint(1, self.num_items + 1)
                neg_targets.append(neg)
        neg_targets = np.array(neg_targets, dtype=np.int64)
        
        return torch.tensor(input_ids), torch.tensor(pos_targets), torch.tensor(neg_targets)

# Function to run one single epoch of training
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0
    
    for batch in dataloader:
        input_ids, pos_targets, neg_targets = batch
        input_ids = input_ids.to(device)
        pos_targets = pos_targets.to(device)
        neg_targets = neg_targets.to(device)
        
        optimizer.zero_grad()
        
        # features shape: (batch_size, seq_len-1, hidden_size)
        feats = model(input_ids)
        
        # Positive and negative item embeddings
        pos_embs = model.item_emb(pos_targets) # (batch_size, seq_len-1, hidden_size)
        neg_embs = model.item_emb(neg_targets) # (batch_size, seq_len-1, hidden_size)
        
        # Affinity logits
        pos_logits = (feats * pos_embs).sum(dim=-1) # (batch_size, seq_len-1)
        neg_logits = (feats * neg_embs).sum(dim=-1) # (batch_size, seq_len-1)
        
        # Calculate loss only for non-padded elements (pos_targets != 0)
        non_pad_mask = (pos_targets != 0)
        
        pos_loss = -torch.log(torch.sigmoid(pos_logits[non_pad_mask]) + 1e-24)
        neg_loss = -torch.log(1.0 - torch.sigmoid(neg_logits[non_pad_mask]) + 1e-24)
        
        loss = (pos_loss + neg_loss).mean()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
        
    return total_loss / count if count > 0 else 0.0

# Verify dataset availability
data_path = os.path.join("data", "processed.pkl")

# Sidebar - Hyperparameters Layout
with st.sidebar:
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_column_width=True)
    else:
        st.markdown("## SASRec Studio")
        
    st.markdown("---")
    st.markdown("### Hyperparameters")
    
    num_epochs = st.slider("Number of Epochs", 5, 100, 10)
    hidden_size = st.selectbox("Hidden Size (hidden_size)", [32, 64, 128], index=1)
    num_heads = st.selectbox("Number of Heads (num_heads)", [1, 2, 4], index=1)
    num_layers = st.selectbox("Number of Layers (num_layers)", [1, 2, 3], index=1)
    learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0005, format="%.4f")
    batch_size = st.selectbox("Batch Size", [64, 128, 256], index=1)

    st.markdown("---")
    st.markdown("*SASRec Studio v1.0.0*")

# Main content header
st.markdown('<div class="page-title">Model Training</div>', unsafe_allow_html=True)

if not os.path.exists(data_path):
    st.warning("Preprocessed dataset 'data/processed.pkl' not found. Please run Step 1 (Data Preprocessing) first.")
else:
    # Initialize session states for training control
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    if 'current_epoch' not in st.session_state:
        st.session_state.current_epoch = 0
    if 'epoch_history' not in st.session_state:
        st.session_state.epoch_history = []
    if 'train_model' not in st.session_state:
        st.session_state.train_model = None
    if 'train_optimizer' not in st.session_state:
        st.session_state.train_optimizer = None

    # Load processed data
    with open(data_path, 'rb') as f:
        dataset_data = pickle.load(f)
        
    train_seqs = dataset_data['train']
    val_seqs = dataset_data['val']
    test_seqs = dataset_data['test']
    item2id = dataset_data['item2id']
    all_items = list(item2id.values())
    num_items = len(item2id)
    
    st.info(f"Preprocessed dataset loaded: **{len(train_seqs)}** users and **{num_items}** unique products.")

    # Control buttons (Start / Stop)
    col_ctrl1, col_ctrl2 = st.columns([1, 8])
    with col_ctrl1:
        if not st.session_state.is_training:
            if st.button("Start Training", type="primary"):
                # Reset training state
                st.session_state.is_training = True
                st.session_state.training_completed = False
                st.session_state.current_epoch = 0
                st.session_state.epoch_history = []
                
                # Instantiate PyTorch Model
                st.session_state.train_model = SASRec(
                    num_items=num_items,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    max_len=50,
                    dropout=0.2
                )
                
                # Optimizer
                st.session_state.train_optimizer = torch.optim.Adam(
                    st.session_state.train_model.parameters(), 
                    lr=learning_rate
                )
                st.rerun()
        else:
            if st.button("Stop", type="secondary", help="Stop training after this epoch"):
                st.session_state.is_training = False
                st.info("Training interrupted by user.")
                st.rerun()

    # Metric placeholders
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    with col_metric1:
        metric_epoch = st.empty()
    with col_metric2:
        metric_loss = st.empty()
    with col_metric3:
        metric_hr = st.empty()
    with col_metric4:
        metric_ndcg = st.empty()

    # Progress bar placeholder
    progress_bar = st.progress(0)
    
    # Chart placeholder
    chart_placeholder = st.empty()

    # Status Message
    status_msg = st.empty()

    # Training state logic
    if st.session_state.is_training:
        # Run one epoch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Prepare PyTorch DataLoader
        train_dataset = SASRecTrainingDataset(train_seqs, num_items, max_len=50)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        epoch = st.session_state.current_epoch
        model = st.session_state.train_model
        optimizer = st.session_state.train_optimizer
        
        status_msg.text(f"Training epoch {epoch + 1}/{num_epochs} on {device}...")
        
        # Run epoch training
        loss_val = train_one_epoch(model, train_loader, optimizer, device)
        
        # Evaluate model on validation set
        # Reformat val_seqs for evaluation: for each user, val input is seq_train, target is item_val
        val_data_eval = {}
        for u in val_seqs:
            if val_seqs[u] is not None:
                val_data_eval[u] = (train_seqs[u], val_seqs[u])
                
        # Evaluate on validation data
        val_metrics = evaluate_model(model, val_data_eval, all_items, device, k=10)
        
        hr10_val = val_metrics['HR@10']
        ndcg10_val = val_metrics['NDCG@10']
        hr5_val = val_metrics['HR@5']
        ndcg5_val = val_metrics['NDCG@5']
        
        # Save historical metrics
        st.session_state.epoch_history.append({
            'Epoch': epoch + 1,
            'Loss': loss_val,
            'HR@10': hr10_val,
            'NDCG@10': ndcg10_val,
            'HR@5': hr5_val,
            'NDCG@5': ndcg5_val
        })
        
        # Increment epoch
        st.session_state.current_epoch += 1
        
        # Check if training completed
        if st.session_state.current_epoch >= num_epochs:
            st.session_state.is_training = False
            st.session_state.training_completed = True
            
            # Save model to disk
            model_save_path = os.path.join("data", "sasrec_model.pt")
            torch.save(model.state_dict(), model_save_path)
            
            # Save model config for dynamic loading in pages 3 and 4
            config_save_path = os.path.join("data", "model_config.pkl")
            try:
                with open(config_save_path, 'wb') as f_cfg:
                    pickle.dump({
                        'hidden_size': hidden_size,
                        'num_heads': num_heads,
                        'num_layers': num_layers,
                        'max_len': 50
                    }, f_cfg)
            except Exception as e_cfg:
                st.warning(f"Could not save model configuration: {e_cfg}")
            
            # Save global state metrics to share with landing/eval page
            st.session_state.model_trained = True
            st.session_state.model_metrics = {
                "Loss": loss_val,
                "NDCG@10": ndcg10_val,
                "Recall@10": hr10_val, # Standard notation in recommender papers
            }
            
            st.balloons()
            
        # Rerun Streamlit page to show updates
        st.rerun()

    # Draw metrics and charts from history if they exist
    history = st.session_state.get('epoch_history', [])
    if len(history) > 0:
        latest = history[-1]
        
        # Update metrics cards
        with col_metric1:
            st.markdown(f'<div class="metric-card"><h5>Epoch</h5><h3>{latest["Epoch"]}/{num_epochs}</h3></div>', unsafe_allow_html=True)
        with col_metric2:
            st.markdown(f'<div class="metric-card"><h5>Loss</h5><h3>{latest["Loss"]:.4f}</h3></div>', unsafe_allow_html=True)
        with col_metric3:
            st.markdown(f'<div class="metric-card"><h5>HR@10 (Val)</h5><h3>{latest["HR@10"]:.4f}</h3></div>', unsafe_allow_html=True)
        with col_metric4:
            st.markdown(f'<div class="metric-card"><h5>NDCG@10 (Val)</h5><h3>{latest["NDCG@10"]:.4f}</h3></div>', unsafe_allow_html=True)
            
        # Update progress bar
        progress_bar.progress(latest["Epoch"] / num_epochs)
        
        # Draw Plotly Live Chart
        epochs_x = [h['Epoch'] for h in history]
        losses_y = [h['Loss'] for h in history]
        hr_y = [h['HR@10'] for h in history]
        ndcg_y = [h['NDCG@10'] for h in history]
        
        # Make secondary Y axis subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=epochs_x, y=losses_y, name="Loss", line=dict(color="#EF4444", width=3)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=epochs_x, y=hr_y, name="HR@10 (Val)", line=dict(color="#10B981", width=3)),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=epochs_x, y=ndcg_y, name="NDCG@10 (Val)", line=dict(color="#06B6D4", width=2, dash='dash')),
            secondary_y=True,
        )
        
        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            yaxis2_title="Evaluation Metrics",
            legend=dict(x=0.01, y=0.99, orientation="h"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=400
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)

    # Show Final Summary Metrics Table
    if st.session_state.get('training_completed', False) and len(history) > 0:
        latest = history[-1]
        st.success(f"Training completed! The model has been saved automatically to `data/sasrec_model.pt`.")
        
        st.subheader("Final Metrics Summary (Validation)")
        
        summary_df = pd.DataFrame({
            "Metric": ["Training Loss", "Hit Rate @ 10 (HR@10)", "NDCG @ 10 (NDCG@10)", "Hit Rate @ 5 (HR@5)", "NDCG @ 5 (NDCG@5)"],
            "Value": [
                f"{latest['Loss']:.6f}",
                f"{latest['HR@10']:.4f}",
                f"{latest['NDCG@10']:.4f}",
                f"{latest['HR@5']:.4f}",
                f"{latest['NDCG@5']:.4f}"
            ]
        })
        st.table(summary_df)
