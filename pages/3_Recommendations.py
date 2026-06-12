import streamlit as st
import os
import pickle
import pandas as pd
import torch
import plotly.graph_objects as go

from models.sasrec import SASRec

# Page config
st.set_page_config(
    page_title="Recommendations — SASRec Studio",
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
        margin-bottom: 1.5rem;
    }
    .rec-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .rec-card:hover {
        transform: translateY(-2px);
        border-color: #06B6D4;
    }
    .rec-rank {
        font-weight: bold;
        font-size: 1.15rem;
        color: #A855F7;
    }
    .rec-id {
        color: #F8FAFC;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .rec-score {
        float: right;
        font-weight: bold;
        color: #06B6D4;
    }
</style>
""", unsafe_allow_html=True)

# Cache raw CSV loading to speed up history retrieval (invalidates cache if file changes on disk)
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
st.markdown('<div class="page-title">Predictions & Recommendations</div>', unsafe_allow_html=True)

model_path = os.path.join("data", "sasrec_model.pt")
data_path = os.path.join("data", "processed.pkl")

# Check if required files exist
if not os.path.exists(model_path):
    st.warning("No trained model weights found in `data/sasrec_model.pt`. Please train the model first in the **2. Model Training** section.")
elif not os.path.exists(data_path):
    st.warning("Preprocessed data file `data/processed.pkl` not found. Please complete step 1 (Data Preprocessing) first.")
else:
    # Load processed data
    with open(data_path, 'rb') as f:
        dataset_data = pickle.load(f)
        
    train_seqs = dataset_data['train']
    val_seqs = dataset_data['val']
    item2id = dataset_data['item2id']
    id2item = dataset_data['id2item']
    item2name = dataset_data.get('item2name', {})
    num_items = len(item2id)
    raw_filename = dataset_data.get('dataset_name', 'ratings_Electronics.csv')
    
    # Rebuild item2name from raw CSV if it is empty
    if not item2name:
        raw_mtime = os.path.getmtime(raw_filename) if os.path.exists(raw_filename) else 0
        raw_df = load_raw_data_cached(raw_filename, raw_mtime)
        if raw_df is not None and 'name' in raw_df.columns:
            product_names = raw_df[['productId', 'name']].dropna().drop_duplicates(subset=['productId'])
            item2name = dict(zip(product_names['productId'], product_names['name']))
            
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

    # Load Model Weights
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

    # Form with modes
    mode = st.radio(
        "Choose recommendation mode:",
        ["Existing User", "Manual Sequence"],
        horizontal=True
    )

    sequence_to_recommend = []
    selected_user = None

    if mode == "Existing User":
        st.subheader("Existing User Mode")
        
        # Select User ID from list of users
        user_list = sorted(list(train_seqs.keys()))
        selected_user = st.selectbox("Select user ID (userId):", user_list)
        
        # Load user history from processed sequence
        user_history = train_seqs[selected_user].copy()
        if val_seqs.get(selected_user) is not None:
            user_history.append(val_seqs[selected_user])
            
        # Display chronological history
        st.markdown("### Chronological Purchase History")
        
        # Convert internal IDs to productIds strings
        user_history_pids = [id2item[idx] for idx in user_history if idx in id2item]
        
        # Try loading raw data to display exact timestamps
        raw_mtime = os.path.getmtime(raw_filename) if os.path.exists(raw_filename) else 0
        raw_df = load_raw_data_cached(raw_filename, raw_mtime)
        if raw_df is not None:
            user_raw_df = raw_df[raw_df['userId'] == selected_user].sort_values(by='timestamp').copy()
            user_raw_df['Purchase Date'] = pd.to_datetime(user_raw_df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Populate Product Name from name column or map from item2name dict
            if 'name' in user_raw_df.columns:
                user_raw_df['Product Name'] = user_raw_df['name']
            else:
                user_raw_df['Product Name'] = user_raw_df['productId'].map(lambda pid: item2name.get(pid, pid))
                
            cols_to_show = ['productId', 'Product Name', 'Purchase Date']
            
            # Rename columns for display
            display_cols = {
                'productId': 'Product ID',
                'Product Name': 'Product Name',
                'Purchase Date': 'Purchase Date'
            }
            user_raw_df_rename = user_raw_df[cols_to_show].rename(columns=display_cols)
            st.dataframe(user_raw_df_rename, use_container_width=True)
        else:
            # Fallback: display list of products
            history_rows = []
            for idx in user_history_pids:
                row = {"Product ID": idx}
                if idx in item2name:
                    row["Product Name"] = item2name[idx]
                history_rows.append(row)
            history_df = pd.DataFrame(history_rows)
            st.dataframe(history_df, use_container_width=True)
            
        # Store history sequence for model input
        sequence_to_recommend = user_history

    else:
        st.subheader("Manual Sequence Mode")
        st.write("Build a custom sequence of viewed products to simulate a new user's behavior.")
        
        # Multiselect for products
        product_options = sorted(list(item2id.keys()))
        
        def format_product_option(pid):
            title = item2name.get(pid, pid)
            if title != pid:
                trunc_title = title[:60] + "..." if len(title) > 60 else title
                return f"{pid} - {trunc_title}"
            return pid
            
        selected_pids = st.multiselect(
            "Select one or more products in chronological order:",
            options=product_options,
            format_func=format_product_option
        )
        
        # Convert string productIds to internal IDs
        sequence_to_recommend = [item2id[pid] for pid in selected_pids if pid in item2id]
        
        # Display current constructed sequence
        if len(selected_pids) > 0:
            st.write("Current sequence: " + " -> ".join([f"`{p}`" for p in selected_pids]))

    # Trigger recommendation button
    st.markdown("<br>", unsafe_allow_html=True)
    btn_recommend = st.button("Generate Recommendations", type="primary")

    if btn_recommend:
        if len(sequence_to_recommend) == 0:
            st.warning("The input sequence is empty. Please select products to generate recommendations.")
        else:
            # Perform PyTorch inference
            max_len = model.max_len
            
            # Pad sequence
            if len(sequence_to_recommend) > max_len:
                padded_seq = sequence_to_recommend[-max_len:]
            else:
                padded_seq = [0] * (max_len - len(sequence_to_recommend)) + sequence_to_recommend
                
            input_tensor = torch.tensor([padded_seq], dtype=torch.long)
            
            # Predict scores for all items
            with torch.no_grad():
                # Shape: (1, num_items + 1)
                scores = model.predict(input_tensor, item_ids=None).squeeze(0)
                
            # Mask padding token 0
            scores[0] = -float('inf')
            
            # Mask history if in "Existing User" mode
            if mode == "Existing User":
                scores[sequence_to_recommend] = -float('inf')
                
            # Softmax to get probability distribution
            probs = torch.softmax(scores, dim=-1)
            
            # Extract top 10 valid items (we fetch more and filter out 0/invalid items)
            k_fetch = min(20, num_items)
            top_probs, top_indices = torch.topk(probs, k=k_fetch)
            
            rec_pids = []
            rec_scores = []
            
            for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
                if idx in id2item and len(rec_pids) < 10:
                    rec_pids.append(id2item[idx])
                    rec_scores.append(prob)
                
            # Display Recommendations in cards
            st.markdown("### Top-10 Recommended Products")
            
            col_cards, col_plot = st.columns([1, 1])
            
            with col_cards:
                for rank, (pid, score) in enumerate(zip(rec_pids, rec_scores), 1):
                    # Progress bar normalized to top recommendation score
                    normalized_progress = score / rec_scores[0] if rec_scores[0] > 0 else 0.0
                    
                    p_title = item2name.get(pid, f"Product {pid}")
                    st.markdown(f"""
                    <div class="rec-card" style="padding: 0.8rem 1rem; margin-bottom: 0.8rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="display: flex; align-items: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-right: 15px;">
                                <span class="rec-rank" style="font-weight: bold; font-size: 1.15rem; color: #A855F7; margin-right: 0.6rem;">#{rank}</span>
                                <span title="Product ID: {pid}" style="color: #F8FAFC; font-weight: 600; font-size: 1.05rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; cursor: help;">{p_title}</span>
                            </div>
                            <span class="rec-score" style="font-weight: bold; color: #06B6D4; white-space: nowrap;">Score: {score:.5f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(normalized_progress)
                    
            with col_plot:
                # Plotly Bar Chart
                x_labels = [item2name.get(pid, pid)[:15] + "..." if len(item2name.get(pid, pid)) > 15 else item2name.get(pid, pid) for pid in rec_pids]
                fig = go.Figure(data=[
                    go.Bar(
                        x=x_labels,
                        y=rec_scores,
                        hovertext=[f"ID: {pid}<br>{item2name.get(pid, pid)}" for pid in rec_pids],
                        marker=dict(
                            color=rec_scores,
                            colorscale='Viridis'
                        ),
                        text=[f"{s:.5f}" for s in rec_scores],
                        textposition='outside'
                    )
                ])
                fig.update_layout(
                    title="Distribution of Top-10 Scores",
                    template="plotly_dark",
                    xaxis_title="Product",
                    yaxis_title="Softmax Probability",
                    margin=dict(l=10, r=10, t=50, b=10),
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Export Button CSV
                export_data = {
                    "Rank": list(range(1, len(rec_pids) + 1)),
                    "Product ID": rec_pids
                }
                if any(pid in item2name for pid in rec_pids):
                    export_data["Product Name"] = [item2name.get(pid, "") for pid in rec_pids]
                export_data["Similarity Score"] = rec_scores
                export_df = pd.DataFrame(export_data)
                
                csv_data = export_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Export Recommendations (CSV)",
                    data=csv_data,
                    file_name=f"recommendations_user_{selected_user}.csv" if mode == "Existing User" else "recommendations_manual.csv",
                    mime="text/csv",
                    use_container_width=True
                )
