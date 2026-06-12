import streamlit as st
import os
import pickle
import pandas as pd
import plotly.graph_objects as go
from utils.preprocessing import (
    load_dataset,
    filter_min_interactions,
    build_sequences,
    encode_items,
    train_test_split_sequential,
    save_processed
)

st.set_page_config(
    page_title="Data Preprocessing — SASRec Studio",
    layout="wide"
)

# Custom CSS for premium design
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
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h5 {
        color: #94A3B8;
        font-size: 0.9rem;
        margin: 0;
        font-weight: 400;
    }
    .metric-card h3 {
        color: #F8FAFC;
        font-size: 1.6rem;
        margin: 0.5rem 0 0 0;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_column_width=True)
    st.markdown("### Model Status")
    if st.session_state.get('model_trained', False):
        st.success("Model Trained")
    else:
        st.error("Model Not Trained")
    st.markdown("---")
    st.markdown("*SASRec Studio v1.0.0*")

st.markdown('<div class="page-title">Data Preprocessing</div>', unsafe_allow_html=True)
st.info("This section allows you to load raw interaction datasets, analyze their properties, and configure the k-core filtering pipeline to prepare sequences for SASRec.")

# 1. Scan for CSV files in the workspace
workspace_csvs = [f for f in os.listdir(".") if f.endswith(".csv")]
if not workspace_csvs:
    st.error("No `.csv` files found in the workspace root. Please add your datasets.")
else:
    # Let the user select the dataset
    st.subheader("Raw Dataset Selection")
    
    # Prefer Datafiniti if it exists
    default_idx = 0
    for idx, f in enumerate(workspace_csvs):
        if "Datafiniti" in f:
            default_idx = idx
            break
            
    selected_file = st.selectbox(
        "Select the CSV file to explore and preprocess:",
        workspace_csvs,
        index=default_idx
    )
    
    file_size_mb = os.path.getsize(selected_file) / (1024 * 1024)
    st.write(f"**Selected File**: `{selected_file}` | **Size**: `{file_size_mb:.2f} MB`")
    
    # Cache loading of data to speed up interactive changes
    @st.cache_data(show_spinner=False)
    def load_cached_raw_data(file_path):
        return load_dataset(file_path)
        
    with st.spinner("Loading and analyzing file..."):
        try:
            df_raw = load_cached_raw_data(selected_file)
            load_success = True
        except Exception as e:
            st.error(f"Error loading file: {e}")
            load_success = False
            
    if load_success:
        # --- EXPLORATORY DATA ANALYSIS (EDA) ---
        st.markdown("### Exploratory Data Analysis (EDA)")
        
        # Display raw statistics in cards
        raw_rows = len(df_raw)
        raw_users = df_raw['userId'].nunique()
        raw_items = df_raw['productId'].nunique()
        raw_sparsity = (1.0 - raw_rows / (raw_users * raw_items)) * 100
        
        col_raw1, col_raw2, col_raw3, col_raw4 = st.columns(4)
        with col_raw1:
            st.markdown(f'<div class="metric-card"><h5>Total Interactions</h5><h3>{raw_rows:,}</h3></div>', unsafe_allow_html=True)
        with col_raw2:
            st.markdown(f'<div class="metric-card"><h5>Unique Users</h5><h3>{raw_users:,}</h3></div>', unsafe_allow_html=True)
        with col_raw3:
            st.markdown(f'<div class="metric-card"><h5>Unique Products</h5><h3>{raw_items:,}</h3></div>', unsafe_allow_html=True)
        with col_raw4:
            st.markdown(f'<div class="metric-card"><h5>Graph Sparsity</h5><h3>{raw_sparsity:.4f}%</h3></div>', unsafe_allow_html=True)
            
        # Display data preview and plots side-by-side
        col_preview, col_plots = st.columns([4, 6])
        
        with col_preview:
            st.markdown("##### Data Preview (First Rows)")
            st.dataframe(df_raw.head(10), use_container_width=True)
            
            st.markdown("##### Ratings Distribution")
            rating_counts = df_raw['rating'].value_counts().sort_index()
            fig_ratings = go.Figure(data=[
                go.Bar(
                    x=rating_counts.index.tolist(),
                    y=rating_counts.values.tolist(),
                    marker_color='#06B6D4'
                )
            ])
            fig_ratings.update_layout(
                template="plotly_dark",
                xaxis_title="Rating",
                yaxis_title="Number of Reviews",
                margin=dict(l=10, r=10, t=20, b=10),
                height=230
            )
            st.plotly_chart(fig_ratings, use_container_width=True, key="fig_ratings")
            
        with col_plots:
            st.markdown("##### Most Popular Products")
            top_items = df_raw['productId'].value_counts().head(10).sort_values(ascending=True)
            
            # Map product ID to name for plotting if 'name' column is available
            y_labels = []
            hover_labels = []
            if 'name' in df_raw.columns:
                id_to_name = dict(zip(df_raw['productId'], df_raw['name']))
                for pid in top_items.index:
                    name = id_to_name.get(pid, pid)
                    hover_labels.append(name)
                    y_labels.append(name[:30] + "..." if len(name) > 30 else name)
            else:
                y_labels = top_items.index.tolist()
                hover_labels = top_items.index.tolist()
                
            fig_items = go.Figure(data=[
                go.Bar(
                    x=top_items.values.tolist(),
                    y=y_labels,
                    hovertext=hover_labels,
                    orientation='h',
                    marker_color='#A855F7'
                )
            ])
            fig_items.update_layout(
                template="plotly_dark",
                xaxis_title="Number of Interactions",
                yaxis_title="Product",
                margin=dict(l=10, r=10, t=20, b=10),
                height=260
            )
            st.plotly_chart(fig_items, use_container_width=True, key="fig_items")
            
            st.markdown("##### User Interaction Density")
            user_counts = df_raw['userId'].value_counts()
            fig_user_dist = go.Figure(data=[
                go.Histogram(
                    x=user_counts.values,
                    nbinsx=30,
                    marker_color='#EC4899'
                )
            ])
            fig_user_dist.update_layout(
                template="plotly_dark",
                xaxis_title="Interactions per User",
                yaxis_title="Number of Users (Log)",
                yaxis_type="log",
                margin=dict(l=10, r=10, t=20, b=10),
                height=200
            )
            st.plotly_chart(fig_user_dist, use_container_width=True, key="fig_user_dist")
            
        st.markdown("---")
        
        # --- CONFIGURATION & RUN ---
        st.subheader("K-Core & Sequence Configuration")
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        with col_cfg1:
            user_min = st.slider(
                "User K-Core (min user interactions):",
                min_value=1,
                max_value=15,
                value=3,
                help="Removes users with fewer than N reviews/purchases. Recommended: 3 for Datafiniti."
            )
        with col_cfg2:
            item_min = st.slider(
                "Item K-Core (min item interactions):",
                min_value=1,
                max_value=15,
                value=3,
                help="Removes products reviewed fewer than N times. Recommended: 3 for Datafiniti."
            )
        with col_cfg3:
            max_len = st.slider(
                "Max sequence length (max_len):",
                min_value=5,
                max_value=100,
                value=50,
                help="Maximum history length retained per user for SASRec training."
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Preprocessing execution
        if st.button("Preprocess and Generate SASRec Dataset", type="primary", use_container_width=True):
            with st.spinner("Executing preprocessing pipeline..."):
                # 1. Filter interactions
                df_filtered = filter_min_interactions(df_raw, min_user=user_min, min_item=item_min)
                
                if len(df_filtered) == 0:
                    st.error("Error: The selected k-core thresholds are too high and eliminated all data. Please reduce min_user or min_item.")
                else:
                    # 2. Build chronological sequences
                    seqs = build_sequences(df_filtered, max_len=max_len)
                    
                    # 3. Encode items
                    seqs_encoded, item2id, id2item = encode_items(seqs)
                    
                    # 4. Sequential split train/val/test
                    train, val, test = train_test_split_sequential(seqs_encoded)
                    
                    # Extract item2name mapping if name column is available
                    if 'name' in df_filtered.columns:
                        product_names = df_filtered[['productId', 'name']].dropna().drop_duplicates(subset=['productId'])
                        item2name = dict(zip(product_names['productId'], product_names['name']))
                    else:
                        item2name = {}

                    # 5. Package metadata and save
                    packaged_data = {
                        'train': train,
                        'val': val,
                        'test': test,
                        'item2id': item2id,
                        'id2item': id2item,
                        'item2name': item2name,
                        'dataset_name': selected_file,
                        'min_user': user_min,
                        'min_item': item_min,
                        'max_len': max_len,
                        'raw_stats': {
                            'num_rows': raw_rows,
                            'num_users': raw_users,
                            'num_items': raw_items
                        }
                    }
                    
                    save_processed(packaged_data, "data/processed.pkl")
                    
                    # Invalidate model training in session state
                    st.session_state.model_trained = False
                    if 'model_metrics' in st.session_state:
                        st.session_state.model_metrics = {"NDCG@10": 0.0, "Recall@10": 0.0, "Loss": 0.0}
                    if 'epoch_history' in st.session_state:
                        st.session_state.epoch_history = []
                        
                    st.success("🎉 Preprocessing completed successfully! Preprocessed data saved to `data/processed.pkl`.")
                    st.balloons()
                    
                    # Show statistics of the newly processed dataset
                    st.markdown("### Preprocessed Dataset Properties")
                    
                    total_int_processed = sum(len(seq) for seq in train.values()) + \
                                          sum(1 for v in val.values() if v is not None) + \
                                          sum(1 for t in test.values() if t is not None)
                                          
                    users_processed = len(train)
                    items_processed = len(item2id)
                    sparsity_processed = (1.0 - total_int_processed / (users_processed * items_processed)) * 100
                    
                    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
                    with col_p1:
                        st.metric("Filtered Users", f"{users_processed:,}", delta=f"{users_processed - raw_users:,}")
                    with col_p2:
                        st.metric("Filtered Products", f"{items_processed:,}", delta=f"{items_processed - raw_items:,}")
                    with col_p3:
                        st.metric("Filtered Interactions", f"{total_int_processed:,}", delta=f"{total_int_processed - raw_rows:,}")
                    with col_p4:
                        st.metric("Final Sparsity", f"{sparsity_processed:.3f}%", delta=f"{sparsity_processed - raw_sparsity:.3f}%")
                        
                    st.markdown(
                        f"""
                        - **Total number of sequences (users)**: {users_processed}
                        - **Total number of products (items) to recommend**: {items_processed}
                        - **Training set size (Train)**: {len(train)} sequences
                        - **Validation set size (Val)**: {sum(1 for v in val.values() if v is not None)} targets
                        - **Test set size (Test)**: {sum(1 for t in test.values() if t is not None)} targets
                        
                        *Note: The model has been reset. Please navigate to the **⚙️ Model Training** page to train the model on this new dataset.*
                        """
                    )
