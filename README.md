# NextBuy — SASRec Studio

**NextBuy** (SASRec Studio) is a turnkey interactive platform designed for experimenting, training, visualizing, and evaluating the state-of-the-art **SASRec** (*Self-Attentive Sequential Recommendation*) model.

Developed using **Streamlit** and **PyTorch**, this studio enables you to transform users' historical purchase behavior into highly personalized and contextualized product recommendations.

---

## 🔬 SASRec Model Architecture (Simplified)

The **SASRec** model solves the sequential recommendation task using a **Transformer** architecture (specifically, the self-attention mechanism). Unlike Recurrent Neural Networks (RNNs/GRUs) that process items sequentially in a rigid manner, or Markov Chains that only remember the last action:

1. **Item & Positional Embeddings**: Viewed product IDs are converted into dense vectors (*embeddings*), which are combined with *positional embeddings* to inject the temporal order of interactions.
2. **Causal Attention Layers**: The model utilizes multi-head self-attention blocks. A causal attention mask (upper triangular) prevents the model from "cheating" by looking at future items during training.
3. **Dynamic Attention Weights**: To predict the next item of interest, the model computes an influence score for each previously viewed product. This allows it to place high importance on an item purchased long ago (long-term preferences) while still capturing recent clicks (short-term interests).
4. **Weight-Tying**: The final contextual representation of the sequence is dot-producted with the item embedding matrix to generate recommendation probability scores for all candidate products.

---

## 🛠️ Installation Instructions (Step-by-Step)

### 1. Clone or Navigate to the Workspace
Open your terminal and navigate to the project directory:
```bash
cd "d:\own project\SASRec Studio"
```

### 2. Configure Python and Install Dependencies
The application requires **Python 3.8+** (recommended 3.11.9). Install the required packages using `pip`:

```bash
# Install Streamlit and Pandas
python -m pip install streamlit pandas

# Install PyTorch (CPU version is faster to install locally)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Plotly for interactive charts
python -m pip install plotly
```

---

## 🚀 Launch Instructions

Start the local Streamlit server by running the following command at the root of the project:

```bash
python -m streamlit run app.py
```

The application will automatically open in your default browser at:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 📂 Complete Project Structure

Here is the final file structure of the studio:

```
d:\own project\SASRec Studio\
├── app.py                      # Main landing page
├── README.md                   # This file (Complete English documentation)
├── ratings_Electronics (1).csv  # Raw Amazon Electronics dataset (318 MB)
├── assets/                     # Media & temp exports
├── data/                       # Working data directory
│   ├── processed.pkl           # Preprocessed final pickle file (train/val/test/mappings)
│   └── sasrec_model.pt         # Saved PyTorch trained model weights checkpoint
├── models/                     # Model architecture definitions
│   └── sasrec.py               # SASRec class and custom attention layer
├── utils/                      # Utility helper modules
│   ├── preprocessing.py        # K-core filtering, sequencing, and encoding pipeline
│   ├── metrics.py              # Vectorized HR@K and NDCG@K evaluation metrics
│   └── dataset.py              # Data access wrapper (EcommerceDataset class)
└── pages/                      # Streamlit application pages
    ├── 1_Data_Preprocessing.py  # Data preprocessing and EDA steps
    ├── 2_Training.py           # Interactive PyTorch model training dashboard
    ├── 3_Recommendations.py    # Predictions & Recommendations engine
    ├── 4_Attention_Visualizer.py # Multi-head self-attention visualizer
    └── 5_Evaluation.py         # Model evaluation and benchmark page
```

---

## 📖 Description of the 5 Studio Pages

1. **📊 Data Preprocessing (`1_Data_Preprocessing.py`)**: Allows you to explore the interaction dataset, configure the iterative k-core filtering threshold, split purchase sequences, and generate processed files.
2. **⚙️ Model Training (`2_Training.py`)**: An interactive PyTorch training dashboard. Customize hyperparameters (layers, heads, batch size, learning rate, epochs), start or stop training dynamically, monitor the loss/metrics curves in real-time, and save model checkpoints.
3. **🔮 Predictions & Recommendations (`3_Recommendations.py`)**: Predicts next items using two modes: loading a real user's purchase history or building custom sequences manually via a product multiselect box. Displays the Top-10 recommendations in CSS-styled cards with scores and progress bars, plots score distributions using Plotly, and exports recommendations as CSV.
4. **🔍 Attention Visualizer (`4_Attention_Visualizer.py`)**: Extracts and maps the multi-head self-attention weights computed by the Transformer. Displays interactive Plotly heatmaps (excluding zero-padding), shows past items' influence on the final prediction, and supports side-by-side comparison of different attention layers.
5. **🎯 Evaluation & Metrics (`5_Evaluation.py`)**: Evaluates the trained model on the test dataset using metrics at different values of K, and benchmarks SASRec against classic recommendation baselines.

---

## 📊 Dataset & Metrics Obtained

### Used Dataset: Amazon Electronics
The dataset used comes from the University of California, San Diego (UCSD) research repositories managed by Professor Julian McAuley. It contains user reviews and interactions on products within the Electronics category.
* **Official UCSD Download Link**: [Amazon Webpage Dataset](http://jmcauley.ucsd.edu/data/amazon/)

### Obtained Metrics (Literature Benchmarks)
The table below summarizes typical performances on the Amazon Electronics dataset after 5-core filtering (sequence length $\le 50$):

| Model | HR @ 10 (Recall@10) | NDCG @ 10 |
| :--- | :---: | :---: |
| **Popularity (Non-sequential baseline)** | 0.1840 | 0.1250 |
| **FPMC (Markov Chains)** | 0.4210 | 0.3120 |
| **GRU4Rec (Sequential RNN)** | 0.5120 | 0.4180 |
| **SASRec (Self-Attention)** | **0.5912** | **0.4827** |

---

## 📝 References

* **Original Research Paper**:
  > Wang-Cheng Kang, Julian McAuley (2018). *Self-Attentive Sequential Recommendation*. In Proceedings of the IEEE International Conference on Data Mining (ICDM'18).
  > [PDF Paper Link](https://arxiv.org/pdf/1808.09781.pdf)
