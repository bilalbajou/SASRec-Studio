import torch
import torch.nn as nn
import torch.nn.functional as F

class MyTransformerEncoderLayer(nn.Module):
    """
    Couche d'encodage Transformer personnalisée qui capture les poids d'attention
    de la couche MultiheadAttention lors du passage forward.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.2, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=batch_first
        )
        
        # Point-wise Feed-Forward Network (FFN)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer Normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
        
        # Variable d'instance pour stocker la matrice d'attention
        self.last_attn_weights = None
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None, *args, **kwargs):
        # 1. Sous-couche de Self-Attention avec connexion résiduelle
        attn_output, attn_weights = self.self_attn(
            query=src, 
            key=src, 
            value=src, 
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        
        # Sauvegarde des poids d'attention pour la visualisation
        self.last_attn_weights = attn_weights.detach()
        
        # Connexion résiduelle + Normalisation (Post-LN comme dans le papier SASRec)
        x = src + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 2. Sous-couche Feed-Forward avec connexion résiduelle
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x

class SASRec(nn.Module):
    """
    Modèle SASRec (Self-Attentive Sequential Recommendation) basé sur Kang & McAuley (ICDM 2018).
    """
    def __init__(self, num_items, hidden_size=64, num_heads=2, num_layers=2, max_len=50, dropout=0.2):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout_rate = dropout
        
        # Item Embedding (0 est réservé pour le padding)
        self.item_emb = nn.Embedding(
            num_embeddings=num_items + 1, 
            embedding_dim=hidden_size, 
            padding_idx=0
        )
        
        # Positional Embedding (de 0 à max_len - 1)
        self.pos_emb = nn.Embedding(
            num_embeddings=max_len, 
            embedding_dim=hidden_size
        )
        
        self.emb_dropout = nn.Dropout(p=dropout)
        
        # Transformer Encoder avec notre couche personnalisée qui enregistre l'attention
        encoder_layer = MyTransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_layers
        )
        
    def _generate_causal_mask(self, seq_len, device):
        """
        Génère un masque causal triangulaire supérieur (les valeurs à -inf bloquent l'attention).
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, input_ids):
        """
        Prend en entrée une séquence d'identifiants d'items de forme (batch_size, seq_len)
        et retourne les représentations contextuelles correspondantes de forme (batch_size, seq_len, hidden_size).
        """
        batch_size, seq_len = input_ids.size()
        
        # 1. Embeddings d'items
        item_embeddings = self.item_emb(input_ids) # (batch_size, seq_len, hidden_size)
        
        # 2. Embeddings de positions
        # Les positions vont de 0 à seq_len-1. Elles sont broadcastées sur le batch.
        positions = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.pos_emb(positions) # (1, seq_len, hidden_size)
        
        # Addition des embeddings et dropout
        x = item_embeddings + pos_embeddings
        x = self.emb_dropout(x)
        
        # 3. Masquage
        # Masque d'attention causal (pour empêcher de regarder dans le futur)
        causal_mask = self._generate_causal_mask(seq_len, input_ids.device)
        
        # 4. Passage dans l'encodeur Transformer (sans masque de padding pour éviter les NaNs sur les séquences courtes)
        outputs = self.encoder(
            src=x, 
            mask=causal_mask
        )
        
        return outputs
        
    def predict(self, input_ids, item_ids=None):
        """
        Prédit les scores d'affinité pour des items candidats à la fin de la séquence d'entrée.
        input_ids : tenseur de forme (batch_size, seq_len)
        item_ids : tenseur de forme (batch_size, num_candidates) ou None.
                   Si None, calcule les scores pour TOUS les items du catalogue (num_items + 1).
                   
        Retourne : scores de forme (batch_size, num_candidates) ou (batch_size, num_items + 1)
        """
        # Obtenir les représentations contextuelles de la séquence
        features = self.forward(input_ids) # (batch_size, seq_len, hidden_size)
        
        # On utilise la représentation contextuelle du dernier élément de la séquence
        final_feat = features[:, -1, :] # (batch_size, hidden_size)
        
        if item_ids is None:
            # Recommandation globale : produit scalaire avec TOUS les embeddings d'items (Weight Tying)
            # self.item_emb.weight a pour taille (num_items + 1, hidden_size)
            scores = torch.matmul(final_feat, self.item_emb.weight.t()) # (batch_size, num_items + 1)
        else:
            # Recommandation parmi des candidats ciblés
            if item_ids.dim() == 1:
                item_ids = item_ids.unsqueeze(0).expand(input_ids.size(0), -1)
                
            candidate_embeddings = self.item_emb(item_ids) # (batch_size, num_candidates, hidden_size)
            
            # Produit scalaire batch-wise entre final_feat et candidate_embeddings
            # final_feat: (batch_size, 1, hidden_size)
            # candidate_embeddings: (batch_size, num_candidates, hidden_size)
            scores = (final_feat.unsqueeze(1) * candidate_embeddings).sum(dim=-1) # (batch_size, num_candidates)
            
        return scores
        
    def get_attention_weights(self, input_ids):
        """
        Exécute un passage forward et extrait les poids d'attention stockés dans chaque couche d'encodage.
        Retourne une liste de tenseurs de taille (batch_size, seq_len, seq_len).
        Chaque tenseur de la liste correspond à une couche du Transformer.
        """
        # Passage forward pour mettre à jour les matrices d'attention stockées
        _ = self.forward(input_ids)
        
        # Récupération des poids d'attention sauvegardés
        attn_weights = []
        for layer in self.encoder.layers:
            if hasattr(layer, 'last_attn_weights') and layer.last_attn_weights is not None:
                attn_weights.append(layer.last_attn_weights)
                
        return attn_weights
