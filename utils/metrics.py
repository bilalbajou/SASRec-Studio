import math
import torch

def hit_rate_at_k(recommended_items, target_item, k=10):
    """
    Calcule le Hit Rate à K (HR@K) pour une seule recommandation.
    
    Formule mathématique :
        HR@K = I(target_item in recommended_items[:k])
        Où I est la fonction indicatrice (1 si vrai, 0 si faux).
    """
    top_k_recs = recommended_items[:k]
    return 1 if target_item in top_k_recs else 0

def ndcg_at_k(recommended_items, target_item, k=10):
    """
    Calcule le Normalized Discounted Cumulative Gain à K (NDCG@K) pour une cible unique.
    
    Formule mathématique générale :
        NDCG@K = DCG@K / IDCG@K
        
    Dans le cas de l'évaluation Leave-One-Out (une seule cible d'intérêt, pertinence binaire 0 ou 1) :
        - Si la cible est recommandée à la position 'rank' (1-indexed) dans les K premières recommandations :
          DCG@K = 1 / log_2(rank + 1)
        - Si la cible n'est pas dans le top-K :
          DCG@K = 0
        - Le cas idéal (IDCG@K) place l'unique recommandation pertinente en première position (rank = 1) :
          IDCG@K = 1 / log_2(1 + 1) = 1.0
          
    Ainsi, la formule simplifiée pour une cible unique est :
        NDCG@K = 1 / log_2(rank + 1)   si target_item est dans recommended_items[:k] (où rank = index + 1)
        NDCG@K = 0                     sinon
    """
    top_k_recs = recommended_items[:k]
    if target_item in top_k_recs:
        # index() est 0-indexed, on rajoute 1 pour avoir le rang 1-indexed (position 1, 2, ...)
        rank = top_k_recs.index(target_item) + 1
        return 1.0 / math.log2(rank + 1)
    return 0.0

def evaluate_model(model, test_data, all_items, device, k=10):
    """
    Évalue le modèle SASRec sur tout le jeu de test.
    Calcule et retourne un dictionnaire contenant HR@10, NDCG@10, HR@5 et NDCG@5.
    
    Formules mathématiques appliquées :
        - Pour chaque utilisateur, on calcule le rang r (1-indexed) de l'item cible parmi tous les items.
        - HR@K = 1 si r <= K, sinon 0
        - NDCG@K = 1 / log_2(r + 1) si r <= K, sinon 0
        - Les métriques globales sont obtenues en calculant la moyenne arithmétique sur tous les utilisateurs.
        
    Paramètres :
        model: Instance de nn.Module (modèle SASRec entraîné ou en cours d'évaluation).
        test_data: Soit un dictionnaire de la forme {user_id: (input_seq, target_item)},
                   soit un tuple (input_seqs, test_targets) où input_seqs est un dictionnaire
                   des séquences d'interactions historiques d'évaluation et test_targets est un
                   dictionnaire des produits cibles de test.
        all_items: Liste ou ensemble de tous les identifiants d'items uniques présents dans le jeu de données.
        device: Périphérique sur lequel exécuter les tenseurs ('cpu' ou 'cuda').
        k: Profondeur d'évaluation par défaut pour l'affichage (10 par défaut).
    """
    model.eval()
    model.to(device)
    
    # Unification du format d'entrée test_data
    if isinstance(test_data, tuple):
        input_seqs, targets = test_data
    else:
        input_seqs = {u: val[0] for u, val in test_data.items()}
        targets = {u: val[1] for u, val in test_data.items()}
        
    user_ids = list(input_seqs.keys())
    num_users = len(user_ids)
    
    # Initialisation des compteurs cumulés
    hr_10_sum, ndcg_10_sum = 0.0, 0.0
    hr_5_sum, ndcg_5_sum = 0.0, 0.0
    
    # Taille des batches pour le calcul vectorisé
    batch_size = 256
    max_len = model.max_len
    
    print(f"Début de l'évaluation séquentielle sur {num_users} utilisateurs...")
    
    with torch.no_grad():
        for i in range(0, num_users, batch_size):
            batch_users = user_ids[i:i + batch_size]
            
            # Préparation du tenseur d'entrée paddé
            batch_inputs = []
            for user_id in batch_users:
                seq = input_seqs[user_id]
                # Troncature si trop long, padding à gauche (0) si trop court
                if len(seq) > max_len:
                    padded_seq = seq[-max_len:]
                else:
                    padded_seq = [0] * (max_len - len(seq)) + seq
                batch_inputs.append(padded_seq)
                
            batch_inputs_t = torch.tensor(batch_inputs, dtype=torch.long, device=device)
            
            # Cibles associées
            batch_targets = [targets[user_id] for user_id in batch_users]
            batch_targets_t = torch.tensor(batch_targets, dtype=torch.long, device=device)
            
            # Prédiction des scores d'affinité pour tous les items du catalogue
            # scores de forme: (batch_size, num_items + 1)
            scores = model.predict(batch_inputs_t, item_ids=None)
            
            # Masquer l'indice 0 (padding token) en attribuant un score de -infini
            scores[:, 0] = -float('inf')
            
            # Masquer les items de l'historique d'interactions (ne pas recommander un produit déjà acheté)
            for idx, user_id in enumerate(batch_users):
                history = input_seqs[user_id]
                scores[idx, history] = -float('inf')
                
            # Extraire les scores prédits pour les items cibles correspondants
            target_scores = scores[torch.arange(len(batch_users), device=device), batch_targets_t] # (batch_size,)
            
            # Calculer le rang (1-indexed) de la cible parmi tous les scores.
            # Le rang est de 1 + le nombre d'items dont le score est strictement supérieur à celui de la cible.
            # scores > target_scores.unsqueeze(1) donne un tenseur booléen de taille (batch_size, num_items + 1)
            better_scores_count = (scores > target_scores.unsqueeze(1)).sum(dim=-1)
            ranks = better_scores_count + 1 # (batch_size,)
            
            # Cumul des métriques
            for r in ranks.tolist():
                if r <= 10:
                    hr_10_sum += 1.0
                    ndcg_10_sum += 1.0 / math.log2(r + 1)
                if r <= 5:
                    hr_5_sum += 1.0
                    ndcg_5_sum += 1.0 / math.log2(r + 1)
                    
    metrics = {
        'HR@10': hr_10_sum / num_users if num_users > 0 else 0.0,
        'NDCG@10': ndcg_10_sum / num_users if num_users > 0 else 0.0,
        'HR@5': hr_5_sum / num_users if num_users > 0 else 0.0,
        'NDCG@5': ndcg_5_sum / num_users if num_users > 0 else 0.0
    }
    
    print(f"[Metrics] Évaluation complétée.")
    print(f"          - HR@10   : {metrics['HR@10']:.4f}")
    print(f"          - NDCG@10 : {metrics['NDCG@10']:.4f}")
    print(f"          - HR@5    : {metrics['HR@5']:.4f}")
    print(f"          - NDCG@5  : {metrics['NDCG@5']:.4f}")
    
    return metrics
