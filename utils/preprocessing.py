import os
import pickle
import pandas as pd

def load_dataset(path):
    """
    Charge le dataset (Amazon Electronics ou Datafiniti Amazon).
    Si le fichier spécifié n'existe pas, tente de trouver un fichier similaire.
    """
    if not os.path.exists(path):
        directory = os.path.dirname(path) or "."
        filename = os.path.basename(path)
        base, ext = os.path.splitext(filename)
        
        files = os.listdir(directory)
        matched_files = []
        if "Datafiniti" in filename or "Datafiniti" in base:
            matched_files = [f for f in files if "Datafiniti" in f and f.endswith(ext)]
        elif "ratings_Electronics" in filename:
            matched_files = [f for f in files if "ratings_Electronics" in f and f.endswith(ext)]
            
        if not matched_files:
            matched_files = [f for f in files if base in f and f.endswith(ext)]
            
        if matched_files:
            fallback_path = os.path.join(directory, matched_files[0])
            print(f"[Alerte] Le fichier '{path}' est introuvable. Utilisation du fichier trouvé : '{fallback_path}'")
            path = fallback_path
        else:
            raise FileNotFoundError(f"Le fichier de données '{path}' est introuvable.")
            
    print(f"Chargement du dataset depuis '{path}'...")
    
    # Lecture d'un échantillon pour détecter le format des colonnes
    sample = pd.read_csv(path, nrows=2)
    
    if 'reviews.username' in sample.columns and 'asins' in sample.columns:
        print("Format détecté : Datafiniti Amazon Reviews (avec en-tête)")
        df = pd.read_csv(path)
        
        # Nettoyage des valeurs manquantes dans les colonnes d'identifiants
        df = df.dropna(subset=['reviews.username', 'asins', 'reviews.date'])
        
        # Mappage des colonnes vers le format unifié
        df = df.rename(columns={
            'reviews.username': 'userId',
            'asins': 'productId',
            'reviews.rating': 'rating'
        })
        
        # Conversion robuste de la date ISO string en timestamp Unix (secondes)
        df['timestamp'] = pd.to_datetime(df['reviews.date']).apply(lambda x: int(x.timestamp()))
        
        # Sélection des colonnes pertinentes (conserve 'name' pour le titre si disponible)
        cols_to_keep = ['userId', 'productId', 'rating', 'timestamp']
        if 'name' in df.columns:
            cols_to_keep.append('name')
        df = df[cols_to_keep]
    else:
        print("Format détecté : Amazon Electronics (sans en-tête)")
        df = pd.read_csv(path, names=['userId', 'productId', 'rating', 'timestamp'], header=None)
    
    print(f"[Load] Chargement terminé.")
    print(f"       - Nombre total d'interactions : {len(df)}")
    print(f"       - Nombre d'utilisateurs uniques : {df['userId'].nunique()}")
    print(f"       - Nombre d'items uniques : {df['productId'].nunique()}")
    return df

def filter_min_interactions(df, min_user=5, min_item=5):
    """
    Filtre récursivement les utilisateurs et les produits ayant moins de min_user et min_item interactions.
    (Filtrage k-core itératif).
    """
    print(f"Filtrage k-core : conservation des utilisateurs avec >= {min_user} interactions et produits avec >= {min_item} interactions...")
    initial_interactions = len(df)
    
    iteration = 1
    while True:
        num_users_before = df['userId'].nunique()
        num_items_before = df['productId'].nunique()
        
        user_counts = df['userId'].value_counts()
        item_counts = df['productId'].value_counts()
        
        # Filtrer
        df_filtered = df[
            df['userId'].isin(user_counts[user_counts >= min_user].index) & 
            df['productId'].isin(item_counts[item_counts >= min_item].index)
        ]
        
        num_users_after = df_filtered['userId'].nunique()
        num_items_after = df_filtered['productId'].nunique()
        
        print(f"  Iteration {iteration} :")
        print(f"    Users: {num_users_before} -> {num_users_after}")
        print(f"    Items: {num_items_before} -> {num_items_after}")
        print(f"    Interactions: {len(df)} -> {len(df_filtered)}")
        
        # Si le nombre d'utilisateurs et de produits ne change plus, le k-core est stabilisé
        if num_users_before == num_users_after and num_items_before == num_items_after:
            break
            
        df = df_filtered
        iteration += 1
        
    print(f"[Filter] Filtrage k-core terminé en {iteration} itérations.")
    print(f"         - Interactions conservées : {len(df)} ({len(df)/initial_interactions*100:.2f}%)")
    print(f"         - Utilisateurs uniques restants : {df['userId'].nunique()}")
    print(f"         - Items uniques restants : {df['productId'].nunique()}")
    return df

def build_sequences(df, max_len=50):
    """
    Groupe les interactions par utilisateur, trie chronologiquement par timestamp,
    et crée les séquences d'interactions d'une longueur maximale de max_len.
    """
    print(f"Génération des séquences d'achats chronologiques (max_len={max_len})...")
    
    # Tri par utilisateur et timestamp pour avoir un ordre chronologique
    df_sorted = df.sort_values(by=['userId', 'timestamp'])
    
    # Groupement par utilisateur
    user_groups = df_sorted.groupby('userId')['productId'].apply(list)
    
    sequences = {}
    truncated_count = 0
    for user_id, items in user_groups.items():
        if len(items) > max_len:
            sequences[user_id] = items[-max_len:]
            truncated_count += 1
        else:
            sequences[user_id] = items
            
    print(f"[Sequence] Génération des séquences terminée.")
    print(f"           - Nombre total de séquences (utilisateurs) : {len(sequences)}")
    print(f"           - Séquences tronquées à {max_len} items : {truncated_count} ({truncated_count/len(sequences)*100:.2f}%)")
    return sequences

def encode_items(sequences):
    """
    Encode les productIds de type string en entiers consécutifs.
    Important : Réserve la valeur 0 pour le padding (nécessaire pour SASRec).
    """
    print("Encodage des productIds en identifiants entiers consécutifs (0 réservé pour le padding)...")
    
    # Extraction de tous les items uniques présents dans les séquences
    unique_items = set()
    for items in sequences.values():
        unique_items.update(items)
        
    # Tri des items pour un encodage reproductible
    sorted_items = sorted(unique_items)
    
    # Création des tables de hachage de correspondance (mapping)
    # L'indexation commence à 1 (0 est le token de padding)
    item2id = {item: idx + 1 for idx, item in enumerate(sorted_items)}
    id2item = {idx + 1: item for idx, item in enumerate(sorted_items)}
    
    # Encodage des séquences
    sequences_encoded = {}
    for user_id, items in sequences.items():
        sequences_encoded[user_id] = [item2id[item] for item in items]
        
    print(f"[Encode] Encodage terminé.")
    print(f"         - Nombre total d'items uniques encodés : {len(item2id)}")
    print(f"         - Plage d'identifiants : 1 à {len(item2id)}")
    return sequences_encoded, item2id, id2item

def train_test_split_sequential(sequences):
    """
    Découpe séquentiel Leave-One-Out :
    Pour chaque utilisateur :
      - Entraînement : sequence[:-2]
      - Validation : sequence[-2]
      - Test : sequence[-1]
    """
    print("Séparation séquentielle (Leave-One-Out) en ensembles d'entraînement, validation et test...")
    train = {}
    val = {}
    test = {}
    
    short_sequences_count = 0
    for user_id, items in sequences.items():
        if len(items) >= 3:
            train[user_id] = items[:-2]
            val[user_id] = items[-2]
            test[user_id] = items[-1]
        else:
            # S'il y a moins de 3 éléments (ne devrait pas arriver avec min_user >= 5)
            train[user_id] = items
            val[user_id] = None
            test[user_id] = None
            short_sequences_count += 1
            
    print(f"[Split] Découpe terminée.")
    print(f"        - Séquences d'entraînement (train) : {len(train)}")
    print(f"        - Cibles de validation (val) : {len(val)} (séquences trop courtes ignorées : {short_sequences_count})")
    print(f"        - Cibles de test (test) : {len(test)}")
    return train, val, test

def save_processed(data, path):
    """
    Sauvegarde l'objet de données traité au format pickle.
    """
    print(f"Sauvegarde des données prétraitées dans '{path}'...")
    
    # Création du dossier parent si inexistant
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"[Save] Sauvegarde Pickle réussie dans '{path}'.")
