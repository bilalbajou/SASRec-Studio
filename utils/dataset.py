import os
import pickle

class EcommerceDataset:
    """
    Wrapper pour charger et accéder facilement aux données prétraitées
    issues du pipeline SASRec Studio.
    """
    def __init__(self, pickle_path="data/processed.pkl"):
        self.pickle_path = pickle_path
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Le fichier prétraité '{pickle_path}' est introuvable. Veuillez exécuter le prétraitement des données.")
            
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
        self.train = data.get('train', {})
        self.val = data.get('val', {})
        self.test = data.get('test', {})
        self.item2id = data.get('item2id', {})
        self.id2item = data.get('id2item', {})
        
    def get_user_sequence(self, user_id):
        """
        Reconstruit et retourne la séquence complète d'interactions d'un utilisateur donné
        (entraînement + validation + test) sous forme d'une liste d'identifiants numériques.
        """
        seq = []
        if user_id in self.train:
            seq.extend(self.train[user_id])
        if user_id in self.val and self.val[user_id] is not None:
            seq.append(self.val[user_id])
        if user_id in self.test and self.test[user_id] is not None:
            seq.append(self.test[user_id])
        return seq
        
    def get_all_users(self):
        """
        Retourne la liste complète de tous les identifiants d'utilisateurs uniques (userId).
        """
        return list(self.train.keys())
        
    def get_item_name(self, item_id):
        """
        Retourne le productId original (string) correspondant à l'identifiant numérique interne.
        """
        return self.id2item.get(item_id, None)
        
    def get_dataset_stats(self):
        """
        Retourne un dictionnaire contenant les statistiques globales du jeu de données.
        """
        num_users = len(self.train)
        num_items = len(self.item2id)
        
        # Somme des longueurs des séquences d'entraînement
        train_interactions = sum(len(seq) for seq in self.train.values())
        val_interactions = sum(1 for v in self.val.values() if v is not None)
        test_interactions = sum(1 for t in self.test.values() if t is not None)
        total_interactions = train_interactions + val_interactions + test_interactions
        
        # Longueur moyenne des séquences complètes
        seq_lens = [len(self.get_user_sequence(u)) for u in self.train]
        avg_seq_len = sum(seq_lens) / len(seq_lens) if len(seq_lens) > 0 else 0.0
        
        return {
            'num_users': num_users,
            'num_items': num_items,
            'num_train_interactions': train_interactions,
            'num_val_interactions': val_interactions,
            'num_test_interactions': test_interactions,
            'num_total_interactions': total_interactions,
            'avg_seq_len': avg_seq_len
        }
