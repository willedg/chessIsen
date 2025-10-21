# WP4.3_tain.py

# Objectif : entraîner le réseau de neurones à reproduire la politique (π)
# et la valeur (v) issues des parties self-play générées par le MCTS.
# Nombre de parties : 10 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# === Configuration des paramètres d'entraînement ===
DATA_PATH = "/.../selfplay_data.npz"   # données générées par le MCTS
MODEL_SAVE_PATH = "/.../trained_model_wp4_3.pt"  # chemin de sauvegarde du modèle
ACTION_SIZE = 4864      # taille de la sortie de la politique (nombre total d'actions possibles)
BATCH_SIZE = 64         # taille du lot d'entraînement
EPOCHS = 20             # nombre d'époques
LR = 1e-3               # taux d'apprentissage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # utilisation du GPU si disponible

print(f" Entraînement sur : {DEVICE}")

# ===============================
# Dataset personnalisé
# ===============================
# Le dataset charge les positions (états du plateau), les politiques π,
# et les valeurs z (résultat final) depuis le fichier .npz.
# Chaque élément correspond à un état de jeu observé durant le self-play.
class ChessDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.states = data["states"]   # états du plateau (18x8x8)
        self.pis = data["pis"]         # distributions de probas MCTS (longueur = ACTION_SIZE)
        self.zs = data["zs"]           # résultats finaux (+1, -1, 0)
        print(f" {len(self.states)} positions chargées depuis {data_path}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # Conversion en tenseurs PyTorch
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        pi = torch.tensor(self.pis[idx], dtype=torch.float32)
        z = torch.tensor(self.zs[idx], dtype=torch.float32)
        return state, pi, z

# ===============================
# Fonctions de perte
# ===============================
# 1. Policy Loss : mesure à quel point la politique prédite s'éloigne de celle du MCTS
# 2. Value Loss  : mesure l'erreur de prédiction du résultat final de la partie
def loss_policy(pred_logits, target_pi):
    """Cross-entropy entre la politique prédite et celle du MCTS"""
    log_probs = torch.log_softmax(pred_logits, dim=1)
    return -torch.mean(torch.sum(target_pi * log_probs, dim=1))

def loss_value(pred_value, target_z):
    """Erreur quadratique (MSE) entre la valeur prédite et le résultat réel"""
    return torch.mean((pred_value.squeeze() - target_z) ** 2)

# ===============================
# Entraînement du modèle
# ===============================
def train_model():
    # Chargement du dataset
    dataset = ChessDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialisation du réseau de neurones AlphaZero-like
    model = NeuralNetwork(input_channels=18, board_size=8, num_actions=ACTION_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Boucle principale d'entraînement
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_policy_loss, total_value_loss = 0.0, 0.0

        # Parcours de tous les lots
        for states, target_pis, target_zs in loader:
            states = states.to(DEVICE)
            target_pis = target_pis.to(DEVICE)
            target_zs = target_zs.to(DEVICE)

            optimizer.zero_grad()

            # Prédiction du réseau : logits (pour la policy) et valeur scalaire
            pred_logits, pred_value = model(states)

            # Calcul des deux pertes
            p_loss = loss_policy(pred_logits, target_pis)
            v_loss = loss_value(pred_value, target_zs)
            loss = p_loss + v_loss   # perte totale combinée

            # Rétropropagation et mise à jour des poids
            loss.backward()
            optimizer.step()

            # Accumulation des pertes pour affichage
            total_policy_loss += p_loss.item()
            total_value_loss += v_loss.item()

        # Affichage du suivi de la convergence
        print(f"Epoch {epoch}/{EPOCHS} | Policy Loss: {total_policy_loss/len(loader):.4f} | "
              f"Value Loss: {total_value_loss/len(loader):.4f}")

    # Sauvegarde finale du modèle entraîné
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n Modèle sauvegardé sous {MODEL_SAVE_PATH}")

# ===============================
# Exécution de l'entraînement
# ===============================
train_model()
