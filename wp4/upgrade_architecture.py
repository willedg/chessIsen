# wp4/upgrade_architecture.py
import torch
import torch.optim as optim
from wp2.model import AlphaChessNet
from wp4.trainer import ShardDataset, train_one_epoch
from torch.utils.data import DataLoader
import os

def upgrade():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[upgrade] Target device: {device}")

    # 1. Créer le NOUVEAU modèle (automatiquement 12/128/SE avec les nouveaux defaults)
    model = AlphaChessNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[upgrade] New architecture created. Params: {total_params:,}")

    # 2. Charger les données (les 20 000 shards)
    shard_dir = "data/selfplay/shards"
    print(f"[upgrade] Loading shards from {shard_dir}...")
    dataset = ShardDataset(shard_dir, max_shards=20000)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # 3. Entraînement initial "intensif"
    # On va faire 10 epochs pour que le nouveau réseau assimile bien les données
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10

    print(f"[upgrade] Starting initial training (10 epochs)...")
    for epoch in range(epochs):
        loss_total, loss_policy, loss_value = train_one_epoch(model, loader, optimizer, device)
        print(f"[upgrade] Epoch {epoch+1}/{epochs} | Loss: {loss_total:.4f} (P:{loss_policy:.4f}, V:{loss_value:.4f})")
        
        # Sauvegarde intermédiaire
        torch.save({"model_state_dict": model.state_dict()}, "checkpoints/current_v2_temp.pt")

    # 4. Finalisation
    # On remplace l'ancien current.pt (pensez à faire une copie de l'ancien si vous voulez le garder !)
    if os.path.exists("checkpoints/current.pt"):
        os.rename("checkpoints/current.pt", "checkpoints/current_old_v1.pt")
        print("[upgrade] Renamed old current.pt to current_old_v1.pt")
    
    torch.save({"model_state_dict": model.state_dict()}, "checkpoints/current.pt")
    print("[upgrade] SUCCESS. New architecture is now active as current.pt")

if __name__ == "__main__":
    upgrade()
