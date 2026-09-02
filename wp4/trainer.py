# wp4/trainer.py

import argparse
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from wp2.model import AlphaChessNet


from torch.utils.data import DataLoader, IterableDataset
import random

class FastShardDataset(IterableDataset):
    def __init__(self, shard_dir, max_shards=None, target_planes=19):
        # On s'assure que le chemin est absolu pour éviter les surprises
        abs_shard_dir = os.path.abspath(shard_dir)
        print(f"[trainer] Scanning directory: {abs_shard_dir}")
        
        all_shards = sorted(glob.glob(os.path.join(abs_shard_dir, "*.npz")))
        print(f"[trainer] Found {len(all_shards)} .npz files total.")
        
        if max_shards and len(all_shards) > max_shards: 
            all_shards = all_shards[-max_shards:]
        
        self.filenames = all_shards
        self.target_planes = target_planes
        print(f"[trainer] Selected {len(self.filenames)} shards for training.")
        random.shuffle(self.filenames)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.filenames
        
        # Si on a plusieurs workers, on divise les fichiers
        if worker_info is not None:
            per_worker = int(np.ceil(len(files) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(files))
            files = files[iter_start:iter_end]

        for path in files:
            try:
                # mmap_mode='r' pour une lecture éclair
                data = np.load(path, mmap_mode='r')
                s_arr, p_arr, z_arr = data["states"], data["pis"], data["zs"]
                
                # Vérification des plans pour éviter les erreurs de format (Nitro vs Old)
                if s_arr.shape[1] != self.target_planes:
                    continue

                idx = np.random.permutation(len(z_arr))
                for i in idx:
                    # Correction: torch.from_numpy ne supporte pas les scalaires numpy
                    # On utilise torch.tensor pour les valeurs simples ou on convertit en array
                    s = torch.from_numpy(s_arr[i]).float()
                    p = torch.from_numpy(p_arr[i]).float()
                    z = torch.tensor(z_arr[i], dtype=torch.float32)
                    yield s, p, z
            except Exception as e:
                if worker_info is None or worker_info.id == 0:
                    print(f"[trainer] Error reading {path}: {e}")
                continue


# ------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss, total_policy, total_value, n_batches = 0, 0, 0, 0
    use_amp = scaler is not None and device.type == 'cuda'
    for states, pis, zs in loader:
        states = states.to(device, non_blocking=True)
        pis = pis.to(device, non_blocking=True)
        zs = zs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        
        # New syntax for AMP
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=use_amp):
            logits, value = model(states)
            log_p = F.log_softmax(logits, dim=1)
            policy_loss = - (pis * log_p).sum(dim=1).mean()
            value_loss = F.mse_loss(value, zs)
            loss = policy_loss + value_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_policy += policy_loss.item()
        total_value += value_loss.item()
        n_batches += 1

    if n_batches == 0:
        return 0, 0, 0

    return (
        total_loss / n_batches,
        total_policy / n_batches,
        total_value / n_batches
    )



# ------------------------------------------------------------
# Programme principal
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_dir", required=True)
    parser.add_argument("--init_model", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_shards", type=int, default=100)
    parser.add_argument("--log_file", default=None, help="CSV file to append training results")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--cycle", type=int, default=0, help="Current cycle number")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load base model
    print(f"[trainer] loading base model {args.init_model}")
    model = AlphaChessNet().to(device)

    checkpoint = torch.load(args.init_model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[trainer] Learning rate: {args.lr}")
    
    # Mixed Precision scaler for faster training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    if scaler:
        print("[trainer] Mixed Precision (AMP) enabled - ~2x faster training!")

    # Load shards into dataset (Nitro Fast Loading)
    dataset = FastShardDataset(args.shard_dir, max_shards=args.max_shards, target_planes=19)
    
    # Optimized DataLoader: num_workers for parallel loading, pin_memory for faster GPU transfer
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=2, # Réduit pour éviter les freezes sur Windows
        pin_memory=True if device.type == 'cuda' else False
    )

    # Training
    for epoch in range(args.epochs):
        loss_total, loss_policy, loss_value = train_one_epoch(model, loader, optimizer, device, scaler)
        print(f"[trainer] epoch {epoch+1} "
            f"total={loss_total:.4f} | "
            f"policy={loss_policy:.4f} | "
            f"value={loss_value:.4f}")


    # Save candidate
    torch.save({"model_state_dict": model.state_dict()}, args.output_model)
    print(f"[trainer] saved candidate model → {args.output_model}")

    # Log results
    if args.log_file:
        import csv
        file_exists = os.path.isfile(args.log_file)
        with open(args.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["cycle", "epoch", "total_loss", "policy_loss", "value_loss"])
            # Log current stats with cycle number
            writer.writerow([args.cycle, "last", f"{loss_total:.4f}", f"{loss_policy:.4f}", f"{loss_value:.4f}"])


if __name__ == "__main__":
    main()
