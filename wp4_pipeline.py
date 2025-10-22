# wp4_pipeline_parallel.py
"""
WP4 pipeline (parallel generation using existing parallele_self_play.py).

Workflow (WP4):
  - create next generation folder genX
  - SP1: generate NUM_SELFPLAY games in selfplay_runs/genX/ using parallele_self_play (no model)
  - merge -> selfplay_runs/genX/dataset.npz
  - train -> save model to selfplay_runs/genX/model_genX.pt
  - create genX+1
  - SP1 of genX+1: generate NUM_SELFPLAY games using the new model (workers load model on CPU)
  - merge -> selfplay_runs/genX+1/dataset.npz
  - STOP (WP4 ends here)
"""

import os
import sys
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# imports from project
from model.WP2_Architecture_Neurone import NeuralNetwork
from model.action_encoding import ACTION_SIZE

# import the parallel generator module (must be importable)
import parallele_self_play

# config (change these to tune)
NUM_SELFPLAY = 25       # number of games for each generation
EPOCHS = 20             # training epochs for model after SP1
BATCH_SIZE = 64
LR = 1e-3
BASE_OUT = "selfplay_runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# make sure base folders exist
os.makedirs(BASE_OUT, exist_ok=True)


# ------------------------- helpers -------------------------
def next_gen_folder(base=BASE_OUT):
    """Return next available generation folder name like gen1, gen2, ..."""
    i = 1
    while True:
        candidate = os.path.join(base, f"gen{i}")
        if not os.path.exists(candidate):
            return candidate, i
        i += 1


def list_chunks_in_dir(d):
    return sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")])


def merge_chunks_to_npz(chunks_dir, out_npz_path):
    files = list_chunks_in_dir(chunks_dir)
    if not files:
        raise RuntimeError(f"No chunk files found in {chunks_dir}")
    all_states, all_pis, all_zs, all_players = [], [], [], []
    total = 0
    for f in files:
        d = np.load(f)
        s = d["states"]; p = d["pis"]; z = d["zs"]
        pl = d["players"] if "players" in d else np.zeros((s.shape[0],), dtype=np.int8)
        all_states.append(s); all_pis.append(p); all_zs.append(z); all_players.append(pl)
        total += s.shape[0]
        print(f"[MERGE] loaded {os.path.basename(f)} -> {s.shape[0]} positions")
    print(f"[MERGE] concatenating {len(files)} files, total positions = {total}")
    ALL_states = np.concatenate(all_states, axis=0)
    ALL_pis = np.concatenate(all_pis, axis=0)
    ALL_zs = np.concatenate(all_zs, axis=0)
    ALL_players = np.concatenate(all_players, axis=0)
    np.savez_compressed(out_npz_path, states=ALL_states, pis=ALL_pis, zs=ALL_zs, players=ALL_players)
    print(f"[MERGE] saved merged dataset to {out_npz_path}")
    return out_npz_path


class SelfPlayDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.states = data["states"].astype(np.float32)
        self.pis = data["pis"].astype(np.float32)
        self.zs = data["zs"].astype(np.float32)
        assert self.states.shape[0] == self.pis.shape[0] == self.zs.shape[0]

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.states[idx])   # (18,8,8)
        pi = torch.from_numpy(self.pis[idx])    # (ACTION_SIZE,)
        z = torch.tensor(self.zs[idx], dtype=torch.float32)
        return s, pi, z


def train_model_on_npz(npz_path, model_save_path, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    print(f"[TRAIN] Loading dataset {npz_path}")
    ds = SelfPlayDataset(npz_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = NeuralNetwork(input_channels=18, board_size=8, num_actions=ACTION_SIZE)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = running_policy = running_value = 0.0
        nb = 0
        for states, pis, zs in dl:
            states = states.to(DEVICE); pis = pis.to(DEVICE); zs = zs.to(DEVICE)
            opt.zero_grad()
            logits, vals = model(states)
            vals = vals.view(-1)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = - (pis * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(vals, zs)
            loss = policy_loss + value_loss
            loss.backward()
            opt.step()
            running_loss += float(loss.item()); running_policy += float(policy_loss.item()); running_value += float(value_loss.item())
            nb += 1
        print(f"[TRAIN] Epoch {epoch}/{epochs} - loss={running_loss/nb:.4f} (policy={running_policy/nb:.4f} value={running_value/nb:.4f})")

    # save model inside generation folder
    torch.save(model.state_dict(), model_save_path)
    print(f"[TRAIN] saved model to {model_save_path}")
    return model_save_path


# ------------------------- parallel driver -------------------------
def call_parallele_self_play(out_dir, n_games, model_path=None):
    """
    Configure parallele_self_play module globals and call its main().
    This assumes parallele_self_play reads globals: OUT_DIR, NUM_GAMES, GAMES_PER_WORKER, NUM_WORKERS, SIMULATIONS, CPUCT, MODEL_PATH.
    """
    # ensure dir exists
    os.makedirs(out_dir, exist_ok=True)

    mod = parallele_self_play
    # backup original values
    orig = {}
    for name in ("OUT_DIR", "NUM_GAMES", "GAMES_PER_WORKER", "NUM_WORKERS", "SIMULATIONS", "CPUCT", "MODEL_PATH"):
        orig[name] = getattr(mod, name, None)

    # configure
    mod.OUT_DIR = out_dir
    mod.NUM_GAMES = n_games
    mod.GAMES_PER_WORKER = max(1, min(n_games, getattr(mod, "GAMES_PER_WORKER", 5)))
    mod.NUM_WORKERS = min(getattr(mod, "NUM_WORKERS", max(1, os.cpu_count() - 1)), n_games)
    # leave SIMULATIONS, CPUCT as module defaults unless you want to override here
    if model_path is not None:
        mod.MODEL_PATH = model_path
    else:
        # ensure MODEL_PATH is None if no model
        mod.MODEL_PATH = None

    print(f"[PARALLEL] launching parallele_self_play.main() -> OUT_DIR={mod.OUT_DIR} NUM_GAMES={mod.NUM_GAMES} NUM_WORKERS={mod.NUM_WORKERS} MODEL_PATH={mod.MODEL_PATH}")
    # run
    mod.main()

    # restore originals
    for name, val in orig.items():
        if val is None:
            try:
                delattr(mod, name)
            except Exception:
                pass
        else:
            setattr(mod, name, val)


# ------------------------- main WP4 flow -------------------------
def wp4_single_cycle(num_selfplay=NUM_SELFPLAY, epochs=EPOCHS):
    # determine generation folders
    gen1_folder, gen1_idx = next_gen_folder(BASE_OUT)  # gen1
    gen2_folder = os.path.join(BASE_OUT, f"gen{gen1_idx + 1}")  # gen2

    print(f"[WP4] Will run generation {gen1_idx} -> folders:\n  GEN1: {gen1_folder}\n  GEN2: {gen2_folder}")
    # 1) SP1 for gen1 (no model)
    print("[WP4] STEP 1 - Generating SP1 for GEN1 (no model)")
    call_parallele_self_play(gen1_folder, num_selfplay, model_path=None)

    # merge gen1
    merged1 = os.path.join(gen1_folder, "dataset.npz")
    print("[WP4] STEP 2 - Merging GEN1 chunks ->", merged1)
    merge_chunks_to_npz(gen1_folder, merged1)

    # train on merged1
    model1_path = os.path.join(gen1_folder, f"model_gen{gen1_idx}.pt")
    print("[WP4] STEP 3 - Training model on GEN1 dataset")
    train_model_on_npz(merged1, model1_path, epochs=epochs)

    # 4) SP1 for gen2 (using trained model1)
    print("[WP4] STEP 4 - Generating GEN2 (SP1) using the newly trained model")
    call_parallele_self_play(gen2_folder, num_selfplay, model_path=model1_path)

    # merge gen2
    merged2 = os.path.join(gen2_folder, "dataset.npz")
    print("[WP4] STEP 5 - Merging GEN2 chunks ->", merged2)
    merge_chunks_to_npz(gen2_folder, merged2)

    print("[WP4] Done. Results:")
    print(f"  GEN1 dataset: {merged1}")
    print(f"  GEN1 model: {model1_path}")
    print(f"  GEN2 dataset: {merged2}")
    return {
        "gen1_folder": gen1_folder,
        "gen1_dataset": merged1,
        "gen1_model": model1_path,
        "gen2_folder": gen2_folder,
        "gen2_dataset": merged2
    }


if __name__ == "__main__":
    # optional overrides via env
    NUM_SELFPLAY = int(os.environ.get("WP4_NUM_SELFPLAY", NUM_SELFPLAY))
    EPOCHS = int(os.environ.get("WP4_EPOCHS", EPOCHS))
    res = wp4_single_cycle(num_selfplay=NUM_SELFPLAY, epochs=EPOCHS)
    print("RESULT:", res)
