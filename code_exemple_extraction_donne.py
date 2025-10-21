import os
import numpy as np

IN_DIR = "selfplay_chunks"
OUT_FILE = "dataset_cycle0.npz"

all_states = []
all_pis = []
all_zs = []
all_players = []

for fname in os.listdir(IN_DIR):
    if not fname.endswith(".npz"):
        continue
    path = os.path.join(IN_DIR, fname)
    data = np.load(path)
    all_states.append(data["states"])
    all_pis.append(data["pis"])
    all_zs.append(data["zs"])
    all_players.append(data["players"])
    print("Loaded", fname, "with", data["states"].shape[0], "positions")

# concat
ALL_states = np.concatenate(all_states, axis=0)
ALL_pis = np.concatenate(all_pis, axis=0)
ALL_zs = np.concatenate(all_zs, axis=0)
ALL_players = np.concatenate(all_players, axis=0)

print("Final shapes:")
print("states :", ALL_states.shape)
print("pis    :", ALL_pis.shape)
print("zs     :", ALL_zs.shape)
print("players:", ALL_players.shape)

np.savez_compressed(
    OUT_FILE,
    states=ALL_states,
    pis=ALL_pis,
    zs=ALL_zs,
    players=ALL_players
)

print("\n✅ Saved merged dataset to", OUT_FILE)
