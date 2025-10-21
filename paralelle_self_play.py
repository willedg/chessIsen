# parallel_selfplay.py
import os
import time
import random
import numpy as np
import chess
from multiprocessing import Pool, current_process
from functools import partial
import torch

# IMPORTS PROJECT (adapte si besoin)
from model.WP2_Architecture_Neurone import NeuralNetwork
from model.action_encoding import ACTION_SIZE
from self_play_wp4 import self_play_one_game, MCTS as MCTS_Class, TorchNetWrapper, ChessEnv

# ---------------- Configuration ----------------
NUM_GAMES = 100                # total games to generate
GAMES_PER_WORKER = 5           # games per worker before returning
NUM_WORKERS = max(1, os.cpu_count() - 1)
SIMULATIONS = 200              # MCTS simulations per move
CPUCT = 1.4
OUT_DIR = "selfplay_chunks"
MODEL_PATH = None              # path to checkpoint (optional)
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------
# Worker initializer
# ------------------------------------------------
def worker_init(model_state_path=None, simulations=200, c_puct=1.4):
    global WORKER_MCTS, WORKER_MODEL, WORKER_WRAPPER, WORKER_ENV

    model = NeuralNetwork(input_channels=18, board_size=8, num_actions=ACTION_SIZE)
    if model_state_path:
        ck = torch.load(model_state_path, map_location="cpu")
        model.load_state_dict(ck)

    wrapper = TorchNetWrapper(model)
    env = ChessEnv()

    WORKER_MCTS = MCTS_Class(wrapper, env, simulations=simulations, c_puct=c_puct)
    WORKER_MODEL = model
    WORKER_WRAPPER = wrapper
    WORKER_ENV = env

    print(f"[{current_process().name}] initialized worker with pid {os.getpid()}")

# ------------------------------------------------
# Worker job
# ------------------------------------------------
def worker_run(args):
    n_games, out_dir, chunk_id = args
    chunk_files = []
    for i in range(n_games):
        start_t = time.time()
        try:
            states, pis, zs, players = self_play_one_game(WORKER_MCTS)
            fname = os.path.join(out_dir, f"{current_process().name}_chunk{chunk_id}_game{i}.npz")
            np.savez_compressed(fname, states=states, pis=pis, zs=zs, players=players)
            took = time.time() - start_t
            print(f"[{current_process().name}] saved {fname} ({states.shape[0]} moves) in {took:.1f}s")
            chunk_files.append(fname)
        except Exception as e:
            print(f"[{current_process().name}] ERROR game {i}: {e}")
    return chunk_files

# ------------------------------------------------
# Main driver
# ------------------------------------------------
def main():
    total = NUM_GAMES
    per_worker = GAMES_PER_WORKER
    tasks = []
    games_left = total
    chunk_id = 0
    while games_left > 0:
        g = min(per_worker, games_left)
        tasks.append((g, OUT_DIR, chunk_id))
        games_left -= g
        chunk_id += 1

    print(f"Launching {NUM_WORKERS} workers for {NUM_GAMES} games total...")

    # create pool
    with Pool(
        processes=NUM_WORKERS,
        initializer=worker_init,
        initargs=(MODEL_PATH, SIMULATIONS, CPUCT),
    ) as pool:
        all_results = pool.map(worker_run, tasks)

    # flatten list
    all_files = [f for sub in all_results for f in sub]
    print(f"\n✅ Completed {len(all_files)} self-play games.")
    print(f"Chunks saved under: {OUT_DIR}")

if __name__ == "__main__":
    main()
