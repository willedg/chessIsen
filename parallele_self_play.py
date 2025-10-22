# parallel_selfplay.py
import os
import time
import random
import numpy as np
import chess
from multiprocessing import Pool, current_process
import torch
from datetime import datetime

# IMPORTS PROJECT (adapter chemin si besoin)
from model.WP2_Architecture_Neurone import NeuralNetwork
from model.action_encoding import ACTION_SIZE
from self_play_wp4 import self_play_one_game, MCTS as MCTS_Class, TorchNetWrapper, ChessEnv

# ---------------- CONFIG ----------------
RUN_ID = "gen1"          # <<< À changer manuellement pour chaque génération
NUM_GAMES = 20
GAMES_PER_WORKER = 5
NUM_WORKERS = max(1, os.cpu_count() - 1)
SIMULATIONS = 200
CPUCT = 1.4
BASE_OUT = "selfplay_all"
MODEL_PATH = None        # Si on veut charger un modèle existant
# ----------------------------------------

OUT_DIR = os.path.join(BASE_OUT, RUN_ID)

def worker_init(model_state_path=None, simulations=200, c_puct=1.4):
    """Init per worker (CPU only) + unique seeds."""
    global WORKER_MCTS, WORKER_MODEL, WORKER_WRAPPER, WORKER_ENV

    seed = int(time.time() * 1000000) ^ os.getpid()
    seed = seed % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = NeuralNetwork(input_channels=18, board_size=8, num_actions=ACTION_SIZE)
    if model_state_path:
        ck = torch.load(model_state_path, map_location="cpu")
        model.load_state_dict(ck)

    wrapper = TorchNetWrapper(model)
    env = ChessEnv()
    WORKER_MCTS = MCTS_Class(wrapper, env, simulations=simulations, c_puct=c_puct)

    print(f"[{current_process().name}] ready (pid={os.getpid()}, seed={seed})")


def worker_run(args):
    n_games, out_dir, chunk_id = args
    for i in range(n_games):
        start_t = time.time()
        states, pis, zs, players = self_play_one_game(WORKER_MCTS)

        ts = datetime.now().strftime("%d-%m-%Y_%Hh%M")  # format option 2
        fname = os.path.join(out_dir, f"worker-{current_process().name}_game{i}_#{ts}.npz")

        np.savez_compressed(fname, states=states, pis=pis, zs=zs, players=players)
        print(f"[{current_process().name}] saved {fname} ({states.shape[0]} moves) in {time.time() - start_t:.1f}s")


def main():
    tasks = []
    games_left = NUM_GAMES
    chunk_id = 0
    while games_left > 0:
        g = min(GAMES_PER_WORKER, games_left)
        tasks.append((g, OUT_DIR, chunk_id))
        games_left -= g
        chunk_id += 1

    print(f"Launching {NUM_WORKERS} workers for {NUM_GAMES} games into {OUT_DIR} ...")

    with Pool(processes=NUM_WORKERS,
              initializer=worker_init,
              initargs=(MODEL_PATH, SIMULATIONS, CPUCT)) as pool:
        pool.map(worker_run, tasks)

    print("\nAll self-play games completed.")


if __name__ == "__main__":
    main()
