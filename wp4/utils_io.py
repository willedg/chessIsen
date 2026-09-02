# wp4/utils_io.py
"""
Utilities for saving PGN and training examples.
"""

import os
import numpy as np
import chess
import chess.pgn
from datetime import datetime
from typing import List, Dict


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_pgn(moves: List[chess.Move], save_dir: str, prefix: str = "game"):
    ensure_dir(save_dir)
    game = chess.pgn.Game()
    game.headers["Event"] = "Self-Play"
    game.headers["Site"] = "Local"
    game.headers["Round"] = ""
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "AlphaChess"
    game.headers["Black"] = "AlphaChess"

    node = game
    for m in moves:
        node = node.add_variation(m)

    # On calcule le résultat après les coups
    result = node.board().result()
    game.headers["Result"] = result

    ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    fname = f"{prefix}_{ts}.pgn"
    path = os.path.join(save_dir, fname)

    with open(path, "w", encoding="utf-8") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)

    return path



def save_training_examples(states, pis, zs, save_dir: str, prefix: str = "example"):
    """
    states: list or array of state tensors (numpy arrays) shape (planes,8,8)
    pis: list/array of policy vectors (len ACTION_SPACE)
    zs: list/array of scalars (+1/-1/0) final outcome from root perspective
    Will save a compressed npz with arrays: states, pis, zs
    """
    ensure_dir(save_dir)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    path = os.path.join(save_dir, f"{prefix}_{ts}.npz")
    np.savez_compressed(path, states=np.array(states, dtype=np.float32),
                        pis=np.array(pis, dtype=np.float32),
                        zs=np.array(zs, dtype=np.float32))
    return path


def list_shards(dir_path: str):
    return sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".npz")])


def load_shard(path: str):
    data = np.load(path)
    return data["states"], data["pis"], data["zs"]
