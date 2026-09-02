# wp4/selfplay_worker.py
import argparse
import os
import numpy as np
import chess
import torch

# set threads slightly higher for the larger model
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

from wp4.utils_io import save_pgn, save_training_examples, ensure_dir
from wp2.encoders import board_to_tensor
from wp2.action_map import move_to_index, ACTION_SPACE
from wp3.predictor import Predictor
from wp3.cpp_mcts import MCTS, encode_batch

def play_one_game(mcts: MCTS, predictor: Predictor, n_sim, c_puct, max_moves=400):
    board = chess.Board()
    moves = []
    states = []
    pis = []
    players = []
    move_count = 0
    all_nps = []
    while not board.is_game_over() and move_count < max_moves:
        # run MCTS
        pi_vec, chosen, stats = mcts.search(board, n_sim)
        if "nps" in stats:
            all_nps.append(stats["nps"])

        # temperature for first 20 moves
        if move_count < 20:
            indices = np.where(pi_vec > 1e-8)[0]
            if len(indices) > 0:
                probs = pi_vec[indices]
                probs /= probs.sum()
                chosen_idx = np.random.choice(indices, p=probs)
                chosen = predictor.move_ucis[chosen_idx]

        # Save training data
        states.append(encode_batch([board])[0])
        pis.append(pi_vec)
        players.append(board.turn)

        # Apply move
        mv_obj = chess.Move.from_uci(chosen)
        board.push(mv_obj)
        moves.append(mv_obj)
        move_count += 1

        # advance_root
        mv_idx = move_to_index(mv_obj)
        mcts.advance_root(mv_idx, mv_obj)

    # Compute Z-values
    result = board.result()
    final_value = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0
    zs = [final_value if p == chess.WHITE else -final_value for p in players]

    avg_nps = sum(all_nps)/len(all_nps) if all_nps else 0
    return moves, states, pis, zs, result, avg_nps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/selfplay")
    parser.add_argument("--n_games", type=int, default=20)
    parser.add_argument("--model", default="checkpoints/current.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_sim", type=int, default=60)
    parser.add_argument("--c_puct", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--cycle", type=int, default=0)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    ensure_dir(args.out_dir)
    shards_dir = os.path.join(args.out_dir, "shards")
    pgn_dir = os.path.join(args.out_dir, "pgns")
    ensure_dir(shards_dir)
    ensure_dir(pgn_dir)

    predictor = Predictor(args.model, device=args.device)

    # Create MCTS once per worker to reuse its memory and board pool
    mcts = MCTS(
        predictor, 
        c_puct=args.c_puct,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed if args.seed is not None else -1
    )

    prefix_pgn = f"cycle{args.cycle}_selfplay" if args.cycle > 0 else "selfplay"
    prefix_shard = f"cycle{args.cycle}_shard" if args.cycle > 0 else "shard"

    for g in range(args.n_games):
        # Reset MCTS tree/root for the new game
        mcts.advance_root(-1, None) 
        
        moves, states, pis, zs, result, avg_nps = play_one_game(
            mcts, predictor, args.n_sim, args.c_puct
        )
        pgn_path = save_pgn(moves, pgn_dir, prefix=prefix_pgn)
        shard_path = save_training_examples(states, pis, zs, shards_dir, prefix=prefix_shard)

        print(f"[worker] game {g+1}/{args.n_games} → {result} ({int(avg_nps)} NPS)")

    print("[worker] finished.")


if __name__ == "__main__":
    main()
