# wp5/evaluation.py
import torch
import multiprocessing as mp
import chess
import numpy as np
from wp3.cpp_mcts import MCTS
from wp3.predictor import Predictor
from wp2.action_map import move_to_index

# Basic settings
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def _play_game(args):
    """Plays one evaluation game between current and candidate models."""
    idx, n_sim, c_puct, batch_size, top_k, current_path, candidate_path, device = args
    
    board = chess.Board()
    candidate_is_white = (idx % 2 == 0)
    
    # Predictors for each model
    pred_curr = Predictor(current_path, device=device)
    pred_cand = Predictor(candidate_path, device=device)

    # Evaluation should be focused but have a tiny bit of noise to avoid repeated games
    mcts_curr = MCTS(pred_curr, c_puct=1.0, batch_size=batch_size, top_k=top_k, dirichlet_eps=0.03)
    mcts_cand = MCTS(pred_cand, c_puct=1.0, batch_size=batch_size, top_k=top_k, dirichlet_eps=0.03)
    
    move_count = 0
    while not board.is_game_over(claim_draw=True) and move_count < 250:
        # Determine which MCTS to use
        is_candidate_turn = (board.turn == chess.WHITE and candidate_is_white) or \
                             (board.turn == chess.BLACK and not candidate_is_white)
        
        active_mcts = mcts_cand if is_candidate_turn else mcts_curr
        
        # Search
        pi, chosen_uci, stats = active_mcts.search(board, n_sim)
        
        if not chosen_uci:
            legal = list(board.legal_moves)
            if not legal: break
            chosen_uci = legal[0].uci()
        
        # Apply move to board
        move = chess.Move.from_uci(chosen_uci)
        board.push(move)
        move_count += 1
        
        # Advance root on BOTH trees (Sync pool with move object)
        mv_idx = move_to_index(move)
        mcts_curr.advance_root(mv_idx, move)
        mcts_cand.advance_root(mv_idx, move)
    
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0 if candidate_is_white else 0.0
    elif res == "0-1":
        return 0.0 if candidate_is_white else 1.0
    else:
        return 0.5

def evaluate(curr_path, cand_path, device="cuda", n_games=100, n_workers=None, 
             n_sim=800, c_puct=1.0, batch_size=512, top_k=10000):
    """Runs a parallel evaluation match."""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), n_games)
        
    print(f"[eval] Current: {curr_path} vs Candidate: {cand_path}")

    args = [(i, n_sim, c_puct, batch_size, top_k, curr_path, cand_path, device) 
            for i in range(n_games)]
    
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        scores = pool.map(_play_game, args)
        
    score_new = sum(scores)
    print(f"[eval] Candidate score: {score_new}/{n_games} ({100*score_new/n_games:.1f}%)")
    return score_new

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", default="checkpoints/current.pt")
    parser.add_argument("--candidate", default="checkpoints/candidate.pt")
    parser.add_argument("--games", type=int, default=100)
    args = parser.parse_args()
    
    evaluate(args.current, args.candidate, n_games=args.games)
