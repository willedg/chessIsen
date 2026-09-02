import chess
import os
import sys
import torch
import time

# Configuration des chemins pour le projet
PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

from wp3.predictor import Predictor
from wp3.cpp_mcts import MCTS
from wp2.action_map import move_to_index

# Suite de tests DTM (Distance to Mate)
TEST_SUITE = [
    {
        "name": "Scholar's Mate (Mate in 1)",
        "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 1 1",
        "expected": "f3f7",
        "sims": 100
    },
    {
        "name": "Classic Mate in 2 (Anastasia's variations)",
        "fen": "r5k1/1pR1Qp1p/p5p1/2pP1b2/P1P2n2/8/1P3PPP/6K1 w - - 0 1",
        "expected": "e7f7",
        "sims": 800
    },
    {
        "name": "Back Rank Mate Setup",
        "fen": "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",
        "expected": "e1e8",
        "sims": 50
    },
    {
        "name": "Philidor Defense Trap",
        "fen": "r1bqkb1r/ppp1p1pp/2np1n2/5p2/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 2 5",
        "expected": "f3g5", # Not mate, but best attacking move
        "sims": 1600
    }
]

def benchmark():
    model_path = "checkpoints/current.pt"
    if not os.path.exists(model_path):
        print(f"ERROR: Model {model_path} not found!")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = Predictor(model_path, device=device)
    
    print(f"\n🚀 Starting DTM/Mate Benchmark on {device}")
    print(f"Model: {model_path}")
    
    success_count = 0
    
    for test in TEST_SUITE:
        print(f"\n--- Running: {test['name']} ---")
        board = chess.Board(test['fen'])
        mcts = MCTS(predictor, c_puct=1.25, batch_size=16, dirichlet_eps=0.0)
        
        start = time.time()
        pi, best_move, stats = mcts.search(board, test['sims'])
        end = time.time()
        
        is_success = best_move.startswith(test['expected'])
        if is_success:
            print(f"SUCCESS: Found {best_move} in {end-start:.2f}s")
            success_count += 1
        else:
            print(f"FAILURE: Found {best_move}, expected {test['expected']}")
            # Find probability of expected move
            try:
                exp_move = chess.Move.from_uci(test['expected'])
                idx = move_to_index(exp_move)
                print(f"   (Exp move {test['expected']} had {pi[idx]*100:.1f}% probability)")
            except: pass

    print(f"\n" + "="*30)
    print(f"BENCHMARK RESULT: {success_count}/{len(TEST_SUITE)} Passed")
    print("="*30)

if __name__ == "__main__":
    benchmark()