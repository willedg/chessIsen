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

# FENs de test
FEN_MATE_IN_1 = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 1 1" # Mate: f3f7
FEN_MATE_IN_2 = "r5k1/1pR1Qp1p/p5p1/2pP1b2/P1P2n2/8/1P3PPP/6K1 w - - 0 1" # Mate in 2: e7f7+ then Qxf7# or similar
FEN_MATE_IN_4 = "6k1/1pR2p1p/p2Q2p1/2pP1b2/P1P5/8/1P3PPP/6K1 w - - 1 1" # Strategically winning, mate in a few

def run_mate_test(fen, name, expected_move_prefix, n_sims=800):
    print(f"\n" + "="*50)
    print(f"TEST: {name}")
    print(f"FEN: {fen}")
    print("="*50)
    
    board = chess.Board(fen)
    
    # Initialisation du modèle et du MCTS (Adapté au code wp2/wp3)
    # Note: On utilise checkpoints/current.pt par défaut
    model_path = "checkpoints/current.pt"
    if not os.path.exists(model_path):
        print(f"ERROR: Model {model_path} not found!")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Predictor charge le modèle et gère l'encodage
    predictor = Predictor(model_path, device=device)
    # MCTS utilise le Predictor pour les évaluations
    mcts = MCTS(predictor, c_puct=1.25, batch_size=64, dirichlet_eps=0.0)
    
    print(f"Running {n_sims} simulations...")
    start_time = time.time()
    # search(board, n_sim) -> (policy_array, best_move_uci, stats)
    pi, best_move, stats = mcts.search(board, n_sims)
    duration = time.time() - start_time
    
    print(f"Done in {duration:.2f}s ({stats.get('nps', 0):.0f} NPS)")
    print(f"Best move found: {best_move}")
    
    # Analyse des probabilités
    legal_moves_probs = []
    for move in board.legal_moves:
        idx = move_to_index(move)
        legal_moves_probs.append((move.uci(), pi[idx]))
        
    legal_moves_probs.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Moves by visits/policy:")
    for move_uci, prob in legal_moves_probs[:5]:
        print(f"  {move_uci}: {prob*100:5.1f}%")
        
    if best_move.startswith(expected_move_prefix):
        print("\nRESULT: SUCCESS (Move matches expected pattern)")
    else:
        print(f"\nRESULT: FAILURE (Expected move starting with {expected_move_prefix})")
    
    # Double check mate if possible
    board.push_uci(best_move)
    if board.is_checkmate():
        print("INFO: This move is an immediate CHECKMATE!")
    elif any(board.is_checkmate() for _ in [None]): # Just a placeholder
        pass
    board.pop()

if __name__ == "__main__":
    # Test Mate in 1
    run_mate_test(FEN_MATE_IN_1, "Mate in 1", "f3f7", n_sims=100)
    
    # Test Mate in 2
    run_mate_test(FEN_MATE_IN_2, "Mate in 2", "e7f7", n_sims=1600)
    
    # Test Complex Position
    run_mate_test(FEN_MATE_IN_4, "Mate in 4 (Complex)", "d6f6", n_sims=3200)