import os
import chess
import chess.pgn
import numpy as np
import random

import torch

from self_play_wp4 import MCTS as MCTS_Class, TorchNetWrapper, ChessEnv
from model.WP2_Architecture_Neurone import NeuralNetwork
from model.action_encoding import ACTION_SIZE, move_to_action_index


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# SAVE GAME TO PGN
# ============================================================

def save_game_pgn(move_list, output_path, white_name="White", black_name="Black"):
    """
    Sauvegarde une partie en PGN pour visualisation dans PyChess.
    move_list = [chess.Move, chess.Move, ...]
    """
    game = chess.pgn.Game()
    game.headers["White"] = white_name
    game.headers["Black"] = black_name

    node = game
    for mv in move_list:
        node = node.add_variation(mv)

    with open(output_path, "w", encoding="utf-8") as f:
        print(game, file=f)

    print(f"[ARENA] Partie sauvegardée dans {output_path}")


# ============================================================
# AGENTS
# ============================================================

class ModelAgent:
    """Agent utilisant ton NeuralNetwork + MCTS."""
    def __init__(self, model_path, simulations=200, c_puct=1.0):
        model = NeuralNetwork(18, 8, ACTION_SIZE)
        sd = torch.load(model_path, map_location="cpu")
        model.load_state_dict(sd)
        model.to(DEVICE)

        self.wrapper = TorchNetWrapper(model, device=DEVICE)
        self.simulations = simulations
        self.c_puct = c_puct
        self.name = os.path.basename(model_path)

    def make_mcts(self, env):
        return MCTS_Class(self.wrapper, env, simulations=self.simulations, c_puct=self.c_puct)


class PureMCTSAgent:
    """
    MCTS pur : prior uniforme, value = 0.
    """
    def __init__(self, simulations=200):
        self.simulations = simulations
        self.name = "PureMCTS"

    def predict(self, board, temperature=1.0):
        legal = list(board.legal_moves)
        priors = np.ones(ACTION_SIZE, dtype=np.float32) * 1e-9

        for mv in legal:
            idx = move_to_action_index(mv)
            if idx is not None:
                priors[idx] = 1.0 / len(legal)

        return priors, 0.0

    def make_mcts(self, env):
        return MCTS_Class(self, env, simulations=self.simulations, c_puct=1.0)


# ============================================================
# PLAY ONE GAME
# ============================================================

def play_one_game(agent_white, agent_black, save_pgn_path=None):
    env = ChessEnv()
    board = chess.Board()

    mcts_white = agent_white.make_mcts(env)
    mcts_black = agent_black.make_mcts(env)

    move_history = []

    while not env.is_terminal(board):
        if board.turn == chess.WHITE:
            pi, value, root = mcts_white.run(board)
        else:
            pi, value, root = mcts_black.run(board)

        moves = list(pi.keys())
        probs = np.array([pi[m] for m in moves], dtype=np.float32)
        probs /= probs.sum() + 1e-12

        move = moves[np.random.choice(len(moves), p=probs)]
        move_history.append(move)
        board.push(move)

    # Sauvegarde PGN si demandé
    if save_pgn_path is not None:
        save_game_pgn(
            move_history,
            save_pgn_path,
            white_name=agent_white.name,
            black_name=agent_black.name
        )

    result = board.result()
    if result == "1-0": return 1
    if result == "0-1": return -1
    return 0


# ============================================================
# ARENA LOOP
# ============================================================

def arena(agentA, agentB, games=20):
    os.makedirs("arena_games", exist_ok=True)

    winsA = winsB = draws = 0

    for i in range(games):
        print(f"\n[ARENA] Partie {i+1}/{games}")

        if i % 2 == 0:
            result = play_one_game(
                agentA, agentB,
                save_pgn_path=f"arena_games/game_{i+1}.pgn"
            )
        else:
            result = -play_one_game(
                agentB, agentA,
                save_pgn_path=f"arena_games/game_{i+1}.pgn"
            )

        if result == 1: winsA += 1
        elif result == -1: winsB += 1
        else: draws += 1

        print(f"[ARENA] Score A:{winsA}  B:{winsB}  D:{draws}")

    # Résultat final
    print("\n=========== RESULTATS FINAUX ===========")
    print(f"A wins : {winsA}")
    print(f"B wins : {winsB}")
    print(f"Draws  : {draws}")
    print(f"Winrate A : {winsA/(winsA+winsB+1e-9):.2%}")

    return winsA, winsB, draws


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Exemple : ton modèle gen1 contre MCTS pur
    A = ModelAgent("selfplay_runs/gen1/model_gen1.pt", simulations=200)
    B = PureMCTSAgent(simulations=200)

    arena(A, B, games=1)
