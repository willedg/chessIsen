# main_example.py
from actions_mapping import build_action_space
from torch_wrapper import TorchNetWrapper
from model.WP2_Architecture_Neurone import NeuralNetwork  # ton réseau
from model.mcts_with_net import MCTS
import chess

# 1) build action space
action_list, action_index = build_action_space()
ACTION_SIZE = len(action_list)
print("ACTION_SIZE =", ACTION_SIZE)  # configure ton réseau avec cette taille

# 2) instantiate model (same arch as tu as)
model = NeuralNetwork(input_channels=18, board_size=8, num_actions=ACTION_SIZE)
# charger poids si tu as un checkpoint :
# model.load_state_dict(torch.load("model_best.pth", map_location="cpu"))

# 3) wrapper
wrapper = TorchNetWrapper(model, action_list, action_index)

# 4) env (ton ChessEnv)
class ChessEnv:
    def get_legal_moves(self, board):
        return list(board.legal_moves)
    def next_state(self, board, move):
        nb = board.copy(stack=False)
        nb.push(move)
        return nb
    def is_terminal(self, board):
        return board.is_game_over()
    def get_result(self, board):
        oc = board.outcome()
        if oc is None: return 0
        if oc.winner is None: return 0
        return 1 if oc.winner else -1

env = ChessEnv()
mcts = MCTS(wrapper, env, simulations=200, c_puct=1.4)

board = chess.Board()
pi, value, root = mcts.run(board)

# pi keys are chess.Move objects; to pick best move:
best_move = max(pi.items(), key=lambda x: x[1])[0]
print("Best move:", best_move)

# affiche tous les coups avec leurs probabilités
for move, prob in pi.items():
    print(f"Move: {move.uci()} | Probability: {prob:.4f}")
