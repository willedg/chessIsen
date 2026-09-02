import argparse
import chess
import chess.engine
import torch
import numpy as np

from wp2.model import AlphaChessNet
from wp2.encoders import board_to_tensor
from wp2.action_map import move_to_index, ACTION_SPACE
from wp3.cpp_mcts import MCTS


# -----------------------------
# Chargement du modèle
# -----------------------------
def load_model(path, device):
    model = AlphaChessNet()
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


# -----------------------------
# Prédiction NN (batch MCTS)
# -----------------------------
def predict_batch(model, boards, device):
    xs = []
    for b in boards:
        x = board_to_tensor(b)
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        xs.append(x)

    x = torch.stack(xs).float().to(device)

    with torch.no_grad():
        logits, values = model(x)

    logits = logits.cpu().numpy()
    values = values.cpu().numpy().reshape(-1)

    outs = []
    for i, board in enumerate(boards):
        probs = np.exp(logits[i] - np.max(logits[i]))
        probs /= probs.sum() + 1e-12

        priors = {}
        for mv in board.legal_moves:
            try:
                idx = move_to_index(mv)
                if 0 <= idx < ACTION_SPACE:
                    priors[mv.uci()] = float(probs[idx])
            except Exception:
                pass

        if not priors:
            legal = list(board.legal_moves)
            u = 1.0 / len(legal)
            for mv in legal:
                priors[mv.uci()] = u

        outs.append((priors, float(values[i])))

    return outs


# -----------------------------
# Un coup joué par ton modèle
# -----------------------------
def model_move(board, model, device, n_sim):
    predict_fn = lambda b: predict_batch(model, b, device)

    mcts = MCTS(
        predict_fn,
        c_puct=1.0,
        batch_size=32,
        top_k=10000,
        dirichlet_alpha=0.0,
        dirichlet_eps=0.0
    )

    pi, best, stats = mcts.search(board, n_sim)

    if best is None:
        legal = list(board.legal_moves)
        best = legal[0].uci()

    return best


# -----------------------------
# Match contre Stockfish
# -----------------------------
def eval_vs_stockfish(model_path, n_games, n_sim, stockfish_path, device):
    model = load_model(model_path, device)

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": 0})

    score = 0.0

    for g in range(n_games):
        board = chess.Board()

        model_color = chess.WHITE if (g % 2 == 0) else chess.BLACK

        while not board.is_game_over(claim_draw=True):
            if board.turn == model_color:
                move = model_move(board, model, device, n_sim)
                board.push_uci(move)
            else:
                result = engine.play(
                    board,
                    chess.engine.Limit(depth=2)
                )
                board.push(result.move)

        res = board.result(claim_draw=True)

        if res == "1-0":
            winner = chess.WHITE
        elif res == "0-1":
            winner = chess.BLACK
        else:
            winner = None

        if winner is None:
            score += 0.5
            outcome = "DRAW"
        elif winner == model_color:
            score += 1.0
            outcome = "MODEL WIN"
        else:
            outcome = "MODEL LOSS"

        print(f"[GAME {g+1}] {outcome} ({res})")



    engine.quit()
    return score / n_games


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--stockfish", default="stockfish/stockfish.exe", help="Path to stockfish binary")
    parser.add_argument("--n_games", type=int, default=10)
    parser.add_argument("--n_sim", type=int, default=1600)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    winrate = eval_vs_stockfish(
        model_path=args.model,
        n_games=args.n_games,
        n_sim=args.n_sim,
        stockfish_path=args.stockfish,
        device=args.device
    )

    print("\n============================")
    print("Winrate vs Stockfish =", winrate)
    print("============================")
