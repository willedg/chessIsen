"""
self_play_wp4.py
Génération de parties self-play (WP4.1)

- Utilise le mapping 64x73 (AlphaZero-like) => ACTION_SIZE calculé automatiquement.
- Utilise tes classes existantes :
    * mcts_with_net.MCTS  (ou equivalent — on instancie avec un wrapper)
    * WP2_Architecture_Neurone (ton modèle PyTorch)
    * node.Node (déjà intégré dans mcts)
- Sauvegarde un fichier compressé 'selfplay_data.npz' contenant :
    states:   (N, 18, 8, 8)  dtype=float32 (CHW)
    pis:      (N, ACTION_SIZE) dtype=float32
    zs:       (N,)             dtype=float32 (+1 / -1 / 0)
    players:  (N,)             dtype=int8   (1=white to move, 0=black)
- Sélection du coup : échantillonnage selon π (sample)
"""

import numpy as np
import chess
import random
import torch
import os
from typing import Tuple, List

# ----- IMPORT DE TES CLASSES (adapte le chemin si nécessaire) -----
# On suppose que :
#  - mcts_with_net contient la classe MCTS compatible avec wrapper (predict -> probs,value)
#  - WP2_Architecture_Neurone est le fichier/class de ton réseau
#  - node existe et est utilisé par mcts_with_net
#
# Exemple d'import (adapte selon l'arborescence de ton projet) :
from model.mcts_with_net import MCTS                        # ta classe MCTS adaptée
from model.WP2_Architecture_Neurone import NeuralNetwork    # ta classe de réseau (PyTorch)
# node import inutile si mcts_with_net s'en charge, sinon:
# from node import Node

# ----------------------------
# 1) Mapping 64 x 73 (AlphaZero)
# ----------------------------
# Par-from-square on définit 73 actions :
#  - 56 queen-like (8 directions × steps 1..7)
#  - 8 coups de cavalier
#  - 9 promotions (3 directions × 3 promotions {q,r,b})
ACTION_PER_SQUARE = 73
NUM_SQUARES = 64
ACTION_SIZE = 4864 

# directions (dr, df) : dr=delta rank, df=delta file
DIRS = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
KNIGHT_DELTAS = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
PROMOTION_PIECES = ('q','r','b')  # on encode 3 types pour les promotions (pas le knight)

def square_rank_file(sq: int) -> Tuple[int,int]:
    """Retourne (rank,file) pour une square index (0..63). rank,file ∡ 0..7"""
    return chess.square_rank(sq), chess.square_file(sq)

def in_board(r:int, f:int) -> bool:
    return 0 <= r <= 7 and 0 <= f <= 7

def from_rf_to_sq(r:int, f:int) -> int:
    """Retourne l'index de square pour rank r et file f (attention ordre dans python-chess)"""
    return chess.square(f, r)

# --- Fonctions de mapping ---
def move_to_action_index(move: chess.Move) -> int:
    """
    Mappe un chess.Move -> action_index dans [0, ACTION_SIZE).
    Renvoie None si le mapping ne peut pas être déterminé (très rare avec ce scheme).
    """
    from_sq = move.from_square
    to_sq = move.to_square
    fr_r, fr_f = square_rank_file(from_sq)
    to_r, to_f = square_rank_file(to_sq)
    dr = to_r - fr_r
    df = to_f - fr_f
    base = from_sq * ACTION_PER_SQUARE

    # 1) directions queen-like (56 slots: dir_i * 7 + (steps-1))
    for dir_i, (drr, dff) in enumerate(DIRS):
        # cas où le mouvement correspond à cette direction
        # gestion générale — vérifier divisibilité
        if drr == 0 and dff == 0:
            continue
        # si l'un des composants est zéro
        if drr == 0:
            if dr != 0:
                continue
            if dff == 0:
                continue
            if df == 0 or (df // dff) * dff != df:
                continue
            steps = abs(df)
        elif dff == 0:
            if df != 0:
                continue
            if dr == 0 or (dr // drr) * drr != dr:
                continue
            steps = abs(dr)
        else:
            if dr == 0 or df == 0:
                continue
            if dr % drr != 0 or df % dff != 0:
                continue
            k1 = dr // drr
            k2 = df // dff
            if k1 != k2:
                continue
            steps = abs(k1)

        if 1 <= steps <= 7:
            idx_within = dir_i * 7 + (steps - 1)  # 0..55
            return base + idx_within

    # 2) Knight moves (offset 56..63)
    for k_i, (drr, dff) in enumerate(KNIGHT_DELTAS):
        if dr == drr and df == dff:
            return base + 56 + k_i

    # 3) Promotions (offset 64..72 ; only for pawn promotions)
    if move.promotion is not None:
        # char(s) promotion : 'q','r','b','n' (we only map q/r/b explicitly)
        prom_char = chess.piece_symbol(move.promotion).lower()
        # normalize knights to queen slot if knight promotion not present in scheme
        if prom_char == 'n':
            prom_idx = 0
        else:
            try:
                prom_idx = PROMOTION_PIECES.index(prom_char)
            except ValueError:
                prom_idx = 0
        # déterminer la 'direction' relative (forward / cap-left / cap-right)
        # df in {-1,0,1} and dr magnitude 1 (pawn promotion move)
        if abs(dr) == 1 and abs(df) <= 1:
            if df == 0:
                dir_type = 0
            elif df == -1:
                dir_type = 1
            else:
                dir_type = 2
            promo_slot = dir_type * 3 + prom_idx  # 0..8
            return base + 64 + promo_slot

    # Si aucun mapping trouvé, retourne None
    return None

def action_index_to_move(board: chess.Board, action_index: int) -> chess.Move:
    """
    Décodage d'un action_index en chess.Move (sans vérifier la légalité).
    Renvoie None si l'index est invalide ou mène hors plateau.
    """
    if action_index < 0 or action_index >= ACTION_SIZE:
        return None
    from_sq = action_index // ACTION_PER_SQUARE
    inner = action_index % ACTION_PER_SQUARE
    fr_r, fr_f = square_rank_file(from_sq)

    if inner < 56:
        dir_i = inner // 7
        step = (inner % 7) + 1
        drr, dff = DIRS[dir_i]
        to_r = fr_r + drr * step
        to_f = fr_f + dff * step
        if not in_board(to_r, to_f):
            return None
        to_sq = from_rf_to_sq(to_r, to_f)
        return chess.Move(from_sq, to_sq)

    if inner < 64:
        k_i = inner - 56
        drr, dff = KNIGHT_DELTAS[k_i]
        to_r = fr_r + drr
        to_f = fr_f + dff
        if not in_board(to_r, to_f):
            return None
        to_sq = from_rf_to_sq(to_r, to_f)
        return chess.Move(from_sq, to_sq)

    # promotions
    promo_slot = inner - 64
    dir_type = promo_slot // 3  # 0 forward, 1 cap-left, 2 cap-right
    prom_idx = promo_slot % 3
    prom_piece_char = PROMOTION_PIECES[prom_idx]
    prom_map = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP}
    prom_piece = prom_map[prom_piece_char]

    if dir_type == 0:
        df = 0
    elif dir_type == 1:
        df = -1
    else:
        df = 1

    # On essaie d'inférer la case d'arrivée :
    # pour simplicité on prend la case finale (rank 7) pour white-like, sinon 0 pour black-like,
    # mais l'appelant devra vérifier la légalité sur le board.
    # On essaie d'abord rank 7 (promotion white), puis rank 0.
    to_r = 7
    to_f = fr_f + df
    if not in_board(to_r, to_f):
        to_r = 0
        to_f = fr_f + df
        if not in_board(to_r, to_f):
            return None
    to_sq = from_rf_to_sq(to_r, to_f)
    return chess.Move(from_sq, to_sq, promotion=prom_piece)

# ----------------------------
# 2) Conversion board -> tenseur (CHW, float32)
# ----------------------------
def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convertit chess.Board -> numpy array (18, 8, 8) dtype=float32
    Ordre des canaux :
      0-5   : pièces blanches P,N,B,R,Q,K
      6-11  : pièces noires
      12..15: droits de roque (w_kingside, w_queenside, b_kingside, b_queenside)
      16    : trait (1=white, 0=black) broadcast sur plan
      17    : en-passant (one-hot)
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    for i, piece_type in enumerate(chess.PIECE_TYPES):
        bb_w = board.pieces(piece_type, chess.WHITE)
        arr_w = np.array([(bb_w >> sq) & 1 for sq in range(64)], dtype=np.float32).reshape(8,8)[::-1]
        planes[i, :, :] = arr_w
        bb_b = board.pieces(piece_type, chess.BLACK)
        arr_b = np.array([(bb_b >> sq) & 1 for sq in range(64)], dtype=np.float32).reshape(8,8)[::-1]
        planes[i+6, :, :] = arr_b

    planes[12, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[13, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[14, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[15, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    planes[16, :, :] = float(board.turn == chess.WHITE)

    if board.ep_square is not None:
        r = 7 - chess.square_rank(board.ep_square)
        f = chess.square_file(board.ep_square)
        planes[17, r, f] = 1.0

    return planes  # shape (18,8,8)

# ----------------------------
# 3) Wrapper PyTorch pour ton réseau
# ----------------------------
class TorchNetWrapper:
    """
    Wrapper qui adapte ton NeuralNetwork à l'API attendue par MCTS:
      predict(board, temperature=1.0) -> (probs_vector (np.array length ACTION_SIZE), value_float)
    """
    def __init__(self, model: torch.nn.Module, device: str = None):
        self.model = model
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, board: chess.Board, temperature: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Convertit le board en tenseur, passe dans le modèle et renvoie :
          - probs : numpy array shape (ACTION_SIZE,) (softmax over logits)
          - value : float (between -1 and 1)
        """
        x = board_to_tensor(board)            # (18,8,8) float32
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1,18,8,8)
        with torch.no_grad():
            logits, val = self.model(xt)      # logits: (1, ACTION_SIZE), val: (1,1)
            logits = logits.squeeze(0).cpu().numpy()
            value = float(val.squeeze(0).squeeze(0).cpu().numpy())

        if temperature != 1.0:
            logits = logits / float(temperature)

        # Softmax stable
        maxl = logits.max()
        exps = np.exp(logits - maxl)
        probs = exps / (exps.sum() + 1e-12)
        return probs, value

# ----------------------------
# 4) Self-play : utiliser ton MCTS (mcts_with_net) et stocker données
# ----------------------------
def pi_dict_from_visit_counts(visit_counts: dict) -> dict:
    """
    Convertit visit_counts (move -> N) en distribution π (move -> prob).
    Normalise la somme à 1.
    """
    total = float(sum(visit_counts.values())) + 1e-12
    return {move: cnt/total for move, cnt in visit_counts.items()}

def self_play_one_game(mcts: MCTS, sample_temperature: float = 1.0):
    """
    Joue une partie complète en self-play :
    - à chaque position, lance MCTS.run(board) pour obtenir pi (visit distribution) et value
    - construit pi_vector (ACTION_SIZE) en mappant chaque move -> move_to_action_index
    - choisit le coup par échantillonnage selon pi (sample)
    - retourne arrays : states (T,18,8,8), pis (T,ACTION_SIZE), zs (T,), players (T,)
    """
    env = mcts.env
    board = chess.Board()
    states = []
    pis = []
    players = []

    while not env.is_terminal(board):
        # Lancer MCTS pour obtenir pi (dict move->prob) et value
        pi_move_dict, value_est, _ = mcts.run(board)   # on suppose mcts.run renvoie (pi_dict, value, root)
        # Si MCTS renvoie pi direct sous forme vectorielle, il faudra adapter — ici on suppose dict

        # Construire pi_vector (ACTION_SIZE) initialisé à 0
        pi_vector = np.zeros(ACTION_SIZE, dtype=np.float32)
        # remplir pi_vector à partir de pi_move_dict
        for mv, p in pi_move_dict.items():
            idx = move_to_action_index(mv)
            if idx is None:
                # Si un move ne mappe pas (peu probable), on l'ignore
                continue
            pi_vector[idx] = p

        # si somme nulle (sûreté), uniformiser sur coups légaux
        s = pi_vector.sum()
        if s <= 0:
            legal = list(env.get_legal_moves(board))
            for mv in legal:
                idx = move_to_action_index(mv)
                if idx is not None:
                    pi_vector[idx] = 1.0 / len(legal)
            s = pi_vector.sum()
        # normalisation
        pi_vector = pi_vector / (s + 1e-12)

        # Stocker état, pi et player
        states.append(board_to_tensor(board))
        pis.append(pi_vector)
        players.append(int(board.turn == chess.WHITE))

        # Choix du coup : échantillonnage selon la distribution pondérée sur coups légaux
        legal_moves = list(env.get_legal_moves(board))
        # construire liste de probabilités pour ces coups
        probs_for_legals = []
        for mv in legal_moves:
            probs_for_legals.append(pi_move_dict.get(mv, 0.0))
        # si toutes 0 -> uniform
        if sum(probs_for_legals) <= 0:
            chosen_move = random.choice(legal_moves)
        else:
            arr = np.array(probs_for_legals, dtype=np.float32)
            arr = arr / (arr.sum() + 1e-12)
            # sample index
            i = np.random.choice(len(legal_moves), p=arr)
            chosen_move = legal_moves[i]

        # pousser le coup choisi
        board.push(chosen_move)

    # La partie est terminée : déterminer résultat final et produire z pour chaque position
    result = board.result()  # "1-0", "0-1", "1/2-1/2"
    if result == "1-0":
        winner = 1
    elif result == "0-1":
        winner = -1
    else:
        winner = 0

    zs = []
    for player in players:
        if winner == 0:
            z = 0.0
        else:
            # z = +1 si le joueur à l'instant considéré a gagné, -1 sinon
            if winner == 1:
                z = 1.0 if player == 1 else -1.0
            else: # winner == -1 (black)
                z = 1.0 if player == 0 else -1.0
        zs.append(z)

    # convertir listes en arrays
    states = np.stack(states).astype(np.float32)  # (T,18,8,8)
    pis = np.stack(pis).astype(np.float32)        # (T,ACTION_SIZE)
    zs = np.array(zs, dtype=np.float32)           # (T,)
    players = np.array(players, dtype=np.int8)     # (T,)
    return states, pis, zs, players

# ----------------------------
# 5) Génération N parties et sauvegarde
# ----------------------------
def generate_self_play_games(mcts: MCTS, num_games: int = 10, out_path: str = "selfplay_data.npz"):
    all_states = []
    all_pis = []
    all_zs = []
    all_players = []
    for i in range(num_games):
        s, p, z, pl = self_play_one_game(mcts)
        all_states.append(s)
        all_pis.append(p)
        all_zs.append(z)
        all_players.append(pl)
        print(f"[{i+1}/{num_games}] partie terminée : coups={s.shape[0]} positions")

    # concaténation
    states = np.concatenate(all_states, axis=0)
    pis = np.concatenate(all_pis, axis=0)
    zs = np.concatenate(all_zs, axis=0)
    players = np.concatenate(all_players, axis=0)

    # sauvegarde compressée
    np.savez_compressed(out_path, states=states, pis=pis, zs=zs, players=players)
    print(f"Sauvegardé {states.shape[0]} positions dans {out_path}")
    return out_path

# ----------------------------
# 6) Exemple d'usage (main)
# ----------------------------
if __name__ == "__main__":
    # 1) instancier le modèle (assure-toi de donner ACTION_SIZE comme output)
    model = NeuralNetwork(input_channels=18, board_size=8, num_actions=ACTION_SIZE)

    # 2) wrapper du model
    wrapper = TorchNetWrapper(model)

    # 3) environnement
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
            if oc is None: return 0.0
            if oc.winner is None: return 0.0
            return 1.0 if oc.winner else -1.0

    env = ChessEnv()

    # 4) MCTS : on utilise la classe MCTS existante (mcts_with_net) ; on lui donne le wrapper et env
    mcts = MCTS(wrapper, env, simulations=200, c_puct=1.4)

    # 5) Lancement de la génération (par exemple 5 parties)
    generate_self_play_games(mcts, num_games=10, out_path="selfplay_data.npz")
