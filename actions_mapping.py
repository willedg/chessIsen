# actions_mapping.py
import chess
import numpy as np
import torch

def build_action_space():
    """
    Construit une liste d'actions UCI fixes, et un dict inverse.
    - on génère toutes les paires from_square -> to_square (64x64)
    - si to_rank in {0,7} on ajoute les suffixes de promotion 'q','r','b','n'
    Retour :
      action_list : list of uci strings (index -> uci)
      action_index : dict uci -> index
    """
    action_list = []
    files = 'abcdefgh'
    ranks = '12345678'

    def square_name(i):
        return chess.square_name(i)

    # base moves: from -> to (no promotion suffix)
    for from_sq in range(64):
        from_name = square_name(from_sq)
        for to_sq in range(64):
            to_name = square_name(to_sq)
            uci = from_name + to_name
            # if to is promotion rank (rank 1 or 8 in human terms -> idx 0 or 7)
            to_rank = chess.square_rank(to_sq)  # 0..7
            if to_rank in (0, 7):
                # add promotions with suffixes q r b n
                for prom in ('q', 'r', 'b', 'n'):
                    action_list.append(uci + prom)
            # also include the plain move (non-promotion)
            action_list.append(uci)

    # make unique and deterministic (should already be unique)
    # create inverse mapping
    action_index = {uci: idx for idx, uci in enumerate(action_list)}
    return action_list, action_index


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convertit un chess.Board en tenseur numpy 8x8x18 (int8).
    Plans :
      0-5 : pièces blanches P,N,B,R,Q,K (in this order)
      6-11: pièces noires
      12..15: castling rights (w_kingside, w_queenside, b_kingside, b_queenside)
      16: side-to-move (1 blanc, 0 noir) -> broadcast to plane
      17: en-passant square (one-hot)
    """
    planes = np.zeros((8, 8, 18), dtype=np.int8)

    for i, piece_type in enumerate(chess.PIECE_TYPES):
        bb_white = board.pieces(piece_type, chess.WHITE)
        arr_w = np.array([(bb_white >> sq) & 1 for sq in range(64)], dtype=np.int8).reshape(8, 8)[::-1]
        planes[:, :, i] = arr_w
        bb_black = board.pieces(piece_type, chess.BLACK)
        arr_b = np.array([(bb_black >> sq) & 1 for sq in range(64)], dtype=np.int8).reshape(8, 8)[::-1]
        planes[:, :, i + 6] = arr_b

    planes[:, :, 12] = int(board.has_kingside_castling_rights(chess.WHITE))
    planes[:, :, 13] = int(board.has_queenside_castling_rights(chess.WHITE))
    planes[:, :, 14] = int(board.has_kingside_castling_rights(chess.BLACK))
    planes[:, :, 15] = int(board.has_queenside_castling_rights(chess.BLACK))

    planes[:, :, 16] = int(board.turn == chess.WHITE)

    if board.ep_square is not None:
        r = 7 - chess.square_rank(board.ep_square)
        f = chess.square_file(board.ep_square)
        planes[r, f, 17] = 1

    return planes
