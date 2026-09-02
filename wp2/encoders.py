import chess
import numpy as np
# torch est importé uniquement si nécessaire dans board_to_tensor

PIECE_PLANES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

from typing import Any

def board_to_tensor(board: chess.Board) -> Any:
    """
    Encode un board en tenseur (19,8,8)
    """
    planes = np.zeros((19, 8, 8), dtype=np.float32)

    # --- 12 plans pièces ---
    for square, piece in board.piece_map().items():
        rank = 7 - (square // 8)
        file = square % 8
        base = 0 if piece.color == chess.WHITE else 6
        planes[base + PIECE_PLANES[piece.piece_type], rank, file] = 1.0

    # --- camp à jouer ---
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # --- droits roque ---
    planes[13, :, :] = board.has_kingside_castling_rights(chess.WHITE)
    planes[14, :, :] = board.has_queenside_castling_rights(chess.WHITE)
    planes[15, :, :] = board.has_kingside_castling_rights(chess.BLACK)
    planes[16, :, :] = board.has_queenside_castling_rights(chess.BLACK)

    # --- 50 move rule ---
    planes[17, :, :] = board.halfmove_clock / 100.0

    # --- repetition indicator ---
    planes[18, :, :] = 1.0 if board.is_repetition(2) else 0.0

    import torch
    return torch.tensor(planes, dtype=torch.float32)

def get_fast_state_info(board: chess.Board):
    """
    Récupère toutes les infos nécessaires pour l'encodage en UN SEUL appel.
    Optimisation majeure pour le C++.
    """
    # 12 bitboards (6 blanc, 6 noir)
    occ_w = board.occupied_co[chess.WHITE]
    occ_b = board.occupied_co[chess.BLACK]
    
    return (
        board.pawns & occ_w,   # 0
        board.knights & occ_w, # 1
        board.bishops & occ_w, # 2
        board.rooks & occ_w,   # 3
        board.queens & occ_w,  # 4
        board.kings & occ_w,   # 5
        board.pawns & occ_b,   # 6
        board.knights & occ_b, # 7
        board.bishops & occ_b, # 8
        board.rooks & occ_b,   # 9
        board.queens & occ_b,  # 10
        board.kings & occ_b,   # 11
        board.turn,            # 12
        board.has_kingside_castling_rights(chess.WHITE), # 13
        board.has_queenside_castling_rights(chess.WHITE),# 14
        board.has_kingside_castling_rights(chess.BLACK), # 15
        board.has_queenside_castling_rights(chess.BLACK),# 16
        board.halfmove_clock,  # 17
        board.is_repetition(2) # 18
    )
