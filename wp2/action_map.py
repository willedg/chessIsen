import chess

# 64*64 moves (0-4095) + promotions (4096-4671)
ACTION_SPACE = 4672

def move_to_index(move: chess.Move) -> int:
    """
    Convertit un move python-chess en index entier.
    Mapping robuste pour les promotions (supporte from_sq et to_sq).
    """
    from_sq = move.from_square
    to_sq = move.to_square

    if move.promotion:
        # On utilise un espace dédié pour les promotions (4096 + ...)
        # On a besoin de distinguer la pièce, et la colonne de départ/arrivée.
        # promo_map: Knight=0, Bishop=1, Rook=2, Queen=3
        promo_map = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3}
        p_idx = promo_map[move.promotion]
        
        from_file = from_sq % 8
        to_file = to_sq % 8
        is_black = (from_sq < 16) # Rank 2
        
        # Mapping: 4096 + (Color * 256) + (Piece * 64) + (From_File * 8) + To_File
        # Total: 4096 + 512 = 4608 (tient dans 4672)
        offset = (1 if is_black else 0) * 256 + p_idx * 64 + from_file * 8 + to_file
        return 4096 + offset

    return from_sq * 64 + to_sq

def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Reconstruit un coup depuis un index.
    """
    if index < 4096:
        from_sq = index // 64
        to_sq = index % 64
        return chess.Move(from_sq, to_sq)

    # Promotions
    offset = index - 4096
    is_black = (offset // 256) == 1
    p_idx = (offset % 256) // 64
    from_file = (offset % 64) // 8
    to_file = offset % 8
    
    promo_map = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    
    from_sq = (1 if is_black else 6) * 8 + from_file
    to_sq = (0 if is_black else 7) * 8 + to_file
    
    return chess.Move(from_sq, to_sq, promotion=promo_map[p_idx])
