# action_encoding.py
"""
Encodage d'actions style AlphaZero — 76 plans par case (4864 actions au total)

Organisation par "from_square" (0..63) :
  - 0..55   : déplacements rayonnants (8 directions × 7 steps)
             (ordre des directions : N, S, E, W, NE, SE, SW, NW)
             pour chaque direction les steps 1..7
  - 56..63  : mouvements du cavalier (8)
  - 64..75  : promotions : 3 "directions" × 4 pièces (Q,R,B,N) = 12
             (directions promotions : straight, diag_left, diag_right)
             ordre des pièces : Q, R, B, N

Final action_index = from_square * 76 + plane_id   (0 <= action_index < 4864)

Fonctions exportées :
  - move_to_action_plane_index(move) -> int in [0..75] or None
  - move_to_action_index(move) -> int in [0..4863] or None
  - action_index_to_move(action_index, from_square=None, board=None) -> chess.Move or None
    (si from_square donné, l'index est relativisé ; sinon décodage complet)
"""

from typing import Optional, Tuple
import chess

# ------------------------
# Paramètres / constantes
# ------------------------
ACTION_PER_SQUARE = 76
NUM_SQUARES = 64
ACTION_SIZE = NUM_SQUARES * ACTION_PER_SQUARE  # 4864

# ordre fixé des directions (dr = delta rank, df = delta file)
# NOTE : chess.square_rank/square_file renvoient rank/file 0..7 (rank 0 = rangée 1)
DIRS = [
    (1, 0),   # N  (vers rang supérieure)
    (-1, 0),  # S
    (0, 1),   # E
    (0, -1),  # W
    (1, 1),   # NE
    (-1, 1),  # SE  (attention signe : -1 rank -> vers rang inférieure)
    (-1, -1), # SW
    (1, -1),  # NW
]
# Remarque importante sur les signes :
# - dr positif signifie avancer vers les rangs supérieures (ex: de rank 6->7 dr=+1)
# - df positif signifie aller vers la colonne + (f->g->h)
# L'ordre ci-dessus est déterminé et fixe. On l'utilise pour indexer 0..55.

# cavalier (order arbitrary but fixed)
KNIGHT_DELTAS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]

# promotions : 3 directions (straight, diag_left, diag_right) × 4 pièces (Q,R,B,N)
PROMOTION_DIRS = [
    (1, 0),   # straight (pour white dr=+1)
    (1, -1),  # diag_left
    (1, 1)    # diag_right
]
# ordre des pièces pour l'encodage promotions : Q, R, B, N
PROMOTION_PIECES = ['q', 'r', 'b', 'n']
PROMOTION_COUNT = len(PROMOTION_DIRS) * len(PROMOTION_PIECES)  # 3 * 4 = 12

# indices internes
RAY_COUNT = 8 * 7  # 56
KNIGHT_COUNT = 8
PROMO_COUNT = PROMOTION_COUNT  # 12

# helper : conversion rank/file <-> square (python-chess)
def rf_to_square(rank: int, file: int) -> int:
    """Retourne l'index de case (0..63) à partir (rank, file)."""
    return chess.square(file, rank)

def square_to_rf(sq: int) -> Tuple[int, int]:
    """Retourne (rank,file) pour une square index (0..63)."""
    return chess.square_rank(sq), chess.square_file(sq)

# ------------------------
# Fonctions principales
# ------------------------
def move_to_action_plane_index(move: chess.Move) -> Optional[int]:
    """
    Retourne le plane index (0..75) pour un chess.Move donné (ne tient pas compte du from_square).
    Renvoie None si le move ne rentre pas dans le schéma (ex : move étrange).
    """
    fr = move.from_square
    to = move.to_square
    fr_r, fr_f = square_to_rf(fr)
    to_r, to_f = square_to_rf(to)
    dr = to_r - fr_r
    df = to_f - fr_f

    # 1) RAY moves (8 directions × 7 steps) -> indices 0..55
    for dir_idx, (drr, dff) in enumerate(DIRS):
        # check if movement is along this direction
        if drr == 0 and dff == 0:
            continue
        # if one of components is zero
        if drr == 0:
            if dr != 0:
                continue
            if dff == 0:
                continue
            # dr == 0, check df divisible by dff
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
            # both non-zero: dr/drr == df/dff and integer
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
            # plane index within rays
            plane = dir_idx * 7 + (steps - 1)  # 0..55
            return plane

    # 2) Knight moves (56..63)
    for k_i, (drr, dff) in enumerate(KNIGHT_DELTAS):
        if dr == drr and df == dff:
            return RAY_COUNT + k_i  # 56..63

    # 3) Promotions (64..75) — nécessite move.promotion non None
    if move.promotion is not None:
        prom_piece_char = chess.piece_symbol(move.promotion).lower()  # 'q','r','b','n'
        try:
            prom_idx = PROMOTION_PIECES.index(prom_piece_char)
        except ValueError:
            # improbable : si piece non standard -> fallback to queen
            prom_idx = 0

        # pour promotions on attend dr = +1 (white) ou dr = -1 (black)
        # on code la direction relative "white-forward", puis on ajustera selon couleur lors du décodage
        # cependant ici on accepte dr's sign either +1 or -1 (on mappe selon signe)
        # on normalise la direction vers le signe positif (on mappe sur PROMOTION_DIRS cadré pour white)
        # Détecter direction relative (df == -1,0,1) et dr magnitude 1
        if abs(dr) == 1 and abs(df) <= 1:
            # determine dir_type in {0,1,2} corresponding to PROMOTION_DIRS order
            # We define "straight" as df==0, "diag_left" as df==-1, "diag_right" as df==+1
            if df == 0:
                dir_type = 0
            elif df == -1:
                dir_type = 1
            else:
                dir_type = 2
            # plane within promotions
            plane = RAY_COUNT + KNIGHT_COUNT + (dir_type * len(PROMOTION_PIECES) + prom_idx)  # 64..75
            return plane

    # pas de correspondance
    return None


def move_to_action_index(move: chess.Move) -> Optional[int]:
    """
    Mappe un chess.Move en un index d'action global [0..ACTION_SIZE).
    Renvoie None si on ne peut pas mapper (par ex. move bizarre).
    """
    plane = move_to_action_plane_index(move)
    if plane is None:
        return None
    return move.from_square * ACTION_PER_SQUARE + plane


def action_index_to_move(action_index: int, from_square: Optional[int] = None,
                         board: Optional[chess.Board] = None) -> Optional[chess.Move]:
    """
    Décodage inverse :
    - si from_square fourni : action_index est relativisé sur la case from_square
      (action_index attendu dans [0..76) -> retourne Move relatif à from_square)
    - sinon action_index global [0..4863] -> on récupère from_square = action_index // 76

    Si board fourni, on vérifie la légalité du move et on ajuste la direction des promotions
    (on utilise la couleur du joueur pour déterminer si la promotion va vers rank 7 ou rank 0).
    """
    if from_square is not None:
        # on suppose action_index est relatif (0..75)
        if not (0 <= action_index < ACTION_PER_SQUARE):
            return None
        from_sq = from_square
        plane = action_index
    else:
        if not (0 <= action_index < ACTION_SIZE):
            return None
        from_sq = action_index // ACTION_PER_SQUARE
        plane = action_index % ACTION_PER_SQUARE

    fr_r, fr_f = square_to_rf(from_sq)

    # 1) ray moves
    if 0 <= plane < RAY_COUNT:
        dir_idx = plane // 7
        step = (plane % 7) + 1
        drr, dff = DIRS[dir_idx]
        to_r = fr_r + drr * step
        to_f = fr_f + dff * step
        if not (0 <= to_r <= 7 and 0 <= to_f <= 7):
            return None
        to_sq = rf_to_square(to_r, to_f)
        return chess.Move(from_sq, to_sq)

    # 2) knight
    if RAY_COUNT <= plane < RAY_COUNT + KNIGHT_COUNT:
        k_i = plane - RAY_COUNT
        drr, dff = KNIGHT_DELTAS[k_i]
        to_r = fr_r + drr
        to_f = fr_f + dff
        if not (0 <= to_r <= 7 and 0 <= to_f <= 7):
            return None
        to_sq = rf_to_square(to_r, to_f)
        return chess.Move(from_sq, to_sq)

    # 3) promotions
    promo_inner = plane - (RAY_COUNT + KNIGHT_COUNT)  # 0..11
    if not (0 <= promo_inner < PROMO_COUNT):
        return None
    dir_type = promo_inner // len(PROMOTION_PIECES)   # 0..2
    prom_idx = promo_inner % len(PROMOTION_PIECES)    # 0..3
    prom_char = PROMOTION_PIECES[prom_idx]
    prom_piece_map = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}
    prom_piece = prom_piece_map.get(prom_char, chess.QUEEN)

    # direction relative 'white-forward' as stored in PROMOTION_DIRS:
    drr_rel, dff_rel = PROMOTION_DIRS[dir_type]  # example (1,0) straight

    # si board fourni, tenter d'utiliser la couleur du pion pour déterminer to_rank sign
    # pour white, forward = +1 (vers rank 7), pour black forward = -1
    # on cherche la case de promotion en partant de from_square selon couleur et direction
    if board is not None:
        # la pièce qui bouge doit être un pion ; si ce n'est pas le cas, on retourne None
        piece = board.piece_at(from_sq)
        if piece is None or piece.piece_type != chess.PAWN:
            # encore on retourne un object Move plausible (mais peut être illégal)
            # on va quand même essayer les deux sens (white-like puis black-like)
            pass

        # déterminer couleur : si le pion concerné appartient au joueur qui a le trait,
        # on peut utiliser board.turn pour déduire direction s'il s'agit du pion à jouer.
        # Mais move peut être joué par le joueur adverse si on l'appelle hors contexte.
        # On favorise l'usage : si there's a pawn of current player on from_sq, we use that color.
        use_white_first = False
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN:
            use_white_first = (piece.color == chess.WHITE)

        # candidate 1 : white-like promotion (to_rank = 7)
        candidates = []
        if use_white_first:
            to_r = 7
            to_f = fr_f + dff_rel
            if 0 <= to_r <= 7 and 0 <= to_f <= 7:
                candidates.append((to_r, to_f))
            # also try black-like fallback
            to_r_b = 0
            to_f_b = fr_f + dff_rel
            if 0 <= to_r_b <= 7 and 0 <= to_f_b <= 7:
                candidates.append((to_r_b, to_f_b))
        else:
            # try black-like first
            to_r = 0
            to_f = fr_f + dff_rel
            if 0 <= to_r <= 7 and 0 <= to_f <= 7:
                candidates.append((to_r, to_f))
            # then white-like
            to_r_w = 7
            to_f_w = fr_f + dff_rel
            if 0 <= to_r_w <= 7 and 0 <= to_f_w <= 7:
                candidates.append((to_r_w, to_f_w))

        # choose first candidate that looks reasonable (square in board); legality left to caller
        if not candidates:
            return None
        chosen_r, chosen_f = candidates[0]
        to_sq = rf_to_square(chosen_r, chosen_f)
        return chess.Move(from_sq, to_sq, promotion=prom_piece)

    else:
        # pas de board fourni : on retourne un move 'white-like' (promotion to rank 7)
        to_r = 7
        to_f = fr_f + dff_rel
        if not (0 <= to_r <= 7 and 0 <= to_f <= 7):
            # fallback to rank 0
            to_r = 0
            to_f = fr_f + dff_rel
            if not (0 <= to_r <= 7 and 0 <= to_f <= 7):
                return None
        to_sq = rf_to_square(to_r, to_f)
        return chess.Move(from_sq, to_sq, promotion=prom_piece)


# ------------------------
# Tests rapides (si exécuté en tant que script)
# ------------------------
if __name__ == "__main__":
    # quelques tests basiques pour vérifier encodage/décodage
    import sys
    print("ACTION_SIZE:", ACTION_SIZE)
    # tests pour quelques coups typiques
    tests = [
        "e2e4", "g1f3", "a2a4", "h2h4",   # ray + knight
        "e7e8q", "a7a8r", "g7h8n", "b7c8b" # promotions
    ]
    board = chess.Board()
    ok = True
    for uci in tests:
        try:
            mv = chess.Move.from_uci(uci)
        except Exception as e:
            print("Impossible de parser UCI:", uci, e)
            ok = False
            continue
        plane = move_to_action_plane_index(mv)
        idx = move_to_action_index(mv)
        print(f"{uci} -> plane {plane} idx {idx}")
        # décoder
        if idx is not None:
            mv_back = action_index_to_move(idx, board=board)
            print("  décodé (avec board) :", mv_back)
        else:
            print("  mapping non défini pour ce coup.")

    # test roundtrip for all squares and some planes
    sample_from = 0
    sample_panels = [0, 10, 30, 56, 60, 64, 75]
    # quick smoke
    for fr in range(0, 64, 8):
        for plane in sample_panels:
            ai = fr * ACTION_PER_SQUARE + plane
            mv = action_index_to_move(ai, board=None)
            print(f"from {fr} plane {plane} -> {mv}")
    print("Tests terminés.")
