import pygame
import chess
import chess.svg
import numpy as np
import random
import io
import cairosvg

# ========= 1) Tensor =========
def board_to_tensor(board: chess.Board) -> np.ndarray:
    planes = np.zeros((8, 8, 18), dtype=np.int8)
    for i, pt in enumerate(chess.PIECE_TYPES):
        planes[:,:,i]   = np.array([(board.pieces(pt,1)>>sq)&1 for sq in range(64)]).reshape(8,8)[::-1]
        planes[:,:,i+6] = np.array([(board.pieces(pt,0)>>sq)&1 for sq in range(64)]).reshape(8,8)[::-1]
    planes[:,:,12]=board.has_kingside_castling_rights(1)
    planes[:,:,13]=board.has_queenside_castling_rights(1)
    planes[:,:,14]=board.has_kingside_castling_rights(0)
    planes[:,:,15]=board.has_queenside_castling_rights(0)
    planes[:,:,16]=board.turn==1
    if board.ep_square:
        r=7-chess.square_rank(board.ep_square)
        f=chess.square_file(board.ep_square)
        planes[r,f,17]=1
    return planes

# ========= 2) pygame loop =========
def run_gui(speed=0.6):
    pygame.init()
    W,H=600,600
    screen = pygame.display.set_mode((W,H))
    pygame.display.set_caption("WP1 — Chess Viewer (pygame)")

    board = chess.Board()
    tensors=[]
    clock = pygame.time.Clock()
    move_count=0

    running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                running=False

        if not board.is_game_over():
            # --- SVG → PNG -> Surface
            svg_bytes = chess.svg.board(board).encode("utf-8")
            png_bytes = cairosvg.svg2png(bytestring=svg_bytes)
            surf = pygame.image.load(io.BytesIO(png_bytes))
            surf = pygame.transform.smoothscale(surf,(W,H))
            screen.blit(surf,(0,0))
            pygame.display.flip()

            tensors.append(board_to_tensor(board))
            pygame.time.wait(int(speed*1000))

            board.push(random.choice(list(board.legal_moves)))
            move_count+=1
        else:
            running=False

        clock.tick(60)

    print(f"Partie terminée : {board.result()} | coups={move_count} | tenseurs={len(tensors)}")
    pygame.quit()
    return tensors


# ========= 3) main =========
if __name__=="__main__":
    ts = run_gui(speed=0.5)
