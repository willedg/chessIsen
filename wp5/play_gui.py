import tkinter as tk
from tkinter import messagebox
import threading
import chess
import chess.engine
import torch
from PIL import Image, ImageTk
import random
import numpy as np

from wp2.model import AlphaChessNet
from wp2.action_map import move_to_index
from wp3.cpp_mcts import MCTS
from wp3.predictor import Predictor

# ===============================
# CONSTANTS
# ===============================
SQUARE_SIZE = 64
LIGHT_COLOR = "#f0d9b5"
DARK_COLOR = "#b58863"

MODE_HUMAN = "human"
MODE_RANDOM = "random"
MODE_STOCKFISH = "stockfish"

# ===============================
# GUI
# ===============================
class ChessGUI:
    def __init__(self, root, predictor, device, n_sim, stockfish_path):
        self.root = root
        self.predictor = predictor
        self.device = device
        self.n_sim = n_sim
        self.stockfish_path = stockfish_path

        self.board = chess.Board()
        self.selected_square = None
        self.pending_promotion = None

        self.game_mode = None
        self.human_color = None
        self.model_color = None

        self.white_player = ""
        self.black_player = ""

        self.engine = None
        self.game_over_shown = False
        self.active_popup = None

        self.images = {}
        self.buttons = {}
        self.frame = None

        self.info_label = tk.Label(self.root, font=("Arial", 12, "bold"), pady=6)
        self.info_label.pack()

        # UI for n_sim
        self.ui_frame = tk.Frame(self.root)
        self.ui_frame.pack(pady=5)
        
        self.sim_slider = tk.Scale(self.ui_frame, from_=10, to=3200, orient=tk.HORIZONTAL, label="Simulations", length=300)
        self.sim_slider.set(n_sim)
        self.sim_slider.pack(side=tk.LEFT, padx=20)
        
        self.nps_label = tk.Label(self.ui_frame, text="NPS: 0", font=("Arial", 10))
        self.nps_label.pack(side=tk.LEFT, padx=10)

        # Control buttons for testing
        self.ctrl_frame = tk.Frame(self.root)
        self.ctrl_frame.pack(pady=5)
        tk.Button(self.ctrl_frame, text="Model Move", bg="#c3e6cb", command=lambda: self.model_turn(force=True)).pack(side=tk.LEFT, padx=5)
        tk.Button(self.ctrl_frame, text="Stockfish Move", bg="#bee5eb", command=self.stockfish_move_single).pack(side=tk.LEFT, padx=5)
        tk.Button(self.ctrl_frame, text="Reset / New Game", command=self.reset_game).pack(side=tk.LEFT, padx=5)
        tk.Button(self.ctrl_frame, text="Quit", bg="#f8d7da", command=self.quit_app).pack(side=tk.LEFT, padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        self.load_images()
        self.root.after(200, self.show_mode_choice_popup)

    def quit_app(self):
        if self.engine:
            try: self.engine.quit()
            except: pass
        self.root.destroy()

    def reset_game(self):
        if self.engine:
            self.engine.quit()
            self.engine = None

        self.board = chess.Board()
        self.selected_square = None
        self.pending_promotion = None
        self.game_mode = None
        self.human_color = None
        self.model_color = None
        self.white_player = ""
        self.black_player = ""
        self.game_over_shown = False

        if self.frame:
            self.frame.destroy()
            self.frame = None

        self.buttons.clear()
        self.info_label.config(text="")

        self.root.after(100, self.show_mode_choice_popup)

    def load_images(self):
        empty = Image.new("RGBA", (SQUARE_SIZE, SQUARE_SIZE), (0, 0, 0, 0))
        self.images["empty"] = ImageTk.PhotoImage(empty)

        for c in ["w", "b"]:
            for p in ["p", "r", "n", "b", "q", "k"]:
                try:
                    img = Image.open(f"wp5/pièces/{c}{p}.png")
                    img = img.resize((SQUARE_SIZE, SQUARE_SIZE), Image.LANCZOS)
                    self.images[c + p] = ImageTk.PhotoImage(img)
                except:
                    pass

    def show_mode_choice_popup(self):
        if self.active_popup and self.active_popup.winfo_exists():
            self.active_popup.lift()
            return
        self.active_popup = tk.Toplevel(self.root)
        self.active_popup.title("Game Mode")
        self.active_popup.grab_set()
        popup = self.active_popup

        tk.Label(popup, text="Choose game mode", font=("Arial", 14, "bold")).pack(pady=10)

        def choose(mode):
            self.game_mode = mode
            if self.active_popup:
                self.active_popup.destroy()
                self.active_popup = None
            if mode == MODE_HUMAN:
                self.show_color_choice_popup()
            else:
                self.show_model_color_popup()

        tk.Button(popup, text="Human vs Model", width=25, command=lambda: choose(MODE_HUMAN)).pack(pady=5)
        tk.Button(popup, text="Model vs Random", width=25, command=lambda: choose(MODE_RANDOM)).pack(pady=5)
        tk.Button(popup, text="Model vs Stockfish", width=25, command=lambda: choose(MODE_STOCKFISH)).pack(pady=5)
        
        tk.Frame(popup, height=10).pack() # Spacer
        tk.Button(popup, text="Load Position (FEN)", width=25, bg="#d1e7dd", command=lambda: [popup.destroy(), setattr(self, 'active_popup', None), self.root.after(10, self.show_fen_popup)]).pack(pady=5)
        
        tk.Frame(popup, height=10).pack() # Spacer
        tk.Button(popup, text="Quit Application", width=25, bg="#f8d7da", command=self.quit_app).pack(pady=5)

    def show_model_color_popup(self):
        if self.active_popup and self.active_popup.winfo_exists():
            self.active_popup.lift()
            return
        self.active_popup = tk.Toplevel(self.root)
        self.active_popup.title("Model Color")
        self.active_popup.grab_set()
        popup = self.active_popup

        tk.Label(popup, text="Model plays which color?", font=("Arial", 14, "bold")).pack(pady=10)

        def choose(color):
            self.model_color = color
            if color == chess.WHITE:
                self.white_player = "AlphaChess"
                self.black_player = "Random" if self.game_mode == MODE_RANDOM else "Stockfish"
            else:
                self.white_player = "Random" if self.game_mode == MODE_RANDOM else "Stockfish"
                self.black_player = "AlphaChess"

            if self.game_mode == MODE_STOCKFISH:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                self.engine.configure({"Skill Level": 0})

            if self.active_popup:
                self.active_popup.destroy()
                self.active_popup = None
            self.update_info_label()
            self.create_board()
            self.update_board()
            self.root.after(300, self.auto_play)

        tk.Button(popup, text="White", width=25, command=lambda: choose(chess.WHITE)).pack(pady=5)
        tk.Button(popup, text="Black", width=25, command=lambda: choose(chess.BLACK)).pack(pady=5)
        tk.Button(popup, text="Quit Application", width=25, bg="#f8d7da", command=self.quit_app).pack(pady=5)

    def show_fen_popup(self):
        if self.active_popup and self.active_popup.winfo_exists():
            self.active_popup.lift()
            return
        self.active_popup = tk.Toplevel(self.root)
        self.active_popup.title("Load Position")
        self.active_popup.grab_set()

        popup = self.active_popup

        tk.Label(popup, text="Enter FEN or select a preset:", font=("Arial", 12, "bold")).pack(pady=10)
        
        entry = tk.Entry(popup, width=60)
        entry.pack(padx=10, pady=5)

        presets = [
            ("Starting Position", chess.STARTING_FEN),
            ("Mat en 1 (Mat du Berger)", "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 5 3"),
            ("Mat en 2 (Mat du Couloir)", "r5k1/5ppp/2n5/8/8/8/5PPP/R3R1K1 w - - 0 1"),
            ("Mat en 3 (Attaque de la Reine)", "r2qk2r/pb4pp/1n2Pb2/2B2Q2/p1p5/2P5/2B2PPP/R3K2R w KQkq - 0 1"),
        ]

        def load_fen(fen):
            try:
                self.board = chess.Board(fen)
                if self.active_popup:
                    self.active_popup.destroy()
                    self.active_popup = None
                # On initialise le plateau directement pour permettre le test manuel
                self.white_player = "Position Loaded"
                self.black_player = "Testing Mode"
                self.human_color = self.board.turn # Permet de bouger les pièces du trait
                self.update_info_label()
                if not self.frame: self.create_board()
                self.update_board()
                # On s'assure que Stockfish est prêt si besoin
                if not self.engine:
                    try: self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                    except: pass
            except ValueError:
                messagebox.showerror("Error", "Invalid FEN")

        for name, fen in presets:
            tk.Button(popup, text=name, width=50, command=lambda f=fen: (entry.delete(0, tk.END), entry.insert(0, f))).pack(pady=2)

        tk.Button(popup, text="LOAD CUSTOM", bg="#0dcaf0", command=lambda: load_fen(entry.get())).pack(pady=10)
        tk.Button(popup, text="Quit Application", width=25, bg="#f8d7da", command=self.quit_app).pack(pady=5)

    def show_color_choice_popup(self):
        if self.active_popup and self.active_popup.winfo_exists():
            self.active_popup.lift()
            return
        self.active_popup = tk.Toplevel(self.root)
        self.active_popup.title("Choose Color")
        self.active_popup.grab_set()
        popup = self.active_popup

        tk.Label(popup, text="Choose your color", font=("Arial", 14, "bold")).pack(pady=10)

        def choose(color):
            self.human_color = color
            self.model_color = not color
            if color == chess.WHITE:
                self.white_player = "Human"
                self.black_player = "AlphaChess"
            else:
                self.white_player = "AlphaChess"
                self.black_player = "Human"

            if self.active_popup:
                self.active_popup.destroy()
                self.active_popup = None
            self.update_info_label()
            self.create_board()
            self.update_board()

            if self.model_color == chess.WHITE:
                self.root.after(200, self.model_turn)

        tk.Button(popup, text="Play as White", width=15, command=lambda: choose(chess.WHITE)).pack(pady=5)
        tk.Button(popup, text="Play as Black", width=15, command=lambda: choose(chess.BLACK)).pack(pady=5)
        tk.Button(popup, text="Quit Application", width=25, bg="#f8d7da", command=self.quit_app).pack(pady=5)

    def update_info_label(self):
        self.info_label.config(text=f"White: {self.white_player} | Black: {self.black_player}")

    def create_board(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack()
        for r in range(8):
            for c in range(8):
                sq = chess.square(c, 7 - r)
                color = DARK_COLOR if (c + r) % 2 == 0 else LIGHT_COLOR
                btn = tk.Button(self.frame, image=self.images["empty"], bg=color, bd=0, command=lambda s=sq: self.on_click(s))
                btn.grid(row=r, column=c)
                self.buttons[sq] = btn

    def update_board(self):
        for sq, btn in self.buttons.items():
            piece = self.board.piece_at(sq)
            if piece:
                key = ("w" if piece.color else "b") + piece.symbol().lower()
                btn.config(image=self.images.get(key, self.images["empty"]))
            else:
                btn.config(image=self.images["empty"])

    def on_click(self, square):
        if self.board.turn != self.human_color or self.pending_promotion:
            return
        if self.selected_square is None:
            if self.board.piece_at(square): self.selected_square = square
            return
        from_sq = self.selected_square
        to_sq = square
        self.selected_square = None
        piece = self.board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
            self.pending_promotion = (from_sq, to_sq)
            self.show_promotion_popup()
            return
        move = chess.Move(from_sq, to_sq)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.update_board()
            self.root.after(200, self.model_turn)

    def show_promotion_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("Promotion")
        popup.grab_set()
        tk.Label(popup, text="Promote to:", font=("Arial", 12, "bold")).pack(pady=10)
        frame = tk.Frame(popup)
        frame.pack(pady=10)
        for name, piece in [("Queen", chess.QUEEN), ("Rook", chess.ROOK), ("Bishop", chess.BISHOP), ("Knight", chess.KNIGHT)]:
            tk.Button(frame, text=name, width=10, command=lambda p=piece: self.complete_promotion(p, popup)).pack(side=tk.LEFT, padx=5)

    def complete_promotion(self, piece_type, popup):
        from_sq, to_sq = self.pending_promotion
        self.pending_promotion = None
        popup.destroy()
        move = chess.Move(from_sq, to_sq, promotion=piece_type)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.update_board()
            self.root.after(200, self.model_turn)

    def model_turn(self, force=False):
        if self.board.is_game_over(): 
            self.show_game_over_popup(); return
        if not force and self.board.turn != self.model_color: return

        # On mémorise la position actuelle pour vérifier si elle change pendant la réflexion
        current_fen = self.board.fen()

        def think():
            # GUI/Eval should NOT use Dirichlet noise and should use c_puct=1.0 for exploitation
            mcts = MCTS(self.predictor, c_puct=1.0, batch_size=64, dirichlet_eps=0.0)
            current_sims = self.sim_slider.get()
            
            # On travaille sur une copie pour éviter les conflits si le board principal change
            analysis_board = self.board.copy()
            _, best, stats = mcts.search(analysis_board, current_sims)
            
            nps = stats.get("nodes_per_second", 0)
            self.root.after(0, lambda: self.nps_label.config(text=f"NPS: {int(nps)}"))
            
            mv = best or random.choice(list(analysis_board.legal_moves)).uci()
            
            def safe_push():
                # On vérifie si la position a changé entre-temps (ex: clic humain)
                if self.board.fen() != current_fen and not force:
                    print("[GUI] Position changed during thinking, move aborted.")
                    return
                
                try:
                    move = self.board.parse_uci(mv)
                    if move in self.board.legal_moves:
                        self.play_move(mv)
                    else:
                        print(f"[GUI] Illegal move suggested: {mv}")
                except Exception as e:
                    print(f"[GUI] Error pushing move: {e}")

            self.root.after(0, safe_push)

        threading.Thread(target=think, daemon=True).start()

    def play_move(self, mv):
        self.board.push_uci(mv)
        self.update_board()
        if self.board.is_game_over(): self.show_game_over_popup()

    def stockfish_move_single(self):
        """Fait jouer un coup à Stockfish (le 'meilleur coup théorique')"""
        if self.board.is_game_over(): return
        if not self.engine:
            try: self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            except: 
                tk.messagebox.showerror("Error", "Stockfish engine not found at " + self.stockfish_path)
                return
        
        def sf_think():
            result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
            self.root.after(0, lambda: self.play_move(result.move.uci()))
        
        threading.Thread(target=sf_think, daemon=True).start()

    def auto_play(self):
        if self.board.is_game_over():
            self.show_game_over_popup(); return
        
        if self.board.turn == self.model_color:
            self.model_turn_sync()
        else:
            # Tour de l'adversaire (Random ou Stockfish)
            mv = random.choice(list(self.board.legal_moves)).uci() if self.game_mode == MODE_RANDOM else self.engine.play(self.board, chess.engine.Limit(depth=2)).move.uci()
            self.play_move(mv)
        
        self.root.after(400, self.auto_play)

    def model_turn_sync(self):
        """Version synchrone du tour du modèle pour l'auto-play"""
        mcts = MCTS(self.predictor, c_puct=1.0, batch_size=64, dirichlet_eps=0.0)
        current_sims = self.sim_slider.get()
        _, mv, stats = mcts.search(self.board, current_sims)
        nps = stats.get("nodes_per_second", 0)
        self.nps_label.config(text=f"NPS: {int(nps)}")
        if not mv:
             mv = random.choice(list(self.board.legal_moves)).uci()
        self.play_move(mv)

    def show_game_over_popup(self):
        if self.game_over_shown: return
        self.game_over_shown = True
        popup = tk.Toplevel(self.root)
        popup.title("Game Over")
        popup.grab_set()
        result = self.board.result()
        msg = "Draw!" if result == "1/2-1/2" else "White wins!" if result == "1-0" else "Black wins!"
        tk.Label(popup, text=msg, font=("Arial", 16, "bold")).pack(padx=20, pady=20)
        tk.Button(popup, text="OK", command=lambda: (popup.destroy(), self.reset_game())).pack(pady=10)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--stockfish", default="stockfish/stockfish.exe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_sim", type=int, default=2000)
    args = parser.parse_args()
    root = tk.Tk()
    root.title("AlphaChessNet")
    predictor = Predictor(args.model, device=args.device)
    ChessGUI(root, predictor, args.device, args.n_sim, args.stockfish)
    root.mainloop()
