import chess
import chess.pgn
import os


class ChessEnv:
    """
    Environnement d'échecs minimal pour self-play ou random-play.
    S'appuie sur python-chess.
    """

    def __init__(self):
        self.board = chess.Board()
        self.moves = []

    def reset(self):
        """Réinitialise la partie et renvoie le board."""
        self.board.reset()
        self.moves = []
        return self.board

    def legal_moves(self):
        """Liste des coups légaux (objets chess.Move)."""
        return list(self.board.legal_moves)

    def step(self, move: chess.Move):
        """
        Joue un coup, renvoie (next_board, reward, done, info).
        Reward = 0 jusqu’à la fin de partie.
        """
        self.board.push(move)
        self.moves.append(move)

        done = self.board.is_game_over()
        reward = 0

        if done:
            result = self.board.result()  # "1-0", "0-1", "1/2-1/2"
            if result == "1-0":
                reward = 1
            elif result == "0-1":
                reward = -1
            else:
                reward = 0

        return self.board, reward, done, {}

    def render(self):
        """Affiche le board en ASCII."""
        print(self.board)

    def save_game(self, path: str):
        """
        Sauvegarde la partie en PGN.
        """
        game = chess.pgn.Game()
        node = game

        board = chess.Board()
        for mv in self.moves:
            node = node.add_variation(mv)

        # Créer dossier si besoin
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            exporter = chess.pgn.FileExporter(f)
            game.accept(exporter)


if __name__ == "__main__":
    import random
    
    # 1. Initialisation de l'environnement
    env = ChessEnv()
    board = env.reset()
    done = False
    
    print("--- Démarrage d'une partie aléatoire (WP1) ---")
    env.render()
    print("\n")
    
    # 2. Boucle de jeu (Livrable WP1)
    move_count = 0
    while not done:
        legal_moves = env.legal_moves()
        if not legal_moves:
            break
            
        move = random.choice(legal_moves)
        board, reward, done, info = env.step(move)
        
        move_count += 1
        # Optionnel : Afficher de temps en temps
        if move_count % 10 == 0:
            print(f"Coup {move_count} joué : {move}")
            
    print("\n--- Partie terminée ---")
    env.render()
    print(f"Résultat : {board.result()}")
    print(f"Nombre de coups : {move_count}")
    
    # 3. Sauvegarde (Livrable WP1)
    pgn_path = "data/wp1_random_game.pgn"
    env.save_game(pgn_path)
    print(f"Partie sauvegardée dans : {os.path.abspath(pgn_path)}")
