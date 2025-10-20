# mcts_with_net.py
import math
import random
from model.node import Node  # ton Node existant
import numpy as np

class MCTS:
    def __init__(self, network_wrapper, env, simulations=800, c_puct=1.4):
        """
        network_wrapper: instance de TorchNetWrapper (ou équivalent) fournissant .predict(board)
        env: ChessEnv avec get_legal_moves(board), next_state(board,move), is_terminal(board), get_result(board)
        """
        self.network = network_wrapper
        self.env = env
        self.simulations = simulations
        self.c_puct = c_puct

    def _get_priors_for_legal(self, board):
        """
        Appelle le réseau, récupère la policy sur l'ensemble des actions,
        puis renvoie un dict {move (chess.Move): prior}
        """
        policy_dict, value = self.network.predict(board)
        legal_moves = list(self.env.get_legal_moves(board))  # list of chess.Move
        if len(legal_moves) == 0:
            return {}, value

        priors = {}
        # map network's probabilities to actual Move objects by their uci
        for mv in legal_moves:
            u = mv.uci()
            p = policy_dict.get(u, 1e-8)  # fallback tiny prob if network didn't produce it
            priors[mv] = p
        # normalize priors on legal moves
        s = sum(priors.values())
        if s <= 0:
            # uniform
            k = 1.0 / len(priors)
            for m in priors:
                priors[m] = k
        else:
            for m in priors:
                priors[m] = priors[m] / s
        return priors, value

    def run(self, root_state):
        root = Node(root_state)

        # don't expand terminal
        if self.env.is_terminal(root_state):
            return {}, self.env.get_result(root_state), root

        priors, _ = self._get_priors_for_legal(root_state)
        root.expand(priors, self.env)

        for _ in range(self.simulations):
            node = root
            search_path = [node]

            # Selection
            while not node.is_leaf() and not self.env.is_terminal(node.state):
                move, node = node.best_child(self.c_puct)
                search_path.append(node)

            # Evaluation / Expansion
            if self.env.is_terminal(node.state):
                value = self.env.get_result(node.state)
            else:
                priors, value = self._get_priors_for_legal(node.state)
                if priors:
                    node.expand(priors, self.env)

            # Backpropagation
            for n in reversed(search_path):
                n.update(value)
                value = -value

        # final policy pi from visit counts
        if not root.children:
            return {}, root.Q, root

        visit_counts = {move: child.N for move, child in root.children.items()}
        total_visits = float(sum(visit_counts.values())) + 1e-8
        pi = {move: child.N / total_visits for move, child in root.children.items()}

        return pi, root.Q, root
