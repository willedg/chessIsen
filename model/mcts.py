import numpy as np
from model.node import Node


class MCTS:
    def __init__(self, network, env, simulations=50, c_puct=1.0, reuse_tree=False):
        """
        MCTS optimisé :
          - minimise les appels réseau et copies d'état
          - réutilisation optionnelle de l’arbre entre tours
          - sélection vectorisée pour accélérer les itérations
        """
        self.network = network
        self.env = env
        self.simulations = simulations
        self.c_puct = c_puct
        self.reuse_tree = reuse_tree  # pour AlphaZero-like reuse

        # permet de réutiliser le sous-arbre si activé
        self.previous_root = None


    def run(self, root_state):
        """
        Lancement du MCTS sur un état initial.
        Retourne :
          - pi : distribution normalisée des visites
          - value : valeur estimée de la position
          - root : nœud racine (pour éventuelle réutilisation)
        """
        # Réutilisation d’arbre (si activée et même position)
        if self.reuse_tree and self.previous_root and self.previous_root.state == root_state:
            root = self.previous_root
        else:
            root = Node(root_state)
            self.previous_root = root

        # Cas terminal : ne pas développer
        if self.env.is_terminal(root_state):
            return {}, self.env.get_result(root_state), root

        # Initialisation des priorités avec le réseau
        policy, value = self.network.predict(root_state)
        legal_moves = list(self.env.get_legal_moves(root_state))
        if not legal_moves:
            return {}, value, root

        priors = {move: policy.get(move.uci(), 1 / len(legal_moves)) for move in legal_moves}
        root.expand(priors, self.env)

        # Boucle principale des simulations
        for _ in range(self.simulations):
            node = root
            search_path = [node]

            # --- Sélection ---
            while not node.is_leaf() and not self.env.is_terminal(node.state):
                move, node = self._select_best_child(node)
                search_path.append(node)

            # --- Évaluation / Expansion ---
            if self.env.is_terminal(node.state):
                value = self.env.get_result(node.state)
            else:
                policy, value = self.network.predict(node.state)
                legal_moves = list(self.env.get_legal_moves(node.state))
                if legal_moves:
                    priors = {move: policy.get(move.uci(), 1 / len(legal_moves)) for move in legal_moves}
                    node.expand(priors, self.env)

            # --- Rétropropagation ---
            for n in reversed(search_path):
                n.update(value)
                value = -value  # inversion de perspective

        # Distribution de probabilités (π)
        if not root.children:
            return {}, root.Q, root

        visit_counts = np.array([child.N for child in root.children.values()], dtype=np.float64)
        total_visits = np.sum(visit_counts) + 1e-8
        pi = {move: child.N / total_visits for move, child in root.children.items()}

        return pi, root.Q, root


    # --- Sélection vectorisée ---
    def _select_best_child(self, node):
        """
        Retourne (move, child) avec la plus haute valeur UCB.
        Calcul vectorisé pour réduire la latence de sélection.
        """
        children = list(node.children.values())
        moves = list(node.children.keys())

        Qs = np.array([child.Q for child in children], dtype=np.float64)
        Ns = np.array([child.N for child in children], dtype=np.float64)
        Ps = np.array([child.prior for child in children], dtype=np.float64)

        total_N = np.sum(Ns) + 1e-8
        ucb = Qs + self.c_puct * Ps * np.sqrt(total_N) / (1 + Ns)

        best_idx = int(np.argmax(ucb))
        return moves[best_idx], children[best_idx]
