import math

class Node:
    """
    Représente un nœud dans l'arbre de recherche MCTS.
    Chaque nœud correspond à un état de jeu donné.
    """

    def __init__(self, state, parent=None, prior=0.0):
        """
        Args:
            state: l'état actuel du jeu (objet environnement)
            parent: le nœud parent (None si racine)
            prior: probabilité a priori (issue de la tête Policy du réseau)
        """
        self.state = state
        self.parent = parent
        self.prior = prior  # P(s, a)
        self.children = {}  # move -> Node
        self.N = 0          # nombre de visites
        self.W = 0.0        # somme des valeurs de simulation
        self.Q = 0.0        # valeur moyenne (W / N)

    def is_leaf(self):
        """Renvoie True si le nœud n’a pas encore d’enfants."""
        return len(self.children) == 0

    def expand(self, action_priors, env):
        """
        Crée les enfants du nœud à partir des coups possibles.
        Args:
            action_priors: dict {move: probabilité}
            env: environnement de jeu (doit implémenter next_state)
        """
        for move, prob in action_priors.items():
            next_state = env.next_state(self.state, move)
            self.children[move] = Node(next_state, parent=self, prior=prob)

    def update(self, value):
        """
        Met à jour les statistiques du nœud après une simulation.
        Args:
            value: valeur retournée par la simulation (+1, -1, 0)
        """
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

    def best_child(self, c_puct):
        """
        Sélectionne le meilleur enfant selon la formule UCB1 (AlphaZero).
        Args:
            c_puct: coefficient de pondération exploration/exploitation
        Returns:
            (meilleur_move, meilleur_enfant)
        """
        best_score = -float("inf")
        best_move, best_node = None, None
        total_N = sum(child.N for child in self.children.values()) + 1e-8

        for move, child in self.children.items():
            ucb = child.Q + c_puct * child.prior * math.sqrt(total_N) / (1 + child.N)
            if ucb > best_score:
                best_score = ucb
                best_move, best_node = move, child

        return best_move, best_node
