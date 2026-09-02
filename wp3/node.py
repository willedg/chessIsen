# wp3/node.py
from typing import Dict, Optional

class Node:
    """
    Noeud MCTS (Monte Carlo Tree Search).
    Stocke:
        - prior P (probabilité issue du réseau)
        - N (visits)
        - W (sum of values)
        - Q (W / N) (computed on the fly)
        - children: dict(move_uci -> Node)
    """

    def __init__(self, prior: float = 0.0, parent: Optional["Node"] = None, move_uci: Optional[str] = None):
        self.P = float(prior)
        self.N = 0
        self.W = 0.0
        self.parent = parent
        self.move_uci = move_uci  # string UCI of the move that led here from parent
        self.children: Dict[str, Node] = {}

    @property
    def Q(self):
        return 0.0 if self.N == 0 else self.W / self.N

    def expand(self, priors: dict):
        """
        priors: mapping move_uci -> prior_probability (float)
        Ajoute des enfants pour chaque move présent dans priors (s'il n'existe pas déjà).
        """
        for m_uci, p in priors.items():
            if m_uci not in self.children:
                self.children[m_uci] = Node(prior=p, parent=self, move_uci=m_uci)

    def is_leaf(self):
        return len(self.children) == 0

    def backup(self, value: float):
        """
        Backpropagate value up to root. value is from the perspective of the node's side to play at root.
        We assume the value is from the viewpoint of the node where evaluate() was called:
        - At each parent step we propagate with sign flip: v -> -v
        """
        node = self
        cur_value = value
        while node is not None:
            node.N += 1
            node.W += cur_value
            cur_value = -cur_value  # flip for opponent
            node = node.parent

    def best_child(self, c_puct: float):
        """
        Return child_key, child_node maximizing UCB = Q + U
        U = c_puct * P * sqrt(sum_N) / (1 + N_child)
        """
        sum_N = sum(child.N for child in self.children.values())
        best_score = -float("inf")
        best_move = None
        best_node = None
        for m_uci, child in self.children.items():
            u = c_puct * child.P * ( (sum_N ** 0.5) / (1 + child.N) )
            score = child.Q + u
            if score > best_score:
                best_score = score
                best_move = m_uci
                best_node = child
        return best_move, best_node
