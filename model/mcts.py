import numpy as np
from model.node import Node

class MCTS:
    def __init__(self, network, env, simulations=50, c_puct=1.0):
        self.network = network
        self.env = env
        self.simulations = simulations
        self.c_puct = c_puct

    def run(self, root_state):
        root = Node(root_state)

        if self.env.is_terminal(root_state):
            return {}, self.env.get_result(root_state), root

        policy, value = self.network.predict(root_state)
        legal_moves = self.env.get_legal_moves(root_state)

        priors = {}
        for move in legal_moves:
            move_uci = move.uci()
            priors[move] = policy.get(move_uci, 1 / len(legal_moves))

        root.expand(priors, self.env)

        for _ in range(self.simulations):
            node = root
            search_path = [node]

            while not node.is_leaf() and not self.env.is_terminal(node.state):
                move, node = node.best_child(self.c_puct)
                search_path.append(node)

            if self.env.is_terminal(node.state):
                value = self.env.get_result(node.state)
            else:
                policy, value = self.network.predict(node.state)
                legal_moves = self.env.get_legal_moves(node.state)

                priors = {}
                for move in legal_moves:
                    move_uci = move.uci()
                    priors[move] = policy.get(move_uci, 1 / len(legal_moves))

                node.expand(priors, self.env)

            for n in reversed(search_path):
                n.update(value)
                value = -value

        # garde-fou si aucun enfant
        if not root.children:
            return {}, root.Q

        visit_counts = {move: child.N for move, child in root.children.items()}
        total_visits = sum(visit_counts.values()) + 1e-8
        pi = {move: N / total_visits for move, N in visit_counts.items()}

        return pi, root.Q, root
    
