import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    """
    Réseau de neurones de type AlphaZero :
    - Prend en entrée un état de jeu (8x8x18)
    - Retourne :
        * une politique (probabilités sur les coups)
        * une valeur (évaluation de la position entre -1 et 1)
    """

    def __init__(self, input_channels=18, board_size=8, num_actions=4672):
        super(NeuralNetwork, self).__init__()

        # === 1.Corps convolutionnel (feature extractor) ===
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # === 2.Policy Head ===
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1),   # réduit les canaux
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, num_actions)
        )

        # === 3.Value Head ===
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # valeur entre -1 et 1
        )

    def forward(self, x):
        """
        Passe avant : prend un état du jeu et retourne la policy et la value.
        """
        # 1. Extraction des features du plateau
        features = self.conv_block(x)

        # 2. Policy (probabilités de coups)
        policy_logits = self.policy_head(features)

        # 3. Value (évaluation de la position)
        value = self.value_head(features)

        return policy_logits, value


if __name__ == "__main__":
    net = NeuralNetwork()
    x = torch.randn(1, 18, 8, 8)  # batch de 1 état
    policy, value = net(x)
    print("Policy shape:", policy.shape)
    print("Value shape:", value.shape)
