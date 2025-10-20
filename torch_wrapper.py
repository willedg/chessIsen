# torch_wrapper.py
import torch
import numpy as np

# import your NeuralNetwork class
# from model.neural_network import NeuralNetwork  # adapte le chemin si besoin

class TorchNetWrapper:
    """
    Wrappe ton NeuralNetwork PyTorch pour fournir la méthode predict(board).
    predict(board) -> (policy_dict, value_float)
    policy_dict: {uci_str: probability}
    value_float: scalar in [-1,1]
    """
    def __init__(self, model, action_list, action_index, device=None):
        self.model = model
        self.action_list = action_list
        self.action_index = action_index
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def board_to_tensor_torch(self, board):
        # re-use board_to_tensor from actions_mapping.py
        from actions_mapping import board_to_tensor
        arr = board_to_tensor(board).astype(np.float32)  # shape (8,8,18)
        # PyTorch expects (C, H, W)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, C, 8,8)
        return tensor.to(self.device)

    def predict(self, board, temperature=1.0):
        """
        Retourne (policy_dict, value)
        policy_dict maps UCI string -> probability (float)
        Note: we compute softmax over the full action space.
        """
        tensor = self.board_to_tensor_torch(board)
        with torch.no_grad():
            logits, value_t = self.model(tensor)  # logits shape (1, ACTION_SIZE), value (1,1)
            logits = logits.squeeze(0).cpu().numpy()  # (ACTION_SIZE,)
            value = float(value_t.squeeze(0).squeeze(0).cpu().numpy())

        # temperature softmax
        if temperature != 1.0:
            logits = logits / float(temperature)

        # numerical stable softmax
        maxl = logits.max()
        exps = np.exp(logits - maxl)
        probs = exps / (exps.sum() + 1e-12)

        # build dict only for convenience; MCTS will query only legal moves
        policy_dict = {uci: float(probs[idx]) for idx, uci in enumerate(self.action_list)}

        return policy_dict, value
