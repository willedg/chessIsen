import torch
import numpy as np
import os
import chess
from typing import Union, List, Dict
from wp2.action_map import move_to_index, ACTION_SPACE
from wp2.model import AlphaChessNet

class Predictor:
    """Fast PyTorch model wrapper for MCTS using C++ encode_batch."""
    def __init__(self, model_or_path: Union[str, torch.nn.Module], device: str = "cuda"):
        # DISABLE TORCH DYNAMO/INDUCTOR ON WINDOWS
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
        except ImportError:
            pass

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.is_trt = False
        
        if isinstance(model_or_path, str):
            if model_or_path.endswith(".ts") or model_or_path.endswith(".trt"):
                # Potential TensorRT / TorchScript engine
                try:
                    self.model = torch.jit.load(model_or_path, map_location=self.device)
                    self.is_trt = True
                    print(f"[Predictor] Loaded Compiled Engine: {model_or_path}")
                except Exception as e:
                    print(f"[Predictor] Failed to load compiled engine {model_or_path}: {e}")
                    self.model = AlphaChessNet()
            else:
                self.model = AlphaChessNet()
                if os.path.exists(model_or_path):
                    checkpoint = torch.load(model_or_path, map_location=self.device, weights_only=True)
                    state_dict = checkpoint.get("model_state_dict", checkpoint)
                    self.model.load_state_dict(state_dict)
                    print(f"[Predictor] Loaded model: {model_or_path}")
        else:
            self.model = model_or_path
            
        self.model.to(self.device).eval()

        # --- Optimisations de vitesse ---
        if self.device.type == 'cuda':
            # 1. Utiliser le FP16 (Demi-précision)
            try:
                self.model = self.model.half()
                print("[Predictor] Mode Demi-Précision (FP16) activé")
            except Exception as e:
                print(f"[Predictor] FP16 non supporté: {e}")

            # 2. TorchScript Tracing (Fusion d'opérations)
            # Environ 20-30% plus rapide et très stable sur Windows
            try:
                print("[Predictor] Tracing du modèle (optimisation graphique)...")
                example_input = torch.randn(1, 19, 8, 8).to(self.device).half()
                with torch.no_grad():
                    self.model = torch.jit.trace(self.model, example_input)
                print("[Predictor] TorchScript Tracing réussi")
            except Exception as e:
                print(f"[Predictor] Tracing échoué: {e}")

        # PRE-CACHE EVERYTHING
        from wp2.action_map import index_to_move
        print("[Predictor] Pre-caching all moves and UCI strings...")
        self.move_objects = [None] * ACTION_SPACE
        self.move_ucis = [""] * ACTION_SPACE
        dummy_board = chess.Board()
        for i in range(ACTION_SPACE):
            try:
                m = index_to_move(i, dummy_board)
                self.move_objects[i] = m
                self.move_ucis[i] = m.uci()
            except:
                pass
        print("[Predictor] Caches ready.")

    def optimize_with_tensorrt(self, batch_size: int, output_path: str = None):
        """Compile the model to TensorRT FP16 for the host GPU."""
        try:
            import torch_tensorrt
            print(f"[Predictor] Optimizing for TensorRT (Batch={batch_size}, FP16)...")
            
            example_input = torch.randn(batch_size, 19, 8, 8).to(self.device)
            # Trace first for stability on Windows
            traced = torch.jit.trace(self.model, example_input)
            
            # Compile
            trt_model = torch_tensorrt.compile(
                traced,
                inputs=[torch_tensorrt.Input(example_input.shape)],
                enabled_precisions={torch.float16}
            )
            
            self.model = trt_model
            self.is_trt = True
            
            if output_path:
                torch.jit.save(trt_model, output_path)
                print(f"[Predictor] Engine saved to: {output_path}")
                
            print("[Predictor] TensorRT optimization successful.")
            return True
        except Exception as e:
            print(f"[Predictor] TensorRT optimization failed: {e}")
            return False

    @torch.no_grad()
    def __call__(self, boards):
        """Batched prediction. Returns raw logits and move lookup data."""
        if not boards:
            return []
            
        from wp3.cpp_mcts import encode_batch
        x_np = encode_batch(boards)
        x = torch.from_numpy(x_np).to(self.device, non_blocking=True)
        
        # S'assurer que les données d'entrée correspondent à la précision du modèle
        if next(self.model.parameters()).dtype == torch.float16:
            x = x.half()
        else:
            x = x.float()

        logits, values = self.model(x)
        logits_np = logits.detach().cpu().float().numpy()
        values_np = values.detach().cpu().float().numpy().flatten()
        
        outputs = []
        # Optimization: Only send lookup tables if the C++ side hasn't cached them yet
        # Since we can't easily track C++ state, we send them only in the first batch of the call
        # or just rely on the fact that MCTS caches them on first receive.
        for i, board in enumerate(boards):
            idx_list = [move_to_index(mv) for mv in board.legal_moves]
            # Send lookup tables only for the first element of the very first batch
            if i == 0:
                outputs.append((logits_np[i], float(values_np[i]), idx_list, self.move_objects, self.move_ucis))
            else:
                outputs.append((logits_np[i], float(values_np[i]), idx_list, None, None))
            
        return outputs
