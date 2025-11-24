# wp5.py
"""
WP5 — Repeated generation & training loop (AlphaZero-like).

- Reuses parallele_self_play.py to generate self-play data (workers CPU).
- Trains model on replay buffer built from last REPLAY_K generations (GPU if available).
- Optionally evaluates new model vs previous model.
- Directory layout:
    selfplay_runs/gen1/dataset.npz
    selfplay_runs/gen1/model_gen1.pt
    selfplay_runs/gen2/dataset.npz
    ...
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import chess
import random

# project imports (adjust paths if different)
from model.WP2_Architecture_Neurone import NeuralNetwork
from model.action_encoding import ACTION_SIZE

import parallele_self_play
from self_play_wp4 import MCTS as MCTS_Class, TorchNetWrapper, ChessEnv

# ------------------- CONFIG -------------------
NUM_CYCLES = 1           # number of training cycles (each cycle produces model_gen{i})
NUM_SELFPLAY = 1000      # games generated per generation
REPLAY_K = 2             # keep last K generations in replay buffer (K >=1)
EPOCHS = 2               # epochs per training
BATCH_SIZE = 64
LR = 1e-3
EVAL_GAMES = 10          # games to evaluate new model vs previous model (0 = skip)
EVAL_SIMULATIONS = 200   # lower simulations for faster eval
BASE_OUT = "selfplay_runs"
MODELS_DIR = "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACCEPT_THRESHOLD = 0.55  # new model must win at least this fraction to be accepted

os.makedirs(BASE_OUT, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# safety checks
assert REPLAY_K >= 1, "REPLAY_K must be >= 1"


# ------------------- Utilities -------------------
def next_gen_folder(base=BASE_OUT):
    """Return next available generation folder name and index like (path, idx)."""
    i = 1
    while True:
        candidate = os.path.join(base, f"gen{i}")
        if not os.path.exists(candidate):
            return candidate, i
        i += 1


def gen_folder_for_index(idx, base=BASE_OUT):
    return os.path.join(base, f"gen{idx}")


def list_chunks_in_dir(d):
    """
    Liste uniquement les chunks de self-play (.npz) à fusionner,
    en excluant tout fichier de replay, dataset global ou autre artefact.
    """
    valid_files = []
    for f in os.listdir(d):
        if not f.endswith(".npz"):
            continue
        # exclure les fichiers finaux ou non-chunks
        if f.startswith("replay_last") or f == "dataset.npz" or f.startswith("model_gen"):
            continue
        valid_files.append(os.path.join(d, f))
    return sorted(valid_files)



def merge_chunks_to_npz(chunks_dir, out_npz_path):
    files = list_chunks_in_dir(chunks_dir)
    if not files:
        raise RuntimeError(f"No chunk files found in {chunks_dir}")
    all_states, all_pis, all_zs, all_players = [], [], [], []
    total = 0
    for f in files:
        d = np.load(f)
        s = d["states"]; p = d["pis"]; z = d["zs"]
        pl = d["players"] if "players" in d else np.zeros((s.shape[0],), dtype=np.int8)
        all_states.append(s); all_pis.append(p); all_zs.append(z); all_players.append(pl)
        total += s.shape[0]
        print(f"[MERGE] loaded {os.path.basename(f)} -> {s.shape[0]} positions")
    print(f"[MERGE] concatenating {len(files)} files, total positions = {total}")
    ALL_states = np.concatenate(all_states, axis=0)
    ALL_pis = np.concatenate(all_pis, axis=0)
    ALL_zs = np.concatenate(all_zs, axis=0)
    ALL_players = np.concatenate(all_players, axis=0)
    np.savez_compressed(out_npz_path, states=ALL_states, pis=ALL_pis, zs=ALL_zs, players=ALL_players)
    print(f"[MERGE] saved merged dataset to {out_npz_path}")
    return out_npz_path


class SelfPlayDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.states = data["states"].astype(np.float32)
        self.pis = data["pis"].astype(np.float32)
        self.zs = data["zs"].astype(np.float32)
        assert self.states.shape[0] == self.pis.shape[0] == self.zs.shape[0]

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        s = torch.from_numpy(self.states[idx])   # (18,8,8)
        pi = torch.from_numpy(self.pis[idx])    # (ACTION_SIZE,)
        z = torch.tensor(self.zs[idx], dtype=torch.float32)
        return s, pi, z


def train_model_on_npz(npz_path, model_save_path, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    print(f"[TRAIN] Loading dataset {npz_path}")
    ds = SelfPlayDataset(npz_path)
    num_workers = 0 if os.name == "nt" else 4  # Windows → éviter multiprocessing
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    model = NeuralNetwork(input_channels=18, board_size=8, num_actions=ACTION_SIZE)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = running_policy = running_value = 0.0
        nb = 0
        for states, pis, zs in dl:
            states = states.to(DEVICE); pis = pis.to(DEVICE); zs = zs.to(DEVICE)
            opt.zero_grad()
            logits, vals = model(states)
            vals = vals.view(-1)
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = - (pis * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(vals, zs)
            loss = policy_loss + value_loss
            loss.backward()
            opt.step()
            running_loss += float(loss.item()); running_policy += float(policy_loss.item()); running_value += float(value_loss.item())
            nb += 1
        print(f"[TRAIN] Epoch {epoch}/{epochs} - loss={running_loss/nb:.4f} (policy={running_policy/nb:.4f} value={running_value/nb:.4f})")

    torch.save(model.state_dict(), model_save_path)
    print(f"[TRAIN] saved model to {model_save_path}")
    return model_save_path


# ------------------- call parallele_self_play -------------------
def call_parallele_self_play(out_dir, n_games, model_path=None):
    """
    Lance parallele_self_play pour générer n_games parties,
    en reprenant automatiquement s'il existe déjà des fichiers partiels dans out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    mod = parallele_self_play

    # compter les fichiers existants
    existing_chunks = [f for f in os.listdir(out_dir) if f.endswith(".npz")]
    completed_games = len(existing_chunks)
    remaining = max(0, n_games - completed_games)

    if remaining == 0:
        print(f"[PARALLEL] {completed_games}/{n_games} parties déjà présentes, skip self-play.")
        return

    print(f"[PARALLEL] reprise self-play : {completed_games} déjà faits, {remaining} restants.")

    # sauvegarde des variables globales de parallele_self_play
    orig = {}
    for name in ("OUT_DIR", "NUM_GAMES", "GAMES_PER_WORKER", "NUM_WORKERS", "SIMULATIONS", "CPUCT", "MODEL_PATH"):
        orig[name] = getattr(mod, name, None)

    # configuration temporaire
    mod.OUT_DIR = out_dir
    mod.NUM_GAMES = remaining
    mod.GAMES_PER_WORKER = max(1, min(remaining, getattr(mod, "GAMES_PER_WORKER", 5)))
    mod.NUM_WORKERS = min(getattr(mod, "NUM_WORKERS", max(1, os.cpu_count() - 1)), remaining)
    mod.MODEL_PATH = model_path if model_path is not None else None

    print(f"[PARALLEL] launching parallele_self_play.main() -> OUT_DIR={mod.OUT_DIR}, GAMES={remaining}, MODEL={mod.MODEL_PATH}")
    mod.main()

    # restauration du module
    for name, val in orig.items():
        if val is None:
            try:
                delattr(mod, name)
            except Exception:
                pass
        else:
            setattr(mod, name, val)



# ------------------- Build replay dataset from last K gens -------------------
def build_replay_dataset(latest_gen_idx, k=REPLAY_K, out_merged_path=None):
    """
    Build merged dataset from generations:
      keep generations: max(1, latest_gen_idx - k + 1) .. latest_gen_idx
    If out_merged_path provided, save to that path; otherwise return (states,pis,zs) arrays.
    """
    start = max(1, latest_gen_idx - k + 1)
    npz_paths = []
    for idx in range(start, latest_gen_idx + 1):
        gen_folder = gen_folder_for_index(idx)
        ds_path = os.path.join(gen_folder, "dataset.npz")
        if not os.path.exists(ds_path):
            raise RuntimeError(f"Missing dataset for gen{idx} at {ds_path}")
        npz_paths.append(ds_path)

    all_states, all_pis, all_zs = [], [], []
    for p in npz_paths:
        d = np.load(p)
        all_states.append(d["states"])
        all_pis.append(d["pis"])
        all_zs.append(d["zs"])
        print(f"[REPLAY] include {p} -> {d['states'].shape[0]} positions")
    S = np.concatenate(all_states, axis=0)
    P = np.concatenate(all_pis, axis=0)
    Z = np.concatenate(all_zs, axis=0)

    if out_merged_path:
        np.savez_compressed(out_merged_path, states=S, pis=P, zs=Z)
        print(f"[REPLAY] saved merged replay to {out_merged_path}")
        return out_merged_path
    return S, P, Z


# ------------------- Simple evaluation routine -------------------
def evaluate_models(model_a_path, model_b_path, n_games=EVAL_GAMES, simulations=EVAL_SIMULATIONS):
    """
    Play n_games between model_a (new) and model_b (baseline), alternating colors.
    Uses a small MCTS (SIMULATIONS per move) for speed. Returns (wins_a, wins_b, draws).
    """
    wins_a = wins_b = draws = 0

    class QuickWrapper(TorchNetWrapper):
        def __init__(self, model_path):
            # instantiate the model and wrapper, load weights on CPU then .to(device) if needed
            model = NeuralNetwork(input_channels=18, board_size=8, num_actions=ACTION_SIZE)
            sd = torch.load(model_path, map_location="cpu")
            model.load_state_dict(sd)
            model.to(DEVICE)
            super().__init__(model)

    # reduce simulations for speed: use a lightweight MCTS (we reuse your MCTS but with fewer sims)
    def play_one_game(wrapper_white, wrapper_black):
        env = ChessEnv()
        mcts_white = MCTS_Class(wrapper_white, env, simulations=simulations, c_puct=1.0)
        mcts_black = MCTS_Class(wrapper_black, env, simulations=simulations, c_puct=1.0)
        board = chess.Board()
        while not env.is_terminal(board):
            if board.turn == chess.WHITE:
                pi, v, _ = mcts_white.run(board)
            else:
                pi, v, _ = mcts_black.run(board)
            # pick best move by visit counts if available
            if isinstance(pi, dict) and pi:
                moves = list(pi.keys()); probs = np.array([pi[m] for m in moves], dtype=np.float32)
                probs = probs / (probs.sum() + 1e-12)
                idx = np.random.choice(len(moves), p=probs)
                move = moves[idx]
            else:
                # fallback: random legal
                legal = list(env.get_legal_moves(board))
                move = random.choice(legal)
            board.push(move)
        result = board.result()
        if result == "1-0": return 1
        if result == "0-1": return -1
        return 0

    for i in range(n_games):
        # alternate colors
        if i % 2 == 0:
            a_white = QuickWrapper(model_a_path)
            b_black = QuickWrapper(model_b_path)
            res = play_one_game(a_white, b_black)
            if res == 1: wins_a += 1
            elif res == -1: wins_b += 1
            else: draws += 1
        else:
            a_black = QuickWrapper(model_a_path)
            b_white = QuickWrapper(model_b_path)
            res = play_one_game(b_white, a_black)
            if res == 1: wins_b += 1
            elif res == -1: wins_a += 1
            else: draws += 1
        print(f"[EVAL] game {i+1}/{n_games} done -> (wins_a={wins_a}, wins_b={wins_b}, draws={draws})")
    return wins_a, wins_b, draws


# ------------------- MAIN WP5 loop -------------------
def wp5_run(num_cycles=NUM_CYCLES, num_selfplay=NUM_SELFPLAY, replay_k=REPLAY_K, epochs=EPOCHS):
    """
    WP5 : Boucle d'entraînement par générations successives, reprise sûre.
    Détection intelligente du dossier genX existant : si le dernier genX est incomplet,
    on reprend dedans ; sinon on crée gen(X+1).
    """
    # lister dossiers gen*
    gens = sorted([d for d in os.listdir(BASE_OUT) if d.startswith("gen")])
    if not gens:
        # aucun dossier -> démarrer à 1
        start_idx = 1
        resume_gen_idx = None
    else:
        # extraire indices valides
        idxs = []
        for d in gens:
            try:
                idxs.append(int(d.replace("gen", "")))
            except ValueError:
                pass
        if not idxs:
            start_idx = 1
            resume_gen_idx = None
        else:
            max_idx = max(idxs)
            last_folder = gen_folder_for_index(max_idx)
            # fichiers importants qui indiquent complétude d'une génération
            dataset_path = os.path.join(last_folder, "dataset.npz")
            model_path = os.path.join(last_folder, f"model_gen{max_idx}.pt")
            # Si dataset ou model manquent -> on reprend dans ce dossier
            if not os.path.exists(dataset_path) or not os.path.exists(model_path):
                start_idx = max_idx           # reprendre dans gen{max_idx}
                resume_gen_idx = max_idx
                print(f"[WP5] Found existing incomplete generation folder: gen{max_idx} -> will resume there.")
            else:
                start_idx = max_idx + 1       # dossier complet -> créer gen{max_idx+1}
                resume_gen_idx = None
                print(f"[WP5] Last generation gen{max_idx} complete -> will create gen{start_idx}.")

    print(f"[WP5] Starting at generation index {start_idx} (device={DEVICE})")

    # Si on reprend depuis une génération existante et qu'un modèle existe, on l'utilise comme current_model
    current_model = None
    if start_idx > 1:
        # si on reprend IN-PLACE, start_idx peut être la gen existante
        prev_idx = start_idx - 1 if resume_gen_idx is None else start_idx - 1
        prev_folder = gen_folder_for_index(prev_idx)
        prev_model = os.path.join(prev_folder, f"model_gen{prev_idx}.pt")
        if os.path.exists(prev_model):
            current_model = prev_model
            print(f"[WP5] Using previous accepted model: {current_model}")
        else:
            # pas de modèle précédent (peut arriver si on reprend gen1) -> None
            current_model = None

    # Boucle sur les cycles demandés
    for cycle in range(num_cycles):
        gen_idx = start_idx + cycle
        gen_folder = gen_folder_for_index(gen_idx)

        # Si dossier existe et est incomplet, on le réutilise. Sinon on le crée.
        if not os.path.exists(gen_folder):
            os.makedirs(gen_folder, exist_ok=True)
            print(f"[WP5] Created folder {gen_folder}")
        else:
            print(f"[WP5] Reusing existing folder {gen_folder}")

        print(f"\n[WP5] --- CYCLE {cycle+1}/{num_cycles} -> GEN{gen_idx} ---")

        dataset_path = os.path.join(gen_folder, "dataset.npz")
        replay_path = os.path.join(gen_folder, f"replay_last{replay_k}.npz")
        model_path = os.path.join(gen_folder, f"model_gen{gen_idx}.pt")

        # -------- Step 1 : Self-play (reprendre partiellement si déjà des chunks) --------
        # call_parallele_self_play gère désormais la reprise partielle (générés déjà)
        if not os.path.exists(dataset_path):
            if gen_idx == 1 and current_model is None:
                print(f"[WP5] Generating self-play for gen{gen_idx} without model")
                call_parallele_self_play(gen_folder, num_selfplay, model_path=None)
            else:
                # si current_model est None but gen_idx>1, try to find previous model
                if current_model is None and gen_idx > 1:
                    prev_folder = gen_folder_for_index(gen_idx - 1)
                    prev_model_try = os.path.join(prev_folder, f"model_gen{gen_idx-1}.pt")
                    if os.path.exists(prev_model_try):
                        current_model = prev_model_try
                        print(f"[WP5] Found previous model {current_model} to use for generation.")
                print(f"[WP5] Generating self-play for gen{gen_idx} using model: {current_model}")
                call_parallele_self_play(gen_folder, num_selfplay, model_path=current_model)
            # une fois les chunks générés (ou s'il y en avait déjà), on fusionne
            merge_chunks_to_npz(gen_folder, dataset_path)
        else:
            print(f"[WP5] dataset.npz already present for gen{gen_idx} -> skipping self-play/merge")

        # -------- Step 2 : build replay dataset (if missing) --------
        if not os.path.exists(replay_path):
            build_replay_dataset(gen_idx, k=replay_k, out_merged_path=replay_path)
        else:
            print(f"[WP5] replay dataset already present for gen{gen_idx}")

        # -------- Step 3 : train model (if missing) --------
        if not os.path.exists(model_path):
            train_model_on_npz(replay_path, model_path, epochs=epochs)
        else:
            print(f"[WP5] model already present for gen{gen_idx}")

        # -------- Step 4 : gating / evaluation --------
        # if we have a previous accepted model, evaluate new vs old
        if gen_idx > 1 and EVAL_GAMES > 0 and current_model is not None:
            print(f"[WP5] Evaluating model_gen{gen_idx} vs previous model")
            wins_new, wins_old, draws = evaluate_models(
                model_path, current_model,
                n_games=EVAL_GAMES, simulations=EVAL_SIMULATIONS
            )
            total = wins_new + wins_old
            win_rate = wins_new / total if total > 0 else 0.0
            print(f"[WP5] Result: new={wins_new}, old={wins_old}, draws={draws}, win_rate={win_rate:.2%}")

            if win_rate >= ACCEPT_THRESHOLD:
                print(f"[WP5] New model accepted")
                current_model = model_path
            else:
                print(f"[WP5] New model rejected -> keeping previous model")
                try:
                    os.remove(model_path)
                except Exception:
                    pass
        else:
            # accept first model automatically if there was none
            if current_model is None:
                current_model = model_path
                print(f"[WP5] Accepted model {current_model} as current baseline")

    print(f"\n[WP5] All cycles finished. Last accepted model: {current_model}")
    return True


if __name__ == "__main__":
    NUM_CYCLES = int(os.environ.get("WP5_NUM_CYCLES", NUM_CYCLES))
    NUM_SELFPLAY = int(os.environ.get("WP5_NUM_SELFPLAY", NUM_SELFPLAY))
    REPLAY_K = int(os.environ.get("WP5_REPLAY_K", REPLAY_K))
    EPOCHS = int(os.environ.get("WP5_EPOCHS", EPOCHS))

    wp5_run(num_cycles=NUM_CYCLES, num_selfplay=NUM_SELFPLAY, replay_k=REPLAY_K, epochs=EPOCHS)
