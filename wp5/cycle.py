# wp5/cycle.py
import os
import subprocess
import time
import torch
import random
from wp5.evaluation import evaluate
from wp4.utils_io import list_shards

def spawn_workers(n_workers, games_per_worker, sims, model, cycle_num, batch_size=256):
    """Subprocess-based worker spawning with resuming capability."""
    
    # 1. Count existing games for this cycle
    shards_dir = "data/selfplay/shards"
    os.makedirs(shards_dir, exist_ok=True)
    existing_shards = [f for f in os.listdir(shards_dir) if f.startswith(f"cycle{cycle_num}_shard")]
    n_existing = len(existing_shards)
    
    total_target = n_workers * games_per_worker
    remaining = total_target - n_existing
    
    if remaining <= 0:
        print(f"[cycle] Cycle {cycle_num} already has {n_existing}/{total_target} games. Skipping self-play.")
        return

    print(f"[cycle] Cycle {cycle_num} progress: {n_existing}/{total_target} games. Generating {remaining} more.")
    
    # 2. Divide remaining work among workers
    # We use all workers but reduce the games per worker
    rem_per_worker = (remaining + n_workers - 1) // n_workers
    
    processes = []
    for i in range(n_workers):
        # Last worker might do less if it's not perfectly divisible
        current_worker_games = min(rem_per_worker, remaining - len(processes) * rem_per_worker)
        if current_worker_games <= 0: break
        
        worker_seed = random.randint(0, 1000000)
        p = subprocess.Popen([
            "python", "-m", "wp4.selfplay_worker",
            "--n_games", str(current_worker_games),
            "--model", model,
            "--device", "cuda",
            "--n_sim", str(sims),
            "--batch_size", str(batch_size),
            "--seed", str(worker_seed),
            "--cycle", str(cycle_num)
        ])
        processes.append(p)

    for p in processes:
        p.wait()


import json

def load_state():
    state_path = "progress.json"
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            return json.load(f)
    return {"current_cycle": 0, "stage": "start", "games_done": 0}

def save_state(cycle, stage, games_done=0):
    with open("progress.json", "w") as f:
        json.dump({"current_cycle": cycle, "stage": stage, "games_done": games_done}, f, indent=4)

def spawn_workers(n_workers, games_per_worker, sims, model, cycle_num, batch_size=256):
    """Lancement des workers avec mise à jour 'Live' du JSON toutes les 10s."""
    total_target = n_workers * games_per_worker
    shards_dir = "data/selfplay/shards"
    os.makedirs(shards_dir, exist_ok=True)

    # 1. Calcul du reste à faire
    n_prefixed_start = len([f for f in os.listdir(shards_dir) if f.startswith(f"cycle{cycle_num}_shard")])
    state = load_state()
    # On mémorise combien on en avait au total au démarrage (incluant les parties sans préfixe)
    initial_total_done = max(n_prefixed_start, state["games_done"] if state["current_cycle"] == cycle_num else 0)
    
    remaining = total_target - initial_total_done
    
    if remaining <= 0:
        print(f"[cycle] Cycle {cycle_num} déjà complet.")
        save_state(cycle_num, "train", total_target)
        return True

    print(f"[cycle] Reprise Cycle {cycle_num} : {initial_total_done}/{total_target} reconnues. Reste {remaining}.")

    # 2. Lancement des processus
    rem_per_worker = (remaining + n_workers - 1) // n_workers
    processes = []
    for i in range(n_workers):
        count = min(rem_per_worker, remaining - len(processes) * rem_per_worker)
        if count <= 0: break
        p = subprocess.Popen([
            "python", "-m", "wp4.selfplay_worker",
            "--n_games", str(count), "--model", model, "--device", "cuda",
            "--n_sim", str(sims), "--batch_size", str(batch_size),
            "--seed", str(random.randint(0, 10**6)), "--cycle", str(cycle_num)
        ])
        processes.append(p)

    # 3. Boucle de surveillance "Live"
    print(f"[cycle] Surveillance en cours ({total_target} parties visées)...")
    while any(p.poll() is None for p in processes):
        time.sleep(10) # On vérifie toutes les 10 secondes
        # On calcule : parties initiales + nouvelles parties préfixées
        n_prefixed_now = len([f for f in os.listdir(shards_dir) if f.startswith(f"cycle{cycle_num}_shard")])
        new_games = n_prefixed_now - n_prefixed_start
        current_total = initial_total_done + new_games
        
        # On met à jour le JSON
        save_state(cycle_num, "selfplay", current_total)
    
    # 4. Finalisation
    n_prefixed_final = len([f for f in os.listdir(shards_dir) if f.startswith(f"cycle{cycle_num}_shard")])
    final_total = initial_total_done + (n_prefixed_final - n_prefixed_start)
    save_state(cycle_num, "train", final_total)
    print(f"[cycle] Self-play terminé : {final_total} parties au total.")
    return True

def run_wp5_loop(n_cycles):
    CURRENT = "checkpoints/current.pt"
    CANDIDATE = "checkpoints/candidate.pt"

    # Chargement de l'état
    state = load_state()
    
    # Détection si on doit reprendre un cycle au milieu
    if state["current_cycle"] > 0 and state["stage"] != "eval_done":
        print(f"--- REPRISE DÉTECTÉE : Cycle {state['current_cycle']} (Phase: {state['stage']}) ---")
        # On commence directement à ce cycle
        initial_cycle = state["current_cycle"]
    else:
        # Sinon, on cherche le dernier cycle complété dans le log
        last_log_cycle = 0
        if os.path.exists("eval_log.csv"):
            try:
                with open("eval_log.csv", "r") as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_log_cycle = int(lines[-1].strip().split(",")[0])
            except: pass
        initial_cycle = last_log_cycle + 1
    
    for i in range(n_cycles):
        cycle_num = initial_cycle + i
        
        print(f"\n===================")
        print(f"===== CYCLE {cycle_num} =====")
        print("===================\n")

        # 1) SELF-PLAY
        if state["current_cycle"] < cycle_num or state["stage"] == "selfplay":
            # IMPORTANT: On ne reset pas à 0 si c'est le cycle actuel détecté au démarrage
            g_done = state["games_done"] if state["current_cycle"] == cycle_num else 0
            save_state(cycle_num, "selfplay", g_done)
            spawn_workers(1, 800, 2000, CURRENT, cycle_num, 1024)

        # 2) TRAIN
        state = load_state() # Rafraîchir l'état
        if state["current_cycle"] == cycle_num and state["stage"] in ["selfplay", "train"]:
            print("===================\nTRAIN\n===================")
            save_state(cycle_num, "train")
            subprocess.call([
                "python", "-m", "wp4.trainer",
                "--shard_dir", "data/selfplay/shards",
                "--max_shards", "8000",
                "--init_model", CURRENT,
                "--output_model", CANDIDATE,
                "--device", "cuda",
                "--epochs", "5",
                "--lr", "1e-5",
                "--log_file", "train_log.csv",
                "--cycle", f"{cycle_num}"
            ])
            save_state(cycle_num, "evaluate")

        # 3) EVALUATE
        state = load_state()
        if state["current_cycle"] == cycle_num and state["stage"] == "evaluate":
            print("===================\nEVALUATION\n===================")
            n_games = 40
            score_new = evaluate(CURRENT, CANDIDATE, device="cuda", n_games=n_games, n_workers=2, n_sim=1600)
            rate = score_new / n_games
            
            with open("eval_log.csv", "a") as f:
                f.write(f"{cycle_num},{rate}\n")
            
            subprocess.call(["python", "wp5/visualize.py"])

            if rate > 0.50:
                print("ACCEPTED")
                if os.path.exists(CURRENT): os.remove(CURRENT)
                os.replace(CANDIDATE, CURRENT)
            else:
                print("REJECTED")
                if os.path.exists(CANDIDATE): os.remove(CANDIDATE)
            
            save_state(cycle_num, "eval_done")

        # Nettoyage fenêtre glissante
        shards = list_shards("data/selfplay/shards")
        if len(shards) > 50000:
            for s in shards[:-50000]:
                try: os.remove(s)
                except: pass
        
        time.sleep(2)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=1)
    args = p.parse_args()
    run_wp5_loop(args.cycles)