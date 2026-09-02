import time
import subprocess

def run_cycle():
    while True:

        print("=== SELFPLAY ===")
        subprocess.call([
            "python","-m","wp4.selfplay_worker",
            "--n_games","20",
            "--model","checkpoints/latest.pt",
            "--device","cpu",
            "--n_sim","60"
        ])

        print("=== TRAINING ===")
        subprocess.call([
            "python","-m","wp4.trainer",
            "--shard_dir","data/selfplay/shards",
            "--ckpt_dir","checkpoints",
            "--device","cpu",
            "--epochs","1"
        ])

        print("=== LOOP END ===")
        time.sleep(1)


if __name__ == "__main__":
    run_cycle()
