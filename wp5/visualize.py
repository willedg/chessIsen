import os
import csv
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt

def read_csv(filepath):
    data = []
    if not os.path.exists(filepath):
        return data
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def main():
    train_data = read_csv('train_log.csv')
    eval_data = read_csv('eval_log.csv')

    if not train_data and not eval_data:
        print("No log data found to visualize.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.4)

    # 1. Training stats
    if train_data:
        cycles = [int(row['cycle']) for row in train_data]
        total_loss = [float(row['total_loss']) for row in train_data]
        policy_loss = [float(row['policy_loss']) for row in train_data]
        value_loss = [float(row['value_loss']) for row in train_data]

        ax1.plot(cycles, total_loss, label='Total Loss', marker='o', color='#2ecc71', linewidth=2)
        ax1.plot(cycles, policy_loss, label='Policy Loss', marker='s', color='#3498db', alpha=0.7)
        ax1.plot(cycles, value_loss, label='Value Loss', marker='^', color='#e74c3c', alpha=0.7)
        
        ax1.set_title("Training Loss Over Cycles", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Cycle Number")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
    else:
        ax1.text(0.5, 0.5, "No training data yet", ha='center', va='center')

    # 2. Evaluation winrate
    if eval_data:
        cycles = [int(row['cycle']) for row in eval_data]
        winrates = [float(row['winrate']) for row in eval_data]

        ax2.plot(cycles, winrates, label='Candidate Win Rate', marker='o', color='#f1c40f', linewidth=3)
        ax2.axhline(y=0.5, color='r', linestyle='--', label='Acceptance Threshold (Approx)')
        
        ax2.set_title("Candidate Win Rate vs Current Model", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Cycle Number")
        ax2.set_ylabel("Win Rate")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "No evaluation data yet", ha='center', va='center')

    plt.suptitle("AlphaChess Training Progress", fontsize=18, fontweight='bold', y=0.98)
    
    output_path = 'training_progress.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Progress chart updated: {output_path}")

if __name__ == "__main__":
    main()
