import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def parse_arguments():
    """Parse command line arguments for the metric calculation script."""
    parser = argparse.ArgumentParser(description='Calculate and plot performance metrics from evaluation results')
    
    # Input/Output paths
    parser.add_argument('--eval-dir', type=str, default='eval_output',
                       help='Directory containing evaluation output CSV files')
    parser.add_argument('--filename')
    parser.add_argument('--output-filename', type=str, default='level_performance_plots.png',
                       help='Output filename for the plot')
    parser.add_argument('--output-dir', type=str, default='eval_output',
                       help='Directory to save the output plot')
    parser.add_argument('--seq', type=str, default='top', help='level choices')
    
    return parser.parse_args()

# Parse arguments
args = parse_arguments()

if args.seq == "lr=3e-4":
    level_paths=("worlds/l/mjc_half_cheetah.json", 
                "worlds/l/mjc_swimmer.json",  
                "worlds/l/chain_lander.json",
                "worlds/l/cartpole_thrust.json",)
elif args.seq == "lr=1e-4":
    level_paths=("worlds/l/catapult.json", 
                "worlds/l/mjc_walker.json", 
                "worlds/l/h17_unicycle.json",
                "worlds/l/car_launch.json",)
elif args.seq == "lr=3e-5":
    level_paths=("worlds/l/catcher_v3.json",
                "worlds/l/grasp_easy.json", )
elif args.seq == "lr=1e-5":
    level_paths=("worlds/l/trampoline.json", 
                "worlds/l/hard_lunar_lander.json",)
elif args.seq == "all":
    level_paths = (
        "worlds/l/grasp_easy.json",
        "worlds/l/catapult.json",
        "worlds/l/cartpole_thrust.json",
        "worlds/l/hard_lunar_lander.json",
        "worlds/l/mjc_half_cheetah.json",
        "worlds/l/mjc_swimmer.json",
        "worlds/l/mjc_walker.json",
        "worlds/l/h17_unicycle.json",
        "worlds/l/chain_lander.json",
        "worlds/l/catcher_v3.json",
        "worlds/l/trampoline.json",
        "worlds/l/car_launch.json",
    )


df_async_trial = pd.read_csv(f"{args.eval_dir}/debug_hard.csv")
df_naive = pd.read_csv("async_naive/debug_hard.csv")
df_bid = pd.read_csv("async_bid/debug_hard.csv")
df_rtc = pd.read_csv("async_rtc/debug_hard.csv")

def ep_solved_plot():

    avg_naive = (
        df_naive.groupby(["level", "delay"])
        ["returned_episode_solved"]
        .mean()
        .reset_index(name="avg_solved")
    )


    avg_bid = (
        df_bid.groupby(["level", "delay"])
        ["returned_episode_solved"]
        .mean()
        .reset_index(name="avg_solved")
    )

    avg_rtc = (
        df_rtc.groupby(["level", "delay"])
        ["returned_episode_solved"]
        .mean()
        .reset_index(name="avg_solved")
    )


    # Calculate averages for Ours
    avg_async_trial_1 = (
        df_async_trial.groupby(["level", "delay"])
        ["returned_episode_solved"]
        .mean()
        .reset_index(name="avg_solved")
    )

    # Get unique levels
    levels = [level for level in avg_naive['level'].unique() if level in level_paths]

    # Create subplots for each level in 3 rows, 4 columns per row
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, level in enumerate(levels):
        # Filter data for current level
        naive_data = avg_naive[avg_naive['level'] == level]
        bid_data = avg_bid[avg_bid['level'] == level]
        rtc_data = avg_rtc[avg_rtc['level'] == level]
        trial_data = avg_async_trial_1[avg_async_trial_1['level'] == level]
        
        # Create plot
        ax = axes[i]
        
        # Plot RTC Naive data
        ax.plot(naive_data['delay'], naive_data['avg_solved'], 
                marker='o', linewidth=2, markersize=8, label='Naive', color='blue')
        
        # Plot RTC (supplemented) data
        ax.plot(rtc_data['delay'], rtc_data['avg_solved'], 
                marker='D', linewidth=2, markersize=8, label='RTC', color='cyan')
        
        ax.plot(bid_data['delay'], bid_data['avg_solved'], 
                marker='s', linewidth=2, markersize=8, label='BID', color='purple')
        
        # Plot Ours data
        ax.plot(trial_data['delay'], trial_data['avg_solved'], 
                marker='^', linewidth=2, markersize=8, label='Ours', color='green')
        
        ax.set_xlabel('Delay')
        ax.set_ylabel('Average Episodes Solved')
        ax.set_title(f'Level: {level}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)

    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")




def ep_length_plot():
    """Plot average episode lengths for different methods and delays."""
    
    avg_naive = (
        df_naive.groupby(["level", "delay"])
        ["returned_episode_lengths"]
        .mean()
        .reset_index(name="avg_length")
    )

    avg_bid = (
        df_bid.groupby(["level", "delay"])
        ["returned_episode_lengths"]
        .mean()
        .reset_index(name="avg_length")
    )

    avg_rtc = (
        df_rtc.groupby(["level", "delay"])
        ["returned_episode_lengths"]
        .mean()
        .reset_index(name="avg_length")
    )

    # Calculate averages for Ours
    avg_async_trial_1 = (
        df_async_trial.groupby(["level", "delay"])
        ["returned_episode_lengths"]
        .mean()
        .reset_index(name="avg_length")
    )

    # Get unique levels
    levels = [level for level in avg_naive['level'].unique() if level in level_paths]

    # Create subplots for each level in 3 rows, 4 columns per row
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, level in enumerate(levels):
        # Filter data for current level
        naive_data = avg_naive[avg_naive['level'] == level]
        bid_data = avg_bid[avg_bid['level'] == level]
        rtc_data = avg_rtc[avg_rtc['level'] == level]
        trial_data = avg_async_trial_1[avg_async_trial_1['level'] == level]
        
        # Create plot
        ax = axes[i]
        
        # Plot RTC Naive data
        ax.plot(naive_data['delay'], naive_data['avg_length'], 
                marker='o', linewidth=2, markersize=8, label='Naive', color='blue')
        
        # Plot RTC (supplemented) data
        ax.plot(rtc_data['delay'], rtc_data['avg_length'], 
                marker='D', linewidth=2, markersize=8, label='RTC', color='cyan')
        
        ax.plot(bid_data['delay'], bid_data['avg_length'], 
                marker='s', linewidth=2, markersize=8, label='BID', color='purple')
        
        # Plot Ours data
        ax.plot(trial_data['delay'], trial_data['avg_length'], 
                marker='^', linewidth=2, markersize=8, label='Ours', color='green')
        
        ax.set_xlabel('Delay')
        ax.set_ylabel('Average Episode Lengths')
        ax.set_title(f'Level: {level}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)

    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'length_'+args.output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')



def solved_all():
    overall_naive = (
        df_naive.groupby("delay")["returned_episode_solved"]
        .mean()
        .reset_index(name="avg_solved")
    )

    overall_bid = (
        df_bid.groupby("delay")["returned_episode_solved"]
        .mean()
        .reset_index(name="avg_solved")
    )

    overall_rtc = (
        df_rtc.groupby("delay")["returned_episode_solved"]
        .mean()
        .reset_index(name="avg_solved")
    )

    overall_async_trial = (
        df_async_trial.groupby("delay")["returned_episode_solved"]
        .mean()
        .reset_index(name="avg_solved")
    )

    # --- optional: plot overall averages in one figure ---
    plt.figure(figsize=(8, 6))
    plt.plot(overall_naive["delay"], overall_naive["avg_solved"], marker="o", label="Naive", color="blue")
    plt.plot(overall_rtc["delay"], overall_rtc["avg_solved"], marker="D", label="RTC", color="cyan")
    plt.plot(overall_bid["delay"], overall_bid["avg_solved"], marker="s", label="BID", color="purple")
    plt.plot(overall_async_trial["delay"], overall_async_trial["avg_solved"], marker="^", label="Ours", color="green")

    plt.xlabel("Delay")
    plt.ylabel("Average Episodes Solved (all levels)")
    plt.title("Overall Average Across Levels")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "overall_" + args.output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Overall average plot saved to: {output_path}")



def length_all():
    overall_naive = (
        df_naive.groupby("delay")["returned_episode_lengths"]
        .mean()
        .reset_index(name="avg_length")
    )

    overall_bid = (
        df_bid.groupby("delay")["returned_episode_lengths"]
        .mean()
        .reset_index(name="avg_length")
    )

    overall_rtc = (
        df_rtc.groupby("delay")["returned_episode_lengths"]
        .mean()
        .reset_index(name="avg_length")
    )

    overall_async_trial = (
        df_async_trial.groupby("delay")["returned_episode_lengths"]
        .mean()
        .reset_index(name="avg_length")
    )

    # --- optional: plot overall averages in one figure ---
    plt.figure(figsize=(8, 6))
    plt.plot(overall_naive["delay"], overall_naive["avg_length"], marker="o", label="Naive", color="blue")
    plt.plot(overall_rtc["delay"], overall_rtc["avg_length"], marker="D", label="RTC", color="cyan")
    plt.plot(overall_bid["delay"], overall_bid["avg_length"], marker="s", label="BID", color="purple")
    plt.plot(overall_async_trial["delay"], overall_async_trial["avg_length"], marker="^", label="Ours", color="green")

    plt.xlabel("Delay")
    plt.ylabel("Average Episodes Solved (all levels)")
    plt.title("Overall Average Across Levels")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "length_overall_" + args.output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Overall average plot saved to: {output_path}")


if __name__=="__main__":
    ep_solved_plot()
    ep_length_plot()
    solved_all()
    length_all()