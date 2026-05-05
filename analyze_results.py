import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_results(results_dir):
    all_data = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                try:
                    data = json.load(f)
                    all_data.append(data)
                except json.JSONDecodeError:
                    print(f"Error decoding {filename}")
    
    df = pd.DataFrame(all_data)
    return df

def plot_results(df):
    os.makedirs('plots', exist_ok=True)
    sns.set_theme(style="whitegrid")

    tasks = df['task'].unique()
    for task in tasks:
        task_df = df[df['task'] == task]
        
        # 1. Heatmap of Final Train Loss (Depth vs Width) for a fixed LR
        # We'll pick the median LR or create one for each LR
        for lr in task_df['lr'].unique():
            lr_df = task_df[task_df['lr'] == lr]
            pivot_loss = lr_df.pivot(index="depth", columns="width", values="train_loss")
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_loss, annot=True, fmt=".4f", cmap="YlGnBu_r")
            plt.title(f"Train Loss: {task} (LR={lr})")
            plt.savefig(f"plots/heatmap_loss_{task}_lr{lr:.4f}.png")
            plt.close()

            pivot_acc = lr_df.pivot(index="depth", columns="width", values="test_acc")
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_acc, annot=True, fmt=".2f", cmap="YlGnBu")
            plt.title(f"Test Accuracy: {task} (LR={lr})")
            plt.savefig(f"plots/heatmap_acc_{task}_lr{lr:.4f}.png")
            plt.close()

        # 2. Line plot: Training Loss vs Depth (different lines for Widths)
        # Averaging over LRs or picking the best one? Let's show the trend for the best LR per configuration
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=task_df, x="depth", y="train_loss", hue="width", marker="o")
        plt.title(f"Train Loss vs Depth: {task}")
        plt.yscale('log')
        plt.savefig(f"plots/line_loss_depth_{task}.png")
        plt.close()

        # 3. Loss Milestones Convergence
        # Average loss at milestones across widths for the task
        milestones = [5, 20, 50, 100]
        ms_cols = [f'loss_{m}' for m in milestones]
        
        # Melt the dataframe to have milestones as a variable
        melted_df = task_df.melt(id_vars=['depth', 'width', 'lr'], 
                                 value_vars=ms_cols, 
                                 var_name='milestone', value_name='loss')
        melted_df['milestone'] = melted_df['milestone'].str.extract('(\d+)').astype(int)

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=melted_df, x="milestone", y="loss", hue="depth", style="width")
        plt.title(f"Convergence Milestones: {task}")
        plt.yscale('log')
        plt.savefig(f"plots/convergence_{task}.png")
        plt.close()

def main():
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found.")
        return

    print("Aggregating results...")
    df = aggregate_results(results_dir)
    df.to_csv('summary_results.csv', index=False)
    print("Summary saved to summary_results.csv")

    print("Generating plots...")
    plot_results(df)
    print("Plots saved to the 'plots/' directory.")

    # Quick summary report
    print("\n--- Summary Report ---")
    for task in df['task'].unique():
        print(f"\nTask: {task}")
        best_run = df[df['task'] == task].loc[df[df['task'] == task]['train_loss'].idxmin()]
        print(f"  Best Training Loss: {best_run['train_loss']:.6f} (L={best_run['depth']}, W={best_run['width']}, LR={best_run['lr']})")
        
        # Check depth impact (average loss per depth)
        depth_impact = df[df['task'] == task].groupby('depth')['train_loss'].mean()
        print("  Average Train Loss per Depth:")
        print(depth_impact)

if __name__ == "__main__":
    main()
