import pandas as pd

def generate_md():
    df = pd.read_csv('summary_results.csv')
    
    with open('RESULTS_SUMMARY.md', 'w') as f:
        f.write("# Max-GNN Over-parameterization Research Summary\n\n")
        
        for task in df['task'].unique():
            f.write(f"## Task: {task.capitalize()}\n\n")
            task_df = df[df['task'] == task]
            
            best = task_df.loc[task_df['train_loss'].idxmin()]
            f.write(f"- **Best Training Loss:** {best['train_loss']:.6f}\n")
            f.write(f"- **Optimal Configuration:** Depth={best['depth']}, Width={best['width']}, LR={best['lr']}\n\n")
            
            f.write("| Depth | Width | LR | Params | Train Loss | Test Acc |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            
            # Show top 5 runs
            top_runs = task_df.nsmallest(5, 'train_loss')
            for _, row in top_runs.iterrows():
                f.write(f"| {row['depth']} | {row['width']} | {row['lr']} | {row['params']} | {row['train_loss']:.6f} | {row['test_acc']:.4f} |\n")
            f.write("\n")

if __name__ == "__main__":
    generate_md()
    print("RESULTS_SUMMARY.md generated.")
