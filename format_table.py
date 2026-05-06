import pandas as pd
import argparse

def generate_table(task_name):
    df = pd.read_csv('summary_results.csv')
    df = df[df['task'] == task_name]
    
    # Pivot: Index=[Width, Depth], Columns=[LR], Values=[Loss]
    # We want rows to be Depth for each block of Width
    # Columns need to include params. 
    # Let's get unique LRs
    lrs = sorted(df['lr'].unique())
    
    print(f"\\section*{{Results for {task_name.capitalize()}}}")
    print("\\begin{tabular}{c c | " + " c " * len(lrs) + "}")
    print("Width & Depth & Params & " + " & ".join([f"LR={lr}" for lr in lrs]) + " \\\\ \\hline")
    
    widths = sorted(df['width'].unique())
    for w in widths:
        w_df = df[df['width'] == w]
        depths = sorted(w_df['depth'].unique())
        
        for i, d in enumerate(depths):
            row = w_df[w_df['depth'] == d]
            params = row['params'].iloc[0]
            
            # Formatting depth/width column
            w_col = f"\\multirow{{{len(depths)}}}{{*}}{{{w}}}" if i == 0 else ""
            
            line = f"{w_col} & {d} & {params}"
            for lr in lrs:
                val = row[row['lr'] == lr]['train_loss'].values
                val_str = f"{val[0]:.6f}" if len(val) > 0 else "-"
                line += f" & {val_str}"
            print(line + " \\\\")
        print("\\hline")
    
    print("\\end{tabular}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    args = parser.parse_args()
    generate_table(args.task)
