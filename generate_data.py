import os
import torch
import argparse
from dataset import (
    generate_cycle_data, 
    generate_colored_path_data, 
    generate_k4_data, 
    generate_indset4_data
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['cycle', 'path', 'k4', 'indset4'])
    parser.add_argument('--num_graphs', type=int, default=100)
    parser.add_argument('--num_nodes', type=int, default=20)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--l_target', type=int, default=3)
    args = parser.parse_args()
    
    tasks = {
        'cycle': generate_cycle_data,
        'path': generate_colored_path_data,
        'k4': generate_k4_data,
        'indset4': generate_indset4_data
    }
    
    gen_func = tasks[args.task]
    print(f"Generating {args.num_graphs} graphs for task: {args.task} (nodes={args.num_nodes})")
    
    if args.task in ['cycle', 'path']:
        data = gen_func(num_graphs=args.num_graphs, num_nodes=args.num_nodes, l_target=args.l_target)
    else:
        data = gen_func(num_graphs=args.num_graphs, num_nodes=args.num_nodes)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(data, args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
