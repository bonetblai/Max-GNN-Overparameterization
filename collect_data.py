import os
import torch

def collect_shards(tasks, shards, types=['train', 'test']):
    os.makedirs('data', exist_ok=True)
    for task in tasks:
        for t_type in types:
            combined_data = []
            for s in range(shards):
                shard_path = f'data/shards/{task}_{t_type}_shard{s}.pt'
                if os.path.exists(shard_path):
                    data = torch.load(shard_path)
                    combined_data.extend(data)
                else:
                    print(f"Warning: Shard {shard_path} missing.")
            
            output_path = f'data/{task}_{t_type}.pt'
            torch.save(combined_data, output_path)
            print(f"Saved {len(combined_data)} graphs to {output_path}")

if __name__ == "__main__":
    tasks = ['cycle', 'path', 'k4', 'indset4']
    shards = 10
    collect_shards(tasks, shards)
