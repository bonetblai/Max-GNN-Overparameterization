import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data, DataLoader
import networkx as nx
import numpy as np
import random

def generate_cycle_data(num_graphs=1000, num_nodes=20, l_target=3):
    """
    Labels: 1 if node i belongs to a cycle of length exactly l_target.
    """
    dataset = []
    for _ in range(num_graphs):
        # Higher density to ensure cycles
        G = nx.erdos_renyi_graph(num_nodes, 0.3)
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        # Use networkx to find cycles
        # Note: finding all cycles is expensive, we use a simple approach for small graphs
        cycles = nx.cycle_basis(G)
        for cycle in cycles:
            if len(cycle) == l_target:
                for node in cycle:
                    y[node] = 1
        
        # Simple features: just degree
        degrees = [d for n, d in G.degree()]
        x = torch.tensor(degrees, dtype=torch.float).view(-1, 1)
        
        edge_index = torch_geometric.utils.from_networkx(G).edge_index
        dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset

def generate_colored_path_data(num_graphs=1000, num_nodes=20, l_target=3, num_colors=3):
    """
    Nodes have random colors. 
    Label: 1 if node i is the start of a monochromatic path of length l_target.
    """
    dataset = []
    for _ in range(num_graphs):
        G = nx.erdos_renyi_graph(num_nodes, 0.2)
        colors = [random.randint(0, num_colors - 1) for _ in range(num_nodes)]
        
        # Features: One-hot encoded colors
        x = F.one_hot(torch.tensor(colors), num_classes=num_colors).float()
        
        y = torch.zeros(num_nodes, dtype=torch.long)
        
        # Search for monochromatic paths using DFS
        for start_node in range(num_nodes):
            color = colors[start_node]
            stack = [(start_node, 1)] # node, current_length
            visited = {start_node}
            
            while stack:
                curr, length = stack.pop()
                if length >= l_target:
                    y[start_node] = 1
                    break
                
                for neighbor in G.neighbors(curr):
                    if colors[neighbor] == color and neighbor not in visited:
                        # For paths we shouldn't strictly reuse nodes, but for simplicity:
                        stack.append((neighbor, length + 1))
        
        edge_index = torch_geometric.utils.from_networkx(G).edge_index
        dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset

if __name__ == "__main__":
    dataset = generate_graph_data(num_graphs=10, num_nodes=5, l_target=2)
    print(f"Generated {len(dataset)} graphs.")
    print(f"Example data: {dataset[0]}")
    print(f"Labels of first graph: {dataset[0].y}")
