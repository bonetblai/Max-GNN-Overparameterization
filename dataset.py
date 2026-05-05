import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import random
from itertools import combinations

def generate_cycle_data(num_graphs=1000, num_nodes=20, l_target=3):
    """
    Labels: 1 if node i belongs to a cycle of length exactly l_target.
    """
    dataset = []
    for _ in range(num_graphs):
        G = nx.erdos_renyi_graph(num_nodes, 0.3)
        y = torch.zeros(num_nodes, dtype=torch.long)
        cycles = nx.cycle_basis(G)
        for cycle in cycles:
            if len(cycle) == l_target:
                for node in cycle:
                    y[node] = 1
        
        degrees = [d for n, d in G.degree()]
        x = torch.tensor(degrees, dtype=torch.float).view(-1, 1)
        edge_index = torch_geometric.utils.from_networkx(G).edge_index
        dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset

def generate_colored_path_data(num_graphs=1000, num_nodes=20, l_target=3, num_colors=2):
    """
    Label: 1 if node i is the start of a monochromatic path of length l_target.
    """
    dataset = []
    for _ in range(num_graphs):
        G = nx.erdos_renyi_graph(num_nodes, 0.2)
        colors = [random.randint(0, num_colors - 1) for _ in range(num_nodes)]
        x = F.one_hot(torch.tensor(colors), num_classes=num_colors).float()
        y = torch.zeros(num_nodes, dtype=torch.long)
        
        for start_node in range(num_nodes):
            color = colors[start_node]
            stack = [(start_node, 1, {start_node})] 
            while stack:
                curr, length, visited = stack.pop()
                if length >= l_target:
                    y[start_node] = 1
                    break
                for neighbor in G.neighbors(curr):
                    if colors[neighbor] == color and neighbor not in visited:
                        new_visited = visited.copy()
                        new_visited.add(neighbor)
                        stack.append((neighbor, length + 1, new_visited))
        
        edge_index = torch_geometric.utils.from_networkx(G).edge_index
        dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset

def generate_k4_data(num_graphs=1000, num_nodes=20, num_colors=2):
    """
    Label: 1 if node i belongs to a monochromatic clique of size 4 (K4).
    """
    dataset = []
    for _ in range(num_graphs):
        G = nx.erdos_renyi_graph(num_nodes, 0.4)
        colors = [random.randint(0, num_colors - 1) for _ in range(num_nodes)]
        x = F.one_hot(torch.tensor(colors), num_classes=num_colors).float()
        y = torch.zeros(num_nodes, dtype=torch.long)
        
        for color in range(num_colors):
            nodes_with_color = [n for n in range(num_nodes) if colors[n] == color]
            if len(nodes_with_color) < 4:
                continue
            
            subgraph = G.subgraph(nodes_with_color)
            cliques = nx.find_cliques(subgraph)
            for clique in cliques:
                if len(clique) >= 4:
                    for n in clique:
                        y[n] = 1
        
        edge_index = torch_geometric.utils.from_networkx(G).edge_index
        dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset

def generate_indset4_data(num_graphs=1000, num_nodes=20):
    """
    Label: 1 if node i belongs to an independent set of size 4.
    """
    dataset = []
    for _ in range(num_graphs):
        G = nx.erdos_renyi_graph(num_nodes, 0.15)
        y = torch.zeros(num_nodes, dtype=torch.long)
        
        G_comp = nx.complement(G)
        cliques = nx.find_cliques(G_comp)
        for clique in cliques:
            if len(clique) >= 4:
                for n in clique:
                    y[n] = 1
        
        degrees = [d for n, d in G.degree()]
        x = torch.tensor(degrees, dtype=torch.float).view(-1, 1)
        edge_index = torch_geometric.utils.from_networkx(G).edge_index
        dataset.append(Data(x=x, edge_index=edge_index, y=y))
    return dataset
