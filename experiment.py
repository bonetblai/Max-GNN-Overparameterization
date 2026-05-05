import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from dataset import generate_cycle_data, generate_colored_path_data
from model import MaxGNN
import numpy as np
import argparse
import json
import os

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total_nodes = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total_nodes += data.num_nodes
    return total_loss / len(loader.dataset), correct / total_nodes

def run_single_experiment(task, l_target, L, W, lr, epochs, device):
    # Load pre-generated data
    train_path = f'data/{task}_train.pt'
    test_path = f'data/{task}_test.pt'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset files for {task} not found in data/. Run collect_data.py first.")
        
    train_dataset = torch.load(train_path)
    test_dataset = torch.load(test_path)
    
    # Determine in_channels from the first sample
    in_channels = train_dataset[0].x.shape[1]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = MaxGNN(in_channels=in_channels, hidden_channels=W, out_channels=2, num_layers=L).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_train_loss = float('inf')
    loss_milestones = {}
    milestones = [5, 20, 50, 100]
    
    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, optimizer, criterion, device)
        if loss < best_train_loss:
            best_train_loss = loss
        
        if epoch in milestones:
            loss_milestones[epoch] = loss
        
        if epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    res = {
        'task': task,
        'l_target': l_target,
        'depth': L,
        'width': W,
        'lr': lr,
        'params': num_params,
        'train_loss': best_train_loss,
        'test_loss': test_loss,
        'test_acc': test_acc
    }
    for m in milestones:
        res[f'loss_{m}'] = loss_milestones.get(m, None)
    
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='path', choices=['path', 'cycle', 'k4', 'indset4'])
    parser.add_argument('--l_target', type=int, default=3)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--output', type=str, help='Path to save JSON results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running: Task={args.task}, L={args.depth}, W={args.width}, lr={args.lr:.6f}")
    
    result = run_single_experiment(
        task=args.task,
        l_target=args.l_target,
        L=args.depth,
        W=args.width,
        lr=args.lr,
        epochs=args.epochs,
        device=device
    )
    
    print(f"Final Train Loss: {result['train_loss']:.4f}, Test Acc: {result['test_acc']:.4f}")
    
    if args.output:
        dir_name = os.path.dirname(args.output)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"Results saved to {args.output}")
