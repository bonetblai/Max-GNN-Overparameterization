import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from dataset import generate_cycle_data, generate_colored_path_data
from model import MaxGNN
import numpy as np
import argparse

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

def run_experiment_suite(task='path', l_target=3, depths=[1, 2, 3, 4, 5], widths=[8, 16], num_graphs=1000, epochs=200, base_lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Task: {task}")
    
    # Generate data
    if task == 'path':
        full_dataset = generate_colored_path_data(num_graphs=num_graphs + 200, l_target=l_target)
        in_channels = 3 # num_colors
    elif task == 'cycle':
        full_dataset = generate_cycle_data(num_graphs=num_graphs + 200, l_target=l_target)
        in_channels = 1 # degree
    else:
        raise ValueError("Unknown task")

    train_dataset = full_dataset[:num_graphs]
    test_dataset = full_dataset[num_graphs:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    results = []
    
    for W in widths:
        for L in depths:
            print(f"\n--- Testing L={L}, W={W} ---")
            model = MaxGNN(in_channels=in_channels, hidden_channels=W, out_channels=2, num_layers=L).to(device)
            
            num_params = sum(p.numel() for p in model.parameters())
            # Simple heuristic: scale LR by 1/sqrt(num_params) normalized to a baseline
            # We use a baseline of ~500 params for the scaling factor
            lr_scale = (500.0 / num_params)**0.5
            lr = base_lr * min(1.0, lr_scale)
            
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            print(f"Params: {num_params}, Adjusted LR: {lr:.6f}")
            
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
            print(f"L={L}, W={W}, Final Train Loss: {best_train_loss:.4f}, Test Acc: {test_acc:.4f}")
            
            res = {
                'depth': L,
                'width': W,
                'params': num_params,
                'train_loss': best_train_loss,
                'test_loss': test_loss,
                'test_acc': test_acc
            }
            for m in milestones:
                res[f'loss_{m}'] = loss_milestones.get(m, None)
            results.append(res)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='path', choices=['path', 'cycle'])
    parser.add_argument('--l_target', type=int, default=3)
    parser.add_argument('--depths', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--widths', type=int, nargs='+', default=[8, 16, 32])
    parser.add_argument('--num_graphs', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    args = parser.parse_args()
    
    results = run_experiment_suite(
        task=args.task,
        l_target=args.l_target,
        depths=args.depths,
        widths=args.widths,
        num_graphs=args.num_graphs,
        epochs=args.epochs,
        base_lr=args.lr
    )
    
    print(f"\nSummary of Results (Task={args.task}, L_target={args.l_target}):")
    headers = ['L', 'W', 'Params', 'Loss@5', 'Loss@20', 'Loss@50', 'Loss@100', 'TrainL', 'TestAcc']
    header_fmt = "{:<2} | {:<2} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}"
    print(header_fmt.format(*headers))
    print("-" * 90)
    for res in results:
        print(header_fmt.format(
            res['depth'], res['width'], res['params'],
            f"{res['loss_5']:.4f}" if res['loss_5'] else "-",
            f"{res['loss_20']:.4f}" if res['loss_20'] else "-",
            f"{res['loss_50']:.4f}" if res['loss_50'] else "-",
            f"{res['loss_100']:.4f}" if res['loss_100'] else "-",
            f"{res['train_loss']:.4f}",
            f"{res['test_acc']:.4f}"
        ))
