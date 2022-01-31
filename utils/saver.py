import torch
import re
import os, os.path as osp
from torch_geometric.data import DataLoader

def save_model(model, ckpt_dir, epoch, device, cover=True, upper=5):

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'Epoch{epoch}.pkl'
    model.to(torch.device('cpu'))
    torch.save(model, ckpt_path)
    
    if cover:
        ckpt_path = ckpt_dir / f'best.pkl'
        torch.save(model, ckpt_path)
    
    files = os.listdir(ckpt_dir)
    n_tmp_store = sum(1 for f in files if "Epoch" in f)

    if n_tmp_store > upper:
        f_to_remove = 'Epoch{}.pkl'.format(min([int(re.search(r'[0-9]+', f).group(0)) for f in files if 'best' not in f]))
        os.remove(os.path.join(ckpt_dir, f_to_remove))
    model.to(device)

def save_train_emb(emb_path, gnn_model, train_dataset, device, is_graph=True):
    if osp.exists(emb_path):
        return
    X_train = []    
    y_train = []
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for g in loader:
        g.to(device)
        if is_graph:
            rep = gnn_model.get_graph_rep(g.x, g.edge_index, g.edge_attr, g.batch)
            X_train.append(rep.view(-1))
            y_train.append(g.y.item())
        else:
            rep = gnn_model.get_node_reps(g.x, g.edge_index, g.edge_attr, g.batch)
            X_train.append(rep)
            y_train.append(g.y.item())
    train_set =  [(X_train[i].detach(), y_train[i]) for i in range(len(X_train))]
    torch.save((X_train, y_train), emb_path)
