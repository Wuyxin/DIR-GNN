import os
import argparse
import numpy as np
import os.path as osp
from utils.logger import Logger
from datetime import datetime
from utils.helper import set_seed, args_print
from utils.get_subgraph import split_batch, relabel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.nn import GINEConv
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from gnn import MolHivNet, GINVirtual_node
from utils.mask import set_masks, clear_masks


class CausalAttNet(nn.Module):
    
    def __init__(self, causal_ratio):
        super(CausalAttNet, self).__init__()
        self.gnn_node = GINVirtual_node(num_layers=2, emb_dim=args.channels, dropout=0)
        self.linear = nn.Linear(args.channels*2, 1)
        self.ratio = causal_ratio
    def forward(self, data):
        x = self.gnn_node(data.x, data.edge_index, data.edge_attr, data.batch)

        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        pred_edge_weight = self.linear(edge_rep).view(-1)
        causal_edge_index = torch.LongTensor([[],[]]).to(x.device)
        causal_edge_weight = torch.tensor([]).to(x.device)
        causal_edge_attr = torch.LongTensor([]).to(x.device)
        conf_edge_index = torch.LongTensor([[],[]]).to(x.device)
        conf_edge_weight = torch.tensor([]).to(x.device)
        conf_edge_attr = torch.LongTensor([]).to(x.device)

        edge_indices, _, _, num_edges, cum_edges = split_batch(data)
        for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
            n_reserve =  int(self.ratio * N)
            edge_attr = data.edge_attr[C:C+N]
            single_mask = pred_edge_weight[C:C+N]
            single_mask_detach = pred_edge_weight[C:C+N].detach().cpu().numpy()
            rank = np.argpartition(-single_mask_detach, n_reserve)
            idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]

            causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
            conf_edge_index = torch.cat([conf_edge_index, edge_index[:, idx_drop]], dim=1)
            causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
            conf_edge_weight = torch.cat([conf_edge_weight, -1 * single_mask[idx_drop]])
            causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
            conf_edge_attr = torch.cat([conf_edge_attr, edge_attr[idx_drop]])

        causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)
        conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, data.batch)

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch),\
                pred_edge_weight

if __name__ == "__main__":
        
    # Arguments
    parser = argparse.ArgumentParser(description='Training for Causal Feature Learning')
    parser.add_argument('--cuda', default=0, type=int, help='cuda device')
    parser.add_argument('--datadir', default='data/', type=str, help='directory for datasets.')
    parser.add_argument('--epoch', default=400, type=int, help='training iterations')
    parser.add_argument('--reg', default=1, type=int)
    parser.add_argument('--seed',  nargs='?', default='[1,2,3]', help='random seed')
    parser.add_argument('--channels', default=300, type=int, help='width of network')
    parser.add_argument('--commit', default='', type=str, help='experiment name')
    # hyper 
    parser.add_argument('--pretrain', default=10, type=int, help='pretrain epoch')
    parser.add_argument('--alpha', default=1e-4, type=float, help='invariant loss')
    parser.add_argument('--r', default=0.8, type=float, help='causal_ratio')
    # basic
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--net_lr', default=1e-5, type=float, help='learning rate for the predictor')
    args = parser.parse_args()
    args.seed = eval(args.seed)
    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')
    # dataset
    dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv') 
    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)
    n_train_data, n_val_data, n_test_data = len(train_loader.dataset), len(val_loader.dataset), float(len(test_loader.dataset))

    # logger
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_info = { 'causal_auc':[], 'train_auc':[], 'val_auc':[]}
    experiment_name = f'molhiv.{bool(args.reg)}.{args.commit}.netlr_{args.net_lr}.batch_{args.batch_size}'\
                        f'.channels_{args.channels}.pretrain_{args.pretrain}.r_{args.r}.alpha_{args.alpha}.seed_{args.seed}.{datetime_now}'
    exp_dir = osp.join('local/', experiment_name)
    os.mkdir(exp_dir)
    logger = Logger.init_logger(filename=exp_dir + '/_output_.log')
    args_print(args, logger)
    logger.info(f"# Train: {n_train_data}  #Test: {n_test_data} #Val: {n_val_data}")
    # evaluator
    evaluator = Evaluator('ogbg-molhiv')

    for seed in args.seed:
        set_seed(seed)
        # models and optimizers
        g = MolHivNet(emb_dim=args.channels).to(device)
        att_net = CausalAttNet(args.r).to(device)
        model_optimizer = torch.optim.Adam(
            list(g.parameters()) +
            list(att_net.parameters()),
            lr=args.net_lr)
        conf_opt = torch.optim.Adam(g.conf_lin.parameters(), lr=args.net_lr)
        BCELoss = torch.nn.BCEWithLogitsLoss(reduction="mean")

        def train_mode():
            g.train();att_net.train()
            
        def val_mode():
            g.eval();att_net.eval()
            
        def test_auc(loader, att_net, g):
            y_true = []
            y_pred = []
            for graph in loader:
                graph = graph.to(device)
                with torch.no_grad():
                    (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), _, _ = att_net(graph)
                    set_masks(causal_edge_weight, g)
                    pred = g(x=causal_x, edge_index=causal_edge_index, 
                            edge_attr=causal_edge_attr, batch=causal_batch)
                    clear_masks(g)
                y_true.append(graph.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())

            y_true = torch.cat(y_true, dim = 0).numpy()
            y_pred = torch.cat(y_pred, dim = 0).numpy()

            input_dict = {"y_true": y_true, "y_pred": y_pred}
            return evaluator.eval(input_dict)['rocauc']

        cnt, last_val_auc = 0, 0
        for epoch in range(args.epoch):
                
            causal_edge_weights = torch.tensor([]).to(device)
            conf_edge_weights = torch.tensor([]).to(device)
            alpha_prime = args.alpha * (epoch ** 1.6)
            all_loss, n_bw, all_env_loss, all_causal_loss = 0, 0, 0, 0
            train_mode()
            for graph in train_loader:
                n_bw += 1
                graph.to(device)
                N = graph.num_graphs
                (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), pred_edge_weight = att_net(graph)

                set_masks(causal_edge_weight, g)
                causal_rep = g.get_graph_rep(
                    x=causal_x, edge_index=causal_edge_index, 
                    edge_attr=causal_edge_attr, batch=causal_batch)
                causal_out = g.get_causal_pred(causal_rep)
                clear_masks(g)
                set_masks(conf_edge_weight, g)
                conf_rep = g.get_graph_rep(
                    x=conf_x, edge_index=conf_edge_index, 
                    edge_attr=conf_edge_attr, batch=conf_batch).detach()
                conf_out = g.get_conf_pred(conf_rep)
                clear_masks(g)
                is_labeled = graph.y == graph.y
                causal_loss = BCELoss(
                    causal_out.to(torch.float32)[is_labeled], 
                    graph.y.to(torch.float32)[is_labeled]
                    )
                conf_loss = BCELoss(
                    conf_out.to(torch.float32)[is_labeled], 
                    graph.y.to(torch.float32)[is_labeled]
                    )
                env_loss = 0
                if args.reg:
                    env_loss = torch.tensor([]).to(device)
                    for conf in conf_rep:
                        rep_out = g.get_comb_pred(causal_rep, conf)
                        tmp = BCELoss(rep_out.to(torch.float32)[is_labeled], graph.y.to(torch.float32)[is_labeled])
                        env_loss = torch.cat([env_loss, tmp.unsqueeze(0)])
                    causal_loss += alpha_prime * env_loss.mean()
                    env_loss = alpha_prime * torch.var(env_loss * conf_rep.size(0))
                
                # logger
                all_causal_loss += causal_loss 
                all_env_loss += env_loss
                batch_loss = causal_loss + env_loss
                causal_edge_weights = torch.cat([causal_edge_weights, causal_edge_weight])
                conf_edge_weights = torch.cat([conf_edge_weights, conf_edge_weight])
                
                conf_opt.zero_grad()
                conf_loss.backward()
                conf_opt.step()

                model_optimizer.zero_grad()
                (causal_loss + env_loss).backward()
                model_optimizer.step()
                
            all_env_loss /= n_bw
            all_causal_loss /= n_bw
            all_loss = all_causal_loss + all_env_loss
            torch.cuda.empty_cache()
            val_mode()
            with torch.no_grad():
                train_auc = test_auc(train_loader, att_net, g)
                val_auc = test_auc(val_loader, att_net, g)
                causal_auc = test_auc(test_loader, att_net, g)
                
                logger.info("Epoch [{:3d}/{:d}]  all_loss:{:2.3f}=[XE:{:2.3f}  IL:{:2.6f}]  "
                            "Train_AUC:{:.3f} Test_AUC:{:.3f}  Val_AUC:{:.3f}  "
                            "stats:[{:.3f}  {:.3f}]".format(
                        epoch, args.epoch, all_loss, all_causal_loss, all_env_loss, 
                        train_auc, causal_auc, val_auc,
                        causal_edge_weights.mean(), conf_edge_weights.mean()))
                # activate early stopping
                if epoch >= args.pretrain:
                    if val_auc < last_val_auc:
                        cnt += 1
                    else:
                        cnt = 0
                        last_val_auc = val_auc
                if cnt >= 5:
                    logger.info("Early Stopping")
                    break
        
        all_info['causal_auc'].append(causal_auc)
        all_info['train_auc'].append(train_auc)
        all_info['val_auc'].append(val_auc)
        torch.save(g.cpu(), osp.join(exp_dir, 'predictor-%d.pt' % seed))
        torch.save(att_net.cpu(), osp.join(exp_dir, 'attention_net-%d.pt' % seed))
        logger.info("=" * 100)

    logger.info("Causal AP:{:.4f}-+-{:.4f}  Train AP:{:.4f}-+-{:.4f}  Val AP:{:.4f}-+-{:.4f}".format(
                    torch.tensor(all_info['causal_auc']).mean(), torch.tensor(all_info['causal_auc']).std(),
                    torch.tensor(all_info['train_auc']).mean(), torch.tensor(all_info['train_auc']).std(),
                    torch.tensor(all_info['val_auc']).mean(), torch.tensor(all_info['val_auc']).std()
                ))