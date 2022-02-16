import copy
import torch
import argparse
from torch_geometric.data import DataLoader


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv
from utils.mask import set_masks, clear_masks

import os
import numpy as np
import os.path as osp
from torch.autograd import grad
from utils.logger import Logger
from datetime import datetime
from utils.helper import set_seed, args_print
from utils.get_subgraph import split_batch, relabel
from datasets.graphss2_dataset import get_dataset, get_dataloader  
from gnn import GraphSST2Net

class CausalAttNet(nn.Module):
    
    def __init__(self, causal_ratio):
        super(CausalAttNet, self).__init__()
        self.conv1 = ARMAConv(in_channels=768, out_channels=args.channels)
        self.conv2 = ARMAConv(in_channels=args.channels, out_channels=args.channels)
        self.mlp = nn.Sequential(
            nn.Linear(args.channels*2, args.channels*4),
            nn.ReLU(),
            nn.Linear(args.channels*4, 1)
        )
        self.ratio = causal_ratio
    
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr.view(-1)))
        x = self.conv2(x, data.edge_index, data.edge_attr.view(-1))

        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        pred_edge_weight = self.mlp(edge_rep).view(-1)

        causal_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
        causal_edge_weight = torch.tensor([]).to(data.x.device)
        causal_edge_attr = torch.tensor([]).to(data.x.device)
        conf_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
        conf_edge_weight = torch.tensor([]).to(data.x.device)
        conf_edge_attr = torch.tensor([]).to(data.x.device)

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
            conf_edge_weight = torch.cat([conf_edge_weight,  -1 * single_mask[idx_drop]])

            causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
            conf_edge_attr = torch.cat([conf_edge_attr, edge_attr[idx_drop]])
        causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)
        conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, data.batch)

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch),\
                pred_edge_weight

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training for Causal Feature Learning')
    parser.add_argument('--cuda', default=0, type=int, help='cuda device')
    parser.add_argument('--datadir', default='data/', type=str, help='directory for datasets.')
    parser.add_argument('--epoch', default=400, type=int, help='training iterations')
    parser.add_argument('--reg', default=1, type=int)
    parser.add_argument('--seed',  nargs='?', default='[1]', help='random seed')
    parser.add_argument('--channels', default=128, type=int, help='width of network')
    parser.add_argument('--commit', default='', type=str, help='experiment name')
    parser.add_argument('--type', default='none', type=str, choices=['none', 'micro', 'macro'])
    # hyper 
    parser.add_argument('--pretrain', default=0, type=int, help='pretrain epoch')
    parser.add_argument('--alpha', default=10, type=float, help='invariant loss')
    parser.add_argument('--r', default=0.6, type=float, help='causal_ratio')
    # basic
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--net_lr', default=2*1e-4, type=float, help='learning rate for the predictor')
    args = parser.parse_args()
    args.seed = eval(args.seed)


    # dataset
    num_classes = 2
    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu') 
    dataset = get_dataset(dataset_dir='data/', dataset_name='Graph_SST2', task=None)
    dataloader = get_dataloader(dataset,  
                                batch_size=args.batch_size,
                                degree_bias=True, 
                                seed=args.seed)    
    train_loader = dataloader['train']
    val_loader = dataloader['eval'] 
    test_loader = dataloader['test']
    n_train_data, n_val_data = len(train_loader.dataset), len(val_loader.dataset)
    n_test_data = float(len(test_loader.dataset))

    # log
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_info = { 'causal_acc':[], 'conf_acc':[], 'train_acc':[], 'val_acc':[]}
    experiment_name = f'graphsst2.{args.type}.{bool(args.reg)}.{args.commit}.netlr_{args.net_lr}.batch_{args.batch_size}.channels_{args.channels}.pretrain_{args.pretrain}.r_{args.r}.alpha_{args.alpha}.seed_{args.seed}.{datetime_now}'
    exp_dir = osp.join('local/', experiment_name)
    os.mkdir(exp_dir)
    logger = Logger.init_logger(filename=exp_dir + '/_output_.log')
    args_print(args, logger)

    for seed in args.seed:
        
        set_seed(seed)
        # models and optimizers
        g = GraphSST2Net(args.channels).to(device)
        att_net = CausalAttNet(args.r).to(device)
        model_optimizer = torch.optim.Adam(
            list(g.parameters()) +
            list(g.causal_mlp.parameters()) +
            list(att_net.parameters()),
            lr=args.net_lr)
        conf_opt = torch.optim.Adam(g.conf_mlp.parameters(), lr=args.net_lr)
        CELoss = nn.CrossEntropyLoss(reduction="mean")
        EleCELoss = nn.CrossEntropyLoss(reduction="none")

        def train_mode():
            g.train();att_net.train()
            
        def val_mode():
            g.eval();att_net.eval()

        def test_acc(loader, att_net, predictor):
            acc = 0
            for graph in loader: 
                graph.to(device)
                
                (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), _, _ = att_net(graph)
                set_masks(causal_edge_weight, g)
                out = predictor(x=causal_x, edge_index=causal_edge_index, 
                        edge_attr=causal_edge_attr, batch=causal_batch)
                clear_masks(g)
                
                acc += torch.sum(out.argmax(-1).view(-1) == graph.y.view(-1))
            acc = float(acc) / len(loader.dataset)
            return acc

        logger.info(f"# Train: {n_train_data}  #Test: {n_test_data} #Val: {n_val_data}")
        cnt, last_val_acc = 0, 0
        for epoch in range(args.epoch):
                
            causal_edge_weights = torch.tensor([]).to(device)
            conf_edge_weights = torch.tensor([]).to(device)
            reg = args.reg
            alpha_prime = args.alpha * (epoch ** 1.6)
            all_loss, n_bw, all_env_loss = 0, 0, 0
            all_causal_loss, all_conf_loss, all_var_loss = 0, 0, 0
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
                causal_loss = CELoss(causal_out, graph.y)
                conf_loss = CELoss(conf_out, graph.y)

                env_loss = 0
                if args.reg:
                    env_loss = torch.tensor([]).to(device)
                    for idx, causal in enumerate(causal_rep):
                        rep_out = g.get_comb_pred(causal, conf_rep)
                        tmp = EleCELoss(rep_out, graph.y[idx].repeat(rep_out.size(0)))
                        causal_loss += alpha_prime * tmp.mean() / causal_rep.size(0)
                        env_loss = torch.cat([env_loss, torch.var(tmp).unsqueeze(0)])
                    env_loss = alpha_prime * env_loss.mean()
                
                # logger
                all_conf_loss += conf_loss
                all_causal_loss += causal_loss
                all_env_loss += env_loss
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

                train_acc = test_acc(train_loader, att_net, g)
                val_acc = test_acc(val_loader, att_net, g)
                # testing acc
                causal_acc, conf_acc = 0., 0.
                for graph in test_loader: 
                    graph.to(device)
                    (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                    (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), pred_edge_weight = att_net(graph)
                    
                    set_masks(causal_edge_weight, g)
                    causal_out = g(
                        x=causal_x, edge_index=causal_edge_index, 
                        edge_attr=causal_edge_attr, batch=causal_batch)
                    set_masks(conf_edge_weight, g)
                    conf_out = g(x=conf_x, edge_index=conf_edge_index, 
                            edge_attr=conf_edge_attr, batch=conf_batch)
                    clear_masks(g)
                    causal_acc += torch.sum(causal_out.argmax(-1).view(-1) == graph.y.view(-1)) / n_test_data
                    conf_acc += torch.sum(conf_out.argmax(-1).view(-1) == graph.y.view(-1)) / n_test_data
                        
                        
                logger.info("Epoch [{:3d}/{:d}]  all_loss:{:2.3f}=[XE:{:2.3f}  IL:{:2.6f}]  "
                            "Train_ACC:{:.3f} Test_ACC[{:.3f}  {:.3f}]  Val_ACC:{:.3f}  "
                            "stats[{:.3f}  {:.3f}]".format(
                        epoch, args.epoch, all_loss, all_causal_loss, env_loss, 
                        train_acc, causal_acc, conf_acc, val_acc,
                        causal_edge_weights.mean(), conf_edge_weights.mean()))
            
                # activate early stopping
                if epoch >= args.pretrain:
                    if val_acc < last_val_acc:
                        cnt += 1
                    else:
                        cnt = 0
                        last_val_acc = val_acc
                if cnt >= 5:
                    logger.info("Early Stopping")
                    break

            
        all_info['causal_acc'].append(causal_acc)
        all_info['conf_acc'].append(conf_acc)
        all_info['train_acc'].append(train_acc)
        all_info['val_acc'].append(val_acc)
        torch.save(g.cpu(), osp.join(exp_dir, 'predictor-%d.pt' % seed))
        torch.save(att_net.cpu(), osp.join(exp_dir, 'attention_net-%d.pt' % seed))
        logger.info("=" * 100)

    logger.info("Causal ACC:{:.4f}-+-{:.4f}  Conf ACC:{:.4f}-+-{:.4f}  Train ACC:{:.4f}-+-{:.4f}  Val ACC:{:.4f}-+-{:.4f}".format(
                    torch.tensor(all_info['causal_acc']).mean(), torch.tensor(all_info['causal_acc']).std(),
                    torch.tensor(all_info['conf_acc']).mean(), torch.tensor(all_info['conf_acc']).std(),
                    torch.tensor(all_info['train_acc']).mean(), torch.tensor(all_info['train_acc']).std(),
                    torch.tensor(all_info['val_acc']).mean(), torch.tensor(all_info['val_acc']).std()
                ))
            