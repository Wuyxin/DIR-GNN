import copy
import torch
import argparse
from datasets import SPMotif
from torch_geometric.data import DataLoader


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LEConv, BatchNorm, fps
from utils.mask import set_masks, clear_masks

import os
import numpy as np
import os.path as osp
from torch.autograd import grad
from utils.logger import Logger
from datetime import datetime
from utils.helper import set_seed, args_print
from utils.get_subgraph import split_graph, split_batch, relabel
from gnn import SPMotifNet



class CausalAttNet(nn.Module):
    
    def __init__(self, causal_ratio):
        super(CausalAttNet, self).__init__()
        self.conv1 = LEConv(in_channels=4, out_channels=args.channels)
        self.conv2 = LEConv(in_channels=args.channels, out_channels=args.channels)
        self.mlp = nn.Sequential(
            nn.Linear(args.channels*2, args.channels*4),
            nn.ReLU(),
            nn.Linear(args.channels*4, 1)
        )
        self.ratio = causal_ratio
    
    def forward(self, data):
        # batch_norm
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr.view(-1)))
        x = self.conv2(x, data.edge_index, data.edge_attr.view(-1))

        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_score = self.mlp(edge_rep).view(-1)

        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_score = self.mlp(edge_rep).view(-1)

        (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        (conf_edge_index, conf_edge_attr, conf_edge_weight) = split_graph(data,edge_score, self.ratio)

        causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)
        conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, data.batch)

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch),\
                edge_score

if __name__ == "__main__":
            
    parser = argparse.ArgumentParser(description='Training for Causal Feature Learning')
    parser.add_argument('--cuda', default=0, type=int, help='cuda device')
    parser.add_argument('--datadir', default='data/', type=str, help='directory for datasets.')
    parser.add_argument('--epoch', default=400, type=int, help='training iterations')
    parser.add_argument('--reg', default=1, type=int)
    parser.add_argument('--seed',  nargs='?', default='[1]', help='random seed')
    parser.add_argument('--channels', default=32, type=int, help='width of network')
    parser.add_argument('--commit', default='', type=str, help='experiment name')
    parser.add_argument('--bias', default='0.9', type=str, help='select bias extend')
    # hyper 
    parser.add_argument('--pretrain', default=10, type=int, help='pretrain epoch')
    parser.add_argument('--alpha', default=1e-2, type=float, help='invariant loss')
    parser.add_argument('--r', default=0.25, type=float, help='causal_ratio')
    # basic
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--net_lr', default=1e-3, type=float, help='learning rate for the predictor')
    args = parser.parse_args()
    args.seed = eval(args.seed)
    # dataset
    num_classes = 3
    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu') 
    train_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='train')
    val_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='val')
    test_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    n_train_data, n_val_data = len(train_dataset), len(val_dataset)
    n_test_data = float(len(test_dataset))

    # log
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_info = { 'causal_acc':[], 'conf_acc':[], 'train_acc':[], 'val_acc':[], 'test_prec':[], 'train_prec':[], 'test_mrr':[], 'train_mrr':[]}
    experiment_name = f'spmotif-{args.bias}.{bool(args.reg)}.{args.commit}.netlr_{args.net_lr}.batch_{args.batch_size}'\
                      f'.channels_{args.channels}.pretrain_{args.pretrain}.r_{args.r}.alpha_{args.alpha}.seed_{args.seed}.{datetime_now}'
    exp_dir = osp.join('local/', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger.init_logger(filename=exp_dir + '/_output_.log')
    args_print(args, logger)

    for seed in args.seed:
        
        set_seed(seed)
        # models and optimizers
        g = SPMotifNet(args.channels).to(device)
        att_net = CausalAttNet(args.r).to(device)
        model_optimizer = torch.optim.Adam(
            list(g.parameters()) +
            list(att_net.parameters()),
            lr=args.net_lr)
        conf_opt = torch.optim.Adam(g.conf_fw.parameters(), lr=args.net_lr)
        CELoss = nn.CrossEntropyLoss(reduction="mean")
        EleCELoss = nn.CrossEntropyLoss(reduction="none")

        def train_mode():
            g.train();att_net.train()
            
        def val_mode():
            g.eval();att_net.eval()

        def test_metrics(loader, att_net):
            def metrics_batch(graph, pred_weight, mrr_k=5):
                _, _, _, num_edges, cum_edges = split_batch(graph)
                ground_truth_mask = graph.edge_gt_att.view(-1)
                
                precision, mrr = [], []
                for E, C in zip(num_edges.tolist(), cum_edges.tolist()):
                    # compute precision
                    num_gd = int(ground_truth_mask[C: C + E].sum())
                    pred = pred_weight[C:C + E]
                    _, indices_for_sort = pred.sort(descending=True, dim=-1)
                    idx = indices_for_sort[:num_gd].detach().cpu().numpy()
                    precision.append(ground_truth_mask[C: C + E][idx].sum().float()/num_gd)
                    
                    # compute mrr
                    k = min(pred.size(0), mrr_k)
                    true_sorted_by_preds = torch.gather(
                        graph.edge_gt_att[C: C + E], dim=-1, index=indices_for_sort
                    )
                    true_sorted_by_pred_shrink = true_sorted_by_preds[:k]
                    values, indices = torch.max(true_sorted_by_pred_shrink, dim=0)
                    indices = indices.type_as(values).unsqueeze(dim=0).t()
                    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))
                    zero_sum_mask = values == 0.0
                    result[zero_sum_mask] = 0.0
                    mrr.append(result[0])
                return torch.tensor(precision), torch.tensor(mrr)

            precision_lst, mrr_lst =  torch.FloatTensor([]), torch.FloatTensor([])
            for graph in loader: 
                graph.to(device)
                causal_g, conf_g, edge_score = att_net(graph)
                precision, mrr = metrics_batch(graph, edge_score)
                precision_lst = torch.cat([precision_lst, precision])
                mrr_lst = torch.cat([mrr_lst, mrr])
            return torch.mean(precision_lst), torch.mean(mrr_lst)
            
        def test_acc(loader, att_net, predictor):
            acc = 0
            for graph in loader: 
                graph.to(device)
                
                (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), edge_score = att_net(graph)
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
            dummy_w = nn.Parameter(torch.Tensor([1.0])).to(device)
            train_mode()
            for graph in train_loader:
                n_bw += 1
                graph.to(device)
                N = graph.num_graphs
                (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), edge_score = att_net(graph)

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
                    for conf in conf_rep:
                        rep_out = g.get_comb_pred(causal_rep, conf)
                        env_loss = torch.cat([env_loss, CELoss(rep_out, graph.y).unsqueeze(0)])
                    causal_loss += alpha_prime * env_loss.mean()
                    env_loss = alpha_prime * torch.var(env_loss * conf_rep.size(0))

                # logger
                all_conf_loss += conf_loss
                all_causal_loss += causal_loss
                all_env_loss += env_loss
                causal_edge_weights = torch.cat([causal_edge_weights, causal_edge_weight])
                conf_edge_weights = torch.cat([conf_edge_weights, conf_edge_weight])

            all_env_loss /= n_bw
            all_causal_loss /= n_bw
            all_conf_loss /= n_bw
            all_loss = all_causal_loss + all_env_loss

            conf_opt.zero_grad()
            all_conf_loss.backward()
            conf_opt.step()

            model_optimizer.zero_grad()
            all_loss.backward()
            model_optimizer.step()
            val_mode()
            with torch.no_grad():
                test_prec, test_mrr = test_metrics(test_loader, att_net)
                train_prec, train_mrr = test_metrics(train_loader, att_net)

                train_acc = test_acc(train_loader, att_net, g)
                val_acc = test_acc(val_loader, att_net, g)
                # testing acc
                causal_acc, conf_acc = 0., 0.
                for graph in test_loader: 
                    graph.to(device)
                    (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                    (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), edge_score = att_net(graph)
                    
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
                            "Prec[{:.3f}  {:.3f}]  MRR[{:.4f}  {:.4f}]  stats[{:.3f}  {:.3f}]".format(
                        epoch, args.epoch, all_loss, all_causal_loss, all_env_loss, 
                        train_acc, causal_acc, conf_acc, val_acc,
                        test_prec, train_prec, test_mrr, train_mrr,
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
        all_info['test_prec'].append(test_prec)
        all_info['train_prec'].append(train_prec)
        all_info['test_mrr'].append(test_mrr)
        all_info['train_mrr'].append(train_mrr)
        torch.save(g.cpu(), osp.join(exp_dir, 'predictor-%d.pt' % seed))
        torch.save(att_net.cpu(), osp.join(exp_dir, 'attention_net-%d.pt' % seed))
        logger.info("=" * 100)

    logger.info("Causal ACC:{:.4f}-+-{:.4f}  Conf ACC:{:.4f}-+-{:.4f}  Train ACC:{:.4f}-+-{:.4f}  Val ACC:{:.4f}-+-{:.4f}  "
                "Test Prec:{:.4f}-+-{:.4f}   Train Prec:{:.4f}-+-{:.4f}  Test MRR:{:.4f}-+-{:.4f}   Train MRR:{:.4f}-+-{:.4f}  ".format(
                    torch.tensor(all_info['causal_acc']).mean(), torch.tensor(all_info['causal_acc']).std(),
                    torch.tensor(all_info['conf_acc']).mean(), torch.tensor(all_info['conf_acc']).std(),
                    torch.tensor(all_info['train_acc']).mean(), torch.tensor(all_info['train_acc']).std(),
                    torch.tensor(all_info['val_acc']).mean(), torch.tensor(all_info['val_acc']).std(),
                    torch.tensor(all_info['test_prec']).mean(), torch.tensor(all_info['test_prec']).std(),
                    torch.tensor(all_info['train_prec']).mean(), torch.tensor(all_info['train_prec']).std(),
                    torch.tensor(all_info['test_mrr']).mean(), torch.tensor(all_info['test_mrr']).std(),
                    torch.tensor(all_info['train_mrr']).mean(), torch.tensor(all_info['train_mrr']).std()
                ))
            