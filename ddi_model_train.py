import os
import pickle
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
from ddi_gnn_data import GNN_DATA

from model import ddi_model
from utils.utils import Metrictor_DDI, print_file
from tensorboardX import SummaryWriter#可视化

seed_num = 2
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def multi2big_x(x_ori):
    N = len(x_ori)
    x_cat = torch.zeros(1,55)
    x_num_index = torch.zeros(N)
    for i in range(N):
        x_now = x_ori[i].clone().detach()
        x_num_index[i] = torch.tensor(x_now.size(0))
        x_cat = torch.cat((x_cat, x_now), 0)
    return x_cat[1:], x_num_index

def multi2big_batch(x_num_index):
    N = len(x_num_index)
    num_sum = x_num_index.sum()
    num_sum = num_sum.int()
    batch = torch.zeros(num_sum)
    count = 1
    for i in range(1,N):
        zj1 = x_num_index[:i]
        zj11 = zj1.sum()
        zj11 = zj11.int()
        zj22 = zj11 + x_num_index[i]
        zj22 = zj22.int()
        size1 = x_num_index[i]
        size1 = size1.int()
        tc = count * torch.ones(size1)
        batch[zj11:zj22] = tc
        test = batch[zj11:zj22]
        count = count + 1
    batch = batch.int()
    return batch

def multi2big_edge(edge_ori, num_index):
    N = len(num_index)
    edge_cat = torch.zeros(2, 1)
    edge_num_index = torch.zeros(N)
    for i in range(N):
        edge_index_p = edge_ori[i]
        edge_num_index[i] = torch.tensor(edge_index_p.size(1))
        if i == 0:
            offset = 0
        else:
            zj = num_index[:i].clone().detach()
            offset = zj.sum()
        edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
    return edge_cat[:, 1:], edge_num_index


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def train(batch, p_x_all, p_edge_all, model, graph, loss_fn, optimizer, device,
        result_file_path, summary_writer, save_path,
        batch_size=5000, epochs=500, scheduler=None,
        got=False):
    global_step = 0
    global_best_valid_f1 = 0.0
    global_best_valid_f1_epoch = 0
    # batch = torch.zeros(818994)
    # truth_edge_num = graph.graph_edge_index.shape[1]
    # count = 1
    # for i in range(1, 1552):
    #     num1 = x_num_index[i]
    #     num1 = num1.int()
    #     zj = x_num_index[0:i + 1]
    #     num2 = zj.sum()
    #     num2 = num2.int()
    #     batch[num1:num2] = torch.ones(num2 - num1) * count
    #     count = count + 1
    for epoch in range(epochs):

        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        specificity_sum = 0.0
        loss_sum = 0.0
        Accuracy_sum = 0.0
        # PR_Auc_sum = 0.0
        ROC_Auc_sum = 0.0
        steps = math.ceil(len(graph.train_mask) / batch_size)
        model.train()

        random.shuffle(graph.train_mask)
        random.shuffle(graph.train_mask_got)

        for step in range(steps):
            if step == steps - 1:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size:]
                else:
                    train_edge_id = graph.train_mask[step * batch_size:]
            else:
                if got:
                    train_edge_id = graph.train_mask_got[step * batch_size: step * batch_size + batch_size]
                else:
                    train_edge_id = graph.train_mask[step * batch_size: step * batch_size + batch_size]

            if got:
                output = model(batch, p_x_all, p_edge_all, graph.edge_index_got, train_edge_id)
                label = graph.edge_attr_got[train_edge_id]
            else:
                output = model(batch, p_x_all, p_edge_all, graph.edge_index, train_edge_id)
                label = graph.edge_attr[train_edge_id]
            output = output.squeeze()
            label = label.type(torch.FloatTensor).to(device)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            m = nn.Sigmoid()
            #  # m = nn.LeakyReLU()
            # m = nn.ReLU6()
            # m = nn.ReLU()
            # m = nn.PReLU()
            # m = nn.RReLU()
            # m =nn.Tanh()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

            metrics = Metrictor_DDI(pre_result.cpu().data, label.cpu().data, m(output).cpu().data)

            metrics.show_result()
           
            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            Accuracy_sum += metrics.Accuracy
            f1_sum += metrics.F1
            specificity_sum += metrics.specificity
            ROC_Auc_sum += metrics.ROC_Auc
            # PR_Auc_sum += metrics.PR_Auc
            loss_sum += loss.item()
            
            summary_writer.add_scalar('train/loss', loss.item(), global_step)
            summary_writer.add_scalar('train/Accuracy', metrics.Accuracy, global_step)
            summary_writer.add_scalar('train/precision', metrics.Precision, global_step)
            summary_writer.add_scalar('train/recall', metrics.Recall, global_step)
            summary_writer.add_scalar('train/F1', metrics.F1, global_step)
            summary_writer.add_scalar('train/specificity', metrics.specificity, global_step)
            summary_writer.add_scalar('train/ROC_Auc', metrics.ROC_Auc, global_step)
            # summary_writer.add_scalar('train/PR_Auc', metrics.PR_Auc, global_step)
            global_step += 1
            print_file(
                "epoch: {}, step: {}, Train: label_loss: {},Accuracy:{}，precision: {}, recall: {}, f1: {}, specificity:{},ROC_Auc: {},"
                .format(epoch, step, loss.item(), metrics.Accuracy, metrics.Precision, metrics.Recall, metrics.F1,
                        metrics.specificity, metrics.ROC_Auc,))
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                   os.path.join(save_path, 'gnn_model_train.ckpt'))

        valid_pre_result_list = []
        valid_label_list = []
        true_prob_list = []
        valid_loss_sum = 0.0

        model.eval()

        valid_steps = math.ceil(len(graph.val_mask) / batch_size)

        with torch.no_grad():
            for step in range(valid_steps):
                if step == valid_steps - 1:
                    valid_edge_id = graph.val_mask[step * batch_size:]
                else:
                    valid_edge_id = graph.val_mask[step * batch_size: step * batch_size + batch_size]
                output = model(batch, p_x_all, p_edge_all, graph.edge_index, valid_edge_id)
                output = output.squeeze()
                label = graph.edge_attr[valid_edge_id]
                label = label.type(torch.FloatTensor).to(device)

                loss = loss_fn(output, label)
                valid_loss_sum += loss.item()

                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)
                true_prob_list.append(m(output).cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        true_prob_list = torch.cat(true_prob_list, dim = 0)

        metrics = Metrictor_DDI(valid_pre_result_list, valid_label_list, true_prob_list)

        metrics.show_result()

        recall = recall_sum / steps
        precision = precision_sum / steps
        f1 = f1_sum / steps
        loss = loss_sum / steps
        specificity =specificity_sum / steps
        Accuracy = Accuracy_sum / steps
        valid_loss = valid_loss_sum / valid_steps
        ROC_Auc = ROC_Auc_sum / steps
        # PR_Auc = PR_Auc_sum / steps

        if scheduler != None:
            scheduler.step(loss)
            print_file("epoch: {}, now learning rate: {}".format(epoch, scheduler.optimizer.param_groups[0]['lr']),
                       save_file_path=result_file_path)

        if global_best_valid_f1 < metrics.F1:
            global_best_valid_f1 = metrics.F1
            global_best_valid_f1_epoch = epoch

            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict()},
                       os.path.join(save_path, 'gnn_model_valid_best.ckpt'))

        summary_writer.add_scalar('valid/precision', metrics.Precision, global_step)
        summary_writer.add_scalar('valid/Accuracy', metrics.Accuracy, global_step)
        summary_writer.add_scalar('valid/recall', metrics.Recall, global_step)
        summary_writer.add_scalar('valid/F1', metrics.F1, global_step)
        summary_writer.add_scalar('valid/specificity', metrics.specificity, global_step)
        summary_writer.add_scalar('valid/loss', valid_loss, global_step)
        summary_writer.add_scalar('valid/ROC_Auc', metrics.ROC_Auc, global_step)
        print_file(
            "epoch: {}, Training_avg: label_loss: {}, recall: {}, precision: {}, Accuracy:{},specificity:{},F1: {},ROC_Auc:{}, Validation_avg: loss: {}, recall: {}, precision: {}, F1: {},Accuracy: {},specificity:{},ROC_Auc:{} Best valid_f1: {}, in {} epoch"
            .format(epoch, loss, recall, precision, Accuracy, specificity, f1, ROC_Auc, valid_loss,
                    metrics.Recall,
                    metrics.Precision, metrics.F1, metrics.Accuracy, metrics.specificity, metrics.ROC_Auc,
                    global_best_valid_f1, global_best_valid_f1_epoch), save_file_path=result_file_path)
def main():
    ppi_data = GNN_DATA(ppi_path='./drug_info/ChChMiner.csv')
    ppi_data.get_feature_origin(pseq_path='./drug_info/ChChMiner_smiles.csv')
    ppi_data.generate_data()
    ppi_data.split_dataset(train_valid_index_path='./drug_info/train_val_split_ChChMiner.json', random_new=True,
                           mode='random')
    graph = ppi_data.data
    ppi_list = ppi_data.ppi_list

    graph.train_mask = ppi_data.ppi_split_dict['train_index']
    graph.val_mask = ppi_data.ppi_split_dict['valid_index']



    p_x_all = torch.load('./drug_info/input/d_x_list.pt')
    p_edge_all = torch.load('./drug_info/input/d_edge_list.pt')
    p_x_all, x_num_index = multi2big_x(p_x_all)
    p_edge_all, edge_num_index = multi2big_edge(p_edge_all, x_num_index)
    batch = multi2big_batch(x_num_index)+1




    print("train gnn, train_num: {}, valid_num: {}".format(len(graph.train_mask), len(graph.val_mask)))
    graph.edge_index_got = torch.cat(
        (graph.edge_index[:, graph.train_mask], graph.edge_index[:, graph.train_mask][[1, 0]]), dim=1)
    graph.edge_attr_got = torch.cat((graph.edge_attr[graph.train_mask], graph.edge_attr[graph.train_mask]), dim=0)

    graph.train_mask_got = [i for i in range(len(graph.train_mask))]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    graph.to(device)
    model = ddi_model()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # scheduler = None
    #
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           verbose=True)
    save_path = './result_save'
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    time_stamp = time.strftime("%Y-%m-%d %H-%M-%S")
    save_path = os.path.join(save_path, "gnn_{}".format('training_seed_1'))
    result_file_path = os.path.join(save_path, "valid_results.txt")
    config_path = os.path.join(save_path, "config.txt")
    # os.mkdir(save_path)

    summary_writer = SummaryWriter(save_path)
    train(batch, p_x_all, p_edge_all, model, graph, ppi_list, loss_fn, optimizer, device,
          result_file_path, summary_writer, save_path,
          batch_size=5000, epochs=500, scheduler=scheduler,
          got=False)
    summary_writer.close()


if __name__ == "__main__":
    main()
