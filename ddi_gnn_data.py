import os
import json
import numpy as np
import copy
import torch
import random

from tqdm import tqdm
from utils.utils import UnionFindSet, get_bfs_sub_graph, get_dfs_sub_graph
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader


class GNN_DATA:
    def __init__(self, ddi_path, skip_head=True, p1_index=0, p2_index=1,
                 label_index=2, graph_undirection=True):
        self.ddi_list = []
        self.ddi_dict = {}
        self.ddi_label_list = []
        self.protein_dict = {}
        self.protein_name = {}
        self.ddi_path = ddi_path
        name = 0
        ddi_name = 0
        self.node_num = 0
        self.edge_num = 0

        for line in tqdm(open(ddi_path)):
            if skip_head:
                skip_head = False
                continue
            line = line.strip().split(',')


            # get node and node name
            if line[p1_index] not in self.protein_name.keys():
                self.protein_name[line[p1_index]] = name
                name += 1

            if line[p2_index] not in self.protein_name.keys():
                self.protein_name[line[p2_index]] = name
                name += 1
            if line[p1_index] < line[p2_index]:
                temp_data = line[p1_index] + "__" + line[p2_index]
            else:
                temp_data = line[p2_index] + "__" + line[p1_index]

            if temp_data not in self.ddi_dict.keys():
                self.ddi_dict[temp_data] = ddi_name
                self.ddi_label_list.append(int(line[label_index]))
                ddi_name += 1
            # else:
            #     index = self.ddi_dict[temp_data]
            #     # temp_label = self.ddi_label_list[index]
            #     # # temp_label[class_map[line[label_index]]] = 1
            #     # # temp_label[int(line[label_index]) - 1] = 1
            #     # temp_label = line[label_index]
            #     # temp_label=line[int(temp_label)]
            #     temp_label = int(line[temp_label])
            #     self.ddi_label_list[index] = temp_label
        i = 0
        for ddi in tqdm(self.ddi_dict.keys()):#keys()返回key

            name = self.ddi_dict[ddi]
            assert name == i
            i += 1
            temp = ddi.strip().split('__')
            self.ddi_list.append(temp)

        ddi_num = len(self.ddi_list)
        self.origin_ddi_list = copy.deepcopy(self.ddi_list)
        assert len(self.ddi_list) == len(self.ddi_label_list)
        for i in tqdm(range(ddi_num)):
            seq1_name = self.ddi_list[i][0]
            seq2_name = self.ddi_list[i][1]
            self.ddi_list[i][0] = self.protein_name[seq1_name]
            self.ddi_list[i][1] = self.protein_name[seq2_name]

        if graph_undirection:
            for i in tqdm(range(ddi_num)):
                temp_ddi = self.ddi_list[i][::-1]
                temp_ddi_label = self.ddi_label_list[i]

                self.ddi_list.append(temp_ddi)
                self.ddi_label_list.append(temp_ddi_label)
        self.node_num = len(self.protein_name)
        self.edge_num = len(self.ddi_list)

    def get_protein_aac(self, pseq_path):
        self.pseq_path = pseq_path
        self.pseq_dict = {}
        self.protein_len = []
        for line in tqdm(open(pseq_path)):
            line = line.strip().split(',')
            if line[0] not in self.pseq_dict.keys():
                self.pseq_dict[line[0]] = line[1]
                self.protein_len.append(len(line[1]))

        print("protein num: {}".format(len(self.pseq_dict)))
        print("protein average length: {}".format(np.average(self.protein_len)))
        print("protein max & min length: {}, {}".format(np.max(self.protein_len), np.min(self.protein_len)))

    def embed_normal(self, seq, dim):
        if len(seq) > self.max_len:
            return seq[:self.max_len]
        elif len(seq) < self.max_len:
            less_len = self.max_len - len(seq)
            return np.concatenate((seq, np.zeros((less_len, dim))))
        return seq

    def vectorize(self, vec_path):
        self.acid2vec = {}
        self.dim = None
        for line in open(vec_path):
            line = line.strip().split('\t')
            temp = np.array([float(x) for x in line[1].split()])
            self.acid2vec[line[0]] = temp
            if self.dim is None:
                self.dim = len(temp)
        print("acid vector dimension: {}".format(self.dim))

        self.pvec_dict = {}

        for p_name in tqdm(self.pseq_dict.keys()):
            temp_seq = self.pseq_dict[p_name]
            temp_vec = []
            for acid in temp_seq:
                temp_vec.append(self.acid2vec[acid])
            temp_vec = np.array(temp_vec)
            temp_vec = self.embed_normal(temp_vec, self.dim)
            self.pvec_dict[p_name] = temp_vec

    def get_feature_origin(self, pseq_path):
        self.get_protein_aac(pseq_path)

    def get_connected_num(self):
        self.ufs = UnionFindSet(self.node_num)
        ddi_ndary = np.array(self.ddi_list)
        for edge in ddi_ndary:
            start, end = edge[0], edge[1]
            self.ufs.union(start, end)

    def generate_data(self):
        self.get_connected_num()
        print("Connected domain num: {}".format(self.ufs.count))

        ddi_list = np.array(self.ddi_list)
        ddi_label_list = np.array(self.ddi_label_list)
        self.edge_index = torch.tensor(ddi_list, dtype=torch.long)
        self.edge_attr = torch.tensor(ddi_label_list, dtype=torch.long)
        self.data = Data(edge_index=self.edge_index.T, edge_attr=self.edge_attr)

    def split_dataset(self, train_valid_index_path, test_size=0.2, random_new=True, mode='random'):
        if random_new:
            if mode == 'random':
                ddi_num = int(self.edge_num // 2)
                random_list = [i for i in range(ddi_num)]
                random.shuffle(random_list)

                self.ddi_split_dict = {}
                self.ddi_split_dict['train_index'] = random_list[: int(ddi_num * (1 - test_size))]
                self.ddi_split_dict['valid_index'] = random_list[int(ddi_num * (1 - test_size)):]

                jsobj = json.dumps(self.ddi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()
                print("split done")
            elif mode == 'bfs' or mode == 'dfs':
                print("use {} methed split train and valid dataset".format(mode))
                node_to_edge_index = {}
                edge_num = int(self.edge_num // 2)
                for i in range(edge_num):
                    edge = self.ddi_list[i]
                    if edge[0] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[0]] = []
                    node_to_edge_index[edge[0]].append(i)

                    if edge[1] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[1]] = []
                    node_to_edge_index[edge[1]].append(i)

                node_num = len(node_to_edge_index)

                sub_graph_size = int(edge_num * test_size)
                if mode == 'bfs':
                    selected_edge_index = get_bfs_sub_graph(self.ddi_list, node_num, node_to_edge_index, sub_graph_size)
                elif mode == 'dfs':
                    selected_edge_index = get_dfs_sub_graph(self.ddi_list, node_num, node_to_edge_index, sub_graph_size)

                all_edge_index = [i for i in range(edge_num)]

                unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

                self.ddi_split_dict = {}
                self.ddi_split_dict['train_index'] = unselected_edge_index
                self.ddi_split_dict['valid_index'] = selected_edge_index

                assert len(unselected_edge_index) + len(selected_edge_index) == edge_num

                jsobj = json.dumps(self.ddi_split_dict)
                with open(train_valid_index_path, 'w') as f:
                    f.write(jsobj)
                    f.close()

            else:
                print("your mode is {}, you should use bfs, dfs or random".format(mode))
                return
        else:
            with open(train_valid_index_path, encoding='utf-8-sig',errors='ignore') as f:
                str = f.read()
                self.ddi_split_dict = json.loads(str, strict=False)
                f.close()
