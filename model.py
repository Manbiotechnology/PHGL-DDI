from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from torch_geometric.nn import  GINConv
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch


embed_dim = 128

class GIN_Top(torch.nn.Module):
    def __init__(self,  hidden=512, train_eps=True):
        super(GIN_Top, self).__init__()
        self.train_eps = train_eps
        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(128, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.lin1 = nn.Linear(hidden, hidden)
        self.fc1 = nn.Linear(2 * hidden, 1)
        self.fc2 = nn.Linear(hidden, 1)


    def reset_parameters(self):

        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        self.gin_conv2.reset_parameters()
        # self.gin_conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


    def forward(self, x, edge_index, train_edge_id):
        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)
        # x = self.gin_conv3(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]
        x = torch.mul(x1, x2)
        x = self.fc2(x)
        return x

class GIN_Bottom(nn.Module):
    def __init__(self):
        super(GIN_Bottom, self).__init__()
        hidden = 128
        self.conv1 = GINConv(55, hidden)
        self.conv2 = GINConv(hidden, hidden)
        self.conv3 = GINConv(hidden, hidden)
        self.conv4 = GINConv(hidden, hidden)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        self.sag1 = SAGPooling(hidden, 0.5)
        self.sag2 = SAGPooling(hidden, 0.5)
        self.sag3 = SAGPooling(hidden, 0.5)
        self.sag4 = SAGPooling(hidden, 0.5)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(0.5)
        for param in self.parameters():
            print(type(param), param.size())

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        y = self.sag1(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.bn4(x)
        y = self.sag4(x, edge_index, batch=batch)
        # y = self.sag4(x, edge_index, batch = batch)
        return global_mean_pool(y[0], y[3])

class GCN_Top(torch.nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.conv1 = GCNConv(128, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 1)  # classifier for inner product

    def forward(self, x, edge_index, train_edge_id, p=0.5):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        h = F.dropout(h, p=0.5, training=self.training)
        node_id = edge_index[:, train_edge_id]
        h1 = h[node_id[0]]
        h2 = h[node_id[1]]
        # x = torch.cat([x1, x2], dim=1)
        # x = self.fc1(x)
        h = torch.mul(h1, h2)
        h = self.fc2(h)

        return h

class GCN_Bottom(nn.Module):
    def __init__(self):
        super(GCN_Bottom, self).__init__()
        hidden = 128
        self.conv1 = GCNConv(55, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)
  
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        self.sag1 = SAGPooling(hidden,0.5)
        self.sag2 = SAGPooling(hidden,0.5)
        self.sag3 = SAGPooling(hidden,0.5)
        self.sag4 = SAGPooling(hidden,0.5)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(0.5)
        for param in self.parameters():
            print(type(param), param.size())


    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.bn1(x)
        y = self.sag1(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1] 

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x) 
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]  
        
        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x) 
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x) 
        x = self.bn4(x)
        y = self.sag4(x, edge_index, batch = batch)

        return global_mean_pool(y[0], y[3])

class GIN_Top(torch.nn.Module):
    def __init__(self, hidden=512, train_eps=True, class_num=7):
        super(GIN_Top, self).__init__()
        self.train_eps = train_eps
        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(128, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.lin1 = nn.Linear(hidden, hidden)
        self.fc1 = nn.Linear(2 * hidden, 1)
        self.fc2 = nn.Linear(hidden, 1)

    def reset_parameters(self):
        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        self.gin_conv2.reset_parameters()
        # self.gin_conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x, edge_index, train_edge_id):
        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)
        # x = self.gin_conv3(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        node_id = edge_index[:, train_edge_id]
        x1 = x[node_id[0]]
        x2 = x[node_id[1]]
        x = torch.mul(x1, x2)
        x = self.fc2(x)

        return x

class GIN_Bottom(nn.Module):
    def __init__(self):
        super(GIN_Bottom, self).__init__()
        hidden = 128
        self.conv1 = GINConv(55, hidden)
        self.conv2 = GINConv(hidden, hidden)
        self.conv3 = GINConv(hidden, hidden)
        self.conv4 = GINConv(hidden, hidden)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        self.sag1 = SAGPooling(hidden, 0.5)
        self.sag2 = SAGPooling(hidden, 0.5)
        self.sag3 = SAGPooling(hidden, 0.5)
        self.sag4 = SAGPooling(hidden, 0.5)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(0.5)
        for param in self.parameters():
            print(type(param), param.size())

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        y = self.sag1(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.bn4(x)
        y = self.sag4(x, edge_index, batch=batch)
        return global_mean_pool(y[0], y[3])


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ddi_model(nn.Module):
    def __init__(self):
        super(ddi_model,self).__init__()
        self.BGNN = GCN_Bottom()
        self.TGNN = GIN_Top()

    def forward(self, batch, d_x_all, d_edge_all, edge_index, train_edge_id):
        edge_index = edge_index.to(device)
        batch = batch.to(torch.int64).to(device)
        x = d_x_all.to(torch.float32).to(device)
        edge = torch.LongTensor(d_edge_all.to(torch.int64)).to(device)
        embs = self.BGNN(x, edge, batch - 1)
        final = self.TGNN(embs, edge_index, train_edge_id, p=0.5)
        return final
