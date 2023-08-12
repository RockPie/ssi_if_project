import torch
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool, EdgeConv
from torch.nn import Linear, Parameter, functional as F

class SJEdgeNet(torch.nn.Module):
    def __init__(self, in_size, layer_size):
        super(SJEdgeNet, self).__init__()
        net_layers = []

        # * subnetwork for edge features
        # * linear layer, batch norm, relu
        net_layers.append(torch.nn.Linear(in_size*2, layer_size))
        net_layers.append(torch.nn.BatchNorm1d(layer_size))
        net_layers.append(torch.nn.ReLU())

        net_layers.append(torch.nn.Linear(layer_size, layer_size))
        net_layers.append(torch.nn.BatchNorm1d(layer_size))
        net_layers.append(torch.nn.ReLU())

        net_layers.append(torch.nn.Linear(layer_size, layer_size))
        net_layers.append(torch.nn.BatchNorm1d(layer_size))
        net_layers.append(torch.nn.ReLU())

        self.model = torch.nn.Sequential(*net_layers)

    def forward(self, x):
        return self.model(x)
    
    # def __repr__(self):
    #     return "{}(nn={})".format(self.__class__.__name__, self.model)

class SJN_NTE(torch.nn.Module):
    def __init__(self, dataset, device):
        super().__init__()
        self.device = device

        self.edge_nets = []
        self.kernal_sizes = [64, 128, 256]  
        self.edge_nets.append(SJEdgeNet(dataset.num_features, self.kernal_sizes[0]))
        self.edge_nets.append(SJEdgeNet(self.kernal_sizes[0], self.kernal_sizes[1]))
        self.edge_nets.append(SJEdgeNet(self.kernal_sizes[1], self.kernal_sizes[2]))

        self.edgeconv1 = EdgeConv(self.edge_nets[0], aggr='mean')
        self.edgeconv2 = EdgeConv(self.edge_nets[1], aggr='mean')
        self.edgeconv3 = EdgeConv(self.edge_nets[2], aggr='mean')

        self.linear = torch.nn.Linear(self.kernal_sizes[-1], 1)

    def forward(self, data):
        x           = data.x
        edge_index  = data.edge_index
        edge_attr   = data.edge_attr

        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        x = self.edgeconv1(x, edge_index)
        x = self.edgeconv2(x, edge_index)
        x = self.edgeconv3(x, edge_index)
        x = F.dropout(x, training=self.training)

        x = self.linear(x)

        res = F.sigmoid(x)

        return res