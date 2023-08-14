import torch
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool, EdgeConv, MetaLayer
from torch_geometric.utils import scatter
from torch.nn import Linear, Parameter, functional as F

class EdgeModel(torch.nn.Module):
    def __init__(self, edge_feature_size, node_feature_size,layer_size):
        super().__init__()
        net_layers = []
        in_size = edge_feature_size + 2*node_feature_size + 1

        net_layers.append(torch.nn.Linear(in_size, layer_size))
        net_layers.append(torch.nn.BatchNorm1d(layer_size))
        net_layers.append(torch.nn.ReLU())

        net_layers.append(torch.nn.Linear(layer_size, layer_size))
        net_layers.append(torch.nn.BatchNorm1d(layer_size))
        net_layers.append(torch.nn.ReLU())

        self.model = torch.nn.Sequential(*net_layers)

        self.linear = torch.nn.Linear(layer_size, edge_feature_size)

    def forward(self, src, dst, edge_attr, u, batch):
        # src, dst: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        # print("src size: ", src.size())
        # print("dst size: ", dst.size())
        # print("edge_attr size: ", edge_attr.size())
        # print("batch size: ", batch.size())
        # make batch two dimension
        batch = batch.unsqueeze(1)
        out = torch.cat([src, dst, edge_attr, batch], 1)
        out = self.model(out)
        out = self.linear(out)
        out = torch.nn.ReLU()(out)
        # out = F.dropout(out, training=self.training)

        return out

class NodeModel(torch.nn.Module):
    def __init__(self, edge_feature_size, node_feature_size, layer_size):
        super().__init__()
        self.node_mlp_1 = torch.nn.Sequential(
            torch.nn.Linear(node_feature_size + edge_feature_size + 1, layer_size),
            torch.nn.BatchNorm1d(layer_size),
            torch.nn.ReLU()
        )
        self.node_mlp_2 = torch.nn.Sequential(
            torch.nn.Linear(node_feature_size + layer_size + 1, node_feature_size + layer_size + 1),
            torch.nn.BatchNorm1d(layer_size + node_feature_size + 1),
            torch.nn.ReLU()
        )

        self.linear = torch.nn.Linear(layer_size + node_feature_size + 1, node_feature_size)
    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        # print("node number: ", x.size(0))
        # print("edge number: ", edge_index.size(1))
        row, col = edge_index
        batch = batch.unsqueeze(1)
        out = torch.cat([x[row], batch[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0),
                      reduce='mean')
        out = torch.cat([x, batch, out], dim=1)
        out = self.node_mlp_2(out)
        out = self.linear(out)
        out = torch.nn.ReLU()(out)
        # out = F.dropout(out, training=self.training)
        return out

class GlobalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.global_mlp = torch.nn.Linear(1, 1)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        # out = torch.cat([
        #     u,
        #     scatter(x, batch, dim=0, reduce='mean'),
        # ], dim=1)
        return u # * do nothing

class SJN_Meta(torch.nn.Module):
    def __init__(self, dataset, device):
        super().__init__()
        self.device = device
        node_feature_size = 16
        edge_feature_size = 19

        self.metalayer1 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 64), 
            NodeModel(edge_feature_size, node_feature_size, 64), 
            GlobalModel())
        self.metalayer2 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 256), 
            NodeModel(edge_feature_size, node_feature_size, 256), 
            GlobalModel())
        self.metalayer3 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 128), 
            NodeModel(edge_feature_size, node_feature_size, 128), 
            GlobalModel())

        self.x_linear = torch.nn.Linear(16, 1)

        self.edge_linear = torch.nn.Linear(19, 1)


    def forward(self, x, edge_index, edge_attr, u, batch):
        x           = x.to(self.device)
        edge_index  = edge_index.to(self.device)
        edge_attr   = edge_attr.to(self.device)
        # u           = u.to(self.device)
        batch       = batch.to(self.device)
        # print("x datatype: ", x.dtype)
        # print("edge_index datatype: ", edge_index.dtype)
        # print("edge_attr datatype: ", edge_attr.dtype)
        # print("batch datatype: ", batch.dtype)

        x, edge_attr, u = self.metalayer1(x, edge_index, edge_attr, u, batch)
        x, edge_attr, u = self.metalayer2(x, edge_index, edge_attr, u, batch)
        x, edge_attr, u = self.metalayer3(x, edge_index, edge_attr, u, batch)
        
        x = F.dropout(x, p=0.05, training=self.training)
        edge_attr = F.dropout(edge_attr, p=0.05, training=self.training)

        # row, col = edge_index
        # hybrid_feature = torch.cat([x[row], x[col], edge_attr], dim=1)
        # hybrid_feature = torch.nn.Linear(51, 256)(hybrid_feature)
        # hybrid_feature = torch.nn.BatchNorm1d(256)(hybrid_feature)
        # hybrid_feature = torch.nn.ReLU()(hybrid_feature)
        # hybrid_feature = torch.nn.Linear(256, 256)(hybrid_feature)
        # hybrid_feature = torch.nn.BatchNorm1d(256)(hybrid_feature)
        # hybrid_feature = torch.nn.ReLU()(hybrid_feature)

        # edge_label_pred = torch.nn.Linear(256, 1)(hybrid_feature)
        # edge_label_pred = F.sigmoid(edge_label_pred)

        x = self.x_linear(x)
        # print(x)
        y_pred = F.sigmoid(x)
        # softmax
        # y_pred = F.softmax(x, dim=1)
        edge_attr = self.edge_linear(edge_attr)
        edge_label_pred = F.sigmoid(edge_attr)
        # edge_label_pred = F.softmax(edge_attr, dim=1)

        return y_pred, edge_label_pred
    
def DrielsmaLoss(model_outputs, labels):
    _batch_size = model_outputs.size()[0]
    _loss_array = labels * torch.log(model_outputs) + (1 - labels) * torch.log(1 - model_outputs)
    _loss = -torch.sum(_loss_array) / _batch_size
    return _loss
