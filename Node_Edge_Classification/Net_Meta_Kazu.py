
import torch
from torch_geometric.nn import MessagePassing, GCNConv, global_mean_pool, EdgeConv, MetaLayer
from torch_geometric.utils import scatter
from torch.nn import Linear, Parameter, functional as F

class EdgeModel(torch.nn.Module):
    def __init__(self, edge_feature_size, node_feature_size,layer_size, leakiness=0.1):
        super().__init__()
        in_size = edge_feature_size + 2*node_feature_size

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_size),

            torch.nn.Linear(in_size, layer_size),
            torch.nn.LeakyReLU(leakiness),
            torch.nn.BatchNorm1d(layer_size),
            
            torch.nn.Linear(layer_size,   layer_size),
            torch.nn.LeakyReLU(leakiness),
            torch.nn.BatchNorm1d(layer_size),
            
            torch.nn.Linear(layer_size,   edge_feature_size)
        )

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
        # batch = batch.unsqueeze(1)
        out = torch.cat([src, dst, edge_attr], dim=1)
        # out = torch.cat([src, dst, edge_attr, u[batch]], dim=1)
        out = self.edge_mlp(out)
        return out

class NodeModel(torch.nn.Module):
    def __init__(self, edge_feature_size, node_feature_size, layer_size, leakiness=0.1):
        super().__init__()
        _entry_size = node_feature_size + edge_feature_size
        _second_entry_size = node_feature_size + layer_size
        self.node_mlp_1 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(_entry_size),
            torch.nn.Linear(_entry_size, layer_size),
            torch.nn.LeakyReLU(leakiness),
            torch.nn.BatchNorm1d(layer_size),

            torch.nn.Linear(layer_size, layer_size),
            torch.nn.LeakyReLU(leakiness),
            torch.nn.BatchNorm1d(layer_size),
            
            torch.nn.Linear(layer_size, layer_size)
        )
        self.node_mlp_2 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(_second_entry_size),
            torch.nn.Linear(_second_entry_size, _second_entry_size),
            torch.nn.LeakyReLU(leakiness),
            torch.nn.BatchNorm1d(_second_entry_size),

            torch.nn.Linear(_second_entry_size, _second_entry_size),
            torch.nn.LeakyReLU(leakiness),
            torch.nn.BatchNorm1d(_second_entry_size),
            
            torch.nn.Linear(_second_entry_size, node_feature_size)
        )
    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        # print("node number: ", x.size(0))
        # print("edge number: ", edge_index.size(1))
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0))
        # out = torch.cat([x, batch, out], dim=1)
        out = torch.cat([x, out], dim=1)
        # out = torch.cat([x, out, u[batch]], dim=1)
        out = self.node_mlp_2(out)
        # out = F.dropout(out, training=self.training)
        return out
    
class NodeModel_Enhanced(torch.nn.Module):
    def __init__(self, edge_feature_size, node_feature_size, layer_size, leakiness=0.1):
        super().__init__()
        _entry_size = node_feature_size + edge_feature_size
        _second_entry_size = node_feature_size + layer_size + 7
        self.node_mlp_1 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(_entry_size),
            torch.nn.Linear(_entry_size, layer_size),
            torch.nn.LeakyReLU(leakiness),
            torch.nn.BatchNorm1d(layer_size),

            torch.nn.Linear(layer_size, layer_size),
            torch.nn.LeakyReLU(leakiness),
            torch.nn.BatchNorm1d(layer_size),
            
            torch.nn.Linear(layer_size, layer_size)
        )
        self.node_mlp_2 = torch.nn.Sequential(
            torch.nn.BatchNorm1d(_second_entry_size),
            torch.nn.Linear(_second_entry_size, _second_entry_size),
            torch.nn.LeakyReLU(leakiness),
            torch.nn.BatchNorm1d(_second_entry_size),

            torch.nn.Linear(_second_entry_size, _second_entry_size),
            torch.nn.LeakyReLU(leakiness),
            torch.nn.BatchNorm1d(_second_entry_size),

            torch.nn.Linear(_second_entry_size, _second_entry_size),
            torch.nn.LeakyReLU(leakiness),
            torch.nn.BatchNorm1d(_second_entry_size),
            
            torch.nn.Linear(_second_entry_size, node_feature_size)
        )
    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        # print("node number: ", x.size(0))
        # print("edge number: ", edge_index.size(1))
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0))
        # out = torch.cat([x, out], dim=1)
        out = torch.cat([x, out, u[batch]], dim=1)
        out = self.node_mlp_2(out)
        # out = F.dropout(out, training=self.training)
        return out

class GlobalModel(torch.nn.Module):
    def __init__(self, leakiness = 0.1):
        super().__init__()
        global_feature_size = 7
        entry_size = 23
        layer_node_size = 128
        self.global_mlp = torch.nn.Sequential(
            torch.nn.BatchNorm1d(entry_size),
            torch.nn.Linear(entry_size, layer_node_size),
            torch.nn.LeakyReLU(leakiness),
            torch.nn.BatchNorm1d(layer_node_size),

            torch.nn.Linear(layer_node_size, layer_node_size),
            torch.nn.LeakyReLU(leakiness),
            torch.nn.BatchNorm1d(layer_node_size),
            
            torch.nn.Linear(layer_node_size, global_feature_size)
        )

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
        # return self.global_mlp(out)
        return u

class SJN_Meta_2(torch.nn.Module):
    def __init__(self, dataset, device):
        super().__init__()
        self.device = device
        node_feature_size = 16
        edge_feature_size = 19
        global_leakiness = 0.1

        self.metalayer1 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            NodeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            GlobalModel(global_leakiness))
        self.metalayer2 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            NodeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            GlobalModel(global_leakiness))
        self.metalayer3 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            NodeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            GlobalModel(global_leakiness))
        self.metalayer4 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            NodeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            GlobalModel(global_leakiness))
        self.metalayer5 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            NodeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            GlobalModel(global_leakiness))
        self.metalayer6 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            NodeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            GlobalModel(global_leakiness))
        self.metalayer7 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            NodeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            GlobalModel(global_leakiness))
        self.metalayer8 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            NodeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            GlobalModel(global_leakiness))
        self.metalayer9 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            NodeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            GlobalModel(global_leakiness))
        self.metalayer10 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            NodeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            GlobalModel(global_leakiness))
        self.metalayer11 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            NodeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            GlobalModel(global_leakiness))
        self.metalayer12 = MetaLayer(
            EdgeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            NodeModel(edge_feature_size, node_feature_size, 64, global_leakiness), 
            GlobalModel(global_leakiness))

        self.x_linear = torch.nn.Linear(16, 1)

        self.edge_linear = torch.nn.Linear(19, 1)


    def forward(self, x, edge_index, edge_attr, u, batch):
        x           = x.to(self.device)
        edge_index  = edge_index.to(self.device)
        edge_attr   = edge_attr.to(self.device)
        # batch = None
        u = None
        # u           = u.to(self.device)
        batch       = batch.to(self.device)

        x, edge_attr, u = self.metalayer1(x, edge_index, edge_attr, u, batch)
        x, edge_attr, u = self.metalayer2(x, edge_index, edge_attr, u, batch)
        x, edge_attr, u = self.metalayer3(x, edge_index, edge_attr, u, batch)


        # for i in range(12):
        #     x, edge_attr, u = self.metalayer_array[i](x, edge_index, edge_attr, u, batch)


        x = self.x_linear(x)
        y_pred = torch.nn.Sigmoid()(x)
        # y_pred = torch.nn.Softmax(dim=1)(x)[1]

        edge_attr = self.edge_linear(edge_attr)
        edge_label_pred = torch.nn.Sigmoid()(edge_attr)
        # edge_label_pred = torch.nn.Softmax(dim=1)(edge_attr)[1]

        return y_pred, edge_label_pred
    
def DrielsmaLoss(model_outputs, labels):
    _batch_size = model_outputs.size()[0]
    _loss_array = labels * torch.log(model_outputs) + (1 - labels) * torch.log(1 - model_outputs)
    _loss = -torch.sum(_loss_array) / _batch_size
    return _loss

def extractGlobalFeatures(voxel_data, input_size, batch_size):
    u_generated = []
    for i in range(0, input_size):
        voxel_event = torch.cat(voxel_data[i]['voxels'], dim=0)
        _voxel_num = len(voxel_event)
        # _voxel_info_num = len(voxel_event[0])
        _voxel_x_min = torch.min(voxel_event[:, 0]).item()
        _voxel_x_max = torch.max(voxel_event[:, 0]).item()
        _voxel_y_min = torch.min(voxel_event[:, 1]).item()
        _voxel_y_max = torch.max(voxel_event[:, 1]).item()
        _voxel_z_min = torch.min(voxel_event[:, 2]).item()
        _voxel_z_max = torch.max(voxel_event[:, 2]).item()
        u_generated.append([_voxel_num, _voxel_x_min, _voxel_x_max, _voxel_y_min, _voxel_y_max, _voxel_z_min, _voxel_z_max])
    u_generated_tensor = torch.tensor(u_generated, dtype=torch.float)
    u_generated_tensor_sliced_by_batch = torch.split(u_generated_tensor, batch_size)
    # print("sliced size: ", len(u_generated_tensor_sliced_by_batch))
    # for batch_info in u_generated_tensor_sliced_by_batch:
    #     print(len(batch_info))
    # print an example of event #5 in batch #4
    return u_generated_tensor_sliced_by_batch