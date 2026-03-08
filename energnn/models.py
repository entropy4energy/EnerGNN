import torch.nn as nn
import torch_geometric.nn as tgnn
import torch as tc
from torch_geometric.data import Data
import torch_geometric as tcg

class LeakySiLU(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.silu = nn.SiLU()
    def forward(self, x):
        return self.silu(x) + self.alpha*x
class SoftAbs(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return (x**2+self.alpha)**0.5-self.alpha**0.5
class EnerGMP(tcg.nn.conv.MessagePassing):
    def __init__(self, nn:nn.Module, aggr:str="add"):
        """
        - **nn**: the module that input have shape 2*input_channel, output have shape output_channel.
        """
        super().__init__(aggr=aggr)
        self.nn = nn
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        """
        - x_i: target node
        - x_j: source node
        """
        x = tc.cat((x_i, x_j), dim=1)
        x = self.nn(x)
        return x
    def update(self, aggr_out):
        return aggr_out
class EnerGDev(nn.Module):
    """This model only predic the total energy of a structure (graph) in forward."""
    def __init__(self):
        super().__init__()
        conv1nn = nn.Sequential(nn.Linear(8, 256), nn.BatchNorm1d(256), LeakySiLU(0.05), nn.Dropout1d(0.1), nn.Linear(256, 32), nn.BatchNorm1d(32), LeakySiLU(0.05)) 
        conv2nn = nn.Sequential(nn.Linear(64, 512), nn.BatchNorm1d(512), LeakySiLU(0.05), nn.Dropout1d(0.1), nn.Linear(512, 128), nn.BatchNorm1d(128), LeakySiLU(0.05)) 
        conv3nn = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), LeakySiLU(0.05), nn.Dropout1d(0.1), nn.Linear(128, 64), nn.BatchNorm1d(64), LeakySiLU(0.05))
        conv4nn = nn.Sequential(nn.Linear(128, 256), nn.BatchNorm1d(256), LeakySiLU(0.05), nn.Dropout1d(0.1), nn.Linear(256, 128), nn.BatchNorm1d(128), LeakySiLU(0.05))
        conv5nn = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256), LeakySiLU(0.05), nn.Dropout1d(0.1), nn.Linear(256, 128), nn.BatchNorm1d(128), LeakySiLU(0.05))
        #conv1nn = nn.Sequential(nn.Linear(8, 32))
        #conv2nn = nn.Sequential(nn.Linear(64, 128))
        #conv3nn = nn.Sequential(nn.Linear(256, 64))
        #conv4nn = nn.Sequential(nn.Linear(128, 128))
        #conv5nn = nn.Sequential(nn.Linear(256, 128))
        self.conv1 = EnerGMP(nn=conv1nn, aggr="mean")
        self.conv2 = EnerGMP(nn=conv2nn, aggr="mean")
        self.conv3 = EnerGMP(nn=conv3nn, aggr="mean")
        self.conv4 = EnerGMP(nn=conv4nn, aggr="mean")
        self.conv5 = EnerGMP(nn=conv5nn, aggr="mean")
        self.interlayer1 = nn.Sequential(nn.Linear(32, 512), LeakySiLU(0.05), nn.Dropout1d(0.1), nn.Linear(512, 32), LeakySiLU(0.05), nn.BatchNorm1d(32))
        self.interlayer2 = nn.Sequential(nn.Linear(128, 512), LeakySiLU(0.05), nn.Dropout1d(0.1), nn.Linear(512, 128), LeakySiLU(0.05), nn.BatchNorm1d(128))
        self.interlayer3 = nn.Sequential(nn.Linear(64, 512), LeakySiLU(0.05), nn.Dropout1d(0.1), nn.Linear(512, 64), LeakySiLU(0.05), nn.BatchNorm1d(64))
        self.interlayer4 = nn.Sequential(nn.Linear(128, 512), LeakySiLU(0.05), nn.Dropout1d(0.1), nn.Linear(512, 128), LeakySiLU(0.05), nn.BatchNorm1d(128))
        self.fc1 = tgnn.Linear(128, 128)
        self.fc2 = tgnn.Linear(128, 1)
        self.silu = LeakySiLU(0.1)
        self.softabs = SoftAbs(1.0)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
    def node_features_to_edge_weight(self, node_features, edge_index):
        ans = tc.zeros(edge_index.shape[1], 3).to(node_features.device)
        ans = (node_features[edge_index[1, :]] - node_features[edge_index[0, :]])[:, 1:]
        return ans
    def forward(self, data:Data):
        # we don't using edge_weight here, we calculate edge_weight from node_features.
        model_device = next(self.parameters()).device
        batch_index = data.batch.to(model_device)
        x = data.x.to(model_device)
        matrix = data.matrix.to(model_device)
        # Convert the fractional coordination to absolute.
        def cvt_frac_to_abs(x, matrix, batch_index):
            matrix_group = matrix.float().reshape(-1, 3, 3)
            abs_position = (x[:, 1:].reshape(-1, 1, 3)@matrix_group[batch_index]).reshape(-1, 3)
            ans = tc.cat((x[:, 0].reshape(-1, 1), abs_position), dim=1)
            return  ans
        node_features = cvt_frac_to_abs(x, matrix, batch_index)
        #edge_weight_cal = self.node_features_to_edge_weight(node_features, data.edge_index).to(model_device)
        edge_index = data.edge_index.to(model_device)
        node_features = self.conv1(node_features, edge_index)
        node_features = self.silu(node_features)
        node_features = self.interlayer1(tmp:=node_features) + tmp # Resnet
        node_features = self.conv2(node_features, edge_index)
        node_features = self.silu(node_features)
        node_features = self.interlayer2(tmp:=node_features) + tmp # Resnet
        node_features = self.conv3(node_features, edge_index)
        node_features = self.silu(node_features)
        node_features = self.interlayer3(tmp:=node_features) + tmp # Resnet
        node_features = self.conv4(node_features, edge_index)
        node_features = self.silu(node_features)
        node_features = self.interlayer4(tmp:=node_features) + tmp
        node_features = self.conv5(node_features, edge_index)
        node_features = self.silu(node_features)

        #x = [tc.zeros((node_features.shape[1],), device=next(self.parameters()).device) for _ in range(batch_index.max().item()+1)]
        #for i in range(node_features.shape[0]):
        #    x[batch_index[i]] += node_features[i]
        #x = tc.stack(x, dim=0)
        x = tgnn.global_add_pool(node_features, batch_index)
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        x = x - 100
        return x