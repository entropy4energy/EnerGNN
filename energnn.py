# Matbench all in one
# note: this file MUST located at the root of project.
# region: load and init
import torch as tc
import torch.nn as nn
import torch_geometric.nn as tgnn
import torch_geometric as tcg
import tqdm, os, json, io
import numpy as np
from ase.data import atomic_numbers # this is a dict
import ase
from ase.neighborlist import neighbor_list
from pymatgen.core.lattice import Lattice
import torch_geometric as tcg
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.loader as tcg_loader
from ultraimport.ultraimport import ultraimport
import matbench_discovery.data as matbench_data
import wandb
import pickle
import pandas as pd
from functools import reduce
import itertools
# Configuration here.
config = {
    "seed":0,
    "runing_device":"cuda",

}
# endregion

# region: Toolbelt
# These functions are very useful.
class Toolbelt:
    @staticmethod
    def read_json(path:str) -> dict:
        with open(path, "r") as f:
            ans = json.load(f)
        return ans
    @staticmethod
    def cleanup():
        "This function will clean up all temp file in project"
        os.system(f"rm -r {Toolbelt.get_root_path()}/tmp")
        return
    @staticmethod
    def ase_atoms_to_graph(atoms:ase.atoms.Atoms, cutoff:float=30.0) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor]:
        'This function will turn the ase Atoms into Data'
        # return Data(node_feature=..., edge_index=..., edge_weight=..., cell_vectors=) # this is the input of model.
        atomic_numbers = tc.tensor(atoms.get_atomic_numbers(), dtype=tc.float)
        atom_fractional_position = tc.tensor(atoms.get_scaled_positions(), dtype=tc.float)
        node_features = tc.cat([
            atomic_numbers.reshape(-1, 1),
            atom_fractional_position],
            dim=1)
        # i is the index of central atom, and j is the index of nearby atoms, d is the distance between each other.
        i, j, d = neighbor_list('ijd', atoms, cutoff, self_interaction=True)
        edge_index = tc.tensor([i, j], dtype=tc.long)
        edge_weight = tc.tensor(d, dtype=tc.float)
        #print(f"node_features shape:{node_features.shape}, {type(node_features)}")
        #print(f"edge_index shape:{edge_index.shape}, {edge_index}")
        #print(f"edge_weight shape:{edge_weight.shape}, {edge_weight}")
        return (
            node_features,
            edge_index,
            edge_weight,
            atoms.get_cell().tolist(),
        )
    @staticmethod
    def get_root_path()->str:
        """Return something like '/usr/aylwin/matbench' """
        return os.path.dirname(os.path.abspath(__file__))
    @staticmethod
    def download_environment_files():
        """
        This will download all files that needed. All files will be installed in project_root/tmp
        > This function only need to be excuted at first time.
        """
        # Download alexandria 
        # Make sure target directory exist.
        for file_num in ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009',
                         '010', '011', '012', '013', '014', '015', '016', '017', '018', '019',
                         '020', '021', '022', '023', '024', '025', '026', '027', '028', '029',
                         '030', '031', '032', '033', '034', '035', '036', '037', '038', '039',
                         '040', '041', '042', '043', '044', '045', '046', '047', '048', '049']:
            os.system(f"wget -P {Toolbelt.get_root_path()}/tmp/Alexandria/ https://alexandria.icams.rub.de/data/pbe/2024.12.15/alexandria_{file_num}.json.bz2")
        # unzip the .bz2 files
        print("Extract Alexandria dataset...")
        os.system(f"bzip2 -d {Toolbelt.get_root_path()}/Alexandria/*.bz2")
    @staticmethod
    def pk_dump(obj, path:str, absolute:bool=False):
        """
        Dumplicate python object to path by using pickle.dump.
        - obj: The python object wants to dump.
        - path: file name you want to save.
        - absolute: if true, path should be a absolute path.
        ```
        # These two lines are the same.
        Toolbelt.pk_dump(obj, "/aaa.dump")
        Toolbelt.pk_dump(obj, f"{Toolbelt.get_root_path()}/aaa.dump", absolute=True)
        ```
        """
        abs_path = path if absolute else f"{Toolbelt.get_root_path()}/tmp{path}"
        with open(abs_path, "wb") as f:
            pickle.dump(obj, f)
    @staticmethod
    def pk_load(path:str, absolute:bool=False):
        """
        Load the python object that was dumped.
        - path: file name you want to save.
        - absolute: if true, path should be a absolute path.
        ```
        Toolbelt.pk_load("/aaa.dump")
        Toolbelt.pk_load(f"{Toolbelt.get_root_path()}/aaa.dump", absolute=True)
        ```
        """
        abs_path = path if absolute else f"{Toolbelt.get_root_path()}/tmp{path}"
        with open(abs_path, "rb") as f:
            return pickle.load(f)
# endregion

# region: Datasets.
class DatasetAlexandria(Dataset):
    def __init__(self, cutoff:float=5.0, load_files=['000', '001', '002']):
        self.energy = []
        self.e_above_hull = []
        self.graph = []
        self.cutoff = cutoff
        self.matrix = []
        print("Loading Alexandria dataset...")
        for i in load_files:
            jdict = Toolbelt.read_json(f"{os.path.dirname(__file__)}/tmp/Alexandria/alexandria_{i}.json")
            for each_structure in tqdm.tqdm(jdict['entries'][:]):  # HERE: you can set to read the part of the data
                energy = each_structure['energy']
                e_above_hull = each_structure['data']['e_above_hull']
                atoms = []
                for each_atom in each_structure['structure']['sites']:
                    atoms.append({
                        'element': each_atom['species'][0]['element'],
                        'xyz': each_atom['xyz'],
                        'abc': each_atom['abc']
                    })
                matrix = each_structure['structure']['lattice']['matrix']
                self.graph.append(self.atoms_to_graph(atoms, matrix))
                self.matrix.append(matrix)
                self.energy.append(energy)
                self.e_above_hull.append(e_above_hull)
        return
    def atoms_to_graph(self, atoms:list, matrix:list[list]):
        def pbc_distance(a, b, matrix):
            lat = Lattice(matrix)
            dist, _ = lat.get_distance_and_image(a, b)
            return dist
        num_atoms = len(atoms)
        atoms_list = []
        bonds_list = []
        # build atoms_list
        for atom in atoms:
            atoms_list.append([
                atomic_numbers[atom['element']],
                atom['xyz'],
            ])
        # build bonds_list
        for y in range(num_atoms):
            for x in range(num_atoms):
                dist = pbc_distance(atoms[y]['abc'], atoms[x]['abc'], matrix) # WARN 'abc' or 'xyz'?
                if dist >= 0:
                    bonds_list.append([y, x, dist])
        # Here we have atoms_list and bonds_list.
        # Process the cutoff
        # Set all distance < cutoff to 0
        nodes_features = []
        edges_index = [[], []]
        edges_weight = []
        for atom in atoms_list:
            nodes_features.append([atom[0], atom[1][0], atom[1][1], atom[1][2]])
        for b in bonds_list:
            if b[2]<self.cutoff:
                edges_index[0].append(b[0])
                edges_index[1].append(b[1])
                edges_weight.append(b[2])
        return [nodes_features, edges_index, edges_weight]
    def __len__(self) -> int:
        return self.energy.__len__()
    def __getitem__(self, x:int|slice):
        ans = Data(
            node_features = tc.tensor(self.graph[x][0]).float(),
            edge_index = tc.tensor(self.graph[x][1]),
            edge_weight = tc.tensor(self.graph[x][2]).float(),
            y = tc.tensor([[self.energy[x], self.e_above_hull[x]]]).float()
        )
        return ans
    def dump(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return

class DatasetWBM(Dataset): 
    def __init__(self, cutoff=5.0):
        print("Init the Dataset WBM.")
        from ase.io import read as ase_read # This function read .extxyz file to ase.atoms.Atoms
        import zipfile # because the ase dataset is in zip type.
        DataFiles = matbench_data.DataFiles
        df_wbm_summary:pd.DataFrame = pd.read_csv(DataFiles.wbm_summary.path) # Load the wbm summary dataframe
        # Load the graphs: wbm_relax_dataset_list
        wbm_relax_dataset_list:list[tuple[ase.atoms.Atoms, str]] = []
        with zipfile.ZipFile(DataFiles.wbm_relaxed_atoms.path, 'r') as z:
            for file_name in (bar:=tqdm.tqdm(z.namelist())):
                #bar.set_description(f"Init the WBM data")
                with z.open(file_name) as f:
                    text_data = io.TextIOWrapper(f)
                    wbm_relaxed_atoms = ase_read(text_data, index=":", format='extxyz')
                    wbm_relax_dataset_list.append((wbm_relaxed_atoms[0], file_name[:-7])) # [(<ase.atoms.Atoms>, <mateiral_id>), (...), ...]
        #self.df_wbm_summary = df_wbm_summary
        self.wbm_relax_dataset_list = wbm_relax_dataset_list
        self.cutoff = cutoff
        #df_wbm_summary_indexed = self.df_wbm_summary.set_index('material_id').sort_index()
        self.e_above_hull_wbm:dict = dict(zip(df_wbm_summary['material_id'], df_wbm_summary['e_above_hull_wbm']))
        self.e_form_per_atom_wbm:dict = dict(zip(df_wbm_summary['material_id'], df_wbm_summary['e_form_per_atom_wbm']))
        self.uncorrected_enery:dict = dict(zip(df_wbm_summary['material_id'], df_wbm_summary['uncorrected_energy']))
        return
    def __len__(self) -> int:
        return self.wbm_relax_dataset_list.__len__()
    def __getitem__(self, index:int) -> Data: # Not support Slice currently.
        if type(index)==slice: raise Exception("Not support slice currently")
        node_features, edge_index, edge_weight, cell = Toolbelt.ase_atoms_to_graph(self.wbm_relax_dataset_list[index][0], cutoff=self.cutoff)
        material_id:str = self.wbm_relax_dataset_list[index][1]
        e_above_hull_wbm = self.e_above_hull_wbm[material_id]
        e_form_per_atom_wbm = self.e_form_per_atom_wbm[material_id]
        uncorrected_energy = self.uncorrected_enery[material_id]
        return Data(
            x = node_features, 
            edge_index = edge_index,
            edge_attr = edge_weight,
            matrix = tc.tensor(self.wbm_relax_dataset_list[index][0].get_cell().tolist()).float(),
            # TODO: Add stress and force to WBM dataset.
            #y = tc.tensor([[uncorrected_energy, e_above_hull_wbm]]).float()
            y = tc.tensor([[uncorrected_energy]]).float()
        )

def datasetAlexandriaNeo(load_files:list[str]=["000"], cutoff=6.0)->list[Data]:
    """
    This function will return a list as dataset.
    - **load_files**: A list of file number from "000"-"049", the number in list will be load in one dataset.
    - **cutoff**: The maximum length of bond.
    - **return**: list[Data(x, edge_index, edge_attr, matrix, abc, y)]
        - x: node features, shape [n, 4], where 4 for (atomic_number, fractional_x, fractional_y, fractional_z)
        - edge_index: shape [2, m], where m is the number of edges.
        - edge_attr: I think this may not been used in future because we already have node_features and matrix. But I still keep it.
        - matrix: the lattice matrix of the structure, shape [3, 3]
    """
    print("Loading Alexandria dataset ==== ")
    def structure_to_graph_data(structure, cutoff=5.0):
        # 1. Get all neighbors, (PBC considered).
        # nbs return list of list: (neighbor_site, distance, index, image)
        import torch
        all_neighbors = structure.get_all_neighbors(r=cutoff)
        edge_index = []
        edge_weight = []
        for i, neighbors in enumerate(all_neighbors):
            for nb in neighbors:
                edge_index.append([i, nb.index]) # Source node i -> target node nb.index
                edge_weight.append(nb.coords - structure[i].coords)
                
        # Convert to Tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        # 2. Generate Node Features.
        # You can add your node features here
        node_features = torch.tensor([np.insert(site.frac_coords, 0, site.specie.Z) for site in structure], dtype=torch.float).view(-1, 4) # [<atomic_number>, <fractional x>, <fractional y>, <fractional z>]
        # Generate the matrix
        matrix = torch.tensor(structure.lattice.matrix)
        xyz = torch.tensor([atom.coords for atom in structure]).view(-1, 3)
        return node_features, edge_index, edge_weight, matrix, xyz
    from monty.serialization import loadfn
    ans = []
    for i in load_files:
        file_path = f"{Toolbelt.get_root_path()}/tmp/Alexandria/alexandria_{i}.json"
        data_dict = loadfn(file_path)
        for entry in tqdm.tqdm(data_dict['entries']): # in each entry
            node_features, edge_index, edge_weight, matrix, xyz = structure_to_graph_data(entry.structure, cutoff=cutoff)
            stress = entry.data['stress']
            ans.append(Data(
                x = node_features, # [<atomic_number>, <fractional_x>, <fractional_y>, <fractional_z>]
                edge_index = edge_index,
                edge_attr = edge_weight,
                matrix = matrix,
                stress = stress, # The stress of the structure.
                xyz = xyz,
                # TODO: Add force.
                y = entry.energy, # The energy without correction.
            ))
    return ans


# region: Modules
# This model is depreciated.
class LeakySiLU(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.silu = nn.SiLU()
    def forward(self, x):
        return self.silu(x) + self.alpha*x
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
class EnergDev(nn.Module):
    """This model only predic the total energy of a structure (graph) in forward."""
    def __init__(self):
        super().__init__()
        conv1nn = nn.Sequential(nn.Linear(8, 64), LeakySiLU(0.05), nn.Linear(64, 32), LeakySiLU(0.05)) 
        conv2nn = nn.Sequential(nn.Linear(64, 64), LeakySiLU(0.05), nn.Linear(64, 128), LeakySiLU(0.05)) 
        conv3nn = nn.Sequential(nn.Linear(256, 64), LeakySiLU(0.05), nn.Linear(64, 64), LeakySiLU(0.05))
        conv4nn = nn.Sequential(nn.Linear(128, 64), LeakySiLU(0.05), nn.Linear(64, 128), LeakySiLU(0.05))
        self.conv1 = EnerGMP(nn=conv1nn, aggr="add")
        self.conv2 = EnerGMP(nn=conv2nn, aggr="add")
        self.conv3 = EnerGMP(nn=conv3nn, aggr="add")
        self.conv4 = EnerGMP(nn=conv4nn, aggr="add")
        self.interlayer1 = nn.Sequential(nn.Linear(32, 128), LeakySiLU(0.05), nn.Linear(128, 32), LeakySiLU(0.05))
        self.interlayer2 = nn.Sequential(nn.Linear(128, 128), LeakySiLU(0.05), nn.Linear(128, 128), LeakySiLU(0.05))
        self.interlayer3 = nn.Sequential(nn.Linear(64, 128), LeakySiLU(0.05), nn.Linear(128, 64), LeakySiLU(0.05))
        self.fc1 = tgnn.Linear(128, 1)
        self.silu = LeakySiLU(0.1)
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
        x = [tc.zeros((node_features.shape[1],), device=next(self.parameters()).device) for _ in range(batch_index.max().item()+1)]
        for i in range(node_features.shape[0]):
            x[batch_index[i]] += node_features[i]
        x = tc.stack(x, dim=0)
        x = self.fc1(x)
        x = -self.silu(x) # minus because the energies are negative. and -5 is to push center away from zero.
        return x*0.1

class EnerG(nn.Module):
    """This model only predic the total energy of a structure (graph) in forward."""
    def __init__(self):
        super().__init__()
        conv1nn = nn.Sequential(nn.Linear(3, 64), LeakySiLU(0.05), nn.Linear(64, 4*8), LeakySiLU(0.05)) 
        conv2nn = nn.Sequential(nn.Linear(3, 64), LeakySiLU(0.05), nn.Linear(64, 8*64), LeakySiLU(0.05)) 
        conv3nn = nn.Sequential(nn.Linear(3, 64), LeakySiLU(0.05), nn.Linear(64, 64*128), LeakySiLU(0.05))
        conv4nn = nn.Sequential(nn.Linear(3, 64), LeakySiLU(0.05), nn.Linear(64, 128*128), LeakySiLU(0.05))
        self.conv1 = tgnn.NNConv(4, 8, nn=conv1nn, aggr="add")
        self.conv2 = tgnn.NNConv(8, 64, nn=conv2nn, aggr="add")
        self.conv3 = tgnn.NNConv(64, 128, nn=conv3nn, aggr="add")
        self.conv4 = tgnn.NNConv(128, 128, nn=conv4nn, aggr="add")
        self.interlayer1 = nn.Sequential(nn.Linear(8, 128), LeakySiLU(0.05), nn.Linear(128, 8), LeakySiLU(0.05))
        self.interlayer2 = nn.Sequential(nn.Linear(64, 128), LeakySiLU(0.05), nn.Linear(128, 64), LeakySiLU(0.05))
        self.interlayer3 = nn.Sequential(nn.Linear(128, 128), LeakySiLU(0.05), nn.Linear(128, 128), LeakySiLU(0.05))
        self.fc1 = tgnn.Linear(128, 1)
        self.silu = LeakySiLU(0.1)
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
        edge_weight_cal = self.node_features_to_edge_weight(node_features, data.edge_index).to(model_device)
        edge_index = data.edge_index.to(model_device)
        node_features = self.conv1(node_features, edge_index, edge_weight_cal)
        node_features = self.silu(node_features)
        node_features = self.interlayer1(tmp:=node_features) + tmp # Resnet
        node_features = self.conv2(node_features, edge_index, edge_weight_cal)
        node_features = self.silu(node_features)
        node_features = self.interlayer2(tmp:=node_features) + tmp # Resnet
        node_features = self.conv3(node_features, edge_index, edge_weight_cal)
        node_features = self.silu(node_features)
        node_features = self.interlayer3(tmp:=node_features) + tmp # Resnet
        x = [tc.zeros((node_features.shape[1],), device=next(self.parameters()).device) for _ in range(batch_index.max().item()+1)]
        for i in range(node_features.shape[0]):
            x[batch_index[i]] += node_features[i]
        x = tc.stack(x, dim=0)
        x = self.fc1(x)
        x = -self.silu(x) # minus because the energies are negative. and -5 is to push center away from zero.
        return x
    def force(self, input_data:Data):
        """This function will evaluate the force of struct"""
        model_device = next(self.parameters()).device
        edge_index = edge_index.to(model_device)
        x = input_data.x
        edge_index = input_data.edge_index
        edge_weight = input_data.edge_attr
        batch_index = input_data.batch
        # Convert fractional coordination to absolute.
        def cvt_frac_to_abs(x, matrix, batch_index):
            matrix_group = matrix.float().reshape(-1, 3, 3)
            abs_position = (x[:, 1:].reshape(-1, 1, 3)@matrix_group[batch_index]).reshape(-1, 3)
            ans = tc.cat((x[:, 0].reshape(-1, 1), abs_position), dim=1)
            return  ans
        node_features = cvt_frac_to_abs(x, input_data.matrix, batch_index).detach().clone().to(model_device).requires_grad_(True)
        edge_weight = edge_weight.detach().clone().to(model_device).requires_grad_(True)
        batch_index = batch_index.to(model_device)
        node_features.grad.zero_() if node_features.grad!=None else "Do nothing"# clean the grad
        energy = self.forward(Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight, batch_index=batch_index))
        energy.sum().backward() # backward.
        force_ans = node_features.grad[:, 1:]
        return force_ans
    def stress(self, input_data:Data):
        # TODO
        return 
    def relax(self, node_features:tc.Tensor, edge_index:tc.Tensor, edge_weight:tc.Tensor, batch_index:tc.Tensor):
        """This function will relax a structure"""
        # TODO
        return
    
# endregion
# region: Trainer and Tester
def tcg_trainer(model:tc.nn.Module, dataset:Dataset, optimizer, 
                loss_fn=tc.nn.L1Loss(),
                epoch:int=100, 
                batch_size=32,
                num_workers=1,
                device='cpu') -> tuple[tc.nn.Module, dict]:
    """This trainer apply tcg_loader.dataloader to dataset."""
    print('trainer start...')
    log = {'avr_loss':[]}
    model = model.to(device)
    dataloader = tcg_loader.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # Train loop
    for each_epoch in range(epoch):
        sum_loss, n = 0.0, 0
        print(f"Epoch {each_epoch} ===========>")
        for i in (bar:=tqdm.tqdm(dataloader)):
            y = i.y.to(device)
            pred = model(i)
            loss = loss_fn(y, pred)
            # refresh parameters.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += float(loss)
            n+=1
            bar.set_postfix_str(f"avr_loss:{sum_loss/n} loss:{float(loss)}")
        log['avr_loss'] += [sum_loss/n]
    return model, log
def tcg_tester(model:tc.nn.Module, dataset:Dataset, loss_fn,# loss_fn here should be a closure (return the real loss_fn).
               device='cpu', 
               batch_size=64,
               num_workers=1):
    """This tester apply the tcg_loader.dataloader to dataset"""
    loss_fn = loss_fn()
    with tc.no_grad(): # Test don't need the grad
        model = model.to(device)
        dataloader = tcg_loader.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        for i in (bar:=tqdm.tqdm(dataloader)):
            y = i.y.to(device)
            pred = model(i)
            loss = loss_fn(y, pred)
            bar.set_description_str(f"loss_fn_output={loss}")            
    return
# endregion


# region: playground

def try_to_load_data_from_matbench_discovery():
    "I want to know how to load dataset from matbench_discovery"
    DataFiles = matbench_data.DataFiles
    df_wbm_summary = pd.read_csv(DataFiles.wbm_summary.path)
    #print(type(df_wbm_summary))
    #print(f"wbm_summary shape:{df_wbm_summary.shape}")
    #---
    # Load the wbm dataset.
    from ase.io import read as ase_read
    import zipfile
    wbm_relax_dataset_list:list[tuple[ase.atoms.Atoms, str]] = []
    with zipfile.ZipFile(DataFiles.wbm_relaxed_atoms.path, 'r') as z:
        for file_name in z.namelist():
            with z.open(file_name) as f:
                #print(file_name)
                # df_wbm_summary.material_id
                #breakpoint()
                text_data = io.TextIOWrapper(f)
                wbm_relaxed_atoms = ase_read(text_data, index=":", format='extxyz')
                wbm_relax_dataset_list.append((wbm_relaxed_atoms[0], file_name[0:-7])) # [(<ase.atoms.Atoms>, <mateiral_id>), (...), ...]
        #print("wbm_relax_atoms length is:", len(wbm_relax_dataset_list))
        #print(wbm_relax_dataset_list[0])
        #print(type(wbm_relax_dataset_list[0]))
    def ase_atoms_to_graph(atoms:ase.atoms.Atoms, cutoff:float=30.0) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor]:
        'This function will turn the ase Atoms into Data'
        # return Data(node_feature=..., edge_index=..., edge_weight=...) # this is the input of model.
        atomic_numbers = tc.tensor(atoms.get_atomic_numbers(), dtype=tc.float)
        atom_position = tc.tensor(atoms.get_positions(), dtype=tc.float)
        node_features = tc.cat([
            atomic_numbers.reshape(-1, 1),
            atom_position],
            dim=1)
        # i is the index of central atom, and j is the index of nearby atoms, d is the distance between each other.
        i, j, d = neighbor_list('ijd', atoms, cutoff)
        edge_index = tc.tensor([i, j], dtype=tc.long)
        edge_weight = tc.tensor(d, dtype=tc.float)
        print(f"node_features shape:{node_features.shape}, {type(node_features)}")
        print(f"edge_index shape:{edge_index.shape}, {edge_index}")
        print(f"edge_weight shape:{edge_weight.shape}, {edge_weight}")
        return (
            node_features,
            edge_index,
            edge_weight
        )
    data = ase_atoms_to_graph(wbm_relax_dataset_list[0][0], cutoff=5.0)
    for (ase_atoms, material_id) in tqdm.tqdm(wbm_relax_dataset_list[:10000]):
        # get e_above_hull_wbm and e_form_per_atom_wbm
        df_wbm_summary_indexed = df_wbm_summary.set_index('material_id')
        e_above_hull_wbm = df_wbm_summary_indexed.loc[material_id, 'e_above_hull_wbm']
        e_form_per_atom_wbm = df_wbm_summary_indexed.loc[material_id, 'e_form_per_atom_wbm']
    return

if __name__=='__main__':
    # Playground
    #load_train_test()
    try_to_load_data_from_matbench_discovery()
    pass
# endregion
