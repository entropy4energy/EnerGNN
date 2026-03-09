from . import utils
import tqdm
from torch_geometric.data import Data
import numpy as np
import matbench_discovery.data as matbench_data
import ase, io
import pandas as pd
import torch as tc
from .utils import Toolbelt
import fairchem as fc
def alexandria(load_files:list[str]=["000"], cutoff=6.0)->list[Data]:
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
        file_path = f"{utils.Toolbelt.get_tmp_path()}/Alexandria/alexandria_{i}.json"
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

def wbm(cutoff=6.0)->list[Data]:
    """
    Return the list contain the Data in WBM.
    - cutoff: the largest bond between atoms (A).
    """
    def init()->tuple[dict, list]: # dict here is energy where key is material_id, and value is energy.
        """
        Return the energy and list[(ase.atoms.Atoms, str)]
        """
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
        energy:dict = dict(zip(df_wbm_summary["material_id"], df_wbm_summary["uncorrected_energy"]))
        atoms:dict = wbm_relax_dataset_list
        return atoms, energy
    atoms, energy_dict = init()
    ans:list[Data] = []
    for ase_atom, material_id in tqdm.tqdm(atoms):
        energy = energy_dict[material_id]
        node_features, edge_index, edge_weight, cell = Toolbelt.ase_atoms_to_graph(ase_atom, cutoff=cutoff)
        ans.append(Data(
            x = node_features,
            edge_index = edge_index,
            edge_attr = edge_weight,
            matrix = tc.tensor(cell).float(),
            y = tc.tensor([energy]).float()
        ))
    return ans

def omat24()->list[Data]:
    
    return