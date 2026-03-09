import torch as tc
import torch_geometric.loader as tcg_loader
import tqdm
from torch.utils.data import Dataset
import os, pickle, json
import ase

# region: toolbelt
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
    def ase_atoms_to_graph(atoms:ase.atoms.Atoms, cutoff:float=6.0) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor]:
        """
        This function will turn the ase Atoms into Data
        - **atoms**: the ase:atoms.Atoms
        - **cutoff**: the maximum length of bond.
        - return
            - node_features: Tensor[[<atomic number, fractional_x, fractional_y, fractional_z>], ...]
            - edge_index:
            - edge_weight:
            - cell: """
        # return Data(node_feature=..., edge_index=..., edge_weight=..., cell_vectors=) # this is the input of model.
        atomic_numbers = tc.tensor(atoms.get_atomic_numbers(), dtype=tc.float)
        atom_fractional_position = tc.tensor(atoms.get_scaled_positions(), dtype=tc.float)
        node_features = tc.cat([
            atomic_numbers.reshape(-1, 1),
            atom_fractional_position],
            dim=1)
        # i is the index of central atom, and j is the index of nearby atoms, d is the distance between each other.
        i, j, d = ase.neighbor_list('ijd', atoms, cutoff, self_interaction=True)
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
        this_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(this_path, ".."))
    @staticmethod
    def get_tmp_path()->str:
        """Return the path of tmp files located."""
        return Toolbelt.get_root_path()+"/tmp"
    @staticmethod
    def download_environment_files():
        """
        This will download all files that needed. All files will be installed in project_root/tmp
        > This function only need to be excuted at project setup.
        """
        # Download alexandria 
        # Make sure target directory exist.
        for file_num in ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009',
                         '010', '011', '012', '013', '014', '015', '016', '017', '018', '019',
                         '020', '021', '022', '023', '024', '025', '026', '027', '028', '029',
                         '030', '031', '032', '033', '034', '035', '036', '037', '038', '039',
                         '040', '041', '042', '043', '044', '045', '046', '047', '048', '049']:
            os.system(f"wget -P {Toolbelt.get_tmp_path()}/Alexandria/ https://alexandria.icams.rub.de/data/pbe/2024.12.15/alexandria_{file_num}.json.bz2")
        # unzip the .bz2 files
        print("Extract Alexandria dataset...")
        os.system(f"bzip2 -d {Toolbelt.get_tmp_path()}/Alexandria/*.bz2")
        # Download the OMAT24 dataset.
        download_url_train = [
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000-subsampled.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500-subsampled.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300-subsampled.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-npt.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-nvt.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-relax.tar.gz'
            # TODO
        ]
        download_url_valid = [
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000-subsampled.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-500.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-500-subsampled.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-300.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-300-subsampled.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-1000-npt.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-1000-nvt.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-3000-npt.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-3000-nvt.tar.gz',
            'https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-relax.tar.gz',
        ]
        for train_url in download_url_train:
            os.system(f"wget -P {Toolbelt.get_tmp_path()}/omat24/train/ {train_url}")
        for vaild_url in download_url_valid:
            os.system(f"wget -P {Toolbelt.get_tmp_path()}/omat24/vaild/ {vaild_url}")
        
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
        Toolbelt.pk_dump(obj, f"{Toolbelt.get_tmp_path()}/aaa.dump", absolute=True)
        ```
        """
        abs_path = path if absolute else f"{Toolbelt.get_tmp_path()}{path}"
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
        Toolbelt.pk_load(f"{Toolbelt.get_tmp_path()}/aaa.dump", absolute=True)
        ```
        """
        abs_path = path if absolute else f"{Toolbelt.get_tmp_path()}{path}"
        with open(abs_path, "rb") as f:
            return pickle.load(f)
# endregion

#region: trainer and tester
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
    model.eval()
    with tc.no_grad(): # Test don't need the grad
        model = model.to(device)
        dataloader = tcg_loader.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        for i in (bar:=tqdm.tqdm(dataloader)):
            y = i.y.to(device)
            pred = model(i)
            loss = loss_fn(y, pred)
            bar.set_description_str(f"loss_fn_output={loss}")            
    return
#endrigion