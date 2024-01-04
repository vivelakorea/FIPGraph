
import glob
import pickle
import re
import networkx as nx
from matplotlib.pyplot import sca
import numpy as np
import torch
import torch_geometric

MAIN_DIR = "C:\\Users\\Gyu-Jang Sim\\Documents\\FIPGraph"

torchdata_dir = f'{MAIN_DIR}\\preprocessing\\graph_torchdata'
graph_dir = f'{MAIN_DIR}\\preprocessing\\graph_sves'
fip_dir = f'{MAIN_DIR}\\preprocessing\\FIPtables'
scaler_dir = f'{MAIN_DIR}\\preprocessing\\scalers'
textures = ['160']
train_fraction = 1.0
seed = 42

def __natural_sort(l):
    """Sort list in Natural order.

    Parameters
    ----------
    l : {list}
        List of strings.

    Returns
    -------
    l : {ndarray}
        Sorted list of strings.
    """
    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


datalist = [] # elements: {'x': [.., .., .., ...], 'edge_index':[[...],[...]], 'e': [.., .., .., ...], 'fip': [..., ..., ...]}

for texture in textures:
    # feature
    graph_files = glob.glob(f'{graph_dir}\\{texture}\\*')
    graph_files = __natural_sort(graph_files)

    # fip
    fip_files = glob.glob(f'{fip_dir}\\{texture}\\*')
    fip_files = __natural_sort(fip_files)


    ## Fill in datalist while doing scaling

    assert(len(graph_files) == len(fip_files))
    for i in range(len(graph_files)):
        graph_file = graph_files[i]
        fip_file = fip_files[i]

        G = nx.read_gpickle(graph_file)
        data = torch_geometric.utils.from_networkx(G)
        x = data.x
        edge_index = data.edge_index
        e = data.e

        fip = np.loadtxt(fip_file, delimiter=',')[:,1]

        with open(file=f'{scaler_dir}\\{texture}\\nfeat_scaler.pickle', mode='rb') as f:
            nfeat_scaler = pickle.load(f)
        x = nfeat_scaler.transform(x)

        with open(file=f'{scaler_dir}\\{texture}\\efeat_scaler.pickle', mode='rb') as f:
            efeat_scaler = pickle.load(f)
        e = efeat_scaler.transform(e[:,None])

        with open(file=f'{scaler_dir}\\{texture}\\fip_scaler.pickle', mode='rb') as f:
            fip_scaler = pickle.load(f)
        fip = fip_scaler.transform(fip[:,None])
        

        # rewrite data with type casting
        data.x = torch.from_numpy(x).float()
        data.edge_index = edge_index
        data.e = torch.from_numpy(e).float()
        data.fip = torch.from_numpy(fip).float()

        datalist.append(data)


## Shuffle, split and save
np.random.seed(seed)
bnd_idx = int(len(datalist)*train_fraction)

# shuffle
rand_ids = np.random.choice(len(datalist), len(datalist), replace=False)
datalist_new = [datalist[i] for i in rand_ids]

# split
train_datalist = datalist_new[:bnd_idx]
val_datalist = datalist_new[bnd_idx:]

foldername = '_'.join(textures)+'_test'

with open(f'{torchdata_dir}\\{foldername}\\test_datalist.pickle', 'wb') as f:
    pickle.dump(train_datalist, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open(f'{torchdata_dir}\\{foldername}\\val_datalist.pickle', 'wb') as f:
#     pickle.dump(val_datalist, f, protocol=pickle.HIGHEST_PROTOCOL)
