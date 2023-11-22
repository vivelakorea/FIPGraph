
import glob
import pickle
import re
import warnings
import h5py
from matplotlib import scale
from matplotlib.pyplot import sca
import numpy as np
import networkx as nx
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch_geometric


def __make_pdc(micro):
    '''Pads 3D array with opposing faces.

    Parameters
    ----------
    micro : {ndarray}
        3D array of grain IDs.

    Returns
    -------
    micro : {ndarray}
        3D array of grain IDs with periodic faces.
    '''
    micro = np.pad(micro, 1)
    micro[0, :, :] = micro[-2, :, :]
    micro[-1, :, :] = micro[1, :, :]
    micro[:, 0, :] = micro[:, -2, :]
    micro[:, -1, :] = micro[:, 1, :]
    micro[:, :, 0] = micro[:, :, -2]
    micro[:, :, -1] = micro[:, :, 1]
    return micro

######################################################################################################################################

def __get_nbrs(micro, periodic=True):
    '''Get neighbors of grains.

    Parameters
    ----------
    micro : {ndarray}
        3D array of grain IDs.
        *DONT USE grain ID == 0*

    Returns
    -------
    nbr_dict : {dictionary} {grain_ID : [nbrs, shared_area]}
        Dictionary giving neighbors and shared area of each neighbor for every grain.
    '''
    if periodic:
        micro = __make_pdc(micro)
    else:
        micro = np.pad(micro, 1)

    dim = micro.shape

    # structure element used to get voxel face neighbors
    s = np.zeros((3, 3, 3))
    s[1, 1, :] = 1
    s[1, :, 1] = 1
    s[:, 1, 1] = 1
    s[1, 1, 1] = 0

    nbr_dict = {}
    for feat in np.unique(micro):
        if feat == 0:
            continue
        nbr_list = []
        for x in range(1, dim[0]-1):
            for y in range(1, dim[1]-1):
                for z in range(1, dim[2]-1):
                    if micro[x, y, z] == feat:
                        nbrs = s*micro[x-1:x+2, y-1:y+2, z-1:z+2]
                        nbrs = nbrs[~np.isin(nbrs, [0, feat])]
                        for nbr in nbrs:
                            nbr_list.append(nbr)

        nbrs, counts = np.unique(np.asarray(nbr_list), return_counts=True)
        nbr_dict[feat] = [nbrs.astype(int), counts]
    return nbr_dict

######################################################################################################################################

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

######################################################################################################################################

def __get_grains(dream3d_fname):
    '''
    Does same thing what utils.assign_grains does,
    but reads information dicectly from dream3d hdf5 file, so that the time complexity is better (utils.assign_grains: O(n^2), utils2.__get_grains: O(n) where n is number of cells), 
    and grain IDs are consistent with hdf5 files' feature IDs.
    '''
    
    f = h5py.File(dream3d_fname, 'r')

    featureIds = np.array(f['DataContainers']['SyntheticVolumeDataContainer']['CellData']['FeatureIds'])
    featureIds = featureIds[:,:,:,0]
    
    eulerAngles = np.array(f['DataContainers']['SyntheticVolumeDataContainer']['CellFeatureData']['EulerAngles'])
    eulerAngles = eulerAngles[:,:] # eulerAngles[1,:] is dummy for indexing grain starting from 1, not 0

    return featureIds, eulerAngles

######################################################################################################################################

def __get_orientation_matrix(euler_angles):
    euler1, euler2, euler3 = euler_angles

    COS = np.cos
    SIN = np.sin

    orientation_matrix = np.array([
        [COS(euler1)*COS(euler3)-SIN(euler1)*SIN(euler3)*COS(euler2), SIN(euler1)*COS(euler3)+COS(euler1)*SIN(euler3)*COS(euler2), SIN(euler2)*SIN(euler3)],
        [-COS(euler1)*SIN(euler3)-SIN(euler1)*COS(euler3)*COS(euler2), -SIN(euler1)*SIN(euler3)+COS(euler1)*COS(euler3)*COS(euler2), SIN(euler2)*COS(euler3)],
        [SIN(euler1)*SIN(euler2), -COS(euler1)*SIN(euler2), COS(euler2)]
    ])

    return orientation_matrix

######################################################################################################################################

def __rotate_stiffness(D11, D12, D44, euler_angles):
    # https://jakubmikula.com/solidmechanics/2017/06/18/Cubic-Elasticity.html
    R_cg = __get_orientation_matrix(euler_angles)
    print(R_cg)
    
    D = np.zeros((6,6))
    D[0,0] = D11; D[0,1] = D12; D[0,2] = D12
    D[1,0] = D12; D[1,1] = D11; D[1,2] = D12
    D[2,0] = D12; D[2,1] = D12; D[2,2] = D11
    D[3,3] = D44; D[4,4] = D44; D[5,5] = D44

    R = np.zeros((6,6))
    R[0,0] = 1; R[1,1] = 1; R[2,2] = 1
    R[3,3] = 2; R[4,4] = 2; R[5,5] = 2

    invR = np.zeros((6,6))
    invR[0,0] = 1; invR[1,1] = 1; invR[2,2] = 1
    invR[3,3] = 0.5; invR[4,4] = 0.5; invR[5,5] = 0.5

    K1 = np.zeros((3,3))
    K2 = np.zeros((3,3))
    K3 = np.zeros((3,3))
    K4 = np.zeros((3,3))

    mmod = np.array([0,1,2,0,1])

    for i in range(3):
        for j in range(3):
            K1[i,j] = R_cg[i,j]**2
            K2[i,j] = R_cg[i,mmod[j+1]]*R_cg[i,mmod[j+2]]
            K3[i,j] = R_cg[mmod[i+1],j]*R_cg[mmod[i+2],j]
            K4[i,j] = R_cg[mmod[i+1],mmod[j+1]]*R_cg[mmod[i+2],mmod[j+2]] + R_cg[mmod[i+1],mmod[j+2]]*R_cg[mmod[i+2],mmod[j+1]]

    T_cg = np.zeros((6,6))
    T_cg[0:3,0:3] = K1
    T_cg[0:3,3:6] = 2.0*K2
    T_cg[3:6,0:3] = K3
    T_cg[3:6,3:6] = K4

    invT_cg = np.zeros((6,6))
    invT_cg[0:3,0:3] = K1.T
    invT_cg[0:3,3:6] = 2.0*K3.T
    invT_cg[3:6,0:3] = K2.T
    invT_cg[3:6,3:6] = K4.T

    D_rot = T_cg@D@R@invT_cg@invR
    
    return D_rot

######################################################################################################################################

def __get_fcc_schmids(euler_angles, loading_direction): 
    '''
    Calculate all euler angles of fcc
    
    euler_angles : Bunge Euler angles in randian
    loading_direction : ex) [1, 0, 0]
    '''
    
    schmid_factors = []

    crystal_info = [
        {'hkl': [1.,1.,1.], 'uvw': [0.,1.,-1.]},
        {'hkl': [1.,1.,1.], 'uvw': [-1.,0.,1.]},
        {'hkl': [1.,1.,1.], 'uvw': [1.,-1.,0.]},

        {'hkl': [-1.,-1.,1.], 'uvw': [0.,-1.,-1.]},
        {'hkl': [-1.,-1.,1.], 'uvw': [1.,0.,1.]},
        {'hkl': [-1.,-1.,1.], 'uvw': [-1.,1.,0.]},

        {'hkl': [1.,-1.,-1.], 'uvw': [0.,-1.,1.]},
        {'hkl': [1.,-1.,-1.], 'uvw': [-1.,0.,-1.]},
        {'hkl': [1.,-1.,-1.], 'uvw': [1.,1.,0.]},

        {'hkl': [-1.,1.,-1.], 'uvw': [0.,1.,1.]},
        {'hkl': [-1.,1.,-1.], 'uvw': [1.,0.,-1.]},
        {'hkl': [-1.,1.,-1.], 'uvw': [-1.,-1.,0.]},
    ]

    orientation_matrix = __get_orientation_matrix(euler_angles)

    crs_load_dir = np.matmul(orientation_matrix, np.array(loading_direction))
    xb, yb, zb = crs_load_dir
    length = np.sqrt(xb**2. + yb**2. + zb**2.)

    for cry_sys in crystal_info:
        hkl = np.array(cry_sys['hkl'])
        uvw = np.array(cry_sys['uvw'])
        
        cosphi = np.dot(hkl, crs_load_dir)/(np.sqrt(3)*length)
        coslam = np.dot(uvw, crs_load_dir)/(np.sqrt(2)*length)

        schmid = abs(cosphi*coslam)
        schmid_factors.append(schmid)

    return schmid_factors

######################################################################################################################################

def __get_all_data(graph_dir, fip_dir, texture, num_nfeat, num_efeat):
    '''
    From graph_dir, get all data of node features, edge features, and fip labels
    '''
    
    feature_data = np.zeros((0, num_nfeat))
    edge_data = np.zeros((0, num_efeat))
    fip_data = np.zeros((0, 1))

    # feature
    graph_files = glob.glob(f'{graph_dir}\\{texture}\\*')
    graph_files = __natural_sort(graph_files)

    for file in graph_files:
        G = nx.read_gpickle(file)
        data = torch_geometric.utils.from_networkx(G)
        if num_efeat == 1:
            data.e = np.column_stack([data.e]) # transpose 1d array to column vector
        feature_data = np.vstack((feature_data, data.x))
        edge_data = np.vstack((edge_data, data.e))

    # fip    
    fip_files = glob.glob(f'{fip_dir}\\{texture}\\*')
    fip_files = __natural_sort(fip_files)

    for file in fip_files:
        fips = np.loadtxt(file, delimiter=',')
        fips = np.column_stack([fips[:,1]]) # transpose 1d array to column vector
        fip_data = np.vstack((fip_data, fips))

    return feature_data, edge_data, fip_data

######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

def write_nx_graph(graph_dir, hdf5_dir, texture, ith_sve):
    '''
    Read data from data_dir and write graph at graph_dir.
    {texture}'s {batch}th batch which has {batch_size} size of SVEs with {elem_per_side}
    '''

    hdf5_fname = f'{hdf5_dir}\\{texture}\\Output_FakeMatl_{ith_sve}.dream3d'
    
    grain_ids, grain_oris = __get_grains(hdf5_fname)
    nbr_dict = __get_nbrs(grain_ids, periodic=True)

    G = nx.Graph()

    for feat, ori in enumerate(grain_oris):
        if feat == 0: continue

        size = np.sum(grain_ids == feat)

        sorted_schmid = sorted(__get_fcc_schmids(ori, loading_direction=[1, 0, 0]), reverse=True)
        
        G.add_nodes_from([(feat, {"x": np.hstack([ori, size, sorted_schmid])})])
        # G.add_nodes_from([(feat, {"x": np.array([sorted_schmid[-1], size])})])

    for feat, (nbrs, areas) in nbr_dict.items():
        for nbr, area in zip(nbrs, areas):
            G.add_edge(feat, nbr, e=area)
        
        G.nodes[feat]['x'] = np.hstack([G.nodes[feat]['x'], len(nbrs)]) # add number of neighbors for node feature

    nx.write_gpickle(G, f'{graph_dir}\\{texture}\\sve_{ith_sve}')
    
######################################################################################################################################

def write_fip_table(fiptable_dir, fiphdf5_dir, texture, ith_sve):
    
    # https://materialscommons.org/public/datasets/248/overview

    hdf5s = {
        '30': 
            {
            'fname': fiphdf5_dir + '\\30_voxels_per_side_200_samples_275_avg_num_grains.hdf5',
            'featname': '275_grain_data'
            },
        '45': 
            {
            'fname': fiphdf5_dir + '\\45_voxels_per_side_200_samples_950_avg_num_grains.hdf5', 
            'featname': '950_grain_data'
            },
        '90': 
            {
            'fname': fiphdf5_dir + '\\90_voxels_per_side_200_samples_7500_avg_num_grains.hdf5', 
            'featname': '7500_grain_data'
            },
        '160':
            {
            'fname': fiphdf5_dir + '\\160_voxels_per_side_100_samples_41000_avg_num_grains.hdf5', 
            'featname': '41000_grain_data'
            },
        '200':
            {
            'fname': fiphdf5_dir + '\\200_voxels_per_side_4_samples_80000_avg_num_grains.hdf5', 
            'featname': '80000_grain_data'
            },
        '250':
            {
            'fname': fiphdf5_dir + '\\250_voxels_per_side_2_samples_160000_avg_num_grains.hdf5', 
            'featname': '160000_grain_data'
            },
    }

    hdf5 = hdf5s[texture]['fname']
    featname = hdf5s[texture]['featname']
    elem_per_side = int(texture)

    f = h5py.File(hdf5, 'r')
    data = np.array(f.get(featname))
    f.close()

    elemFIPs = data[ith_sve, 1, :].reshape((elem_per_side, elem_per_side, elem_per_side))
    grainIDs = data[ith_sve, 2, :].reshape((elem_per_side, elem_per_side, elem_per_side))

    fips = {} # grainID: elemFIP
    for i in range(elem_per_side):
        for j in range(elem_per_side):
            for k in range(elem_per_side):
                grainID = int(grainIDs[i, j, k])
                elemFIP = elemFIPs[i, j, k]
                if grainID not in fips:
                    fips[grainID] = elemFIP
                else: # test if all element in same grain has same fip
                    assert(elemFIP == fips[grainID])

    f = open(f'{fiptable_dir}\\fip_{ith_sve}.csv', 'w')
    
    numGrains = len(fips.keys())
    
    for i in range(1, numGrains+1):
        f.write(f'{i},{fips[i]}\n')

    f.close()

######################################################################################################################################


def make_scaler(scaler_dir, graph_dir, fip_dir, texture, num_nfeat, num_efeat):
    nfeat_data, efeat_data, fip_data = __get_all_data(graph_dir, fip_dir, texture, num_nfeat, num_efeat)

    # For euler angles, they will be divided by pi, 2*pi, pi without considering data
    # For Schmid factor, it will be multiplied by 2.0 without considering data

    # min_schmid = np.min(nfeat_data[:, 0])
    # with open(file=f'{scaler_dir}\\{texture}\\min_schmid.pickle', mode='wb') as f:
    #     pickle.dump(min_schmid, f, protocol=pickle.HIGHEST_PROTOCOL)

    # avg_schmid = np.average(nfeat_data[:, 0])
    # with open(file=f'{scaler_dir}\\{texture}\\avg_schmid.pickle', mode='wb') as f:
    #     pickle.dump(avg_schmid, f, protocol=pickle.HIGHEST_PROTOCOL)

    # std_schmid = np.std(nfeat_data[:, 0])
    # with open(file=f'{scaler_dir}\\{texture}\\std_schmid.pickle', mode='wb') as f:
    #     pickle.dump(std_schmid, f, protocol=pickle.HIGHEST_PROTOCOL)




    # # Size of grain will be divided by maximum size
    # max_grainsize = np.max(nfeat_data[:, 1])
    # with open(file=f'{scaler_dir}\\{texture}\\max_grainsize.pickle', mode='wb') as f:
    #     pickle.dump(max_grainsize, f, protocol=pickle.HIGHEST_PROTOCOL)

    # avg_grainsize = np.average(nfeat_data[:, 1])
    # with open(file=f'{scaler_dir}\\{texture}\\avg_grainsize.pickle', mode='wb') as f:
    #     pickle.dump(avg_grainsize, f, protocol=pickle.HIGHEST_PROTOCOL)

    # std_grainsize = np.std(nfeat_data[:, 1])
    # with open(file=f'{scaler_dir}\\{texture}\\std_grainsize.pickle', mode='wb') as f:
    #     pickle.dump(std_grainsize, f, protocol=pickle.HIGHEST_PROTOCOL)




    # # number of neighbors will be divided by maximum number of neighbors
    # max_num_neighbors = np.max(nfeat_data[:, 2])
    # with open(file=f'{scaler_dir}\\{texture}\\max_num_neighbors.pickle', mode='wb') as f:
    #     pickle.dump(max_num_neighbors, f, protocol=pickle.HIGHEST_PROTOCOL)

    # avg_num_neighbors = np.average(nfeat_data[:, 2])
    # with open(file=f'{scaler_dir}\\{texture}\\avg_num_neighbors.pickle', mode='wb') as f:
    #     pickle.dump(avg_num_neighbors, f, protocol=pickle.HIGHEST_PROTOCOL)

    # std_num_neighbors = np.std(nfeat_data[:, 2])
    # with open(file=f'{scaler_dir}\\{texture}\\std_num_neighbors.pickle', mode='wb') as f:
    #     pickle.dump(std_num_neighbors, f, protocol=pickle.HIGHEST_PROTOCOL)



    # # Area of grain boundary will be divided by maximum area
    # max_gbarea = np.max(efeat_data[:, 0])
    # with open(file=f'{scaler_dir}\\{texture}\\max_gbarea.pickle', mode='wb') as f:
    #     pickle.dump(max_gbarea, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # # FIP will be divided by maximum FIP
    # max_fip = np.max(fip_data)
    # with open(file=f'{scaler_dir}\\{texture}\\max_fip.pickle', mode='wb') as f:
    #     pickle.dump(max_fip, f, protocol=pickle.HIGHEST_PROTOCOL)

    # avg_fip = np.average(fip_data)
    # with open(file=f'{scaler_dir}\\{texture}\\avg_fip.pickle', mode='wb') as f:
    #     pickle.dump(avg_fip, f, protocol=pickle.HIGHEST_PROTOCOL)

    # std_fip = np.std(fip_data)
    # with open(file=f'{scaler_dir}\\{texture}\\std_fip.pickle', mode='wb') as f:
    #     pickle.dump(std_fip, f, protocol=pickle.HIGHEST_PROTOCOL)

    nfeat_scaler = StandardScaler()
    nfeat_scaler.fit(nfeat_data)

    with open(file=f'{scaler_dir}\\{texture}\\nfeat_scaler.pickle', mode='wb') as f:
        pickle.dump(nfeat_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    efeat_scaler = StandardScaler()
    efeat_scaler.fit(efeat_data)

    with open(file=f'{scaler_dir}\\{texture}\\efeat_scaler.pickle', mode='wb') as f:
        pickle.dump(efeat_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    fip_scaler = StandardScaler()
    fip_scaler.fit(fip_data)

    with open(file=f'{scaler_dir}\\{texture}\\fip_scaler.pickle', mode='wb') as f:
        pickle.dump(fip_scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

######################################################################################################################################

def make_torchdataset(torchdata_dir, graph_dir, fip_dir, scaler_dir, texture, train_fraction, seed):

    # feature
    graph_files = glob.glob(f'{graph_dir}\\{texture}\\*')
    graph_files = __natural_sort(graph_files)

    # fip
    fip_files = glob.glob(f'{fip_dir}\\{texture}\\*')
    fip_files = __natural_sort(fip_files)


    ## Fill in datalist while doing scaling
    datalist = [] # elements: {'x': [.., .., .., ...], 'edge_index':[[...],[...]], 'e': [.., .., .., ...], 'fip': [..., ..., ...]}

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

        # # scale Euler angles
        # x[:,0] /= (2.0*np.pi)
        # x[:,1] /= np.pi
        # x[:,2] /= (2.0*np.pi)

        # # scale grain size
        # # with open(file=f'{scaler_dir}\\{texture}\\avg_grainsize.pickle', mode='rb') as f:
        # #     avg_grainsize = pickle.load(f)
        # # with open(file=f'{scaler_dir}\\{texture}\\std_grainsize.pickle', mode='rb') as f:
        # #     std_grainsize = pickle.load(f)
        # # x[:,1] = (x[:,1]-avg_grainsize)/std_grainsize

        # # scale Schmid factors
        # with open(file=f'{scaler_dir}\\{texture}\\avg_schmid.pickle', mode='rb') as f:
        #     avg_schmid = pickle.load(f)
        # with open(file=f'{scaler_dir}\\{texture}\\std_schmid.pickle', mode='rb') as f:
        #     std_schmid = pickle.load(f)
        # x[:,0] = (x[:,0]-avg_schmid)/std_schmid

        # # scale number of neighbors
        # # with open(file=f'{scaler_dir}\\{texture}\\avg_num_neighbors.pickle', mode='rb') as f:
        # #     avg_num_neighbors = pickle.load(f)
        # # with open(file=f'{scaler_dir}\\{texture}\\std_num_neighbors.pickle', mode='rb') as f:
        # #     std_num_neighbors = pickle.load(f)
        # # x[:,2] = (x[:,2]-avg_num_neighbors)/std_num_neighbors

        # # scale grain boundary area
        # e = e.float()
        # with open(file=f'{scaler_dir}\\{texture}\\max_gbarea.pickle', mode='rb') as f:
        #     max_gbarea = pickle.load(f)
        # e /= max_gbarea

        # # scale fips
        # with open(file=f'{scaler_dir}\\{texture}\\avg_fip.pickle', mode='rb') as f:
        #     avg_fip = pickle.load(f)
        # with open(file=f'{scaler_dir}\\{texture}\\std_fip.pickle', mode='rb') as f:
        #     std_fip = pickle.load(f)
        # fip = (fip-avg_fip)/std_fip

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

    with open(f'{torchdata_dir}\\{texture}\\train_datalist.pickle', 'wb') as f:
        pickle.dump(train_datalist, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{torchdata_dir}\\{texture}\\val_datalist.pickle', 'wb') as f:
        pickle.dump(val_datalist, f, protocol=pickle.HIGHEST_PROTOCOL)

######################################################################################################################################


# from utils_preprocessing import write_fip_table
textures = {
    '30': {'sve_indices': list(range(200))}
}
MAIN_DIR = "C:\\Users\\gyujang95\\Desktop\\Marat\\FIPGraph"
# for texture, info in textures.items():
    
#     sve_indices = info['sve_indices']

#     for ith_sve in sve_indices:
#         write_fip_table(f'{MAIN_DIR}\\fiptables\\{texture}', f'{MAIN_DIR}\\data\\hdf5s_FIP', texture, ith_sve)
    

# for texture, info in textures.items():
    
#     sve_indices = info['sve_indices']

#     for ith_sve in sve_indices:
#         write_nx_graph(f'{MAIN_DIR}\\preprocessing\\graph_sves', f'{MAIN_DIR}\\data\\hdf5s_SVE', texture, ith_sve)
    


# print(__get_fcc_schmids([4.759338337, 0.704240353, 0.441219235], [0, 0, 1]))
    
# a,b,c = (__get_pyg_data(graph_dir=f'{MAIN_DIR}\\preprocessing\\graph_sves', fip_dir=f'{MAIN_DIR}\\preprocessing\\FIPtables', num_nfeat=5, num_efeat=1, texture=30))
# print(a)
# print(b)
# print(c)

# make_scaler(f'{MAIN_DIR}\\preprocessing\\graph_sves', f'{MAIN_DIR}\\preprocessing\\FIPtables', 30, 5, 1)
# make_torchdataset(f'{MAIN_DIR}\\preprocessing\\graph_torchdata', 
#                   f'{MAIN_DIR}\\preprocessing\\graph_sves', 
#                   f'{MAIN_DIR}\\preprocessing\\FIPtables',
#                   f'{MAIN_DIR}\\preprocessing\\scalers', 
#                   30, 0.9, 42)