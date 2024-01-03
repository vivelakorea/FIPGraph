
from utils_preprocessing import write_fip_table
import multiprocessing
import time
import datetime

MAIN_DIR = "C:\\Users\\Gyu-Jang Sim\\Documents\\FIPGraph"

textures = {
    '90': {'sve_indices': list(range(200))},
    '160': {'sve_indices': list(range(100))},
    '200': {'sve_indices': list(range(4))},
    '250': {'sve_indices': list(range(2))}
}

texture = '160'

def func(ith_sve):
    write_fip_table(f'{MAIN_DIR}\\preprocessing\\fiptables\\{texture}', f'{MAIN_DIR}\\data\\hdf5s_FIP', texture, ith_sve)

if __name__ == '__main__':

    info = textures[texture]
    sve_indices = info['sve_indices']

    # for ith_sve in sve_indices:
    #     write_nx_graph(f'{MAIN_DIR}\\preprocessing\\graph_sves', f'{MAIN_DIR}\\data\\hdf5s_SVE', texture, ith_sve)

    print('started at: ', datetime.datetime.now())
    t = time.time()
    pool_obj = multiprocessing.Pool(61)
    pool_obj.map(func, sve_indices)
    print('elapsed time:', time.time()-t)

    # t = time.time()
    # list(map(func, sve_indices))
    # print('time:', time.time()-t)