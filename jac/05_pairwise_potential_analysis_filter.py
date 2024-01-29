import pickle
import numpy as np
from scipy.stats import spearmanr
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

with open('../data/full_seq_dict.json', "r") as json_file:
    seq_dict = json.load(json_file)

with open('/data/shuffled_selected_protein.json', "r") as json_file:
    shuffled = json.load(json_file)

def correlation_calculation(pdb, n_std, jac):
    with open('../results/inv_cov_msa/' + pdb + '_inv_cov_msa.pkl', 'rb') as f:
        msa_tmp = pickle.load(f)
    
    L = len(seq_dict[pdb])
    ic = msa_tmp["ic"].reshape(L, 21, L, 21)[:,:20,:,:20]

    idx = np.triu_indices(L, 1)
    a = jac[idx[0], :, idx[1], :]
    b = ic[idx[0], :, idx[1], :]
    ORDER = msa_tmp["apc"][idx[0], idx[1]].argsort()[::-1]

    cutoff = L # pick the cutoff 
    b_std = np.sqrt(np.square(b[ORDER[:cutoff]]).mean())
    mask = np.abs(b[ORDER[:cutoff]]) > b_std * n_std

    return spearmanr(a[ORDER[:cutoff]][mask].flatten(),
              b[ORDER[:cutoff]][mask].flatten()).statistic

def process_dataset(dataset_name):
    jac_data = {pdb: np.load(f'../results/jac/{dataset_name}/' + pdb + '_esm2_jac.npy') for pdb in tqdm(shuffled)}

    cc_info = {}
    for pdb in tqdm(shuffled):
        cc_info.setdefault(pdb, {})
        for i in range(0, 5):
            cc_info[pdb][i] = correlation_calculation(pdb, i, jac_data[pdb])

    with open(f'../results/{dataset_name}_L_std.json', 'w') as file:
        json.dump(cc_info, file)

datasets = ['150M', '35M', '8M']

for dataset in datasets:
    print(f"Processing dataset: {dataset}")
    process_dataset(dataset)
