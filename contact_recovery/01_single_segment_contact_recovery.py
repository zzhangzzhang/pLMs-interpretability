import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import esm
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np 
import h5py
import json
import re
import shutil
import random
import time

esm_transformer, esm2_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = esm2_alphabet.get_batch_converter()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
esm_transformer = esm_transformer.to(device)

with open('../data/full_seq_dict.json', "r") as json_file:
    seq_dict = json.load(json_file)

with open('../data/selected_protein.json', 'r') as file:
    selected_protein = json.load(file)


def patch_sum(matrix, top_left, size=22):
    i, j = top_left
    return sum(matrix[x][y] for x in range(i, i + size) for y in range(j, j + size))

def find_diagonal_patches(matrix, size=22, threshold=15, sample_numb=3):
    m = len(matrix)
    patches = []

    for i in range(m - size + 1):
        j = i  
        # select those with enough contacts + enough space (10) for explore flanking values 
        if (patch_sum(matrix, (i, j), size) > threshold) and (10 < i < (matrix.shape[0] - size - 10)):
            patches.append((i, j))
            
    sampled_patches = random.sample(patches, min(sample_numb, len(patches))) 

    return sampled_patches


def get_contact(seq): 
    seq_tuple = [(1, seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(seq_tuple)
    #batch_tokens = torch.cat((torch.full((batch_tokens.shape[0], 1), 32), batch_tokens[:, 1:-1], torch.full((batch_tokens.shape[0], 1), 32)), dim=1)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        esm2_predictions = esm_transformer.predict_contacts(batch_tokens)[0].cpu()
    return esm2_predictions.numpy()


def norm_sum_mult(ori_contact_seg, seg_cross_contact):
    ori_mult_new = np.multiply(ori_contact_seg, seg_cross_contact)
    ori_mult_ori = np.multiply(ori_contact_seg, ori_contact_seg)
    return (np.sum(ori_mult_new)/np.sum(ori_mult_ori))

save_file_path = '../results/w_bos_eos.hdf5' 
with h5py.File(save_file_path, 'w') as f:
    for protein in tqdm(selected_protein):
            seq = seq_dict[protein]
            ori_contact = get_contact(seq_dict[protein])
            
            patches = find_diagonal_patches(ori_contact, size=22, threshold=15, sample_numb=3)
            
            for patch in tqdm(patches):  
                patch_start = patch[0] 
                patch_end = patch[0] + 22
                flank_len_range = min(patch_start, len(seq) - patch_end + 1)
                
                for flank_len in range(flank_len_range):
                    seg_start = patch_start - flank_len 
                    seg_end = patch_end + flank_len 
                    seq_mask = seg_start *'<mask>' + seq[seg_start:seg_end] + len(seq[seg_end:])*'<mask>'

                    mask_contact_full = get_contact(seq_mask)

                    ori_contact_seg = ori_contact[patch_start:patch_end, patch_start:patch_end]
                    mask_contact_seg = mask_contact_full[patch_start:patch_end, patch_start:patch_end]

                    norm_sum_mult_value = norm_sum_mult(ori_contact_seg, mask_contact_seg)

                    key1 = f'{protein}/{patch[0]}/{flank_len}/mask_contact_full' 
                    key2 = f'{protein}/{patch[0]}/{flank_len}/ori_contact_seg' 
                    key3 = f'{protein}/{patch[0]}/{flank_len}/mask_contact_seg' 
                    key4 = f'{protein}/{patch[0]}/{flank_len}/norm_sum_mult_value' 

                    f.create_dataset(key1, data=mask_contact_full)
                    f.create_dataset(key2, data=ori_contact_seg)
                    f.create_dataset(key3, data=mask_contact_seg)
                    f.create_dataset(key4, data=norm_sum_mult_value)



