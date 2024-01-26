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
import random

esm_transformer, esm2_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
batch_converter = esm2_alphabet.get_batch_converter()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
esm_transformer = esm_transformer.to(device)

with open('data/full_seq_dict.json', "r") as json_file:
    seq_dict = json.load(json_file)

with open('data/selected_protein.json', 'r') as file:
    selected_protein = json.load(file)


save_file_path = '/data/zhzhang/single_seg_w_bos_eos_randommask.hdf5'


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


with open('data/reproduce_single_recovery_w_bos_eos.json', 'r') as json_file:
    previous_data = json.load(json_file)


# for a quick test
keys_for_test = ['1A8LA', '1BYIA']
test_dict = {key: previous_data[key] for key in keys_for_test if key in previous_data}
print(test_dict)


with h5py.File(save_file_path, 'w') as f:
    #proteins = test_dict.keys()
    proteins = previous_data.keys()
    for i, protein in enumerate(tqdm(proteins)):
    
        seq = seq_dict[protein]
        ori_contact = get_contact(seq_dict[protein])

        patch_start_list = previous_data[protein]

        for patch_start in tqdm(patch_start_list):  
            patch_end = patch_start + 22
            flank_len_range = min(patch_start, len(seq) - patch_end + 1)

            for flank_len in range(flank_len_range):

                # generate the randomly masked sequence
                potential_positions = list(range(0, patch_start)) + list(range(patch_end, len(seq)))
                random_unmask_positions = random.sample(potential_positions, 2 * flank_len)
                seq_mask = ['<mask>'] * len(seq) 
                for pos in random_unmask_positions:
                    seq_mask[pos] = seq[pos]  
                seq_mask[patch_start:patch_end] = seq[patch_start:patch_end]
                seq_mask = ''.join(seq_mask)


                mask_contact_full = get_contact(seq_mask)

                ori_contact_seg = ori_contact[patch_start:patch_end, patch_start:patch_end]
                mask_contact_seg = mask_contact_full[patch_start:patch_end, patch_start:patch_end]

                norm_sum_mult_value = norm_sum_mult(ori_contact_seg, mask_contact_seg)

                key0 = f'{protein}/{patch_start}/{flank_len}/ori_contact_full' 
                key1 = f'{protein}/{patch_start}/{flank_len}/mask_contact_full' 
                key2 = f'{protein}/{patch_start}/{flank_len}/ori_contact_seg' 
                key3 = f'{protein}/{patch_start}/{flank_len}/mask_contact_seg' 
                key4 = f'{protein}/{patch_start}/{flank_len}/norm_sum_mult_value' 
                key5 = f'{protein}/{patch_start}/{flank_len}/masked_seq'


                f.create_dataset(key0, data=ori_contact)
                f.create_dataset(key1, data=mask_contact_full)
                f.create_dataset(key2, data=ori_contact_seg)
                f.create_dataset(key3, data=mask_contact_seg)
                f.create_dataset(key4, data=norm_sum_mult_value)
                f.create_dataset(key5, data=seq_mask)
                    
        if i % 300 == 0:
            f.flush()
            shutil.copy(save_file_path, save_file_path.split('.')[0] + '_' + str(i) + '.hdf5')
            time.sleep(20)

torch.cuda.empty_cache()





