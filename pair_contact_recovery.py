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

dist_range_min = 50
dist_range_max = 100 
expand_type = 'outward' 

save_file_path = 'gremline_mask_' + str(dist_range_min) + '_' + str(dist_range_max) + '_' + expand_type + '_wbos_eos.hdf5'
copy_file_path = 'temp_h5/' + save_file_path.split('.')[0] + '_'

with open('data/ss_dict.json', "r") as json_file:
    ss_dict = json.load(json_file)

with open('data/full_seq_dict.json', "r") as json_file:
    seq_dict = json.load(json_file)

with open('data/selected_protein.json', 'r') as file:
    selected_protein = json.load(file)

# get all SSEs
def get_segments(input_str):
    segments = []
    for match in re.finditer('E+|H+', input_str):
        if (match.group()[0] == 'E' and len(match.group()) > 3) or \
           (match.group()[0] == 'H' and len(match.group()) > 7):
            segments.append((match.start(), match.end()))
    return segments

# get contact map of the full sequence 
def get_original_contact(seq): 
    seq_tuple = [(1, seq)]
    
    # with BOS/EOS 
    batch_labels, batch_strs, batch_tokens = batch_converter(seq_tuple)
    
    # without BOS/EOS 
    #batch_tokens = torch.cat((torch.full((batch_tokens.shape[0], 1), 32), batch_tokens[:, 1:-1], torch.full((batch_tokens.shape[0], 1), 32)), dim=1)
    
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        esm2_predictions = esm_transformer.predict_contacts(batch_tokens)[0].cpu()
    return esm2_predictions.numpy()

# get centers of SSEs 
def get_ss_cents(segments): 
    ss_cents = []
    for seg in segments: 
        ss_cents.append((seg[1] + seg[0])//2) 
    return ss_cents 

# select pairs of SSEs separated by certain distance 
def get_pairs(arr):
    pairs = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if dist_range_min < abs(arr[i] - arr[j]) <= dist_range_max: # look at long separation ones 
                pairs.append((arr[i], arr[j]))
    return pairs


# take 5 res on both sides 
# select segment pairs with enough contacts + leaves enough distances for exploring different flanking lengths 
def select_pairs(cent_pairs, matrix, cutoff, seq_len):
    selected_pairs = []
    for pair in cent_pairs: 
        ss1_start = pair[0] - 5 
        ss2_end = pair[1] + 5 + 1 
        patch_sum = np.sum(matrix[(pair[0] - 5): (pair[0] + 6), (pair[1] - 5): (pair[1] + 6)])
        # check there is enough contact between the two SSE 
        # check there is enough region for expanding to check recovery 
        if (patch_sum > cutoff) and (min(ss1_start, seq_len - ss2_end - 1) > 10):
            selected_pairs.append(pair) 
    n = min(len(selected_pairs), 3)
    selected_pairs = random.sample(selected_pairs, n)
    return selected_pairs 


# get contact matrix of the masked sequence 
def contact_matrix(seq):
    seq_tuple = [(1, seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(seq_tuple)
    #batch_tokens = torch.cat((torch.full((batch_tokens.shape[0], 1), 32), batch_tokens[:, 1:-1], torch.full((batch_tokens.shape[0], 1), 32)), dim=1)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        esm2_predictions_gap2 = esm_transformer.predict_contacts(batch_tokens, gap_info_list = None)[0].cpu()

    return esm2_predictions_gap2.numpy()

# get the masked sequence and then the contact map 
def get_seg_contact(sequence, frag1_start, frag1_end, frag2_start, frag2_end): # flank_len is the amount of residues to add at sides of the segments  
    seg_seq_i = sequence[frag1_start: frag1_end] 
    seg_seq_j = sequence[frag2_start: frag2_end] 
    mask_length = frag2_start - frag1_end 
    full_seq = frag1_start * '<mask>' + seg_seq_i + mask_length * '<mask>' + seg_seq_j + (len(seq) -  frag2_end) * '<mask>'
    contact_map = contact_matrix(full_seq) 
    return contact_map


def norm_sum_mult(ori_contact_seg, seg_cross_contact):
    ori_mult_new = np.multiply(ori_contact_seg, seg_cross_contact)
    ori_mult_ori = np.multiply(ori_contact_seg, ori_contact_seg)
    return (np.sum(ori_mult_new)/np.sum(ori_mult_ori))



with h5py.File(save_file_path, 'w') as f:
    for i, protein_name in enumerate(tqdm(selected_protein)):
        seq = seq_dict[protein_name]
        contact_ori = get_original_contact(seq) 
        ss = ss_dict[protein_name + '.pdb']
        
        segments = get_segments(ss)
        ss_cents = get_ss_cents(segments)
        ss_pairs = get_pairs(ss_cents)
        selected_pairs = select_pairs(ss_pairs, contact_ori, 10, len(seq))
        
        for position in tqdm(selected_pairs):
            ss1_start = position[0] - 5 
            ss1_end = position[0] + 5 + 1 
            ss2_start = position[1] - 5 
            ss2_end = position[1] + 5 + 1 
            
            ori_contact_seg = contact_ori[ss1_start:ss1_end, ss2_start:ss2_end]
            
            #expand outward 
            flank_len_range = min(ss1_start, len(seq) - ss2_end - 1)
            
            for flank_len in range(flank_len_range):

                    # expand outward 
                    frag1_start = ss1_start - flank_len
                    frag1_end = ss1_end
                    frag2_start = ss2_start
                    frag2_end = ss2_end + flank_len

                    seg_contact = get_seg_contact(seq, frag1_start, frag1_end, frag2_start, frag2_end) 

                    # expand both ways / outward / inward
                    seg_cross_contact = seg_contact[ss1_start:ss1_end, ss2_start:ss2_end]

                    sum_diff_value = np.sum(seg_cross_contact) - np.sum(ori_contact_seg)
                    norm_sum_mult_value = norm_sum_mult(ori_contact_seg, seg_cross_contact)

                    key1 = f'{protein_name}/{position[0]}_{position[1]}/{flank_len}/seg_contact' 
                    key2 = f'{protein_name}/{position[0]}_{position[1]}/{flank_len}/seg_cross_contact' 
                    key3 = f'{protein_name}/{position[0]}_{position[1]}/{flank_len}/sum_diff' 
                    key4 = f'{protein_name}/{position[0]}_{position[1]}/{flank_len}/sum_mult' 

                    f.create_dataset(key1, data=seg_contact)
                    f.create_dataset(key2, data=seg_cross_contact)
                    f.create_dataset(key3, data=sum_diff_value)
                    f.create_dataset(key4, data=norm_sum_mult_value)
        if i % 300 == 0:
            f.flush()
            shutil.copy(save_file_path, copy_file_path + str(i) + '.hdf5')
            time.sleep(20)

torch.cuda.empty_cache()