import time

import os 
import torch 
import esm
import json 
from tqdm import tqdm 
import numpy as np
from scipy.special import softmax
from utils import *
import random 
import pickle
import matplotlib.pyplot as plt
import random


with open('../esm_gap_distance/contact-recovery_github/data/full_seq_dict.json', "r") as json_file:
    seq_dict = json.load(json_file)
    

with open('../esm_gap_distance/contact-recovery_github/data/selected_protein.json', "r") as json_file:
    selected = json.load(json_file)

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

# put model on GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = model.eval()
#model.args.token_dropout = False

esm_alphabet = "".join(alphabet.all_toks[4:24])+"-"

def get_logits(seq):
  x,ln = alphabet.get_batch_converter()([("seq",seq)])[-1],len(seq)
  with torch.no_grad():
    f = lambda x: model(x)["logits"][0,1:(ln+1),4:24].cpu().numpy()
    x = x.to(device)
    logits = f(x)
    return logits

def get_masked_logits(seq, p=None, get_pll=False):
  x,ln = alphabet.get_batch_converter()([(None,seq)])[-1],len(seq)
  if p is None: p = ln
  with torch.no_grad():
    def f(x):
      fx = model(x)["logits"][:,1:(ln+1),4:24]
      return fx

    logits = np.zeros((ln,20))
    for n in range(0,ln,p):
      m = min(n+p,ln)
      x_h = torch.tile(torch.clone(x),[m-n,1])
      for i in range(m-n):
        x_h[i,n+i+1] = alphabet.mask_idx
      fx_h = f(x_h.to(device))
      for i in range(m-n):
        logits[n+i] = fx_h[i,n+i].cpu().numpy()
  if get_pll:
    logits = np.log(softmax(logits,-1))
    x = x.cpu().numpy()[0]
    x = x[1:(ln+1)] - 4
    return sum([logits[n,i] for n,i in enumerate(x)])
  else:
    return logits

def get_categorical_jacobian(seq):
  # ∂in/∂out
  x,ln = alphabet.get_batch_converter()([("seq",seq)])[-1],len(seq)
  with torch.no_grad():
    f = lambda x: model(x)["logits"][...,1:(ln+1),4:24].cpu().numpy()
    fx = f(x.to(device))[0]
    x = torch.tile(x,[20,1]).to(device)
    fx_h = np.zeros((ln,20,ln,20))
    for n in range(ln): # for each position
      x_h = torch.clone(x)
      x_h[:,n+1] = torch.arange(4,24) # mutate to all 20 aa
      fx_h[n] = f(x_h)
    return fx_h - fx


def get_data(pdb): 
    seq = seq_dict[pdb]
    # jacobian of the model
    jac = get_categorical_jacobian(seq)
    # center & symmetrize
    for i in range(4): jac -= jac.mean(i,keepdims=True)
    jac = (jac + jac.transpose(2,3,0,1))/2
    jac_contacts = get_contacts(jac)
    #print(jac.shape)
    
    np.save('/ssd/zhidian/jac/8M/' + pdb + '_esm2_jac.npy', jac.astype(np.float16))
    np.save('/ssd/zhidian/jac_contact/8M/' + pdb + '_esm2_jac_contact.npy', jac_contacts.astype(np.float16))


for pdb in tqdm(selected):
    get_data(pdb)





