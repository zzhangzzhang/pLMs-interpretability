import numpy as np
import string
import jax.numpy as jnp
import jax

def parse_fasta(filename, a3m=True):
  '''function to parse fasta file'''
  
  if a3m:
    # for a3m files the lowercase letters are removed
    # as these do not align to the query sequence
    rm_lc = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    
  header, sequence = [],[]
  lines = open(filename, "r")
  for line in lines:
    line = line.rstrip()
    if line[0] == ">":
      header.append(line[1:])
      sequence.append([])
    else:
      if a3m: line = line.translate(rm_lc)
      else: line = line.upper()
      sequence[-1].append(line)
  lines.close()
  sequence = [''.join(seq) for seq in sequence]
  
  return header, sequence
  
def mk_msa(seqs, alphabet=None):
  '''one hot encode msa'''
  if alphabet is None:
    alphabet = "ARNDCQEGHILKMFPSTWYV-"
  states = len(alphabet)  
  a2n = {a:n for n,a in enumerate(alphabet)}
  msa_ori = np.array([[a2n.get(aa, states-1) for aa in seq] for seq in seqs])
  return np.eye(states)[msa_ori]

def _do_apc(x, rm_diag=True):
  '''given matrix do apc correction'''
  if rm_diag: np.fill_diagonal(x,0.0)
  a1 = x.sum(0,keepdims=True)
  a2 = x.sum(1,keepdims=True)
  y = x - (a1*a2)/x.sum()
  np.fill_diagonal(y,0.0)
  return y

def inv_cov(Y):
  '''given one-hot encoded MSA, return contacts'''
  N,L,A = Y.shape
  Y_flat = Y.reshape(N,-1)
  c = np.cov(Y_flat.T)
  shrink = 4.5/np.sqrt(N) * np.eye(c.shape[0])
  ic = -np.linalg.inv(c + shrink)
  raw = np.sqrt(np.square(ic.reshape(L,A,L,A)[:,:20,:,:20]).sum((1,3)))
  return {"c":c, "ic":ic,
          "raw":raw, "apc":_do_apc(raw)}

def inv_cov_jax(Y):
  Y = jnp.asarray(Y)
  N,L,A = Y.shape
  Y_flat = Y.reshape(N,-1)
  c = jnp.cov(Y_flat.T)
  shrink = 4.5/jnp.sqrt(N) * jnp.eye(c.shape[0])
  ic = -jnp.linalg.inv(c + shrink)
  raw = jnp.sqrt(jnp.square(ic.reshape(L,A,L,A)[:,:20,:,:20]).sum((1,3)))
  raw = np.array(raw)
  return {"c":np.array(c),"ic":np.array(ic),
          "raw":raw,"apc":_do_apc(raw)}


def do_apc(x, rm=1):
  '''given matrix do apc correction'''
  # trying to remove different number of components
  # rm=0 remove none
  # rm=1 apc
  x = np.copy(x)
  if rm == 0:
    return x
  elif rm == 1:
    a1 = x.sum(0,keepdims=True)
    a2 = x.sum(1,keepdims=True)
    y = x - (a1*a2)/x.sum()
  else:
    # decompose matrix, rm largest(s) eigenvectors
    u,s,v = np.linalg.svd(x)
    y = s[rm:] * u[:,rm:] @ v[rm:,:]
  np.fill_diagonal(y,0)
  return y

def get_contacts(x, symm=True, center=True, rm=1):
  # convert jacobian (L,A,L,A) to contact map (L,L)
  j = x.copy()
  if center:
    for i in range(4): j -= j.mean(i,keepdims=True)
  j_fn = np.sqrt(np.square(j).sum((1,3)))
  np.fill_diagonal(j_fn,0)
  j_fn_corrected = do_apc(j_fn, rm=rm)
  if symm:
    j_fn_corrected = (j_fn_corrected + j_fn_corrected.T)/2
  return j_fn_corrected
