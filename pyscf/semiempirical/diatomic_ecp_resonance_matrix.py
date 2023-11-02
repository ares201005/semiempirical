#!/usr/bin/env python
# flake8: noqa

'''
whatever
'''

import ctypes
import copy
import math
import numpy as np
import warnings
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf.data.elements import _symbol
from .matprint2d import *

def resonance_integral(betai, betaj, alphai, alphaj, rij):
    return 0.5*(betai+betaj)*np.sqrt(rij)*np.exp(-(alphai+alphaj)*rij**2)

def diatomic_ecp_resonance_matrix(ia, ja, zi, zj, xij, rij, params, rot_mat): 
    gecp = np.zeros((4,4))
    nt = zi + zj
    print(f'Atoms: {zi} {zj}')
    if zi == 1 and zj == 1: # first row - first row: no ECP energy 
       #jcall = 2 
       pass # make this if impossible in omx.py loop
    elif (zi > 1 and zj == 1): # second row - first
       #jcall = 3
       # Only s(ecp)-s will be non-zero
       gssam = resonance_integral(params.beta_s[zj], params.beta_ecp[zi], params.alpha_s[zj], params.alpha_ecp[zi], rij)
       print(f'gssam: {gssam}')
       gecp_rotate = np.einsum('ji,kj,km->im', rot_mat, gecp, rot_mat) 
       #gecp_rotate = tmp_b0[:,0]
       #matrix_print_2d(gecp_rotate, 4, 'gecp_rotate')       
       matrix_print_2d(gecp, 4, 'gecp')       
       return gecp_rotate[:,0].reshape(4,1)

    elif (zi == 1 and zj > 1): # first row - second row
       #jcall = 3
       # Only s-s(ecp) will be non-zero
       # No H-bond parameter if zj is ECP atom
       gssma = resonance_integral(params.beta_s[zi], params.beta_ecp[zj], params.alpha_s[zi], params.alpha_ecp[zj], rij)
       print(f'gssma: {gssma}')
       gecp_rotate = np.einsum('ji,kj,km->im', rot_mat, gecp, rot_mat) 
       #gecp_rotate = tmp_b0[0,:]
       #matrix_print_2d(gecp_rotate, 1, 'gecp_rotate')       
       matrix_print_2d(gecp, 1, 'gecp')       
       return gecp_rotate[0,:].reshape(1,4)

    elif zi > 1 and zj > 1: 
       #jcall = 4 
       gssma = resonance_integral(params.beta_s[zi], params.beta_ecp[zj], params.alpha_s[zi], params.alpha_ecp[zj], rij) #Keep
       gpsma = resonance_integral(params.beta_p[zi], params.beta_ecp[zj], params.alpha_p[zi], params.alpha_ecp[zj], rij) #Keep
       gssam = resonance_integral(params.beta_s[zj], params.beta_ecp[zi], params.alpha_s[zj], params.alpha_ecp[zi], rij) #Keep
       gpsam = resonance_integral(params.beta_p[zj], params.beta_ecp[zi], params.alpha_p[zj], params.alpha_ecp[zi], rij) #Keep
       print(f'gssma: {gssma}')
       print(f'gssam: {gssam}')
       print(f'gpsma: {gpsma}')
       print(f'gpsam: {gpsam}')
       #matrix_print_2d(rot_mat, 4, 'T')       
       #matrix_print_2d(gecp, 4, 'gecp')       
       gecp_rotate = np.einsum('ji,kj,km->im', rot_mat, gecp, rot_mat) 
       #matrix_print_2d(gecp_rotate, 4, 'gecp_rotate')       
       matrix_print_2d(gecp, 4, 'gecp')       
       return gecp_rotate # gecp
    else:
       print('invalid combination of zi and zj')
       exit(-1)

