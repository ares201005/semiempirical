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

def diatomic_resonance_matrix(ia, ja, zi, zj, xij, rij, params, rot_mat): 
    bss = 0.0
    bsp = bps = 0.0
    bpp = 0.0
    bpi = 0.0
    bloc = np.zeros((4,4))
    nt = zi + zj

    if zi == 1 and zj == 1: # first row - first row
       #jcall = 2 
       bss = resonance_integral(params.beta_s[zi], params.beta_s[zj], params.alpha_s[zi], params.alpha_s[zj], rij)
       bloc[0][0] = bss
       b0_rotate = bloc[0][0]
       #matrix_print_2d(b0_rotate, 1, 'b0_rotate')       
       return np.atleast_1d(b0_rotate)
    elif (zi > 1 and zj == 1): # second row - first
       #jcall = 3
       bss = resonance_integral(params.beta_sh[zi], params.beta_s[zj], params.alpha_sh[zi], params.alpha_s[zj], rij)
       bps = resonance_integral(params.beta_ph[zi], params.beta_s[zj], params.alpha_ph[zi], params.alpha_s[zj], rij)
       bloc[0][0] = bss
       bloc[0][1] = bsp
       bloc[1][0] = -1.0*bps
       #print(f'bss: {bss}')
       #print(f'bps: {-bps}')
       #print(f'bsp: {bsp}')
       b0_rotate = np.einsum('ji,jk,km->im', rot_mat, bloc, rot_mat) 
       #b0_rotate = tmp_b0[:,0]
       #matrix_print_2d(b0_rotate, 4, 'b0_rotate')       
       return b0_rotate[:,0].reshape(4,1)

    elif (zi == 1 and zj > 1): # first row - second row
       #jcall = 3
       bss = resonance_integral(params.beta_s[zi], params.beta_sh[zj], params.alpha_s[zi], params.alpha_sh[zj], rij)
       bsp = resonance_integral(params.beta_s[zi], params.beta_ph[zj], params.alpha_s[zi], params.alpha_ph[zj], rij)
       bloc[0][0] = bss
       bloc[0][1] = bsp
       bloc[1][0] = -1.0*bps
       b0_rotate = np.einsum('ji,jk,km->im', rot_mat, bloc, rot_mat) 
       #b0_rotate = tmp_b0[0,:]
       #matrix_print_2d(b0_rotate, 1, 'b0_rotate')       
       return b0_rotate[0,:].reshape(1,4)

    elif zi > 1 and zj > 1: # second row - second row #make else? -CL
       #jcall = 4 
       bss = resonance_integral(params.beta_s[zi], params.beta_s[zj], params.alpha_s[zi], params.alpha_s[zj], rij)
       bsp = resonance_integral(params.beta_s[zi], params.beta_p[zj], params.alpha_s[zi], params.alpha_p[zj], rij)
       bps = resonance_integral(params.beta_p[zi], params.beta_s[zj], params.alpha_p[zi], params.alpha_s[zj], rij)
       bpp = resonance_integral(params.beta_p[zi], params.beta_p[zj], params.alpha_p[zi], params.alpha_p[zj], rij)
       bpi = resonance_integral(params.beta_pi[zi], params.beta_pi[zj], params.alpha_pi[zi], params.alpha_pi[zj], rij)

       bps *= -1.0
       bpp *= -1.0
       bpmpi = (bpp-bpi)

       bloc[0][0] = bss

       bloc[0][1] = bsp 
       bloc[1][0] = bps

       bloc[1][1] = bpmpi+bpi
       bloc[2][2] = bpi 
       bloc[3][3] = bpi

       #matrix_print_2d(rot_mat, 4, 'T')       
       #matrix_print_2d(bloc, 4, 'bloc')       
       b0_rotate = np.einsum('ji,jk,km->im', rot_mat, bloc, rot_mat) 
       #matrix_print_2d(b0_rotate, 4, 'b0_rotate')       
       return b0_rotate # bloc
    else:
       print('invalid combination of zi and zj')
       exit(-1)

