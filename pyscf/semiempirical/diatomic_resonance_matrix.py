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

    if zi == 1 and zj == 1: # first row - first row
       #jcall = 2 
       bss = resonance_integral(params.beta_s[zi], params.beta_s[zj], params.alpha_s[zi], params.alpha_s[zj], rij)
       bloc[0,0] = bss 
    elif (zi > 1 and zj == 1): # second row - first
       #jcall = 3
       if nt == 9 or nt == 8:
          bss = resonance_integral(params.beta_sh[zi], params.beta_s[zj], params.alpha_sh[zi], params.alpha_s[zj], rij)
          bps = resonance_integral(params.beta_ph[zi], params.beta_s[zj], params.alpha_ph[zi], params.alpha_s[zj], rij)
       else:
          bss = resonance_integral(params.beta_s[zi], params.beta_s[zj], params.alpha_s[zi], params.alpha_s[zj], rij)
          bps = resonance_integral(params.beta_p[zi], params.beta_s[zj], params.alpha_p[zi], params.alpha_s[zj], rij)
       bloc[0,0] = bss
       bloc[0][3] = bps # or [3][0] one of corners
    elif (zi == 1 and zj > 1): # first row - second row
       #jcall = 3
       if nt == 9 or nt == 8:
          bss = resonance_integral(params.beta_s[zi], params.beta_sh[zj], params.alpha_s[zi], params.alpha_sh[zj], rij)
          bsp = resonance_integral(params.beta_s[zi], params.beta_ph[zj], params.alpha_s[zi], params.alpha_ph[zj], rij)
       else:
          bss = resonance_integral(params.beta_s[zi], params.beta_s[zj], params.alpha_s[zi], params.alpha_s[zj], rij)
          bsp = resonance_integral(params.beta_s[zi], params.beta_p[zj], params.alpha_s[zi], params.alpha_p[zj], rij)
       bloc[0,0] = bss
       bloc[3][0] = bsp
    elif zi > 1 and zj > 1: # second row - second row #make else? -CL
       #jcall = 4 
       bss = resonance_integral(params.beta_s[zi], params.beta_s[zj], params.alpha_s[zi], params.alpha_s[zj], rij)
       bsp = resonance_integral(params.beta_s[zi], params.beta_p[zj], params.alpha_s[zi], params.alpha_p[zj], rij)
       bps = resonance_integral(params.beta_p[zi], params.beta_s[zj], params.alpha_p[zi], params.alpha_s[zj], rij)
       bpp = resonance_integral(params.beta_p[zi], params.beta_p[zj], params.alpha_p[zi], params.alpha_p[zj], rij)
       bpi = resonance_integral(params.beta_pi[zi], params.beta_pi[zj], params.alpha_pi[zi], params.alpha_pi[zj], rij)
       #bloc[0,0] = bss
       #bloc[1][1] = bpp
       #bloc[2][2] = bpp
       #bloc[3][3] = bpipi
       #bloc[0][3] = bps
       #bloc[3][0] = bsp
       print(f'T(1) = {bss}')
       print(f'T(2) = {-bps}')
       print(f'T(3) = {bsp}')
       print(f'T(4) = {-bpp}')
       print(f'T(5) = {bpi}')
       bps *= -1.0
       bpp *= -1.0
       bpmpi = (bpp-bpi)
       print(f'T45 =  {bpmpi}')

       bloc[0,0] = bss

       bloc[0,1] = bsp #sign wrong
       bloc[1,0] = bps

       #bloc[1,0] = bsp
       #bloc[1,1] = bsp
       #bloc[1,2] = bsp

       #bloc[0,1] = bps
       #bloc[1,1] = bps
       #bloc[2,1] = bps

       bloc[1][1] *= bpmpi+bpi #sign wrong
       bloc[2][2] *= bpi #add pemdas
       bloc[3][3] *= bpi
       bloc[0][3] *= bpmpi #maybe ok
       bloc[3][0] *= bpmpi

       matrix_print_2d(rot_mat, 4, 'T')       
       matrix_print_2d(bloc, 4, 'bloc')       
       b0_rotate = np.einsum('ji,kj,km->im', rot_mat, bloc, rot_mat) 
       matrix_print_2d(b0_rotate, 4, 'b0_rotate')       

       #test_t45 = rot_mat*bpmpi
       #matrix_print_2d(test_t45, 4, 'test_t45')

       test_t5 = rot_mat*bpmpi+bpi
       matrix_print_2d(test_t5, 4, 'test_t5')

       test_rot = np.einsum('ji,jk->ik', rot_mat, rot_mat)
       matrix_print_2d(test_rot, 4, 'TtT')
       test_rot[0,0] = bss
       test_rot[0,1] *= bsp
       test_rot[1,0] *= bps
       test_rot[1,1] = test_rot[1,1]*bpmpi+bpi
       test_rot[2,2] = test_rot[2,2]*bpi
       test_rot[3,3] = test_rot[3,3]*bpi
       matrix_print_2d(test_rot, 4, 'TtBT')

       # Indexing rows and columns below may be backwards... -CL
       #bloc[1,1] *= (bpp-bpi)+bpi
       #bloc[1,2] *= (bpp-bpi)
       #bloc[1,3] *= (bpp-bpi)
       #bloc[2,1]  = bloc[1,2]
       #bloc[2,2] *= (bpp-bpi)+bpi
       #bloc[2,3] *= (bpp-bpi)
       #bloc[3,1]  = bloc[1,3]
       #bloc[3,2]  = bloc[2,3]
       #bloc[3,3] *= (bpp-bpi)+bpi

       #bloc[1,1] *= (bpp-bpi)+bpi
       #bloc[2,1] *= (bpp-bpi)
       #bloc[3,1] *= (bpp-bpi)
       #bloc[1,2]  = bloc[1,2]
       #bloc[2,2] *= (bpp-bpi)+bpi
       #bloc[3,2] *= (bpp-bpi)
       #bloc[1,3]  = bloc[1,3]
       #bloc[2,3]  = bloc[2,3]
       #bloc[3,3] *= (bpp-bpi)+bpi

       #bppbpi = bpp-bpi
    else:
       print('invalid combination of zi and zj')
       exit(-1)

    print('bloc:',bloc)

    return b0_rotate # bloc

