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

def resonance_integral(betai, betaj, alphai, alphaj, rij):
    return 0.5*(betai+betaj)*np.sqrt(rij)*np.exp(-(alphai+alphaj)*rij**2)

def diatomic_resonance_matrix(ia, ja, zi, zj, xij, rij, params): #generalize -CL ***
    ''' Need to add pi parameters!!! -CL '''
    if zi == 1 and zj == 1: # first row - first row
       jcall = 2 
       bloc = np.zeros((1,1))
    elif (zi > 1 and zj == 1) or (zi == 1 and zj > 1): # first row - second row
       jcall = 3
       bloc = np.zeros((4,1))
    elif zi > 1 and zj > 1: # second row - second row #make else? -CL
       jcall = 4 
       bloc = np.zeros((4,4))
    else:
       print('invalid combination of zi and zj')
       exit(-1)
    xy = np.linalg.norm(xij[...,:2])
    if xij[2] > 0: tmp = 1.0
    elif xij[2] < 0: tmp = -1.0
    else: tmp = 0.0

    ca = cb = tmp
    sa = sb = 0.0
    if xy > 1.0e-10:
       ca = xij[0]/xy
       cb = xij[2]
       sa = xij[1]/xy
       sb = xy

    sasb = sa*sb
    sacb = sa*cb
    casb = ca*sb
    cacb = ca*cb

    zetas = np.array([params.zeta_s[zi], params.zeta_s[zj]])
    zetap = np.array([params.zeta_p[zi], params.zeta_p[zj]]) #do we need zeta below? -CL

    bloc[0,0] = 1.0
    nt = zi + zj
    if (nt == 8) or (nt == 9):
        bloc[1,0] = ca*sb
        bloc[2,0] = sa*sb
        bloc[3,0] = cb
        if (zi == 7) or (zi == 8):
            bss = resonance_integral(params.beta_sh[zi], params.beta_s[zj], params.alpha_sh[zi], params.alpha_s[zj], rij)
            bloc[0,0] *= bss
            bsp = resonance_integral(params.beta_ph[zi], params.beta_s[zj], params.alpha_ph[zi], params.alpha_s[zj], rij)
            bloc[1:4,0] *= bsp
        elif (zj == 7) or (zj ==8):
            bss = resonance_integral(params.beta_s[zi], params.beta_sh[zj], params.alpha_s[zi], params.alpha_sh[zj], rij)
            bloc[0,0] *= bss
            bsp = resonance_integral(params.beta_s[zi], params.beta_ph[zj], params.alpha_s[zi], params.alpha_ph[zj], rij)
            bloc[1:4,0] *= bsp
    else:   
        bss = resonance_integral(params.beta_s[zi], params.beta_s[zj], params.alpha_s[zi], params.alpha_s[zj], rij)
        bloc[0,0] *= bss
        if jcall == 3: # sp
            bloc[1,0] = ca*sb
            bloc[2,0] = sa*sb
            bloc[3,0] = cb
            bsp = resonance_integral(params.beta_s[zi], params.beta_p[zj], params.alpha_s[zi], params.alpha_p[zj], rij)
            bloc[1:4,0] *= bsp
        elif jcall == 4: # pp
            bloc[1,0] = S211*ca*sb
            bloc[2,0] = S211*sa*sb
            bloc[3,0] = S211*cb
            bloc[0,1] = -S121*casb
            bloc[0,2] = -S121*sasb
            bloc[0,3] = -S121*cb

            bloc[1,1] = -casb**2+(cacb**2+sa**2)
            bloc[1,2] = -casb*sasb+(cacb*sacb-sa*ca)
            bloc[1,3] = -casb*cb-cacb*sb
            bloc[2,1] = -sasb*casb+(sacb*cacb-ca*sa)
            bloc[2,2] = -sasb**2+(sacb**2+ca**2)
            bloc[2,3] = -sasb*cb-sacb*sb
            bloc[3,1] = -cb*casb-sb*cacb
            bloc[3,2] = -cb*sasb-sb*sacb
            bloc[3,3] = -cb**2+sb**2

            bsp = resonance_integral(params.beta_s[zi], params.beta_p[zj], params.alpha_s[zi], params.alpha_p[zj], rij)
            bloc[0,1:4] *= bsp 
            bpp = resonance_integral(params.beta_p[zi], params.beta_p[zj], params.alpha_p[zi], params.alpha_p[zj], rij)
            bloc[1:4,1:4] *= bpp

    print('bloc:',bloc)

    return bloc

