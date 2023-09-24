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

def diatomic_ecp_resonance_matrix(ia, ja, zi, zj, xij, rij, params): #generalize -CL ***
    if zi == 1 and zj == 1: # first row - first row
       jcall = 2 
       gecp = np.zeros((1,1))
    elif (zi > 1 and zj == 1): # i = Second Row, j = H
       jcall = 3
       gecp = np.zeros((4,1))
       gecp_ss = resonance_integral(params.beta_sh[zi], params.beta_ecp[zj], params.alpha_sh[zi], params.alpha_ecp[zj], rij)
       gecp_sp = resonance_integral(params.beta_ph[zi], params.beta_ecp[zj], params.alpha_ph[zi], params.alpha_ecp[zj], rij)
    elif (zi == 1 and zj > 1): # i = H, j = Second Row
       jcall = 3
       gecp = np.zeros((4,1))
       gecp_ss = resonance_integral(params.beta_s[zi], params.beta_ecp[zj], params.alpha_s[zi], params.alpha_ecp[zj], rij)
       gecp_sp = resonance_integral(params.beta_p[zi], params.beta_ecp[zj], params.alpha_p[zi], params.alpha_ecp[zj], rij) # Not needed? -CL
    elif zi > 1 and zj > 1: # second row - second row #make else? -CL
       jcall = 4 
       gecp = np.zeros((4,4))
       gecp_ss = resonance_integral(params.beta_s[zi], params.beta_ecp[zj], params.alpha_s[zi], params.alpha_ecp[zj], rij)
       gecp_sp = resonance_integral(params.beta_p[zi], params.beta_ecp[zj], params.alpha_p[zi], params.alpha_ecp[zj], rij)
       gecp_pp = resonance_integral(params.beta_p[zi], params.beta_ecp[zj], params.alpha_p[zi], params.alpha_ecp[zj], rij)
    #else:
    #   print('invalid combination of zi and zj')
    #   exit(-1)
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

    print('bs, becp, as, aecp', params.beta_s[zi], params.beta_ecp[zj], params.alpha_s[zi], params.alpha_ecp[zj])
    gecp[0,0] *= gecp_ss
    print('gecp_ss', gecp_ss)
    if jcall == 3: # sp
        gecp[1,0] = ca*sb
        gecp[2,0] = sa*sb
        gecp[3,0] = cb
        gecp[1:4,0] *= gecp_sp
    elif jcall == 4: # pp
        gecp[0,1] = -casb
        gecp[0,2] = -sasb
        gecp[0,3] = -cb
        gecp[1,0] = ca*sb
        gecp[2,0] = sa*sb
        gecp[3,0] = cb
        gecp[1,1] = -casb**2+(cacb**2+sa**2)
        gecp[1,2] = -casb*sasb+(cacb*sacb-sa*ca)
        gecp[1,3] = -casb*cb-cacb*sb
        gecp[2,1] = -sasb*casb+(sacb*cacb-ca*sa)
        gecp[2,2] = -sasb**2+(sacb**2+ca**2)
        gecp[2,3] = -sasb*cb-sacb*sb
        gecp[3,1] = -cb*casb-sb*cacb
        gecp[3,2] = -cb*sasb-sb*sacb
        gecp[3,3] = -cb**2+sb**2

        #gecp[0,1:4] *= gecp_sp 
        gecp[1:4,0] *= gecp_sp # ??? -CL
        gecp[1:4,1:4] *= gecp_pp

    print('gecp:',gecp)

    return gecp

