#!/usr/bin/env python
# flake8: noqa

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
    gssam = 0.0
    gssma = 0.0
    gpsam = 0.0
    gpsma = 0.0
    nt = zi + zj

    if (zi > 1 and zj == 1): # second row - first
       #jcall = 3
       # Only s(ecp)-s will be non-zero
       gssam = resonance_integral(params.beta_s[zj], params.beta_ecp[zi], params.alpha_s[zj], params.alpha_ecp[zi], rij)
       return gssam, gssma, gpsam, gpsma

    elif (zi == 1 and zj > 1): # first row - second row
       #jcall = 3
       gssma = resonance_integral(params.beta_s[zi], params.beta_ecp[zj], params.alpha_s[zi], params.alpha_ecp[zj], rij)
       return gssam, gssma, gpsam, gpsma

    elif zi > 1 and zj > 1: 
       #jcall = 4 
       gssma = resonance_integral(params.beta_s[zi], params.beta_ecp[zj], params.alpha_s[zi], params.alpha_ecp[zj], rij) #Keep
       gpsma = resonance_integral(params.beta_p[zi], params.beta_ecp[zj], params.alpha_p[zi], params.alpha_ecp[zj], rij) #Keep
       gssam = resonance_integral(params.beta_s[zj], params.beta_ecp[zi], params.alpha_s[zj], params.alpha_ecp[zi], rij) #Keep
       gpsam = resonance_integral(params.beta_p[zj], params.beta_ecp[zi], params.alpha_p[zj], params.alpha_ecp[zi], rij) #Keep
       return gssam, gssma, gpsam, gpsma

    else:
       print('invalid combination of zi and zj')
       exit(-1)

