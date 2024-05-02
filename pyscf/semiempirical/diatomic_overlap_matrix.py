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
from .compute_hcore_overlap import *

def diatomic_overlap_matrix_ori(ia, ja, zi, zj, xij, rij, params): #generalize -CL ***
    #Plan to generalize: PYSEQM uses zi, zj, xij, rij as arrays and builds with jcall using arrays. 
    #Either call diat_overlap multiple times or pass zi zj arrays. 
    #if zi == 8 and zj == 8: jcall = 4
    if zi == 1 and zj == 1: # first row - first row
       jcall = 2 
       di = np.zeros((1,1))
    elif (zi > 1 and zj == 1) or (zi == 1 and zj > 1): # first row - second row
       jcall = 3
       di = np.zeros((4,1)) # original was 4,1
    elif zi > 1 and zj > 1: # second row - second row #make else? -CL
       jcall = 4 
       di = np.zeros((4,4))
    else:
       print('invalid combination of zi and zj')
       exit(-1)
    #print("jcall:", jcall)
    #print('xij', xij, ia, ja)
    #xy = math.sqrt(xij[ia]*xij[ia] + xij[ja]*xij[ja])
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
    #print("ca, cb, sa, sb=", ca, cb, sa, sb)

    sasb = sa*sb
    sacb = sa*cb
    casb = ca*sb
    cacb = ca*cb

    #print("S", S111, S211, S121, S221, S222)
    #print("ca, cb, sa, sb=", ca, cb, sa, sb) 
    #print('sasb, sacb, casb, cacb',sasb, sacb, casb, cacb)

    zetas = np.array([params.zeta_s[zi], params.zeta_s[zj]])
    #print("zetas:", zetas, zi, zj)
    zetap = np.array([params.zeta_p[zi], params.zeta_p[zj]]) #do we need zeta below? -CL
    zeta = np.array([[zetas[0], zetap[0]], [zetas[1], zetap[1]]]) #np.concatenate(zetas.unsequeeze(1), zetap.unsequeeze(1))
    #print("zeta:", zeta, zeta[0], zeta[1], zeta[0,0], zeta[1,0], zeta[0,1], zeta[1,1])
    #print('Full Zeta:', zeta)
    #if zi == 8 and zj == 8:
    beta = np.array([[params.beta_s[zi],params.beta_p[zi]],[params.beta_s[zj],params.beta_p[zj]]]) / 27.211386
    A111,B111 = SET(rij, zeta[0,0],zeta[1,0])
    #Probably need to make SXX arrays dependent on jcall value. -CL ***

    if jcall == 2:
       S111 = math.pow(zeta[0,0]* zeta[1,0]* rij**2,1.5)* \
                  (A111[2]*B111[0]-B111[2]*A111[0])/4.0
    elif jcall == 3:
       S111 = math.pow(zeta[1,0],1.5)* math.pow(zeta[0,0],2.5)*rij**4 * \
                  (A111[3]*B111[0]-B111[3]*A111[0]+ \
                   A111[2]*B111[1]-B111[2]*A111[1]) / (math.sqrt(3.0)*8.0)
    elif jcall == 4:
       S111 = math.pow(zeta[1,0]*zeta[0,0],2.5)* rij**5 * \
                          (A111[4]*B111[0]+B111[4]*A111[0]-2.0*A111[2]*B111[2])/48.0
    #print("S111:", S111)
    di[0,0] = S111
    if jcall == 3:
       A211,B211 = SET(rij, zeta[0,1],zeta[1,0])
       #print('A211 zeta [0,1] [1,0]', zeta[0,1],zeta[1,0])
       S211 = math.pow(zeta[1,0],1.5)* math.pow(zeta[0,1],2.5)* rij**4 * \
                  (A211[2]*B211[0]-B211[2]*A211[0]+ \
                   A211[3]*B211[1]-B211[3]*A211[1])/8.0
       di[1,0] = S211*ca*sb
       di[2,0] = S211*sa*sb
       di[3,0] = S211*cb
       #print("S211:", S211)
    elif jcall == 4:
       A211,B211 = SET(rij, zeta[0,1],zeta[1,0])
       S211 = math.pow(zeta[1,0]* zeta[0,1],2.5)* rij**5 * \
                  (A211[3]*(B211[0]-B211[2]) \
                  -A211[1]*(B211[2]-B211[4]) \
                  +B211[3]*(A211[0]-A211[2]) \
                  -B211[1]*(A211[2]-A211[4])) \
                  /(16.0*math.sqrt(3.0))
       di[1,0] = S211*ca*sb
       di[2,0] = S211*sa*sb
       di[3,0] = S211*cb
       #print("S211:", S211)
    if jcall == 4:
       A121,B121 = SET(rij, zeta[0,0],zeta[1,1])
       #print('A121 zeta [0,0] [1,1]', zeta[0,0],zeta[1,1])
       S121 = math.pow(zeta[1,1]* zeta[0,0],2.5)* rij**5 * \
                  (A121[3]*(B121[0]-B121[2]) \
                  -A121[1]*(B121[2]-B121[4]) \
                  -B121[3]*(A121[0]-A121[2]) \
                  +B121[1]*(A121[2]-A121[4])) \
                  /(16.0*math.sqrt(3.0))
       #print("S121:", S121)
       di[0,1] = -S121*casb
       di[0,2] = -S121*sasb
       di[0,3] = -S121*cb
    if jcall == 4:
       A22,B22 = SET(rij, zeta[0,1],zeta[1,1]) #Can cause div by 0. Fix with if? -CL
       #print('A22 zeta [0,1] [1,1]', zeta[0,1],zeta[1,1])
       S221 = -math.pow(zeta[1,1]* zeta[0,1],2.5)* rij**5/16.0 * \
                  (B22[2]*(A22[4]+A22[0]) \
                  -A22[2]*(B22[4]+B22[0])) 
       S222 = 0.5*math.pow(zeta[1,1]* zeta[0,1],2.5)* rij**5/16.0 * \
                  (A22[4]*(B22[0]-B22[2]) \
                  -B22[4]*(A22[0]-A22[2]) \
                  -A22[2]*B22[0]+B22[2]*A22[0])
       di[1,1] = -S221*casb**2 \
                        +S222*(cacb**2+sa**2)
       di[1,2] = -S221*casb*sasb \
                        +S222*(cacb*sacb-sa*ca)
       di[1,3] = -S221*casb*cb \
                        -S222*cacb*sb
       di[2,1] = -S221*sasb*casb \
                        +S222*(sacb*cacb-ca*sa)
       di[2,2] = -S221*sasb**2 \
                        +S222*(sacb**2+ca**2)
       di[2,3] = -S221*sasb*cb \
                        -S222*sacb*sb
       di[3,1] = -S221*cb*casb \
                        -S222*sb*cacb
       di[3,2] = -S221*cb*sasb \
                        -S222*sb*sacb
       di[3,3] = -S221*cb**2 \
                        +S222*sb**2
       #print("S221:", S221)
       #print("S222:", S222)

    #print('jcall',jcall)
    #print("di:", di)

    di[0,0] *= (beta[0,0] + beta[1,0]) /2.0
    if jcall >= 3:
       di[1:4,0] *= (beta[0,1] + beta[1,0]) /2.0
    if jcall == 4:
       di[0,1:4] *= (beta[0,0] + beta[1,1]) /2.0
       di[1:4,1:4] *= (beta[0,1] + beta[1,1]) /2.0

    return di

