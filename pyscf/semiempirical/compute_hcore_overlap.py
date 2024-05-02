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

def hcore_overlap(mol, nbas, atom_list_sorted, params):

    h_s = np.zeros((nbas, nbas))
    aoslices = mol.aoslice_by_atom()

    #light-light
    for ia in atom_list_sorted[0]:
        for ja in atom_list_sorted[0]:
            if ja > ia: 
                i0, i1 = aoslices[ia,2:]
                j0, j1 = aoslices[ja,2:]
                #xij and rij should be precomputed and sorted
                xij, rij = get_xij_rij(mol, ia, ja)
                di = diatomic_overlap_matrix_ll(mol.atom_charge(ia),  mol.atom_charge(ja), xij, rij, params)
                h_s[i0:i1,j0:j1] = di 
                h_s[j0:j1,i0:i1] = di.T 

    #light-heavy
    for ia in atom_list_sorted[0]:
        for ja in atom_list_sorted[1]:
            i0, i1 = aoslices[ia,2:]
            j0, j1 = aoslices[ja,2:]
            #xij and rij should be precomputed and sorted
            xij, rij = get_xij_rij(mol, ia, ja)
            di = diatomic_overlap_matrix_lh(mol.atom_charge(ia),  mol.atom_charge(ja), xij, rij, params)
            h_s[i0:i1,j0:j1] = di 
            h_s[j0:j1,i0:i1] = di.T 

    #heavy-heavy
    for ia in atom_list_sorted[1]:
        for ja in atom_list_sorted[1]:
            if ja > ia: 
                i0, i1 = aoslices[ia,2:]
                j0, j1 = aoslices[ja,2:]
                #xij and rij should be precomputed and sorted
                xij, rij = get_xij_rij(mol, ia, ja)
                di = diatomic_overlap_matrix_hh(mol.atom_charge(ia),  mol.atom_charge(ja), xij, rij, params)
                h_s[i0:i1,j0:j1] = di 
                h_s[j0:j1,i0:i1] = di.T 

    return h_s

def diatomic_overlap_matrix(ia, ja, zi, zj, xij, rij, params):

    if zi == 1 and zj == 1: # first row - first row
       di = diatomic_overlap_matrix_ll(zi, zj, xij, rij, params)
    elif zi > 1 and zj == 1:
       di = diatomic_overlap_matrix_hl(zi, zj, xij, rij, params)
    elif zi == 1 and zj > 1:
       di = diatomic_overlap_matrix_lh(zi, zj, xij, rij, params)
    elif zi > 1 and zj > 1: # second row - second row #make else? -CL
       di = diatomic_overlap_matrix_hh(zi, zj, xij, rij, params)
    else:
       print('invalid combination of zi and zj')
       exit(-1)
    
    return di

def diatomic_overlap_matrix_ll(zi, zj, xij, rij, params): #generalize -CL ***

    print("calling diatomic_overlap_matrix_ll")
    di = np.zeros((1,1))
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
    zeta = np.array([[zetas[0], zetap[0]], [zetas[1], zetap[1]]]) #np.concatenate(zetas.unsequeeze(1), zetap.unsequeeze(1))
    beta = np.array([[params.beta_s[zi],params.beta_p[zi]],[params.beta_s[zj],params.beta_p[zj]]]) / 27.211386
    A111,B111 = SET(rij, zeta[0,0],zeta[1,0])

    S111 = math.pow(zeta[0,0]* zeta[1,0]* rij**2,1.5)* \
                  (A111[2]*B111[0]-B111[2]*A111[0])/4.0
    di[0,0] = S111
    di[0,0] *= (beta[0,0] + beta[1,0]) /2.0

    return di

def diatomic_overlap_matrix_hl(zi, zj, xij, rij, params):

    print("calling diatomic_overlap_matrix_hl")
    di = np.zeros((4,1)) # original was 4,1
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

    zetas = np.array([params.zeta_s[zi], params.zeta_s[zj]])
    zetap = np.array([params.zeta_p[zi], params.zeta_p[zj]]) #do we need zeta below? -CL
    zeta = np.array([[zetas[0], zetap[0]], [zetas[1], zetap[1]]]) #np.concatenate(zetas.unsequeeze(1), zetap.unsequeeze(1))
    beta = np.array([[params.beta_s[zi],params.beta_p[zi]],[params.beta_s[zj],params.beta_p[zj]]]) / 27.211386
    A111,B111 = SET(rij, zeta[0,0],zeta[1,0])

    S111 = math.pow(zeta[1,0],1.5)* math.pow(zeta[0,0],2.5)*rij**4 * \
                  (A111[3]*B111[0]-B111[3]*A111[0]+ \
                   A111[2]*B111[1]-B111[2]*A111[1]) / (math.sqrt(3.0)*8.0)
    di[0,0] = S111

    A211,B211 = SET(rij, zeta[0,1],zeta[1,0])
    #print('A211 zeta [0,1] [1,0]', zeta[0,1],zeta[1,0])
    S211 = math.pow(zeta[1,0],1.5)* math.pow(zeta[0,1],2.5)* rij**4 * \
               (A211[2]*B211[0]-B211[2]*A211[0]+ \
                A211[3]*B211[1]-B211[3]*A211[1])/8.0
    di[1,0] = S211*ca*sb
    di[2,0] = S211*sa*sb
    di[3,0] = S211*cb

    di[0,0] *= (beta[0,0] + beta[1,0]) /2.0
    di[1:4,0] *= (beta[0,1] + beta[1,0]) /2.0

    return di

def diatomic_overlap_matrix_lh(zi, zj, xij, rij, params):

    print("calling diatomic_overlap_matrix_hl")
    di = np.zeros((1,4)) # original was 4,1
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

    zetas = np.array([params.zeta_s[zi], params.zeta_s[zj]])
    zetap = np.array([params.zeta_p[zi], params.zeta_p[zj]]) #do we need zeta below? -CL
    zeta = np.array([[zetas[0], zetap[0]], [zetas[1], zetap[1]]]) #np.concatenate(zetas.unsequeeze(1), zetap.unsequeeze(1))
    beta = np.array([[params.beta_s[zi],params.beta_p[zi]],[params.beta_s[zj],params.beta_p[zj]]]) / 27.211386

    A111,B111 = SET(rij,zeta[1,0],zeta[0,0])

    S111 = math.pow(zeta[0,0],1.5)* math.pow(zeta[1,0],2.5)*rij**4 * \
                  (A111[3]*B111[0]-B111[3]*A111[0]+ \
                   A111[2]*B111[1]-B111[2]*A111[1]) / (math.sqrt(3.0)*8.0)
    di[0,0] = S111

    A211,B211 = SET(rij, zeta[1,1],zeta[0,0])
    #print('A211 zeta [0,1] [1,0]', zeta[0,1],zeta[1,0])
    #Yihan, 02/03/24. Somehow we need a negative sign for S211
    S211 = -math.pow(zeta[0,0],1.5)* math.pow(zeta[1,1],2.5)* rij**4 * \
               (A211[2]*B211[0]-B211[2]*A211[0]+ \
                A211[3]*B211[1]-B211[3]*A211[1])/8.0
    di[0,1] = S211*ca*sb
    di[0,2] = S211*sa*sb
    di[0,3] = S211*cb

    di[0,0] *= (beta[0,0] + beta[1,0]) /2.0
    di[0,1:4] *= (beta[1,1] + beta[0,0]) /2.0

    return di

def diatomic_overlap_matrix_hh(zi, zj, xij, rij, params): 

    print("calling diatomic_overlap_matrix_hh")

    di = np.zeros((4,4))
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

    zetas = np.array([params.zeta_s[zi], params.zeta_s[zj]])
    zetap = np.array([params.zeta_p[zi], params.zeta_p[zj]]) #do we need zeta below? -CL
    zeta = np.array([[zetas[0], zetap[0]], [zetas[1], zetap[1]]]) #np.concatenate(zetas.unsequeeze(1), zetap.unsequeeze(1))
    beta = np.array([[params.beta_s[zi],params.beta_p[zi]],[params.beta_s[zj],params.beta_p[zj]]]) / 27.211386
    A111,B111 = SET(rij, zeta[0,0],zeta[1,0])
    S111 = math.pow(zeta[1,0]*zeta[0,0],2.5)* rij**5 * \
                          (A111[4]*B111[0]+B111[4]*A111[0]-2.0*A111[2]*B111[2])/48.0
    di[0,0] = S111

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

    di[0,0] *= (beta[0,0] + beta[1,0]) /2.0
    di[0,1:4] *= (beta[0,0] + beta[1,1]) /2.0
    di[1:4,0] *= (beta[0,1] + beta[1,0]) /2.0
    di[1:4,1:4] *= (beta[0,1] + beta[1,1]) /2.0

    return di

def SET(rij, z1, z2):
    """
    get the A integrals and B integrals for diatom_overlap_matrix
    """
    #alpha, beta below is used in aintgs and bintgs, not the parameters for AM1/MNDO/PM3
    #rij: distance between atom i and j in atomic unit
    #print('SET:')
    #print('rij',rij)
    #print('z1',z1)
    #print('z2',z2)

    alp = 0.5*rij*(z1+z2)
    beta  = 0.5*rij*(z1-z2)
    #print("alpha, beta:", alp, beta)
    A = aintgs(alp)
    B = bintgs(beta)
    #print("A=", A)
    #print("B=", B)
    return A, B


def aintgs(x):
    """
    A integrals for diatom_overlap_matrix
    """
    a1 = np.exp(-x)/x
    a2 = a1 +a1/x
    a3 = a1 + 2.0*a2/x
    a4 = a1+3.0*a3/x
    a5 = a1+4.0*a4/x
    #print("a1, a2, a3, a4, a5:", a1, a2, a3, a4, a5)
    return np.array([a1, a2, a3, a4, a5])

def bintgs(x):
    """
    B integrals for diatom_overlap_matrix
    """
    absx = abs(x)
    #cond1 = absx > 0.5

    b1 = 2.0 
    b2 = 0.0 
    b3 = 2.0/3.0 
    b4 = 0.0 
    b5 = 2.0/5.0 

    if absx>0.5:

       b1 =   np.exp(x)/x - np.exp(-x)/x
       b2 = - np.exp(x)/x - np.exp(-x)/x + b1/x
       b3 =   np.exp(x)/x - np.exp(-x)/x + 2*b2/x
       b4 = - np.exp(x)/x - np.exp(-x)/x + 3*b3/x
       b5 =   np.exp(x)/x - np.exp(-x)/x + 4*b4/x
       #print("406, b1, b2, b3, b4, b5:", b1, b2, b3, b4, b5)

    elif absx > 1.0e-6: 

       b1 = 2.0     + x**2/3.0 + x**4/60.0 + x**6/2520.0
       b3 = 2.0/3.0 + x**2/5.0 + x**4/84.0 + x**6/3240.0
       b5 = 2.0/5.0 + x**2/7.0 + x**4/108.0 + x**6/3960.0

       b2 = -2.0/3.0*x - x**3/15.0 - x**5/420.0
       b4 = -2.0/5.0*x - x**3/21.0 - x**5/540.0
       #print("488, b1, b2, b3, b4, b5:", b1, b2, b3, b4, b5)

    return np.array([b1, b2, b3, b4, b5])

def get_xij_rij(mol, ia, ja):
    xij = mol.atom_coord(ja) - mol.atom_coord(ia)
    rij = np.linalg.norm(xij)
    xij /= rij
    return xij, rij
