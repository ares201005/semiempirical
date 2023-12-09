#!/usr/bin/env python
# flake8: noqa

import os
import ctypes
import copy
import math
import numpy as np
import warnings
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf.data.elements import _symbol, _std_symbol
from pyscf.data.elements import _symbol
from pyscf.semiempirical import mopac_param
from .matprint2d import *

def normalize_gaussians(us_es, cs, l):
    norm = 0.0
    if len(cs) < 2:
        cs[0] = 1.0
    else:
        norm = cs[0]**2
        for idx, ci in enumerate(cs):
            if idx > 0: #Change to remove if
                norm += ci**2
                for jdx, cj in enumerate(cs[0:idx]):
                    sqrtfac = (2*np.sqrt(us_es[idx]*us_es[jdx])/(us_es[idx]+us_es[jdx]))**(l+1.5)
                    norm += 2*ci*cj*sqrtfac
        normco = 1/np.sqrt(norm)
        for idx, ci in enumerate(cs):
            cs[idx] *= normco
    return cs

def scale_bf(us_es, cs, zeta, l):
    gfac = (2/np.pi)**0.75
    if l == 0:
        for idx in range(len(us_es)):
            cs[idx] *= gfac*(us_es[idx]*zeta**2)**0.75
    if l == 1:
        for idx in range(len(us_es)):
            cs[idx] *= gfac*2*(us_es[idx]*zeta**2)**1.25
    return cs

def gaussian_terms(mol, l, zi, zj, zecp, zeta): 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    basisfile = dir_path+'/basis-ecp_om2.dat'
    symb = _std_symbol(zi)
    sqm_basis = gto.basis.load(basisfile,symb)
    #sqm_basis = mol.basis[symb] #orbital basis #problem: uses scaled bf

    sto3g_e = np.array([2.227660584, 0.4057711562, 0.1098175104])
    sto3g_c = np.array([0.1543289673, 0.5353281423, 0.4446345422])
    scaled_e = np.zeros(3)
    scaled_c = np.zeros(3)
    for index in range(len(sto3g_e)):
        scaled_e[index] = sto3g_e[index]*zecp**2
        scaled_c[index] = sto3g_c[index]*(2*scaled_e[index]/np.pi)**0.75
    es_cs = np.array([basval for basval in sqm_basis[l][1:]])
    us_es = es_cs[:,0]
    us_cs = es_cs[:,1]
    s_es = copy.copy(us_es)
    for idx, e in enumerate(us_es):
        s_es[idx] = e*zeta**2
    ncs = normalize_gaussians(us_es, us_cs, l)
    s_cs = scale_bf(us_es, ncs, zeta, l)
    return s_es, s_cs, scaled_e, scaled_c

def ecp_ovlp(mol, rij, orb_ex, orb_c, ecp_ex, ecp_c, l):
    #There are slight differences probably due to declaring XQQ in SPGTO2.f
    #assuming it is a REAL vs. python default double precision. -CL
    ovlp = 0.0
    if l == 0:
        for i in range(len(orb_ex)):
            for j in range(len(orb_ex)):
                fac = (np.pi/(orb_ex[i]+ecp_ex[j]))**1.5
                gauss = np.exp(-(orb_ex[i]*ecp_ex[j])/(orb_ex[i]+ecp_ex[j])*rij**2)
                ovlp += orb_c[i]*ecp_c[j]*fac*gauss #s-ecp
    elif l == 1:
        for i in range(len(orb_ex)):
            for j in range(len(orb_ex)):
                fac = (np.pi/(orb_ex[i]+ecp_ex[j]))**1.5
                gauss = np.exp(-(orb_ex[i]*ecp_ex[j])/(orb_ex[i]+ecp_ex[j])*rij**2)
                ovlp += orb_c[i]*ecp_c[j]*ecp_ex[j]*rij/(orb_ex[i]+ecp_ex[j])*fac*gauss #p-ecp
    return ovlp

def overlap_ecp(mol,zi,zj,params,rij):
    ovlpsam = 0.0
    ovlppam = 0.0
    ovlpsma = 0.0
    ovlppma = 0.0
    if zi < 3: #H-ECP
        l = 0
        orb_ex, orb_c, ecp_ex, ecp_c = gaussian_terms(mol, l, zi, zj, params.zeta_ecp[zj], params.zeta_s[zi]) #orb-ecp
        ovlpsma = ecp_ovlp(mol, rij, orb_ex, orb_c, ecp_ex, ecp_c, l)
        #print('ovlpS:',ovlps)
    elif zj < 3: #ECP-H
        l = 0
        orb_ex, orb_c, ecp_ex, ecp_c = gaussian_terms(mol, l, zj, zi, params.zeta_ecp[zi], params.zeta_s[zj]) #ecp-orb
        ovlpsam = ecp_ovlp(mol, rij, orb_ex, orb_c, ecp_ex, ecp_c, l)
        #print('ovlpS:',ovlps)
    else: #ECP-ORB and ORB-ECP
        orb_ex, orb_c, ecp_ex, ecp_c = gaussian_terms(mol, 0, zi, zj, params.zeta_ecp[zj], params.zeta_s[zi]) #orb-ecp s
        ovlpsma = ecp_ovlp(mol, rij, orb_ex, orb_c, ecp_ex, ecp_c, 0)
        #print('ovlpS:',ovlps)
        orb_ex, orb_c, ecp_ex, ecp_c = gaussian_terms(mol, 1, zi, zj, params.zeta_ecp[zj], params.zeta_p[zi]) #orb-ecp p
        ovlppma = ecp_ovlp(mol, rij, orb_ex, orb_c, ecp_ex, ecp_c, 1)
        #print('ovlpP:',ovlpp)
        orb_ex, orb_c, ecp_ex, ecp_c = gaussian_terms(mol, 0, zj, zi, params.zeta_ecp[zi], params.zeta_s[zj]) #ecp-orb s
        ovlpsam = ecp_ovlp(mol, rij, orb_ex, orb_c, ecp_ex, ecp_c, 0)
        #print('ovlpS:',ovlps)
        orb_ex, orb_c, ecp_ex, ecp_c = gaussian_terms(mol, 1, zj, zi, params.zeta_ecp[zi], params.zeta_p[zj]) #ecp-orb p
        ovlppam = ecp_ovlp(mol, rij, orb_ex, orb_c, ecp_ex, ecp_c, 1)
        #print('ovlpP:',ovlpp)

    return ovlpsam, ovlpsma, ovlppam, ovlppma

def diatomic_ecp_overlap_matrix(mol, zi, zj, params, rij):
    mol_ecp = copy.copy(mol)
    return overlap_ecp(mol_ecp,zi,zj,params,rij)


