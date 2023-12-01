#!/usr/bin/env python
# flake8: noqa

'''
whatever
'''

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
                print('C(I)', ci)
                for jdx, cj in enumerate(cs[0:idx]):
                    print(f'idx: {idx}, jdx: {jdx}')
                    sqrtfac = (2*np.sqrt(us_es[idx]*us_es[jdx])/(us_es[idx]+us_es[jdx]))**(l+1.5)
                    norm += 2*ci*cj*sqrtfac
                    print('E(I)', us_es[idx])
                    print('E(J)', us_es[jdx])
        normco = 1/np.sqrt(norm)
        print('Normalized cs')
        for idx, ci in enumerate(cs):
            cs[idx] *= normco
            print(f'{cs[idx]}')
    return cs

def scale_bf(us_es, cs, zeta, l):
    gfac = (2/np.pi)**0.75
    if l == 0:
        for idx in range(len(us_es)):
            cs[idx] *= gfac*(us_es[idx]*zeta**2)**0.75
    if l == 1:
        for idx in range(len(us_es)):
            cs[idx] *= gfac*2*(us_es[idx]*zeta**2)**1.25
    print('Scaled bf')
    for c in cs:
        print(f'{c}')
    return cs

def ecp_ovlp(rij, orb_ex, orb_c, ecp_ex, ecp_c, l):
    ovlp = 0.0
    if l == 0:
        for i in range(len(orb_ex)):
            fac = (np.pi/(orb_ex[i]+ecp_ex[i]))**1.5
            gauss = np.exp(-(orb_ex[i]*ecp_ex[i])/(orb_ex[i]+ecp_ex[i])*rij**2)
            ovlp += orb_c[i]*ecp_c[i]*fac*gauss #s-ecp
    elif l == 1:
        for i in range(len(orb_ex)):
            fac = (np.pi/(orb_ex[i]+ecp_ex[i]))**1.5
            gauss = np.exp(-(orb_ex[i]*ecp_ex[i])/(orb_ex[i]+ecp_ex[i])*rij**2)
            ovlp += orb_c[i]*ecp_c[i]*ecp_ex[i]*rij/(orb_ex[i]+ecp_ex[i])*fac*gauss #p-ecp
    return ovlp

def _make_mndo_mol_ecp(mol,params,rij):
    assert(not mol.has_ecp())
    def make_sqm_basis_ecp(n, l, charge, zecp, zeta): 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        basisfile = dir_path+'/basis-ecp_om2.dat'
        symb = _std_symbol(charge)
        sqm_basis = gto.basis.load(basisfile,symb)

        sto3g_e = np.array([2.227660584, 0.4057711562, 0.1098175104])
        sto3g_c = np.array([0.1543289673, 0.5353281423, 0.4446345422])
        scaled_e = np.zeros(3)
        scaled_c = np.zeros(3)
        for index in range(len(sto3g_e)):
            scaled_e[index] = sto3g_e[index]*zecp**2
            scaled_c[index] = sto3g_c[index]*(2*scaled_e[index]/np.pi)**0.75
        print(f'scaled_e: {scaled_e}\nscaled_c: {scaled_c}')
        es_cs = np.array([basval for basval in sqm_basis[l][1:]])
        es = es_cs[:,0]
        cs = es_cs[:,1]
        #return [l] + [(e*zeta**2, c) for e, c in zip(es, cs)]

        print(' ')
        if l == 0:
            print(f'S es\t cs')
        else:
            print('P es\t cs')
        for idx in range(len(es)):
            print(f'{es[idx]:>8.5f} {cs[idx]:>8.5f}')
        print('Scaled es Scaled cs')
        for e, c in zip(es,cs):
            print(f'{e*zeta**2:>8.5f} {c:>8.5f}')
        us_es = es_cs[:,0]
        us_cs = es_cs[:,1]
        s_es = copy.copy(us_es)
        print('us_es',us_es)
        for idx, e in enumerate(s_es):
            s_es[idx] = e*zeta**2
        ncs = normalize_gaussians(us_es, us_cs, l)
        s_cs = scale_bf(us_es, ncs, zeta, l)
        #return [l] + [(e, c) for e, c in zip(scaled_e, scaled_c)]
        return s_es, s_cs, scaled_e, scaled_c, l

    def principle_quantum_number(charge):
        if charge < 3:
            return 1
        elif charge < 10:
            return 2
        elif charge < 18:
            return 3
        else:
            return 4

    mndo_mol = copy.copy(mol)
    atom_charges = mndo_mol.atom_charges()
    atom_types = set(atom_charges)
    basis_set = {}
    for charge in atom_types:
        if charge > 1:
            n = principle_quantum_number(charge)
            l = 0
            #sto_6g_function = make_sqm_basis_ecp(n, l, charge, params.zeta_ecp[charge], params.zeta_s[charge])
            #basis = [sto_6g_function]
            orb_ex, orb_c, ecp_ex, ecp_c, l = make_sqm_basis_ecp(n, l, charge, params.zeta_ecp[charge], params.zeta_s[charge])
            ovlp = ecp_ovlp(rij, orb_ex, orb_c, ecp_ex, ecp_c, l)
            print('ovlp:',ovlp)
            if charge > 2:  # include p functions
                l = 1
                #sto_6g_function = make_sqm_basis_ecp(n, l, charge, params.zeta_ecp[charge], params.zeta_p[charge])
                #basis.append(sto_6g_function)
                orb_ex, orb_c, ecp_ex, ecp_c, l = make_sqm_basis_ecp(n, l, charge, params.zeta_ecp[charge], params.zeta_p[charge])
                ovlp = ecp_ovlp(rij, orb_ex, orb_c, ecp_ex, ecp_c, l)
                print('ovlp:',ovlp)
            #basis_set[_symbol(int(charge))] = basis
        print('l = ',l)

    #mndo_mol.basis = basis_set
    #z_eff = mopac_param.CORE[atom_charges]
    #mndo_mol.nelectron = int(z_eff.sum() - mol.charge)
    #mndo_mol.build(0, 0)
    #return mndo_mol
    #return basis_set

def diatomic_ecp_overlap_matrix(mol,params,atom_charges, rij):
    mndo_mol_ecp = copy.copy(mol)
    mol_ecp = _make_mndo_mol_ecp(mndo_mol_ecp,params,rij)
    #ovlp2e = gto.intor_cross('int1e_ovlp', mol, mol_ecp)
    #matrix_print_2d(ovlp2e, 9, 'Overlap2 1e')
    #print(f'mol._bas: {mol._bas}')
    #print(f'mol._env: {mol._env}')
    #print(f'mol_ecp._bas: {mol_ecp._bas}')
    #print(f'mol_ecp._env: {mol_ecp._env}')

    #for i in range(mol.nbas):
    #    print('shell %d on atom %d l = %s has %d contracted GTOs' %
    #        (i, mol.bas_atom(i), mol.bas_angular(i), mol.bas_nctr(i)))
    #return ovlp2e
