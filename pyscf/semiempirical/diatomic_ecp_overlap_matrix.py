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

def _make_mndo_mol_ecp(mol,params):
    assert(not mol.has_ecp())
    def make_sqm_basis_ecp(n, l, charge, zeta): 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        basisfile = dir_path+'/basis-ecp_om2.dat'
        symb = _std_symbol(charge)
        sqm_basis = gto.basis.load(basisfile,symb)

        sto3g_e = np.array([2.227660584, 0.4057711562, 0.1098175104])
        sto3g_c = np.array([0.1543289673, 0.5353281423, 0.4446345422])
        scaled_e = np.zeros(3)
        scaled_c = np.zeros(3)
        for index in range(len(sto3g_e)):
            scaled_e[index] = sto3g_e[index]*zeta**2
            scaled_c[index] = sto3g_c[index]*(2*scaled_e[index]/np.pi)**0.75
        #es_cs = np.array([basval for basval in sqm_basis[l][1:]])
        #es = es_cs[:,0]
        #cs = es_cs[:,1]
        #return [l] + [(e*zeta**2, c) for e, c in zip(es, cs)]
        print('Renorm Scaled')
        if l == 0:
            print(f'S es\t cs')
        else:
            print('P es\t cs')
        for idx in range(len(scaled_e)):
            print(f'{scaled_e[idx]:>8.5f} {scaled_c[idx]:>8.5f}')
        for e, c in zip(scaled_e,scaled_c):
            print(f'{e*zeta**2:>8.5f} {c:>8.5f}')
        return [l] + [(e, c) for e, c in zip(scaled_e, scaled_c)]
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
        n = principle_quantum_number(charge)
        l = 0
        sto_6g_function = make_sqm_basis_ecp(n, l, charge, params.zeta_ecp[charge])
        basis = [sto_6g_function]

        if charge > 2:  # include p functions
            l = 1
            sto_6g_function = make_sqm_basis_ecp(n, l, charge, params.zeta_ecp[charge])
            basis.append(sto_6g_function)

        basis_set[_symbol(int(charge))] = basis

    mndo_mol.basis = basis_set
    z_eff = mopac_param.CORE[atom_charges]
    mndo_mol.nelectron = int(z_eff.sum() - mol.charge)
    mndo_mol.build(0, 0)
    return mndo_mol

def diatomic_ecp_overlap_matrix(mol,params,atom_charges):
    mndo_mol_ecp = copy.copy(mol)
    mol_ecp = _make_mndo_mol_ecp(mndo_mol_ecp,params)
    ovlp2e = gto.intor_cross('int1e_ovlp', mol, mol_ecp)
    #matrix_print_2d(ovlp2e, 9, 'Overlap2 1e')
    #print(f'mol._bas: {mol._bas}')
    #print(f'mol._env: {mol._env}')
    #print(f'mol_ecp._bas: {mol_ecp._bas}')
    #print(f'mol_ecp._env: {mol_ecp._env}')

    #for i in range(mol.nbas):
    #    print('shell %d on atom %d l = %s has %d contracted GTOs' %
    #        (i, mol.bas_atom(i), mol.bas_angular(i), mol.bas_nctr(i)))
    return ovlp2e
