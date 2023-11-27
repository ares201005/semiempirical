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
from .matprint2d import *

def _make_mndo_mol_ecp(mol_ecp,params):
    assert(not mol_ecp.has_ecp())
    def make_sqm_basis(n, l, charge, zeta): 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        basisfile = dir_path+'/basis-ecp_om2.dat'
        symb = _std_symbol(charge)
        sqm_basis = gto.basis.load(basisfile,symb)
        es_cs = np.array([basval for basval in sqm_basis[l][1:]])
        es = es_cs[:,0]
        cs = es_cs[:,1]
        return [l] + [(e*zeta**2, c) for e, c in zip(es, cs)]

    def principle_quantum_number(charge):
        if charge < 3:
            return 1
        elif charge < 10:
            return 2
        elif charge < 18:
            return 3
        else:
            return 4

    mndo_mol_ecp = copy.copy(mol_ecp)
    atom_charges = mndo_mol_ecp.atom_charges()
    atom_types = set(atom_charges)
    ecp_set = {}
    for charge in atom_types:
        n = principle_quantum_number(charge)
        l = 0
        ecp_function = make_sqm_basis(n, l, charge, params.zeta_s[charge])
        print('S basis  function:',ecp_function)
        ecp_function = make_sqm_basis(n, l, charge, params.zeta_ecp[charge])
        print('ECP function:',ecp_function)
        basis = [ecp_function]
        ecp_set[_symbol(int(charge))] = basis

    mndo_mol_ecp.basis = ecp_set
    #z_eff = mopac_param.CORE[atom_charges]
    z_eff = params.tore[atom_charges]
    mndo_mol_ecp.nelectron = int(z_eff.sum() - mol_ecp.charge)
    mndo_mol_ecp.build(0,0)
    return mndo_mol_ecp

def diatomic_ecp_overlap_matrix(mol,params,atom_charges):
    mol_ecp = copy.copy(mol)
    mndo_mol_ecp = _make_mndo_mol_ecp(mol_ecp,params)
    ovlp1e = mndo_mol_ecp.intor("int1e_ovlp")
    matrix_print_2d(ovlp1e, 9, 'Overlap2 1e')
    return ovlp1e

