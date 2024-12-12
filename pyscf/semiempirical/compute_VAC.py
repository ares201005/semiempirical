#!/usr/bin/env python
#
#

import os, sys
import copy
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf.data.elements import _symbol
from pyscf.semiempirical import mopac_param
from .read_param import *
from .mndo_class import *
from .diatomic_overlap_matrix import *
from math import sqrt, atan, acos, sin, cos
write = sys.stdout.write

def get_rij_sij(mol, ia, ja):
    #Xij  = mol.atom_coord(ia) - mol.atom_coord(ja)
    #rij  = numpy.linalg.norm(Xij)
    Xij  = copy.copy(mol.xij[ia,ja])
    rij  = copy.copy(mol.pair_dist[ia,ja])
    sij  = Xij
    sij  *= 1/rij
    return rij, sij

def hcore_VAC(mol, nbas, atom_list_sorted, params):

    h_v = np.zeros((nbas, nbas))
    aoslices = mol.aoslice_by_atom()

    #light-light
    for ia in atom_list_sorted[0]:
        for ja in atom_list_sorted[0]:
            if ja > ia: 
                i0, i1 = aoslices[ia,2:]
                j0, j1 = aoslices[ja,2:]
                #rij and sij should be precomputed and sorted
                rij, sij = get_rij_sij(mol, ia, ja)
                e1b, e2a = compute_VAC_ll(mol.atom_charge(ia), mol.atom_charge(ja), rij, \
                    params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
                h_v[j0:j1,j0:j1] += e2a
                h_v[i0:i1,i0:i1] += e1b

    #light-heavy
    for ia in atom_list_sorted[0]:
        for ja in atom_list_sorted[1]:
            i0, i1 = aoslices[ia,2:]
            j0, j1 = aoslices[ja,2:]
            #rij and sij should be precomputed and sorted
            rij, sij = get_rij_sij(mol, ia, ja)
            e1b, e2a = compute_VAC_lh(mol.atom_charge(ia), mol.atom_charge(ja), rij, sij, \
                    params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
            h_v[j0:j1,j0:j1] += e2a
            h_v[i0:i1,i0:i1] += e1b

    #heavy-heavy
    for ia in atom_list_sorted[1]:
        for ja in atom_list_sorted[1]:
            if ja > ia: 
                i0, i1 = aoslices[ia,2:]
                j0, j1 = aoslices[ja,2:]
                #rij and sij should be precomputed and sorted
                rij, sij = get_rij_sij(mol, ia, ja)
                e1b, e2a = compute_VAC_hh(mol.atom_charge(ia), mol.atom_charge(ja), rij, sij, \
                    params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
                h_v[j0:j1,j0:j1] += e2a
                h_v[i0:i1,i0:i1] += e1b

    return h_v

def compute_VAC(zi, zj, xi, xj, am, ad, aq, dd, qq, tore):

    Xij  = numpy.subtract(xi, xj)
    rij  = numpy.linalg.norm(Xij)
    sij  = Xij
    sij  *= 1/rij

    if zi > 2 and zj > 2:
        return compute_VAC_hh(zi, zj, rij, sij, am, ad, aq, dd, qq, tore)
    elif zi == 1 and zj > 2:
        return compute_VAC_lh(zi, zj, rij, sij, am, ad, aq, dd, qq, tore)
    elif zi > 2 and zj == 1:
        return compute_VAC_hl(zi, zj, rij, sij, am, ad, aq, dd, qq, tore)
    elif zi == 1 and zj == 1:
        return compute_VAC_ll(zi, zj, rij, am, ad, aq, dd, qq, tore)
    else:
        print("not sure how to compute VAC's for this case, zi:", zi, "zj:", zj)
        exit(-1)

def compute_VAC_hh(zi, zj, rij, sij, am, ad, aq, dd, qq, tore):

    #print("calling compute_VAC_hh")
    rij2 = rij*rij
    T2 = numpy.array([[1,0,0,0],[0,-sij[0],0,0],[0,-sij[1],0,0],[0,-sij[2],0,0],
                      [0,0,sij[0]*sij[0],1-sij[0]*sij[0]],[0,0,sij[0]*sij[1],-sij[0]*sij[1]], [0,0,sij[0]*sij[2],-sij[0]*sij[2]],
                      [0,0,sij[1]*sij[1],1-sij[1]*sij[1]],[0,0,sij[1]*sij[2],-sij[1]*sij[2]],
                      [0,0,sij[2]*sij[2],1-sij[2]*sij[2]]])                        

    da = dd[zi] 
    db = dd[zj] 
    qa = qq[zi]*2.0
    qb = qq[zj]*2.0 

    ri   = numpy.zeros(8)
    core = numpy.zeros(8)
    aee = .5 / am[zi] + 0.5/am[zj] 
    aee *= aee
    ri[0] = 1.0 / sqrt(rij2 + aee)

    ade = .5 / ad[zi] + 0.5/am[zj] 
    ade *= ade
    aqe = .5 / aq[zi] + 0.5/am[zj] 
    aqe *= aqe
    ri[1] = 0.5 * (-1/sqrt(pow(rij + da, 2.0)+ade) + 1/sqrt(pow(rij-da, 2.0)+ade))
    ri[2] = ri[0] + 0.25 * ( 1/sqrt(pow(rij + qa, 2.0)+aqe) + 1/sqrt(pow(rij - qa, 2.0)+aqe)) - 0.5 * 1.0/sqrt(rij2 + aqe)
    ri[3] = ri[0] + 0.5  * (1.0/sqrt(rij2 + aqe + qa*qa) -1.0/sqrt(rij2 + aqe))
    #print("ri:", ri[0:4])
    core[0] = -tore[zj] * ri[0]
    core[1] = -tore[zj] * ri[1]
    core[2] = -tore[zj] * ri[2]
    core[3] = -tore[zj] * ri[3]
    e1b_ut = numpy.einsum('ij,j->i',T2, core[0:4])
    e1b = np.zeros((4,4))
    e1b[numpy.triu_indices(4)] = e1b_ut
    e1b = e1b + e1b.transpose() - numpy.diag(numpy.diag(e1b))

    if zj == zi: 
       e2a = numpy.copy(e1b)
       for i in range(1,4):
          e2a[0,i] *= -1.0
          e2a[i,0] *= -1.0
    elif zj != zi:
       aed = .5 / am[zi] + 0.5/ad[zj] 
       aed *= aed
       aeq = .5 / am[zi] + 0.5/aq[zj] 
       aeq *= aeq
       ri[4] = 0.5 * (1/sqrt(pow(rij + db, 2.0)+aed) - 1/sqrt(pow(rij-db, 2.0)+aed))
       ri[5] = ri[0] + 0.25 * ( 1/sqrt(pow(rij + qb, 2.0)+aeq) + 1/sqrt(pow(rij - qb, 2.0)+aeq)) - 0.5 * 1.0/sqrt(rij2 + aeq)
       ri[6] = ri[0] + 0.5  * (1.0/sqrt(rij2 + aeq + qb*qb) -1.0/sqrt(rij2 + aeq))
       core[4] = -tore[zi] * ri[0]
       core[5] = -tore[zi] * ri[4]
       core[6] = -tore[zi] * ri[5]
       core[7] = -tore[zi] * ri[6]
       #print("new core: ", core[4], core[5], core[6], core[7])
       e2a_ut = numpy.einsum('ij,j->i',T2, core[4:8])
       e2a = np.zeros((4,4))
       e2a[numpy.triu_indices(4)] = e2a_ut
       e2a = e2a + e2a.transpose() - numpy.diag(numpy.diag(e2a))
    return e1b, e2a

def compute_VAC_hl(zi, zj, rij, sij, am, ad, aq, dd, qq, tore):

    #print("calling compute_VAC_hl")
    rij2 = rij*rij
    T2 = numpy.array([[1,0,0,0],[0,-sij[0],0,0],[0,-sij[1],0,0],[0,-sij[2],0,0],
                      [0,0,sij[0]*sij[0],1-sij[0]*sij[0]],[0,0,sij[0]*sij[1],-sij[0]*sij[1]], [0,0,sij[0]*sij[2],-sij[0]*sij[2]],
                      [0,0,sij[1]*sij[1],1-sij[1]*sij[1]],[0,0,sij[1]*sij[2],-sij[1]*sij[2]],
                      [0,0,sij[2]*sij[2],1-sij[2]*sij[2]]])                        

    da = dd[zi] 
    qa = qq[zi]*2.0

    ri   = numpy.zeros(8)
    core = numpy.zeros(8)
    aee = .5 / am[zi] + 0.5/am[zj] 
    aee *= aee
    ri[0] = 1.0 / sqrt(rij2 + aee)

    ade = .5 / ad[zi] + 0.5/am[zj] 
    ade *= ade
    aqe = .5 / aq[zi] + 0.5/am[zj] 
    aqe *= aqe
    ri[1] = 0.5 * (-1/sqrt(pow(rij + da, 2.0)+ade) + 1/sqrt(pow(rij-da, 2.0)+ade))
    ri[2] = ri[0] + 0.25 * ( 1/sqrt(pow(rij + qa, 2.0)+aqe) + 1/sqrt(pow(rij - qa, 2.0)+aqe)) - 0.5 * 1.0/sqrt(rij2 + aqe)
    ri[3] = ri[0] + 0.5  * (1.0/sqrt(rij2 + aqe + qa*qa) -1.0/sqrt(rij2 + aqe))
    #print("ri:", ri[0:4])
    core[0] = -tore[zj] * ri[0]
    core[1] = -tore[zj] * ri[1]
    core[2] = -tore[zj] * ri[2]
    core[3] = -tore[zj] * ri[3]
    e1b_ut = numpy.einsum('ij,j->i',T2, core[0:4])
    e1b = np.zeros((4,4))
    e1b[numpy.triu_indices(4)] = e1b_ut
    e1b = e1b + e1b.transpose() - numpy.diag(numpy.diag(e1b))

    e2a = np.zeros((1,1))
    e2a[0,0] = -tore[zi] * ri[0]

    return e1b, e2a

def compute_VAC_lh(zi, zj, rij, sij, am, ad, aq, dd, qq, tore):

    #print("calling compute_VAC_lh")
    rij2 = rij*rij
    T2 = numpy.array([[1,0,0,0],[0,-sij[0],0,0],[0,-sij[1],0,0],[0,-sij[2],0,0],
                      [0,0,sij[0]*sij[0],1-sij[0]*sij[0]],[0,0,sij[0]*sij[1],-sij[0]*sij[1]], [0,0,sij[0]*sij[2],-sij[0]*sij[2]],
                      [0,0,sij[1]*sij[1],1-sij[1]*sij[1]],[0,0,sij[1]*sij[2],-sij[1]*sij[2]],
                      [0,0,sij[2]*sij[2],1-sij[2]*sij[2]]])                        

    db = dd[zj] 
    qb = qq[zj]*2.0 

    ri   = numpy.zeros(8)
    core = numpy.zeros(8)
    aee = .5 / am[zi] + 0.5/am[zj] 
    aee *= aee
    ri[0] = 1.0 / sqrt(rij2 + aee)

    core[0] = - tore[zj] * ri[0]
    e1b = np.zeros((1,1))
    e1b[0,0] = core[0]
    #print("e1b new:", e1b)

    aed = .5 / am[zi] + 0.5/ad[zj] 
    aed *= aed
    aeq = .5 / am[zi] + 0.5/aq[zj] 
    aeq *= aeq
    ri[4] = 0.5 * (1/sqrt(pow(rij + db, 2.0)+aed) - 1/sqrt(pow(rij-db, 2.0)+aed))
    ri[5] = ri[0] + 0.25 * ( 1/sqrt(pow(rij + qb, 2.0)+aeq) + 1/sqrt(pow(rij - qb, 2.0)+aeq)) - 0.5 * 1.0/sqrt(rij2 + aeq)
    ri[6] = ri[0] + 0.5  * (1.0/sqrt(rij2 + aeq + qb*qb) -1.0/sqrt(rij2 + aeq))
    core[4] = -tore[zi] * ri[0]
    core[5] = -tore[zi] * ri[4]
    core[6] = -tore[zi] * ri[5]
    core[7] = -tore[zi] * ri[6]
    #print("new core: ", core[4], core[5], core[6], core[7])
    e2a_ut = numpy.einsum('ij,j->i',T2, core[4:8])
    e2a = np.zeros((4,4))
    e2a[numpy.triu_indices(4)] = e2a_ut
    e2a = e2a + e2a.transpose() - numpy.diag(numpy.diag(e2a))

    return e1b, e2a

def compute_VAC_ll(zi, zj, rij, am, ad, aq, dd, qq, tore):

    #print("calling compute_VAC_ll")
    rij2 = rij*rij
    ri   = numpy.zeros(8)
    core = numpy.zeros(8)
    aee = .5 / am[zi] + 0.5/am[zj] 
    aee *= aee
    ri[0] = 1.0 / sqrt(rij2 + aee)

    core[0] = - tore[zj] * ri[0]
    e1b = np.zeros((1,1))
    e1b[0,0] = core[0]

    e2a = np.zeros((1,1))
    e2a[0,0] = core[0]

    return e1b, e2a
