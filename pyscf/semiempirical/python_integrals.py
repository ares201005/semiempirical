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
from .compute_W import *
from math import sqrt, atan, acos, sin, cos
write = sys.stdout.write

def compute_W_ori(zi, zj, xi, xj, am, ad, aq, dd, qq, tore):

    old_pxpy_pxpy = 0
    if zi > 2 and zj > 2:
        return compute_W_hh(zi, zj, xi, xj, am, ad, aq, dd, qq, tore, old_pxpy_pxpy)
    elif zi == 1 and zj > 2:
        return compute_W_lh(zi, zj, xi, xj, am, ad, aq, dd, qq, tore)
    elif zi > 2 and zj == 1:
        return compute_W_hl(zi, zj, xi, xj, am, ad, aq, dd, qq, tore)
    elif zi == 1 and zj == 1:
        return compute_W_ll(zi, zj, xi, xj, am, ad, aq, dd, qq, tore)
    else:
        print("not sure how to compute W's for this case, zi:", zi, "zj:", zj)
        exit(-1)

def compute_VAC_ori(zi, zj, xi, xj, am, ad, aq, dd, qq, tore):

    print("calling compute_VAC")
    Xij  = numpy.subtract(xi, xj)
    rij  = numpy.linalg.norm(Xij)
    rij2 = rij * rij
    sij  = Xij
    sij  *= 1/rij
    #print("xi: ", xi, "xj:", xj)
    #print("Xij:", Xij, "rij: ", rij, "sij:", sij)
    theta = acos(sij[2]) 
    if abs(sij[0]) < 1e-8: 
       phi = 0.0
    else:
       phi = atan(sij[1]/sij[0])
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

    if zi == 1:
       core[0] = - tore[zj] * ri[0]
       e1b = core[0]
       #print("e1b new:", e1b)
    elif zi>2:
       #electron integrals
       ade = .5 / ad[zi] + 0.5/am[zj] 
       ade *= ade
       aqe = .5 / aq[zi] + 0.5/am[zj] 
       aqe *= aqe
       ri[1] = 0.5 * (-1/sqrt(pow(rij + da, 2.0)+ade) + 1/sqrt(pow(rij-da, 2.0)+ade))
       ri[2] = ri[0] + 0.25 * ( 1/sqrt(pow(rij + qa, 2.0)+aqe) + 1/sqrt(pow(rij - qa, 2.0)+aqe)) - 0.5 * 1.0/sqrt(rij2 + aqe)
       ri[3] = ri[0] + 0.5  * (1.0/sqrt(rij2 + aqe + qa*qa) -1.0/sqrt(rij2 + aqe))
       print("ri:", ri[0:4])
       core[0] = -tore[zj] * ri[0]
       core[1] = -tore[zj] * ri[1]
       core[2] = -tore[zj] * ri[2]
       core[3] = -tore[zj] * ri[3]
       e1b_ut = numpy.einsum('ij,j->i',T2, core[0:4])
       e1b = np.zeros((4,4))
       e1b[numpy.triu_indices(4)] = e1b_ut
       e1b = e1b + e1b.transpose() - numpy.diag(numpy.diag(e1b))
       #print("e1b_ut:", e1b_ut)
       #print("e1b new:", e1b)

    if zj == zi: 
       if zj == 1:
          e2a = core[0]
          #print("e2a new:", e2a)
       elif zj >= 2:
          e2a = numpy.copy(e1b)
          for i in range(1,4):
             e2a[0,i] *= -1.0
             e2a[i,0] *= -1.0
          #print("e2a new:", e2a)
    elif zj != zi:
       if zj == 1:
          e2a = -tore[zi] * ri[0]
          #print("e2a new:", e2a) 
       elif zj>2:
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
          #print("e2a new:", e2a)
    return e1b, e2a

