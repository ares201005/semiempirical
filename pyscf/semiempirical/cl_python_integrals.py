#!/usr/bin/env python
#
#

'''
whatever
'''

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
from .matprint2d import *
write = sys.stdout.write

'''
Might need to transpose rotation matrix.
'''

def compute_W(zi, zj, xi, xj, am, ad, aq, dd, qq, tore):

    old_pxpy_pxpy = 1
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

def compute_W_hh(zi, zj, xi, xj, am, ad, aq, dd, qq, tore, old_pxpy_pxpy):

    w = numpy.zeros((10,10))
   
    Xij  = numpy.subtract(xi, xj)
    rij  = numpy.linalg.norm(Xij)
    r05  = 0.5 * rij
    rij2 = rij * rij
    sij  = Xij
    if rij > 0.0000001: sij  *= 1/rij
    else: return w

    da = dd[zi] 
    db = dd[zj] 
    qa  = qq[zi]
    qa2 = qq[zi]*2.0
    qb  = qq[zj]
    qb2 = qq[zj]*2.0 

    ama = 0.5 / am[zi]
    ada = 0.5 / ad[zi]
    aqa = 0.5 / aq[zi]
    amb = 0.5 / am[zj]
    adb = 0.5 / ad[zj]
    aqb = 0.5 / aq[zj]

    phi_a = [[0,   1.0, ama, 0, 0, -r05], #s
             [1,  -0.5, ada,  da, 0, -r05], #px
             [1,   0.5, ada, -da, 0, -r05], #px
             [3,  -0.5, ada, 0,  da, -r05], #py
             [3,   0.5, ada, 0, -da, -r05], #py
             [6,  -0.5, ada, 0, 0, -r05+da], #pz
             [6,   0.5, ada, 0, 0, -r05-da], #pz
             [2,   1.0, ama, 0, 0, -r05],   #dxx
             [2,  0.25, aqa, qa2, 0, -r05], #dxx
             [2,  -0.5, aqa,   0, 0, -r05], #dxx
             [2,  0.25, aqa,-qa2, 0, -r05], #dxx
             [4,  0.25, aqa,  qa, qa, -r05], #dxy
             [4, -0.25, aqa, -qa, qa, -r05], #dxy
             [4,  0.25, aqa, -qa,-qa, -r05], #dxy
             [4, -0.25, aqa,  qa,-qa, -r05], #dxy
             [7,  0.25, aqa,  qa, 0, -r05+qa], #dxz
             [7, -0.25, aqa, -qa, 0, -r05+qa], #dxz
             [7,  0.25, aqa, -qa, 0, -r05-qa], #dxz
             [7, -0.25, aqa,  qa, 0, -r05-qa], #dxz
             [5,   1.0, ama, 0, 0, -r05],   #dyy
             [5,  0.25, aqa, 0, qa2, -r05], #dyy
             [5,  -0.5, aqa, 0,   0, -r05], #dyy
             [5,  0.25, aqa, 0,-qa2, -r05], #dyy
             [8,  0.25, aqa, 0,  qa, -r05+qa], #dyz
             [8, -0.25, aqa, 0, -qa, -r05+qa], #dyz
             [8,  0.25, aqa, 0, -qa, -r05-qa], #dyz
             [8, -0.25, aqa, 0,  qa, -r05-qa], #dyz
             [9,   1.0, ama, 0, 0, -r05],     #dzz
             [9,  0.25, aqa, 0, 0, -r05+qa2], #dzz
             [9,  -0.5, aqa, 0, 0, -r05],     #dzz
             [9,  0.25, aqa, 0, 0, -r05-qa2], #dzz
            ]

    phi_b = [[0,   1.0, amb, 0, 0, r05], #s
             [1,  -0.5, adb,  db, 0, r05], #px
             [1,   0.5, adb, -db, 0, r05], #px
             [3,  -0.5, adb, 0,  db, r05], #py
             [3,   0.5, adb, 0, -db, r05], #py
             [6,  -0.5, adb, 0, 0, r05+db], #pz
             [6,   0.5, adb, 0, 0, r05-db], #pz
             [2,   1.0, amb, 0, 0, r05],   #dxx
             [2,  0.25, aqb, qb2, 0, r05], #dxx
             [2,  -0.5, aqb,   0, 0, r05], #dxx
             [2,  0.25, aqb,-qb2, 0, r05], #dxx
             [4,  0.25, aqb,  qb, qb, r05], #dxy
             [4, -0.25, aqb, -qb, qb, r05], #dxy
             [4,  0.25, aqb, -qb,-qb, r05], #dxy
             [4, -0.25, aqb,  qb,-qb, r05], #dxy
             [7,  0.25, aqb,  qb, 0, r05+qb], #dxz
             [7, -0.25, aqb, -qb, 0, r05+qb], #dxz
             [7,  0.25, aqb, -qb, 0, r05-qb], #dxz
             [7, -0.25, aqb,  qb, 0, r05-qb], #dxz
             [5,   1.0, amb, 0, 0, r05],   #dyy
             [5,  0.25, aqb, 0, qb2, r05], #dyy
             [5,  -0.5, aqb, 0,   0, r05], #dyy
             [5,  0.25, aqb, 0,-qb2, r05], #dyy
             [8,  0.25, aqb, 0,  qb, r05+qb], #dyz
             [8, -0.25, aqb, 0, -qb, r05+qb], #dyz
             [8,  0.25, aqb, 0, -qb, r05-qb], #dyz
             [8, -0.25, aqb, 0,  qb, r05-qb], #dyz
             [9,   1.0, amb, 0, 0, r05],     #dzz
             [9,  0.25, aqb, 0, 0, r05+qb2], #dzz
             [9,  -0.5, aqb, 0, 0, r05],     #dzz
             [9,  0.25, aqb, 0, 0, r05-qb2], #dzz
            ]

    v = numpy.zeros(4)
    for ka in range(0, len(phi_a)):
        iwa = phi_a[ka][0]
        for kb in range(0, len(phi_b)):
            iwb = phi_b[kb][0]
            v[0] = phi_b[kb][3] - phi_a[ka][3]
            v[1] = phi_b[kb][4] - phi_a[ka][4]
            v[2] = phi_b[kb][5] - phi_a[ka][5]
            v[3] = phi_b[kb][2] + phi_a[ka][2] 
            w[iwa, iwb] += phi_a[ka][1] * phi_b[kb][1] / sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3])
    
    #This approximation is not necessary (Yihan)
    if old_pxpy_pxpy: w[4,4] = 0.5 * (w[2,2] - w[2,5])

    T = rotation_matrix(sij) # CL 

    T2 = numpy.zeros((10,10))
    ii = 0
    for j in range(0, 4):
        for i in range(0, j+1):
            prod = numpy.einsum('m,n->mn', T[:,i], T[:,j])
            kk = 0
            for l in range(0, 4):
                for k in range(0, l+1):
                    T2[kk, ii] = prod[k, l]
                    if k != l: T2[kk, ii] += prod[l, k] 
                    kk += 1    
            ii += 1
    #matrix_print_2d(T2, 5, "T2")

    #matrix_print_2d(w, 5, "w before rotation")
    w = numpy.einsum('ij,ik,kl->jl', T2, w, T2)
    #matrix_print_2d(w, 5, "w after rotation")

    return w;

def compute_W_hl(zi, zj, xi, xj, am, ad, aq, dd, qq, tore):

    w = numpy.zeros((10,10))
   
    #print("calling compute_W_hl")
    Xij  = numpy.subtract(xi, xj)
    rij  = numpy.linalg.norm(Xij)
    r05  = 0.5 * rij
    rij2 = rij * rij
    sij  = Xij
    if rij > 0.0000001: sij  *= 1/rij
    else: return w

    da = dd[zi] 
    qa  = qq[zi]
    qa2 = qq[zi]*2.0

    ama = 0.5 / am[zi]
    ada = 0.5 / ad[zi]
    aqa = 0.5 / aq[zi]
    amb = 0.5 / am[zj]

    phi_a = [[0,   1.0, ama, 0, 0, -r05], #s
             [1,  -0.5, ada,  da, 0, -r05], #px
             [1,   0.5, ada, -da, 0, -r05], #px
             [3,  -0.5, ada, 0,  da, -r05], #py
             [3,   0.5, ada, 0, -da, -r05], #py
             [6,  -0.5, ada, 0, 0, -r05+da], #pz
             [6,   0.5, ada, 0, 0, -r05-da], #pz
             [2,   1.0, ama, 0, 0, -r05],   #dxx
             [2,  0.25, aqa, qa2, 0, -r05], #dxx
             [2,  -0.5, aqa,   0, 0, -r05], #dxx
             [2,  0.25, aqa,-qa2, 0, -r05], #dxx
             [4,  0.25, aqa,  qa, qa, -r05], #dxy
             [4, -0.25, aqa, -qa, qa, -r05], #dxy
             [4,  0.25, aqa, -qa,-qa, -r05], #dxy
             [4, -0.25, aqa,  qa,-qa, -r05], #dxy
             [7,  0.25, aqa,  qa, 0, -r05+qa], #dxz
             [7, -0.25, aqa, -qa, 0, -r05+qa], #dxz
             [7,  0.25, aqa, -qa, 0, -r05-qa], #dxz
             [7, -0.25, aqa,  qa, 0, -r05-qa], #dxz
             [5,   1.0, ama, 0, 0, -r05],   #dyy
             [5,  0.25, aqa, 0, qa2, -r05], #dyy
             [5,  -0.5, aqa, 0,   0, -r05], #dyy
             [5,  0.25, aqa, 0,-qa2, -r05], #dyy
             [8,  0.25, aqa, 0,  qa, -r05+qa], #dyz
             [8, -0.25, aqa, 0, -qa, -r05+qa], #dyz
             [8,  0.25, aqa, 0, -qa, -r05-qa], #dyz
             [8, -0.25, aqa, 0,  qa, -r05-qa], #dyz
             [9,   1.0, ama, 0, 0, -r05],     #dzz
             [9,  0.25, aqa, 0, 0, -r05+qa2], #dzz
             [9,  -0.5, aqa, 0, 0, -r05],     #dzz
             [9,  0.25, aqa, 0, 0, -r05-qa2], #dzz
            ]

    phi_b = [[0,   1.0, amb, 0, 0, r05] #s
            ]

    v = numpy.zeros(4)
    for ka in range(0, len(phi_a)):
        iwa = phi_a[ka][0]
        for kb in range(0, len(phi_b)):
            iwb = phi_b[kb][0]
            v[0] = phi_b[kb][3] - phi_a[ka][3]
            v[1] = phi_b[kb][4] - phi_a[ka][4]
            v[2] = phi_b[kb][5] - phi_a[ka][5]
            v[3] = phi_b[kb][2] + phi_a[ka][2] 
            w[iwa, iwb] += phi_a[ka][1] * phi_b[kb][1] / sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3])

    T = rotation_matrix(sij)

    T2 = numpy.zeros((10,10))
    ii = 0
    for j in range(0, 4):
        for i in range(0, j+1):
            prod = numpy.einsum('m,n->mn', T[:,i], T[:,j])
            kk = 0
            for l in range(0, 4):
                for k in range(0, l+1):
                    T2[kk, ii] = prod[k, l]
                    if k != l: T2[kk, ii] += prod[l, k] 
                    kk += 1    
            ii += 1
    #matrix_print_2d(T2, 5, "T2")

    #matrix_print_2d(w, 5, "w before rotation")
    w = numpy.einsum('ij,ik->jk', T2, w)
    #matrix_print_2d(w, 5, "w after rotation")

    return w;

def compute_W_lh(zi, zj, xi, xj, am, ad, aq, dd, qq, tore):

    w = numpy.zeros((10,10))
   
    #print("calling compute_W_lh")
    Xij  = numpy.subtract(xi, xj)
    rij  = numpy.linalg.norm(Xij)
    r05  = 0.5 * rij
    rij2 = rij * rij
    sij  = Xij
    if rij > 0.0000001: sij  *= 1/rij
    else: return w

    db = dd[zj] 
    qb  = qq[zj]
    qb2 = qq[zj]*2.0 

    ama = 0.5 / am[zi]
    amb = 0.5 / am[zj]
    adb = 0.5 / ad[zj]
    aqb = 0.5 / aq[zj]
    #print("zi: ", zi, "zj:", zj)
    #print("i, ama", ama)
    #print("j, amb, adb, aqb", amb, adb, aqb)

    phi_a = [[0,   1.0, ama, 0, 0, -r05]] #s

    phi_b = [[0,   1.0, amb, 0, 0, r05], #s
             [1,  -0.5, adb,  db, 0, r05], #px
             [1,   0.5, adb, -db, 0, r05], #px
             [3,  -0.5, adb, 0,  db, r05], #py
             [3,   0.5, adb, 0, -db, r05], #py
             [6,  -0.5, adb, 0, 0, r05+db], #pz
             [6,   0.5, adb, 0, 0, r05-db], #pz
             [2,   1.0, amb, 0, 0, r05],   #dxx
             [2,  0.25, aqb, qb2, 0, r05], #dxx
             [2,  -0.5, aqb,   0, 0, r05], #dxx
             [2,  0.25, aqb,-qb2, 0, r05], #dxx
             [4,  0.25, aqb,  qb, qb, r05], #dxy
             [4, -0.25, aqb, -qb, qb, r05], #dxy
             [4,  0.25, aqb, -qb,-qb, r05], #dxy
             [4, -0.25, aqb,  qb,-qb, r05], #dxy
             [7,  0.25, aqb,  qb, 0, r05+qb], #dxz
             [7, -0.25, aqb, -qb, 0, r05+qb], #dxz
             [7,  0.25, aqb, -qb, 0, r05-qb], #dxz
             [7, -0.25, aqb,  qb, 0, r05-qb], #dxz
             [5,   1.0, amb, 0, 0, r05],   #dyy
             [5,  0.25, aqb, 0, qb2, r05], #dyy
             [5,  -0.5, aqb, 0,   0, r05], #dyy
             [5,  0.25, aqb, 0,-qb2, r05], #dyy
             [8,  0.25, aqb, 0,  qb, r05+qb], #dyz
             [8, -0.25, aqb, 0, -qb, r05+qb], #dyz
             [8,  0.25, aqb, 0, -qb, r05-qb], #dyz
             [8, -0.25, aqb, 0,  qb, r05-qb], #dyz
             [9,   1.0, amb, 0, 0, r05],     #dzz
             [9,  0.25, aqb, 0, 0, r05+qb2], #dzz
             [9,  -0.5, aqb, 0, 0, r05],     #dzz
             [9,  0.25, aqb, 0, 0, r05-qb2], #dzz
            ]

    v = numpy.zeros(4)
    for ka in range(0, len(phi_a)):
        iwa = phi_a[ka][0]
        for kb in range(0, len(phi_b)):
            iwb = phi_b[kb][0]
            v[0] = phi_b[kb][3] - phi_a[ka][3]
            v[1] = phi_b[kb][4] - phi_a[ka][4]
            v[2] = phi_b[kb][5] - phi_a[ka][5]
            v[3] = phi_b[kb][2] + phi_a[ka][2] 
            w[iwa, iwb] += phi_a[ka][1] * phi_b[kb][1] / sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3])
    
    T = rotation_matrix(sij)

    T2 = numpy.zeros((10,10))
    ii = 0
    for j in range(0, 4):
        for i in range(0, j+1):
            prod = numpy.einsum('m,n->mn', T[:,i], T[:,j])
            kk = 0
            for l in range(0, 4):
                for k in range(0, l+1):
                    T2[kk, ii] = prod[k, l]
                    if k != l: T2[kk, ii] += prod[l, k] 
                    kk += 1    
            ii += 1
    #matrix_print_2d(T2, 5, "T2")

    #matrix_print_2d(w, 5, "w before rotation")
    w = numpy.einsum('ik,kl->il', w, T2)
    #matrix_print_2d(w, 5, "w after rotation")

    return w;

def compute_W_ll(zi, zj, xi, xj, am, ad, aq, dd, qq, tore):

    w = numpy.zeros((10,10))
   
    #print("calling compute_W_ll")

    Xij  = numpy.subtract(xi, xj)
    rij  = numpy.linalg.norm(Xij)
    r05  = 0.5 * rij
    rij2 = rij * rij
    sij  = Xij
    if rij > 0.0000001: sij  *= 1/rij
    else: return w

    ama = 0.5 / am[zi]
    amb = 0.5 / am[zj]
    #print("zi: ", zi, "zj:", zj)

    phi_a = [[0,   1.0, ama, 0, 0, -r05]] #s
    phi_b = [[0,   1.0, amb, 0, 0, r05]] #s

    v = numpy.zeros(4)
    for ka in range(0, len(phi_a)):
        iwa = phi_a[ka][0]
        for kb in range(0, len(phi_b)):
            iwb = phi_b[kb][0]
            v[0] = phi_b[kb][3] - phi_a[ka][3]
            v[1] = phi_b[kb][4] - phi_a[ka][4]
            v[2] = phi_b[kb][5] - phi_a[ka][5]
            v[3] = phi_b[kb][2] + phi_a[ka][2] 
            w[iwa, iwb] += phi_a[ka][1] * phi_b[kb][1] / sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3])

    return w;

def compute_VAC(zi, zj, xi, xj, am, ad, aq, dd, qq, tore):

    #print("calling compute_VAC")
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
       #print("ri:", ri[0:4])
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
