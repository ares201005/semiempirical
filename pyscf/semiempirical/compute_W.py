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

def compute_W_hh(mol, zi, zj, ia, ja, am, ad, aq, dd, qq, tore, old_pxpy_pxpy=0):

    w = numpy.zeros((10,10))
   
    #print("calling compute_W_hh")
    #Xij  = numpy.subtract(xi, xj)
    #rij  = numpy.linalg.norm(Xij)
    rij = mol.pair_dist[ia,ja]
    r05  = 0.5 * rij
    rij2 = rij * rij
    #sij  = Xij
    sij = numpy.copy(mol.xij[ia,ja])
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
    #print("zi: ", zi, "zj:", zj)
    #print("i, ama, ada, aqa", ama, ada, aqa)
    #print("j, amb, adb, aqb", amb, adb, aqb)

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
    #if old_pxpy_pxpy: w[4,4] = 0.5 * (w[2,2] - w[2,5])
    w[4,4] = 0.5 * (w[2,2] - w[2,5]) # using for now to match. will add kwarg -CL

    T = rotation_matrix(sij)

    T2 = T2_matrix(T)

    w = numpy.einsum('ij,ik,kl->jl', T2, w, T2)
    matrix_print_2d(w*27.21, 8, "Whh after rotation")

    return w

def compute_W_hl(mol, zi, zj, ia, ja, am, ad, aq, dd, qq, tore):

    w = numpy.zeros((10,10))
   
    #print("calling compute_W_hl")
    #Xij  = numpy.subtract(xi, xj)
    #rij  = numpy.linalg.norm(Xij)
    rij = mol.pair_dist[ia, ja]
    r05  = 0.5 * rij
    rij2 = rij * rij
    #sij  = Xij
    #sij = mol.xij[ia, ja]
    sij = numpy.copy(mol.xij[ia,ja])
    if rij > 0.0000001: sij  *= 1/rij
    else: return w
    #theta = acos(sij[2]) 
    #if abs(sij[0]) < 1e-8: 
    #   phi = 0.0
    #else:
    #   phi = atan(sij[1]/sij[0])
    #print("theta:", theta, "phi:", phi)

    da = dd[zi] 
    qa  = qq[zi]
    qa2 = qq[zi]*2.0

    ama = 0.5 / am[zi]
    ada = 0.5 / ad[zi]
    aqa = 0.5 / aq[zi]
    amb = 0.5 / am[zj]
    #print("zi: ", zi, "zj:", zj)
    #print("i, ama, ada, aqa", ama, ada, aqa)
    #print("j, amb", amb)

    #Yihan, 02/04/2024
    #we might want to swap the p charges, so that (p|s) integrals are positive
    phi_a = [[0,   1.0, ama, 0, 0, -r05], #s
             [1,   0.5, ada,  da, 0, -r05], #px
             [1,  -0.5, ada, -da, 0, -r05], #px
             [3,   0.5, ada, 0,  da, -r05], #py
             [3,  -0.5, ada, 0, -da, -r05], #py
             [6,   0.5, ada, 0, 0, -r05+da], #pz
             [6,  -0.5, ada, 0, 0, -r05-da], #pz
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

    T = rotation_matrix(-sij)
    #T = numpy.array([[1.0, 0.0, 0.0, 0.0], [0.0, cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)], 
    #              [0.0, -sin(phi), cos(phi), 0.0], [0.0, sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]])
    #matrix_print_2d(T, 5, "T")

    T2 = T2_matrix(T)
    #matrix_print_2d(T2, 5, "T2")

    #matrix_print_2d(w, 5, "w before rotation")
    w = numpy.einsum('ij,ik->jk', T2, w)
    #matrix_print_2d(w, 5, "w after rotation")

    matrix_print_2d(w*27.21, 8, "Whl after rotation")
    return w

def compute_W_lh(mol, zi, zj, ia, ja, am, ad, aq, dd, qq, tore):

    #print("calling compute_W_lh")
    w = numpy.zeros((10,10))
   
    #print("calling compute_W_lh")
    #Xij  = numpy.subtract(xi, xj)
    #rij  = numpy.linalg.norm(Xij)
    rij = mol.pair_dist[ia, ja] # SEQM BOHR
    r05  = 0.5 * rij
    rij2 = rij * rij
    #sij  = Xij
    #sij = mol.xij[ia, ja]
    sij = numpy.copy(mol.xij[ia,ja]) # SEQM BOHR?
    if rij > 0.0000001: sij  *= 1/rij
    else: return w
    #print("xi: ", xi, "xj:", xj)
    #print("Xij:", Xij, "rij: ", rij, "sij:", sij)
    #theta = acos(sij[2]) 
    #if abs(sij[0]) < 1e-8: 
    #   phi = 0.0
    #else:
    #   phi = atan(sij[1]/sij[0])
    #print("theta:", theta, "phi:", phi)

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
            #print(f'w[{iwa}, {iwb}] = {w[iwa, iwb]*27.21}')
    #print('=========== LINE 315 compute_W.py ===========')
    #print("db: ", db, "adb:", adb, "rij:", rij)
    #print("w:", w[0,:])
    #matrix_print_2d(w,5,'W BEFORE ROTATION')


    T = rotation_matrix(sij)
    
    #matrix_print_2d(numpy.einsum('ji,jk->ik',T,T), 5, "TtT")

    T2 = T2_matrix(T)
    #matrix_print_2d(T2, 8, "T2")

    #matrix_print_2d(w, 5, "w before rotation")
    w = numpy.einsum('ik,kl->il', w, T2)
    #matrix_print_2d(w, 5, "w after rotation")

    #matrix_print_2d(w,5,'W AFTER ROTATION')
    #matrix_print_2d(27.21*w,5,'W_lh (eV) AFTER ROTATION')
    matrix_print_2d(w*27.21, 8, "Wlh after rotation")
    return w

def compute_W_ll(mol, zi, zj, ia, ja, am, ad, aq, dd, qq, tore):

    #print("calling compute_W_ll no rotation")
    w = numpy.zeros((10,10))
    #if numpy.linalg.norm(numpy.subtract(xi, xj)) < 0.0000001: return w
    if mol.pair_dist[ia,ja] < 0.0000001: return w

    # Why are these special? -CL
    # Only use xi xj for W_ll ...
    xi = mol.coords[ia]/0.529167
    xj = mol.coords[ja]/0.529167
    ama = 0.5 / am[zi]
    amb = 0.5 / am[zj]

    phi_a = [[0,   1.0, ama, xi[0], xi[1], xi[2]]] #s
    phi_b = [[0,   1.0, amb, xj[0], xj[1], xj[2]]] #s

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

    #matrix_print_2d(27.21*w,5,'W_ll (eV) AFTER ROTATION')
    #matrix_print_2d(w*27.21, 8, "Wll after rotation")
    return w

def compute_W_lh_no_rotate(zi, zj, xi, xj, am, ad, aq, dd, qq, tore):

    print("calling compute_W_lh no rotation")
    print("wrong integrals, but good for checking the sign of integrals")
    w = numpy.zeros((10,10))
    if numpy.linalg.norm(numpy.subtract(xi, xj)) < 0.0000001: return w
   
    db = dd[zj] 
    qb  = qq[zj]
    qb2 = qq[zj]*2.0 

    ama = 0.5 / am[zi]
    amb = 0.5 / am[zj]
    adb = 0.5 / ad[zj]
    aqb = 0.5 / aq[zj]

    phi_a = [[0,   1.0, ama, xi[0], xi[1], xi[2]]] #s

    # Yihan, 02/04/2023
    # The following put the px, py, pz charges in the standard way, +q along +x, +y, +z axis
    # Note that this is different from get_w_lh, where we want the (s|p) integral along the rotated axis to be positive

    phi_b = [[0,   1.0, amb, 0, 0, 0], #s
             [1,   0.5, adb,  db, 0, 0], #px
             [1,  -0.5, adb, -db, 0, 0], #px
             [3,   0.5, adb, 0,  db, 0], #py
             [3,  -0.5, adb, 0, -db, 0], #py
             [6,   0.5, adb, 0, 0, db], #pz
             [6,  -0.5, adb, 0, 0, -db], #pz
             [2,   1.0, amb, 0, 0, 0],   #dxx
             [2,  0.25, aqb, qb2, 0, 0], #dxx
             [2,  -0.5, aqb,   0, 0, 0], #dxx
             [2,  0.25, aqb,-qb2, 0, 0], #dxx
             [4,  0.25, aqb,  qb, qb, 0], #dxy
             [4, -0.25, aqb, -qb, qb, 0], #dxy
             [4,  0.25, aqb, -qb,-qb, 0], #dxy
             [4, -0.25, aqb,  qb,-qb, 0], #dxy
             [7,  0.25, aqb,  qb, 0, qb], #dxz
             [7, -0.25, aqb, -qb, 0, qb], #dxz
             [7,  0.25, aqb, -qb, 0, -qb], #dxz
             [7, -0.25, aqb,  qb, 0, -qb], #dxz
             [5,   1.0, amb, 0, 0, 0],   #dyy
             [5,  0.25, aqb, 0, qb2, 0], #dyy
             [5,  -0.5, aqb, 0,   0, 0], #dyy
             [5,  0.25, aqb, 0,-qb2, 0], #dyy
             [8,  0.25, aqb, 0,  qb, qb], #dyz
             [8, -0.25, aqb, 0, -qb, qb], #dyz
             [8,  0.25, aqb, 0, -qb, -qb], #dyz
             [8, -0.25, aqb, 0,  qb, -qb], #dyz
             [9,   1.0, amb, 0, 0, 0],     #dzz
             [9,  0.25, aqb, 0, 0, qb2], #dzz
             [9,  -0.5, aqb, 0, 0, 0],     #dzz
             [9,  0.25, aqb, 0, 0, -qb2], #dzz
            ]
    for i in range(len(phi_b)):
       phi_b[i][3] += xj[0]
       phi_b[i][4] += xj[1]
       phi_b[i][5] += xj[2]

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
    
    return w

def rotation_matrix(sij):

    theta = acos(sij[2]) 
    if abs(sij[0]) + abs(sij[1]) < 1e-8: 
       phi = 0.0
    elif abs(sij[0]) < 1e-8: 
       if sij[1] > 0:
          phi = numpy.pi / 2.0
       else: 
          phi = -numpy.pi / 2.0
    else:
       phi = atan(sij[1]/sij[0])
    if abs(phi) < 1e-8 and sij[0] < -1.0e-8:
       phi = numpy.pi

    T = numpy.array([[1.0, 0.0, 0.0, 0.0], [0.0, cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)], 
                  [0.0, -sin(phi), cos(phi), 0.0], [0.0, sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]])
    print('sij',sij,'theta',theta,'phi',phi)
    return T

def T2_matrix(T):
    T2 = numpy.zeros((10,10))
    ii = 0
    for j in range(0, 4):
        for i in range(0, j+1):
            prod = numpy.einsum('m,n->mn', T[:,i], T[:,j])
            kk = 0
            for l in range(0, 4):
                for k in range(0, l+1):
                    T2[kk, ii] = prod[k, l]
                    if k != l:
                        print(f'T2[{kk}, {ii}] = prod[{k}, {l}] = {numpy.round(prod[k, l],4)} + {numpy.round(prod[l, k],4)}')
                    else:
                        print(f'T2[{kk}, {ii}] = prod[{k}, {l}] = {numpy.round(prod[k, l],4)}')
                    if k != l: T2[kk, ii] += prod[l, k] 
                    kk += 1    
            ii += 1
    #matrix_print_2d(T2, 5, "T2")
    return T2

def matrix_print_2d(array, ncols, title):
	""" printing a rectangular matrix, ncols columns per batch """

	write(title+'\n')
	m = array.shape[0]
	n = array.shape[1]
	#write('m=%d n=%d\n' % (m, n))
	nbatches = int(n/ncols)
	if nbatches * ncols < n: nbatches += 1
	for k in range(0, nbatches):
		write('     ')  
		j1 = ncols*k
		j2 = ncols*(k+1)
		if k == nbatches-1: j2 = n 
		for j in range(j1, j2):
			write('   %7d  ' % (j+1))
		write('\n')
		for i in range(0, m): 
			write(' %3d -' % (i+1))
			for j in range(j1, j2):
				if abs(array[i,j]) < 0.000001:  
					write(' %11.6f' % abs(array[i,j]))
				else:
					write(' %11.6f' % array[i,j])
			write('\n')
