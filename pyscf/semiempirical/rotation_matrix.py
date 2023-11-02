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

def rotation_matrix2(zi, zj, xij, rij, am, ad, aq, dd, qq, tore, old_pxpy_pxpy):
    ''' Transform local coordinates to molecular coordinates
    '''	
    xij = -xij
    r05  = 0.5 * rij
    rij2 = rij * rij
    sij  = xij

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

    ca1 = xij[0]/numpy.sqrt(xij[0]**2 + xij[1]**2)
    sa1 = xij[1]/numpy.sqrt(xij[0]**2 + xij[1]**2)
    cb1 = xij[2]/numpy.sqrt(xij[0]**2 + xij[1]**2 + xij[2]**2)
    sb1 = numpy.sqrt(xij[0]**2 + xij[1]**2)/numpy.sqrt(xij[0]**2 + xij[1]**2 + xij[2]**2)

    theta = acos(sij[2]) 
    if abs(sij[0]) < 1e-8: 
       phi = 0.0
    else:
       phi = atan(sij[1]/sij[0])

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

    T = numpy.array([[1.0, 0.0, 0.0, 0.0], [0.0, sb*ca, sb*sa, cb], 
                     [0.0, cb*ca, cb*sa, -sb], [0.0, -sa, ca, 0.0]])
    #matrix_print_2d(T, 5, "P-Matrix (T)")
    #Tt = numpy.einsum('ij->ji', T)
    #matrix_print_2d(Tt, 5, "Pt-Matrix (Tt)")

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

    return T

