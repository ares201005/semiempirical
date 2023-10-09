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
    # Matches MNDO2020 with -xij for NOH molecule.
    xij = -xij
    r05  = 0.5 * rij
    rij2 = rij * rij
    sij  = xij
    #if rij > 0.0000001: sij  *= 1/rij # Normalize
    #else: return w # Return ~0

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
    print(f'xij: {xij[0]} {xij[1]} {xij[2]}')
    print('ca1', ca1)
    print('cb1', cb1)
    print('sa1', sa1)
    print('sb1', sb1)

    print(f'ca1*sb1 {ca1*sb1}')
    print(f'sa1*sb1 {sa1*sb1}')
    print(f'cb1 {cb1}')
    print(f'ca1*cb1 {ca1*cb1}')
    print(f'sa1*cb1 {sa1*cb1}')
    print(f'-sb1 {-sb1}')
    print(f'-sa1 {-sa1}')
    print(f'ca1 {ca1}')
    print(f'0 {0}')
    print(f'\n---\n')

    print(f'ca*sb {ca*sb}')
    print(f'sa*sb {sa*sb}')
    print(f'cb {cb}')
    print(f'ca*cb {ca*cb}')
    print(f'sa*cb {sa*cb}')
    print(f'-sb {-sb}')
    print(f'-sa {-sa}')
    print(f'ca {ca}')
    print(f'0 {0}')
    print(f'\n---\n')
    #theta = acos(cb)
    #phi   = acos(ca)

    theta = acos(sij[2]) 
    if abs(sij[0]) < 1e-8: 
       phi = 0.0
    else:
       phi = atan(sij[1]/sij[0])
    print("theta:", theta, "phi:", phi)
    print(f'ca*sb {cos(phi)*sin(theta)}')
    print(f'sa*sb {sin(phi)*sin(theta)}')
    print(f'cb {cos(theta)}')
    print(f'ca*cb {cos(phi)*cos(theta)}')
    print(f'sa*cb {sin(phi)*cos(theta)}')
    print(f'-sb {-sin(theta)}')
    print(f'-sa {-sin(phi)}')
    print(f'ca {cos(phi)}')
    print(f'0 {0}')

    #print(f'ca: {ca} cb: {cb} sa: {sa} sb: {sb}')
    #print(f'theta: {theta} phi: {phi}')
    #print(f'cos(theta): {cos(theta)} cos(phi): {cos(phi)}')
    #print(f'sin(theta): {sin(theta)} sin(phi): {sin(phi)}')
    #print(f'sij[0]: {sij[0]} sij[1]: {sij[1]} sij[2]: {sij[2]}')

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
    print("zi: ", zi, "zj:", zj)
    print("i, ama, ada, aqa", ama, ada, aqa)
    print("j, amb, adb, aqb", amb, adb, aqb)

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

    #v = numpy.zeros(4)
    #for ka in range(0, len(phi_a)):
    #    iwa = phi_a[ka][0]
    #    for kb in range(0, len(phi_b)):
    #        iwb = phi_b[kb][0]
    #        v[0] = phi_b[kb][3] - phi_a[ka][3]
    #        v[1] = phi_b[kb][4] - phi_a[ka][4]
    #        v[2] = phi_b[kb][5] - phi_a[ka][5]
    #        v[3] = phi_b[kb][2] + phi_a[ka][2] 
    #        sqrt_v2 = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3])
    #        w[iwa, iwb] += phi_a[ka][1] * phi_b[kb][1] / sqrt_v2
    
    #This approximation is not necessary (Yihan)
    #if old_pxpy_pxpy: w[4,4] = 0.5 * (w[2,2] - w[2,5])

    #T = numpy.array([[1.0, 0.0, 0.0, 0.0], [0.0, cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)], 
    #              [0.0, -sin(phi), cos(phi), 0.0], [0.0, sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]])
    #T = numpy.array([[1.0, 0.0, 0.0, 0.0], [0.0, cb*ca, cb*sa, -sb], 
    #                 [0.0, -sa, ca, 0.0], [0.0, sb*ca, sb*sa, cb]])
    T = numpy.array([[1.0, 0.0, 0.0, 0.0], [0.0, sb*ca, sb*sa, cb], 
                     [0.0, cb*ca, cb*sa, -sb], [0.0, -sa, ca, 0.0]])
    matrix_print_2d(T, 5, "P-Matrix (T)")
    Tt = numpy.einsum('ij->ji', T)
    matrix_print_2d(Tt, 5, "Pt-Matrix (Tt)")

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
    matrix_print_2d(T2, 5, "T2")
    matrix_print_2d(T2, 10, "T2")

    #matrix_print_2d(w, 5, "w before rotation")
    #w = numpy.einsum('ij,ik,kl->jl', T2, w, T2)
    #matrix_print_2d(w, 5, "w after rotation")

    #return w;
    return T




#def rotation_matrix(zi, zj, xij):
#    ''' Transform local coordinates to molecular coordinates
#    '''	
#    indx_array = np.array([0,1,3,6,10,15,21,28,36])
#    rot_mat = np.zeros((6,10))
#    if zi == 1 and zj == 1: # ss
#       rot_mat[0][0] = 1.0
#    else:
#       # sp rotate
#       sp_rot = np.zeros((3,3))
#       xy = np.linalg.norm(xij[...,:2])
#       if xij[2] > 0: tmp = 1.0
#       elif xij[2] < 0: tmp = -1.0
#       else: tmp = 0.0
#   
#       ca = cb = tmp
#       sa = sb = 0.0
#       if xy > 1.0e-10:
#          ca = xij[0]/xy
#          cb = xij[2]
#          sa = xij[1]/xy
#          sb = xy
#   
#       sasb = sa*sb
#       sacb = sa*cb
#       casb = ca*sb
#       cacb = ca*cb
#       sp_rot[0][0] = casb
#       sp_rot[0][1] = cacb
#       sp_rot[0][2] = -sa
#       sp_rot[1][0] = sasb
#       sp_rot[1][1] = sacb
#       sp_rot[1][2] = ca
#       sp_rot[2][0] = cb
#       sp_rot[2][1] = -sb
#       sp_rot[2][2] = 0.0
#    if zi > 1 or zj > 1:
#       for i in range(0,3):
#          j = indx_array[i+1]+1
#          rot_mat[j][1] = sp_rot[1][i]
#          rot_mat[j][2] = sp_rot[2][i]
#          rot_mat[j][3] = sp_rot[3][i]
#       for i in range(0,3):
#          j = indx_array[i+1]+i+1
#          rot_mat[j][0] = sp_rot[1][i]*sp_rot[1][i]
#          rot_mat[j][1] = sp_rot[1][i]*sp_rot[2][i]
#          rot_mat[j][2] = sp_rot[2][i]*sp_rot[2][i]
#          rot_mat[j][3] = sp_rot[1][i]*sp_rot[3][i]
#          rot_mat[j][4] = sp_rot[2][i]*sp_rot[3][i]
#          rot_mat[j][5] = sp_rot[3][i]*sp_rot[3][i]
#    return rot_mat 
#
