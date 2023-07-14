#!/usr/bin/env python
# flake8: noqa

'''
MNDO-AM1
(In testing)

Ref:
[1] J. J. Stewart, J. Comp. Chem. 10, 209 (1989)
[2] J. J. Stewart, J. Mol. Model 10, 155 (2004)
'''

import ctypes
import copy
import math
import numpy as np
import warnings
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf.data.elements import _symbol
from pyscf.semiempirical import mindo3#, mopac_param 
from pyscf.semiempirical import mopac_param, read_param

#libsemiempirical = lib.load_library('lib/libsemiempirical.so')

libsemiempirical =lib.load_library('/home/chance/pyscf_ext/semiempirical/pyscf/semiempirical/lib/libsemiempirical.so') 
ndpointer = np.ctypeslib.ndpointer
libsemiempirical.MOPAC_rotate.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ndpointer(dtype=np.double),  # xi
    ndpointer(dtype=np.double),  # xj
    ndpointer(dtype=np.double),  # w
    ndpointer(dtype=np.double),  # e1b
    ndpointer(dtype=np.double),  # e2a
    ndpointer(dtype=np.double),  # enuc
    ndpointer(dtype=np.double),  # alp
    ndpointer(dtype=np.double),  # dd
    ndpointer(dtype=np.double),  # qq
    ndpointer(dtype=np.double),  # am
    ndpointer(dtype=np.double),  # ad
    ndpointer(dtype=np.double),  # aq
    ndpointer(dtype=np.double),  # fn1
    ndpointer(dtype=np.double),  # fn2
    ndpointer(dtype=np.double),  # fn3
    ctypes.c_int
]
repp = libsemiempirical.MOPAC_rotate
# parameterlist from PYSEQM
#parameterlist={'AM1':['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
#                      'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp',
#                      'alpha',
#                      'Gaussian1_K', 'Gaussian2_K', 'Gaussian3_K','Gaussian4_K',
#                      'Gaussian1_L', 'Gaussian2_L', 'Gaussian3_L','Gaussian4_L',
#                      'Gaussian1_M', 'Gaussian2_M', 'Gaussian3_M','Gaussian4_M'
#                     ],  
#                'MNDO':['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
#                        'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha'],
#                'PM3':['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
#                       'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp',
#                       'alpha',
#                       'Gaussian1_K', 'Gaussian2_K',
#                       'Gaussian1_L', 'Gaussian2_L',
#                       'Gaussian1_M', 'Gaussian2_M'
#                      ]} 

# improve parameter list later? -CL
#method = 'AM1'
#method_params = parameterlist[method]
#elements = [8,1,1] #generalize later -CL
#elements = [6, 6, 8, 1, 1, 1, 1, 1, 1]
#elements = np.asarray([mol.atom_charge(i) for i in range(mol.natm)])

#print('TEST', gto._symbol, gto._atom, gto.atom_charge, gto.bas_atom)
#elements = gto._symbol
#print(method, elements)
#constants = read_param.read_constants(elements)
#parameters = read_param.read_param(method, elements)

#atmmass = read_param.constants['atmmass']
#tore = read_param.constants['tore']
#
#U_ss = read_param.parameters['U_ss']/27.211386
#U_pp = read_param.parameters['U_pp']/27.211386
#zeta_s = read_param.parameters['zeta_s']
#zeta_p = read_param.parameters['zeta_p']
#zeta_d = read_param.parameters['zeta_d']
#beta_s = read_param.parameters['beta_s']
#beta_p = read_param.parameters['beta_p']
#g_ss = read_param.parameters['g_ss']/27.211386
#g_sp = read_param.parameters['g_sp']/27.211386
#g_pp = read_param.parameters['g_pp']/27.211386
#g_p2 = read_param.parameters['g_p2']/27.211386
#h_sp = read_param.parameters['h_sp']/27.211386 #div 27.211 ? -CL
#alpha = read_param.parameters['alpha']
#
#if method == 'AM1':
#    K = np.stack((parameters['Gaussian1_K'],parameters['Gaussian2_K'],parameters['Gaussian3_K'],parameters['Gaussian4_K']), axis=1)/27.211386
#    L = np.stack((parameters['Gaussian1_L'],parameters['Gaussian2_L'],parameters['Gaussian3_L'],parameters['Gaussian4_L']), axis=1)
#    M = np.stack((parameters['Gaussian1_M'],parameters['Gaussian2_M'],parameters['Gaussian3_M'],parameters['Gaussian4_M']), axis=1)
#return elements, constants, U_ss, U_pp, zeta_s, zeta_p, zeta_d, beta_s, beta_p, g_ss, g_sp, g_pp, g_p2, h_sp, alpha

#*** Set constants.py to constants2.py and make constants.py a function
#that returns a dict where 'tore' gives np array of tore values.

def SET(rij, jcall, z1, z2):
    """
    get the A integrals and B integrals for diatom_overlap_matrix
    """
    #alpha, beta below is used in aintgs and bintgs, not the parameters for AM1/MNDO/PM3
    #rij: distance between atom i and j in atomic unit
    #print('SET:')
    #print('rij',rij)
    #print('z1',z1)
    #print('z2',z2)

    alp = 0.5*rij*(z1+z2)
    beta  = 0.5*rij*(z1-z2)
    #print("alpha, beta:", alp, beta)
    A = aintgs(alp)
    B = bintgs(beta)
    #print("A=", A)
    #print("B=", B)
    return A, B


def aintgs(x):
    """
    A integrals for diatom_overlap_matrix
    """
    a1 = np.exp(-x)/x
    a2 = a1 +a1/x
    a3 = a1 + 2.0*a2/x
    a4 = a1+3.0*a3/x
    a5 = a1+4.0*a4/x
    #print("a1, a2, a3, a4, a5:", a1, a2, a3, a4, a5)
    return np.array([a1, a2, a3, a4, a5])

def bintgs(x):
    """
    B integrals for diatom_overlap_matrix
    """
    absx = abs(x)
    #cond1 = absx > 0.5

    b1 = 2.0 
    b2 = 0.0 
    b3 = 2.0/3.0 
    b4 = 0.0 
    b5 = 2.0/5.0 

    if absx>0.5:

       b1 =   np.exp(x)/x - np.exp(-x)/x
       b2 = - np.exp(x)/x - np.exp(-x)/x + b1/x
       b3 =   np.exp(x)/x - np.exp(-x)/x + 2*b2/x
       b4 = - np.exp(x)/x - np.exp(-x)/x + 3*b3/x
       b5 =   np.exp(x)/x - np.exp(-x)/x + 4*b4/x
       #print("406, b1, b2, b3, b4, b5:", b1, b2, b3, b4, b5)

    elif absx > 1.0e-6: 

       b1 = 2.0     + x**2/3.0 + x**4/60.0 + x**6/2520.0
       b3 = 2.0/3.0 + x**2/5.0 + x**4/84.0 + x**6/3240.0
       b5 = 2.0/5.0 + x**2/7.0 + x**4/108.0 + x**6/3960.0

       b2 = -2.0/3.0*x - x**3/15.0 - x**5/420.0
       b4 = -2.0/5.0*x - x**3/21.0 - x**5/540.0
       #print("488, b1, b2, b3, b4, b5:", b1, b2, b3, b4, b5)

    return np.array([b1, b2, b3, b4, b5])

def diatomic_overlap_matrix(ia, ja, zi, zj, xij, rij): #generalize -CL ***
    #Plan to generalize: PYSEQM uses zi, zj, xij, rij as arrays and builds with jcall using arrays. 
    #Either call diat_overlap multiple times or pass zi zj arrays. 
    #if zi == 8 and zj == 8: jcall = 4
    if zi == 1 and zj == 1: # first row - first row
       jcall = 2 
       di = np.zeros((1,1))
    elif (zi > 1 and zj == 1) or (zi == 1 and zj > 1): # first row - second row
       jcall = 3
       di = np.zeros((4,1)) # original was 4,1
    elif zi > 1 and zj > 1: # second row - second row #make else? -CL
       jcall = 4 
       di = np.zeros((4,4))
    else:
       print('invalid combination of zi and zj')
       exit(-1)
    print('xij', xij, ia, ja)
    #xy = math.sqrt(xij[ia]*xij[ia] + xij[ja]*xij[ja])
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
    #print("ca, cb, sa, sb=", ca, cb, sa, sb)

    sasb = sa*sb
    sacb = sa*cb
    casb = ca*sb
    cacb = ca*cb

    #print("S", S111, S211, S121, S221, S222)
    #print("ca, cb, sa, sb=", ca, cb, sa, sb) 
    #print('sasb, sacb, casb, cacb',sasb, sacb, casb, cacb)

    zetas = np.array([zeta_s[zi], zeta_s[zj]])
    zetap = np.array([zeta_p[zi], zeta_p[zj]]) #do we need zeta below? -CL
    zeta = np.array([[zetas[0], zetap[0]], [zetas[1], zetap[1]]]) #np.concatenate(zetas.unsequeeze(1), zetap.unsequeeze(1))
    #print("zeta:", zeta, zeta[0], zeta[1], zeta[0,0], zeta[1,0], zeta[0,1], zeta[1,1])
    #print('Full Zeta:', zeta)
    #if zi == 8 and zj == 8:
    beta = np.array([[beta_s[zi],beta_p[zi]],[beta_s[zj],beta_p[zj]]]) / 27.211386
    A111,B111 = SET(rij, jcall, zeta[0,0],zeta[1,0])
    #Probably need to make SXX arrays dependent on jcall value. -CL ***

    if jcall == 2:
       S111 = math.pow(zeta[0,0]* zeta[1,0]* rij**2,1.5)* \
                  (A111[2]*B111[0]-B111[2]*A111[0])/4.0
    elif jcall == 3:
       S111 = math.pow(zeta[1,0],1.5)* math.pow(zeta[0,0],2.5)*rij**4 * \
                  (A111[3]*B111[0]-B111[3]*A111[0]+ \
                   A111[2]*B111[1]-B111[2]*A111[1]) / (math.sqrt(3.0)*8.0)
    elif jcall == 4:
       S111 = math.pow(zeta[1,0]*zeta[0,0],2.5)* rij**5 * \
                          (A111[4]*B111[0]+B111[4]*A111[0]-2.0*A111[2]*B111[2])/48.0
       #print("S111:", S111)
    di[0,0] = S111
    if jcall == 3:
       A211,B211 = SET(rij, jcall, zeta[0,1],zeta[1,0])
       #print('A211 zeta [0,1] [1,0]', zeta[0,1],zeta[1,0])
       S211 = math.pow(zeta[1,0],1.5)* math.pow(zeta[0,1],2.5)* rij**4 * \
                  (A211[2]*B211[0]-B211[2]*A211[0]+ \
                   A211[3]*B211[1]-B211[3]*A211[1])/8.0
       di[1,0] = S211*ca*sb
       di[2,0] = S211*sa*sb
       di[3,0] = S211*cb
    elif jcall == 4:
       A211,B211 = SET(rij, jcall, zeta[0,1],zeta[1,0])
       S211 = math.pow(zeta[1,0]* zeta[0,1],2.5)* rij**5 * \
                  (A211[3]*(B211[0]-B211[2]) \
                  -A211[1]*(B211[2]-B211[4]) \
                  +B211[3]*(A211[0]-A211[2]) \
                  -B211[1]*(A211[2]-A211[4])) \
                  /(16.0*math.sqrt(3.0))
       di[1,0] = S211*ca*sb
       di[2,0] = S211*sa*sb
       di[3,0] = S211*cb
       #print("S211:", S211)
    if jcall == 4:
       A121,B121 = SET(rij, jcall, zeta[0,0],zeta[1,1])
       #print('A121 zeta [0,0] [1,1]', zeta[0,0],zeta[1,1])
       S121 = math.pow(zeta[1,1]* zeta[0,0],2.5)* rij**5 * \
                  (A121[3]*(B121[0]-B121[2]) \
                  -A121[1]*(B121[2]-B121[4]) \
                  -B121[3]*(A121[0]-A121[2]) \
                  +B121[1]*(A121[2]-A121[4])) \
                  /(16.0*math.sqrt(3.0))
       #print("S121:", S121)
       di[0,1] = -S121*casb
       di[0,2] = -S121*sasb
       di[0,3] = -S121*cb
    if jcall == 4:
       A22,B22 = SET(rij, jcall, zeta[0,1],zeta[1,1]) #Can cause div by 0. Fix with if? -CL
       #print('A22 zeta [0,1] [1,1]', zeta[0,1],zeta[1,1])
       S221 = -math.pow(zeta[1,1]* zeta[0,1],2.5)* rij**5/16.0 * \
                  (B22[2]*(A22[4]+A22[0]) \
                  -A22[2]*(B22[4]+B22[0])) 
       S222 = 0.5*math.pow(zeta[1,1]* zeta[0,1],2.5)* rij**5/16.0 * \
                  (A22[4]*(B22[0]-B22[2]) \
                  -B22[4]*(A22[0]-A22[2]) \
                  -A22[2]*B22[0]+B22[2]*A22[0])
       di[1,1] = -S221*casb**2 \
                        +S222*(cacb**2+sa**2)
       di[1,2] = -S221*casb*sasb \
                        +S222*(cacb*sacb-sa*ca)
       di[1,3] = -S221*casb*cb \
                        -S222*cacb*sb
       di[2,1] = -S221*sasb*casb \
                        +S222*(sacb*cacb-ca*sa)
       di[2,2] = -S221*sasb**2 \
                        +S222*(sacb**2+ca**2)
       di[2,3] = -S221*sasb*cb \
                        -S222*sacb*sb
       di[3,1] = -S221*cb*casb \
                        -S222*sb*cacb
       di[3,2] = -S221*cb*sasb \
                        -S222*sb*sacb
       di[3,3] = -S221*cb**2 \
                        +S222*sb**2
       #print("S221:", S221)
       #print("S222:", S222)

    #print('jcall',jcall)
    #print("di:", di)

    di[0,0] *= (beta[0,0] + beta[1,0]) /2.0
    if jcall >= 3:
       di[1:4,0] *= (beta[0,1] + beta[1,0]) /2.0
    if jcall == 4:
       di[0,1:4] *= (beta[0,0] + beta[1,1]) /2.0
       di[1:4,1:4] *= (beta[0,1] + beta[1,1]) /2.0

    return di

@lib.with_doc(scf.hf.get_hcore.__doc__)
def get_hcore(mol,U_ss,U_pp): # Var added
    assert(not mol.has_ecp())
    atom_charges = mol.atom_charges()
    basis_atom_charges = atom_charges[mol._bas[:,gto.ATOM_OF]]

    basis_u = []
    for i, z in enumerate(basis_atom_charges):
        l = mol.bas_angular(i)
        if l == 0:
            basis_u.append(U_ss[z])
        else:
            basis_u.append(U_pp[z])
    # U term
    hcore = np.diag(_to_ao_labels(mol, basis_u))
    #print('U term hcore', hcore)

    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        for ja in range(ia+1,mol.natm): #Was ia -CL
            w, e1b, e2a, enuc = _get_jk_2c_ints(mol, ia, ja, tore, method)
            i0, i1 = aoslices[ia,2:]
            j0, j1 = aoslices[ja,2:]
            hcore[j0:j1,j0:j1] += e2a
            hcore[i0:i1,i0:i1] += e1b
            #print("ia:", ia, "ja:", ja, "e2a:", e2a, "e1b", e1b)
            
            # off-diagonal block 
            zi = mol.atom_charge(ia)
            zj = mol.atom_charge(ja)
            xi = mol.atom_coord(ia) #?*lib.param.BOHR
            xj = mol.atom_coord(ja) #?*lib.param.BOHR
            xij = xj - xi
            rij = np.linalg.norm(xij)
            xij /= rij
            #print("zi, zj:", zi, zj)
            #print("xij:", xij, "rij:", rij)
            di = diatomic_overlap_matrix(ia, ja, zi, zj, xij, rij)
            #print('hcore:',hcore[i0:i1,j0:j1], np.shape(hcore[i0:i1,j0:j1]))
            #print('di core:',di, np.shape(di))
            #hcore[i0:i1,j0:j1] += di.T
            #hcore[j0:j1,i0:i1] += di
            hcore[i0:i1,j0:j1] += di #original -CL
            hcore[j0:j1,i0:i1] += di.T #original -CL
 
    return hcore

def _get_jk_2c_ints(mol, ia, ja, tore, method): #should be ok -CL
    zi = mol.atom_charge(ia)
    zj = mol.atom_charge(ja)
    ri = mol.atom_coord(ia) #?*lib.param.BOHR
    rj = mol.atom_coord(ja) #?*lib.param.BOHR
    w = np.zeros((10,10))
    e1b = np.zeros(10)
    e2a = np.zeros(10)
    enuc = np.zeros(1)
    if method == 'AM1': 
        MODEL = 2
    elif method == 'PM3':
        MODEL = 3
    elif method == 'MNDO': # Add MNDO to repp/rotate.C
        return NotImplementedError
    repp(zi, zj, ri, rj, w, e1b, e2a, enuc,
         alpha, mopac_param.MOPAC_DD, mopac_param.MOPAC_QQ, mopac_param.MOPAC_AM, mopac_param.MOPAC_AD, mopac_param.MOPAC_AQ,
         K, L, M, MODEL) # Replaced AM1_MODEL with MODEL 

    #repp(zi, zj, ri, rj, w, e1b, e2a, enuc,
    #     alpha, mopac_param.MOPAC_DD, mopac_param.MOPAC_QQ, mopac_param.MOPAC_AM, mopac_param.MOPAC_AD, mopac_param.MOPAC_AQ,
    #     mopac_param.MOPAC_IDEA_FN1, mopac_param.MOPAC_IDEA_FN2, mopac_param.MOPAC_IDEA_FN3, AM1_MODEL) #Check params vs am1-h2o.py? -CL
    #print('ri,rj',ri,rj)
    #print('repp zi,zj,ri,rj',zi,zj,ri,rj)
    #print('repp alpha',alpha)
    #print('repp mopac_param.MOPAC_DD',mopac_param.MOPAC_DD)
    #print('repp mopac_param.MOPAC_QQ',mopac_param.MOPAC_QQ)
    #print('repp mopac_param.MOPAC_AM',mopac_param.MOPAC_AM)
    #print('repp mopac_param.MOPAC_AD',mopac_param.MOPAC_AD)
    #print('repp mopac_param.MOPAC_AQ',mopac_param.MOPAC_AQ)
    #print('repp K',K)
    #print('repp L',L)
    #print('repp M',M)
    #print('repp e1b',e1b)
    #print('repp e2a',e2a)
    tril2sq = lib.square_mat_in_trilu_indices(4)
    w = w[:,tril2sq][tril2sq]
    e1b = e1b[tril2sq]
    e2a = e2a[tril2sq]

    if tore[zj] <= 1: #check same as mopac_param.CORE[zj]
        e2a = e2a[:1,:1]
        w = w[:,:,:1,:1]
    if tore[zi] <= 1:
        e1b = e1b[:1,:1]
        w = w[:1,:1]
    #print('w',w)
    #print('e1b',e1b)
    #print('e2a',e2a)
    #print('tore[zj]',tore[zj])
    #print('tore[zi]',tore[zi])
    return w, e1b, e2a, enuc[0]


@lib.with_doc(scf.hf.get_jk.__doc__)
def get_jk(mol, dm):
    dm = np.asarray(dm)
    dm_shape = dm.shape
    nao = dm_shape[-1]

    dm = dm.reshape(-1,nao,nao)
    vj = np.zeros_like(dm)
    vk = np.zeros_like(dm)

    # One-center contributions to the J/K matrices
    atom_charges = mol.atom_charges()
    jk_ints = {z: _get_jk_1c_ints(z) for z in set(atom_charges)}
    #print('jk_ints',jk_ints)

    aoslices = mol.aoslice_by_atom()
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z = atom_charges[ia]
        j_ints, k_ints = jk_ints[z]

        dm_blk = dm[:,p0:p1,p0:p1]
        idx = np.arange(p0, p1)
        # J[i,i] = (ii|jj)*dm_jj
        vj[:,idx,idx] = np.einsum('ij,xjj->xi', j_ints, dm_blk)
        # J[i,j] = (ij|ij)*dm_ji +  (ij|ji)*dm_ij
        vj[:,p0:p1,p0:p1] += 2*k_ints * dm_blk

        # K[i,i] = (ij|ji)*dm_jj
        vk[:,idx,idx] = np.einsum('ij,xjj->xi', k_ints, dm_blk)
        # K[i,j] = (ii|jj)*dm_ij + (ij|ij)*dm_ji
        vk[:,p0:p1,p0:p1] += (j_ints+k_ints) * dm_blk

    # Two-center contributions to the J/K matrices
    for ia, (i0, i1) in enumerate(aoslices[:,2:]):
        w = _get_jk_2c_ints(mol, ia, ia, tore, method)[0]
        vj[:,i0:i1,i0:i1] += np.einsum('ijkl,xkl->xij', w, dm[:,i0:i1,i0:i1])
        vk[:,i0:i1,i0:i1] += np.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,i0:i1])
        for ja, (j0, j1) in enumerate(aoslices[:ia,2:]):
            w = _get_jk_2c_ints(mol, ia, ja, tore, method)[0]
            vj[:,i0:i1,i0:i1] += np.einsum('ijkl,xkl->xij', w, dm[:,j0:j1,j0:j1])
            vj[:,j0:j1,j0:j1] += np.einsum('klij,xkl->xij', w, dm[:,i0:i1,i0:i1])
            vk[:,i0:i1,j0:j1] += np.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,j0:j1])
            vk[:,j0:j1,i0:i1] += np.einsum('klij,xjk->xil', w, dm[:,j0:j1,i0:i1])

    vj = vj.reshape(dm_shape)
    vk = vk.reshape(dm_shape)
    #print("dm:", dm)
    return vj, vk

def _get_gamma(mol, F03=None): #From mindo3.py -CL
    #F03=g_ss causes errors bc undefined -CL
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = np.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * lib.param.BOHR
    #rho = np.array([14.3996/F03[z] for z in atom_charges]) #g_ss was MOPAC_AM
    rho = np.array([14.3996/F03[z]/27.211386 for z in atom_charges]) #g_ss was MOPAC_AM
    #Clean up above line... -CL
    #E2 = 14.399/27.211 coulomb coeff (ev and \AA) to Hartree and \AA
    #multiply 27.211 back to get to eV... just gonna use 14.3996 for now. -CL
    #Also note: MOPAC_AM is in Hartrees. g_ss is in eV. -CL

    #gamma = mopac_param.E2 / np.sqrt(distances_in_AA**2 +
    #                                    (rho[:,None] + rho)**2 * .25)
    gamma = 14.3996/27.211386 / np.sqrt(distances_in_AA**2 +
                                        (rho[:,None] + rho)**2 * .25)
    gamma[np.diag_indices(mol.natm)] = 0  # remove self-interaction terms
    return gamma

#def _get_gamma(mol, Gss=mopac_param.F03):
#    atom_charges = mol.atom_charges()
#    atom_coords = mol.atom_coords()
#    distances = numpy.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
#    distances_in_AA = distances * lib.param.BOHR
#
#    rho = numpy.array([mopac_param.E2/F03[z] for z in atom_charges])
#    gamma = mopac_param.E2 / numpy.sqrt(distances_in_AA**2 +
#                                        (rho[:,None] + rho)**2 * .25)
#    gamma[numpy.diag_indices(mol.natm)] = 0  # remove self-interaction terms
#    return gamma

def energy_nuc(mol,alpha,tore,K,L,M,method):
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = np.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * lib.param.BOHR 
    enuc = 0 
    exp = np.exp
    #gamma = mindo3._get_gamma(mol, atmmass)
    #gamma = _get_gamma(mol, atmmass)
    gamma = _get_gamma(mol)
    for ia in range(mol.natm):
        for ja in range(ia):
            ni = atom_charges[ia]
            nj = atom_charges[ja]
            rij = distances_in_AA[ia,ja]
            nt = ni + nj
            if (nt == 8 or nt == 9): 
            #Check N-H and O-H for nuclear energy. Need scale = ~fij MNDO. Mult rij by exp of N or O.
                if (ni == 7 or ni == 8): 
                    #scale += (rij - 1.) * exp(-alpha[ni] * rij)
                    scale = 1. + rij * exp(-alpha[ni] * rij) + exp(-alpha[nj] * rij) # ~fij MNDO
                elif (nj == 7 or nj == 8): 
                    #scale += (rij - 1.) * exp(-alpha[nj] * rij) # ~fij MNDO
                    scale = 1. + rij * exp(-alpha[nj] * rij) + exp(-alpha[ni] * rij) # ~fij MNDO
            else:
                scale = 1. + exp(-alpha[ni] * rij) + exp(-alpha[nj] * rij) # fij MNDO
            #print('scale', scale)

            #print('scale 2', scale)
            enuc += tore[ni] * tore[nj] * gamma[ia,ja] * scale #EN(A,B) = ZZ*gamma*fij | MNDO enuc
            #print('gamma[ia,ja]',gamma[ia,ja])
            if method == 'AM1' or method == 'PM3': # AM1/PM3 scaling for enuc
                fac1 = np.einsum('i,i->', K[ni], exp(-L[ni] * (rij - M[ni])**2))
                # einsum(i,i->, K, exp(L * (rij-M)**2))
                fac2 = np.einsum('i,i->', K[nj], exp(-L[nj] * (rij - M[nj])**2))
                enuc += tore[ni] * tore[nj] / rij * (fac1 + fac2)
    return enuc

def get_init_guess(mol):
    '''Average occupation density matrix'''
    aoslices = mol.aoslice_by_atom()
    dm_diag = np.zeros(mol.nao)
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z_eff = tore[mol.atom_charge(ia)]
        dm_diag[p0:p1] = float(z_eff) / (p1-p0)
    return np.diag(dm_diag)


def energy_tot(mf, dm=None, h1e=None, vhf=None):
    mol = mf._mindo_mol
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + mf.energy_nuc()
    e_ref = _get_reference_energy(mol)

    mf.e_heat_formation = e_tot * mopac_param.HARTREE2KCAL + e_ref
    logger.debug(mf, 'E(ref) = %.15g  Heat of Formation = %.15g kcal/mol',
                 e_ref, mf.e_heat_formation)
    return e_tot.real


class RNDDO(scf.hf.RHF):
    '''RHF-NDDO for closed-shell systems'''
    def __init__(self, mol, method):
        scf.hf.RHF.__init__(self, mol)
        #self.conv_tol = 1e-5
        print(self.conv_tol)
        self.method = method
        print('class', method)
        #self.method_params = parameterlist[self.method]
        self.elements = np.asarray([mol.atom_charge(i) for i in range(mol.natm)])
        self.parameters = read_param.read_param(method, self.elements)
        self.constants = read_param.read_constants(self.elements)

        #self.atmmass = self.constants['atmmass']
        tore = self.constants['tore']
        self.tore = self.constants['tore']
        
        U_ss = self.parameters['U_ss']/27.211386
        U_pp = self.parameters['U_pp']/27.211386
        zeta_s = self.parameters['zeta_s']
        self.zeta_s = self.parameters['zeta_s']
        zeta_p = self.parameters['zeta_p']
        self.zeta_p = self.parameters['zeta_p']
        zeta_d = self.parameters['zeta_d']
        beta_s = self.parameters['beta_s']
        beta_p = self.parameters['beta_p']
        g_ss = self.parameters['g_ss']/27.211386
        g_sp = self.parameters['g_sp']/27.211386
        g_pp = self.parameters['g_pp']/27.211386
        g_p2 = self.parameters['g_p2']/27.211386
        h_sp = self.parameters['h_sp']/27.211386 #div 27.211 ? -CL
        alpha = self.parameters['alpha']
        
        if method == 'AM1':
            K = np.stack((self.parameters['Gaussian1_K'],
                     self.parameters['Gaussian2_K'],
                     self.parameters['Gaussian3_K'],
                     self.parameters['Gaussian4_K']), axis=1)/27.211386
            L = np.stack((self.parameters['Gaussian1_L'],
                     self.parameters['Gaussian2_L'],
                     self.parameters['Gaussian3_L'],
                     self.parameters['Gaussian4_L']), axis=1)
            M = np.stack((self.parameters['Gaussian1_M'],
                          self.parameters['Gaussian2_M'],
                          self.parameters['Gaussian3_M'],
                          self.parameters['Gaussian4_M']), axis=1)
        elif method == 'PM3':
            K = np.stack((self.parameters['Gaussian1_K'],
                     self.parameters['Gaussian2_K']), axis=1)/27.211386 
            L = np.stack((self.parameters['Gaussian1_L'],
                     self.parameters['Gaussian2_L']), axis=1)
            M = np.stack((self.parameters['Gaussian1_M'],
                          self.parameters['Gaussian2_M']), axis=1)
        elif method == 'MNDO':
            K = None
            L = None
            M = None

        self.e_heat_formation = None
        self._mindo_mol = _make_mindo_mol(mol,tore,zeta_s,zeta_p)
        self._keys.update(['e_heat_formation'])

    #def build(self,tore,zeta_s,zeta_p,mol=None):
    def build(self,mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._mindo_mol = _make_mindo_mol(self.tore,self.zeta_s,self.zeta_p,mol)
        return self

    #def get_param_const(self, mol=None): #CL
    #    return param_const(self._mindo_mol) #CL

    def get_ovlp(self, mol=None):
        return np.eye(self._mindo_mol.nao)

    def get_hcore(self,mol=None):
        return get_hcore(self._mindo_mol,U_ss,U_pp)

    @lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        return get_jk(self._mindo_mol, dm)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self._mindo_mol):
            return scf.hf.get_occ(self, mo_energy, mo_coeff)

    def get_init_guess(self, mol=None, key='minao'):
        return get_init_guess(self._mindo_mol)

    def energy_nuc(self,alpha,tore,K,L,M,method):
        return energy_nuc(self._mindo_mol,alpha,tore,K,L,M,method)

    energy_tot = energy_tot

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        scf.hf.RHF._finalize(self)

        # e_heat_formation was generated in SOSCF object.
        if (getattr(self, '_scf', None) and
            getattr(self._scf, 'e_heat_formation', None)):
            self.e_heat_formation = self._scf.e_heat_formation

        logger.note(self, 'Heat of formation = %.15g kcal/mol, %.15g Eh',
                    self.e_heat_formation,
                    self.e_heat_formation/mopac_param.HARTREE2KCAL)
        return self

    density_fit = None
    x2c = x2c1e = sfx2c1e = None

    def nuc_grad_method(self):
        raise NotImplementedError


class UAM1(scf.uhf.UHF):
    '''UHF-AM1 for open-shell systems'''
    def __init__(self, mol):
        scf.uhf.UHF.__init__(self, mol)
        self.conv_tol = 1e-5
        self.e_heat_formation = None
        self._mindo_mol = _make_mindo_mol(mol)
        self._keys.update(['e_heat_formation'])

    def build(self,tore,zeta_s,zeta_p,mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._mindo_mol = _make_mindo_mol(mol)
        self.nelec = self._mindo_mol.nelec
        return self

    def get_ovlp(self, mol=None):
        return np.eye(self._mindo_mol.nao)

    def get_hcore(self, mol=None):
        return get_hcore(self._mindo_mol)

    @lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        return get_jk(self._mindo_mol, dm)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self._mindo_mol):
            return scf.uhf.get_occ(self, mo_energy, mo_coeff)

    def get_init_guess(self, mol=None, key='minao'):
        dm = get_init_guess(self._mindo_mol) * .5
        return np.stack((dm,dm))

    def energy_nuc(self):
        return energy_nuc(self._mindo_mol)

    energy_tot = energy_tot

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        scf.uhf.UHF._finalize(self)

        # e_heat_formation was generated in SOSCF object.
        if (getattr(self, '_scf', None) and
            getattr(self._scf, 'e_heat_formation', None)):
            self.e_heat_formation = self._scf.e_heat_formation

        logger.note(self, 'Heat of formation = %.15g kcal/mol, %.15g Eh',
                    self.e_heat_formation,
                    self.e_heat_formation/mopac_param.HARTREE2KCAL)
        return self

    density_fit = None
    x2c = x2c1e = sfx2c1e = None

    def nuc_grad_method(self):
        import umindo3_grad
        return umindo3_grad.Gradients(self)


def _make_mindo_mol(mol,tore,zeta_s,zeta_p):
    #gexps and gcoefs for Gaussians? zeta_s/p were ZS3/ZP3. Possible source of problems. -CL
    #check to see how this is being used. May need to remove later. -CL
    assert(not mol.has_ecp())
    def make_sto_6g(n, l, zeta): #why definition nested in definition? -CL
        es = mopac_param.gexps[(n, l)]
        cs = mopac_param.gcoefs[(n, l)]
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

    mindo_mol = copy.copy(mol)
    atom_charges = mindo_mol.atom_charges()
    atom_types = set(atom_charges)
    basis_set = {}
    for charge in atom_types:
        n = principle_quantum_number(charge)
        l = 0
        sto_6g_function = make_sto_6g(n, l, zeta_s[charge])
        #sto_6g_function = make_sto_6g(n, l, mopac_param.ZS3[charge]) # change back if errors persist (MINDO/3 zeta parameters) -CL
        basis = [sto_6g_function]

        if charge > 2:  # include p functions
            l = 1
            sto_6g_function = make_sto_6g(n, l, zeta_p[charge])
            #sto_6g_function = make_sto_6g(n, l, mopac_param.ZP3[charge]) # change back if errors persist (MINDO/3 zeta parameters) -CL
            basis.append(sto_6g_function)

        basis_set[_symbol(int(charge))] = basis
    mindo_mol.basis = basis_set

    z_eff = tore[atom_charges]
    mindo_mol.nelectron = int(z_eff.sum() - mol.charge)

    #mindo_mol.build(0,tore,zeta_s,zeta_p,mol)
    mindo_mol.build(0,0) # Original
    return mindo_mol


def _to_ao_labels(mol, labels):
    ao_loc = mol.ao_loc
    degen = ao_loc[1:] - ao_loc[:-1]
    ao_labels = [[label]*n for label, n in zip(labels, degen)]
    return np.hstack(ao_labels)

def _get_beta0(atnoi,atnoj):
    "Resonanace integral for coupling between different atoms"
    return mopac_param.BETA3[atnoi-1,atnoj-1]

def _get_alpha(atnoi,atnoj):
    "Part of the scale factor for the nuclear repulsion"
    return mopac_param.ALP3[atnoi-1,atnoj-1]

def _get_jk_1c_ints(z):
    if z < 3:  # H, He
        j_ints = np.zeros((1,1))
        k_ints = np.zeros((1,1))
        j_ints[0,0] = g_ss[z]
    else:
        j_ints = np.zeros((4,4))
        k_ints = np.zeros((4,4))
        p_diag_idx = ((1, 2, 3), (1, 2, 3))
        # px,py,pz cross terms
        p_off_idx = ((1, 1, 2, 2, 3, 3), (2, 3, 1, 3, 1, 2))

        j_ints[0,0] = g_ss[z]
        j_ints[0,1:] = j_ints[1:,0] = g_sp[z]
        j_ints[p_off_idx] = g_p2[z]
        j_ints[p_diag_idx] = g_pp[z]

        k_ints[0,1:] = k_ints[1:,0] = h_sp[z]
        #print('g_pp', g_pp)
        #print('g_p2', g_p2)
        k_ints[p_off_idx] = 0.5*(g_pp[z]-g_p2[z]) #save h_pp aka hp2 as parameter in file? -CL
        #k_ints[p_off_idx] = mopac_param.HP2M[z]
    return j_ints, k_ints


def _get_reference_energy(mol):
    '''E(Ref) = heat of formation - energy of atomization (kcal/mol)'''
    atom_charges = mol.atom_charges()
    Hf =  mopac_param.EHEAT3[atom_charges].sum()
    Eat = mopac_param.EISOL3[atom_charges].sum()
    return Hf - Eat * mopac_param.EV2KCAL

#if __name__ == '__main__':
#    print('Water:\n')
#    #mol = gto.M(atom=[(8,(0,0,0)),(1,(1.,0.,0.)),(1,(0.0,0.0,1.0))]) #, spin=1)
#    mol = gto.M(atom='methane2.xyz')
#    mol.verbose = 4
#    elements = np.asarray([mol.atom_charge(i) for i in range(mol.natm)])
#    print('elements', elements)
#    mf = RAM1(mol).run(conv_tol=1e-6)
#    print("Enuc:", mf.energy_nuc()*mopac_param.HARTREE2EV)
#    print("Eelec:", (mf.e_tot-mf.energy_nuc())*mopac_param.HARTREE2EV)
#    print(mf.e_heat_formation)

#if __name__ == '__main__':
#    mol = gto.M(atom='''O  0  0  0
#                        O  0  0  1.0''')
#    mol.verbose = 4
#
#    mf = RAM1(mol).run(conv_tol=1e-6)
#    print(mf.e_heat_formation)
#
