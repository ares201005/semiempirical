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

    T = numpy.array([[1.0, 0.0, 0.0, 0.0], [0.0, sb*ca, sb*sa, cb], 
                     [0.0, cb*ca, cb*sa, -sb], [0.0, -sa, ca, 0.0]])

    return T

def is_zero(k):
  if k in [1,2,3,4,8,12,14]:
    return -1
  else:
    return k

def T2_info():
  T2index = np.zeros((10,10,3*256))
  T2count = np.zeros((10,10))
  ii = 0
  for j in range(0, 4):
    for i in range(0, j+1):
        prod = np.zeros((4,4,2))
        for m in range(0, 4):
            for n in range(0, 4):
                prod[m,n,0] = is_zero(m+4*i)
                prod[m,n,1] = is_zero(n+4*j)
                if prod[m,n,0] > prod[m,n,1]:
                  mn = prod[m,n,0]
                  prod[m,n,0] = prod[m,n,1]
                  prod[m,n,1] = mn
        kk = 0
        for l in range(0, 4):
            for k in range(0, l+1):
                if prod[k,l,0] > -0.5 and prod[k,l,1] > -0.5:
                    count = int(T2count[kk, ii]+0.01)
                    T2index[kk, ii, 3*count+0] = prod[k,l,0]
                    T2index[kk, ii, 3*count+1] = prod[k,l,1]
                    T2index[kk, ii, 3*count+2] = 1
                    T2count[kk, ii] += 1
                if k != l and prod[l,k,0] > -0.5 and prod[k,k,1] > -0.5:
                    count = int(T2count[kk, ii]+0.01)
                    skip = False
                    for c in range(0, count):
                        if abs(prod[l,k,0] - T2index[kk, ii, 3*c+0]) < 0.01 and abs(prod[l,k,1] - T2index[kk, ii, 3*c+1]) < 0.01:
                           T2index[kk, ii, 3*c+2] += 1
                           skip = True
                    if skip != True:
                        T2index[kk, ii, 3*count+0] = prod[l,k,0]
                        T2index[kk, ii, 3*count+1] = prod[l,k,1]
                        T2index[kk, ii, 3*count+2] = 1
                        T2count[kk, ii] += 1
                kk += 1
        ii += 1

  T2info = []
  for ll in range(0, 10):
    for kk in range(0, 10):
      info = []
      count = int(T2count[kk,ll]+0.01)
      for c in range(0, count):
        if c == 0: 
          info.append(kk)
          info.append(ll)
        info.append([int(T2index[kk,ll,3*c+0]+0.01),int(T2index[kk,ll,3*c+1]+0.01),T2index[kk,ll,3*c+2]])
      if count!= 0: T2info.append(info)
  #print("T2info:", T2info)

  return T2info

def T2_matrix_from_index(T, T2info):
  T2 = np.zeros((10,10))
  count = len(T2info)
  print("count",count)
  for i in range(0, count):
    kk = T2info[i][0]
    ll = T2info[i][1]
    nt = len(T2info[i])
    for j in range(2, nt):
      m = T2info[i][j][0] 
      n = T2info[i][j][1]
      c = T2info[i][j][2]
      m2 = int(m/4)
      m1 = m - m2*4
      n2 = int(n/4)
      n1 = n - n2*4
      #print(kk, ll, m1, m2, n1, n2, c)
      T2[kk,ll] += c*T[m1,m2]*T[n1,n2]

  return T2

