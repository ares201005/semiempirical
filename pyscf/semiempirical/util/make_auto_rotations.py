import sys
from math import cos, sin, acos, atan
import random
import numpy as np
write = sys.stdout.write


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

  return T2info

T2info = T2_info()


def rotation_matrix(xij):

    sij = xij / np.linalg.norm(xij)
    theta = acos(sij[2])
    if abs(sij[0]) + abs(sij[1]) < 1e-8:
       phi = 0.0
    elif abs(sij[0]) < 1e-8:
       if sij[1] > 0:
          phi = np.pi / 2.0
       else:
          phi = -np.pi / 2.0
    else:
       phi = atan(sij[1]/sij[0])
    if abs(phi) < 1e-8 and sij[0] < -1.0e-8:
       phi = np.pi

    T = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)],
                  [0.0, -sin(phi), cos(phi), 0.0], [0.0, sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]])
    return T, theta, phi

def fdiff_Tx(xij):
  dThetadX = np.zeros((3))
  dPhidX = np.zeros((3))
  Tx = np.zeros((4,4,3))
  delta = 0.0001
  for i in range(0, 3):
    dx = np.zeros(3)
    dx[i] = delta
    T_p, theta_p, phi_p = rotation_matrix(xij+dx)
    T_m, theta_m, phi_m = rotation_matrix(xij-dx)
    dThetadX[i] = (theta_p-theta_m) / (2*delta)
    dPhidX[i]   = (phi_p-phi_m) / (2*delta)
    Tx[:,:,i] = (T_p-T_m) / (2*delta)
  return Tx

def Tx_values(xij):
    rij = np.linalg.norm(xij)
    sij = xij / rij
    theta = acos(sij[2])
    if theta < 1e-8:
      dThetadX = np.array([0.0, 0.0, 0.0])
    else:
      dThetadX = - sqrt(1+xij[2]*xij[2]/(xij[0]*xij[0]+xij[1]*xij[1])) * \
            np.array([-xij[0]*xij[2]/pow(rij,3.0), -xij[1]*xij[2]/pow(rij,3.0), \
            1.0/rij - xij[2]*xij[2]/pow(rij, 3.0)])
    if abs(sij[0]) + abs(sij[1]) < 1e-8:
       phi = 0.0
    elif abs(sij[0]) < 1e-8:
       if sij[1] > 0:
          phi = np.pi / 2.0
       else:
          phi = -np.pi / 2.0
    else:
       phi = atan(sij[1]/sij[0])
    if abs(phi) < 1e-8 and sij[0] < -1.0e-8:
       phi = np.pi
    xy2 = xij[0]*xij[0] + xij[1]*xij[1]
    if xy2 < 1e-8:
      dPhidX = np.array([0.0, 0.0, 0.0])
    else:
      dPhidX = np.array([-xij[1]/xy2, xij[0]/xy2, 0.0])

    T = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)],
                  [0.0, -sin(phi), cos(phi), 0.0], [0.0, sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]])

    zeros = np.zeros(3)
    Tx = np.array([[zeros, zeros, zeros, zeros],
                      [zeros, -sin(theta)*cos(phi) * dThetadX - cos(theta)*sin(phi) * dPhidX,
                       -sin(theta)*sin(phi) * dThetadX + cos(theta) * cos(phi) * dPhidX, -cos(theta) * dThetadX],
                      [zeros, -cos(phi)*dPhidX, -sin(phi)*dPhidX, zeros],
                      [zeros, cos(theta)*cos(phi)*dThetadX-sin(theta)*sin(phi)*dPhidX,
                       cos(theta)*sin(phi)*dThetadX+sin(theta)*cos(phi)*dPhidX, -sin(theta)*dThetadX]])
    return Tx

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

def T2_matrix(T):
    T2 = np.zeros((10,10))
    ii = 0
    for j in range(0, 4):
        for i in range(0, j+1):
            prod = np.einsum('m,n->mn', T[:,i], T[:,j])
            kk = 0
            for l in range(0, 4):
                for k in range(0, l+1):
                    T2[kk, ii] = prod[k, l]
                    if k != l: 
                        T2[kk, ii] += prod[l, k]
                    kk += 1
            ii += 1
    return T2

def T2_matrix_from_index(T, T2info):
  T2 = np.zeros((10,10))
  count = len(T2info)
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
      T2[kk,ll] += c*T[m1,m2]*T[n1,n2]

  return T2

def rotation_matrix_code_generator(T, T2info):
    with open('rotation_matrix_auto.py','w') as wfile:
        wfile.write('import numpy as np\n')
        wfile.write('from math  import cos, sin, acos, atan\n')
        wfile.write("def rotation_matrix_auto(xij):\n")
        wfile.write("""    sij = xij / np.linalg.norm(xij)
    theta = acos(sij[2])
    if abs(sij[0]) + abs(sij[1]) < 1e-8:
        phi = 0.0
    elif abs(sij[0]) < 1e-8:
        if sij[1] > 0:
            phi = np.pi / 2.0
        else:
            phi = -np.pi / 2.0
    else:
        phi = atan(sij[1]/sij[0])
    if abs(phi) < 1e-8 and sij[0] < -1.0e-8:
        phi = np.pi

    T = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)],
                     [0.0, -sin(phi), cos(phi), 0.0], [0.0, sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]])\n""")
        wfile.write("    T2 = np.zeros((10,10))\n\n")
        
        T2 = np.zeros((10,10))
        count = len(T2info)
        for i in range(0, count):
            kk = T2info[i][0]
            ll = T2info[i][1]
            nt = len(T2info[i])
            wfile.write(f'    T2[{kk}][{ll}] = ')
            for j in range(2, nt):
                m = T2info[i][j][0]
                n = T2info[i][j][1]
                c = T2info[i][j][2]
                m2 = int(m/4)
                m1 = m - m2*4
                n2 = int(n/4)
                n1 = n - n2*4
                T2[kk,ll] += c*T[m1,m2]*T[n1,n2]
                if j > 2:
                    if c > 1.0:
                        wfile.write(f'+{c}*T[{m1},{m2}]*T[{n1},{n2}]')
                    else:
                        wfile.write(f'+T[{m1},{m2}]*T[{n1},{n2}]')
                else:
                    if c > 1.0:
                        wfile.write(f'{c}*T[{m1},{m2}]*T[{n1},{n2}]')
                    else:
                        wfile.write(f'    T[{m1},{m2}]*T[{n1},{n2}]')
            wfile.write('\n')
        wfile.write("\n    return T2")


T2sum =  np.zeros((10,10))
T22sum = np.zeros((10,10))
Tnewsum = np.zeros((10,10))
Tnew = np.zeros((10,10))
T2 = np.zeros((10,10))
T = np.zeros((10,10))
theta = 0.0
phi = 0.0
s = np.array([random.random(), random.random(), random.random()])

T, theta, phi = rotation_matrix(s)
T2 = T2_matrix(T)
#matrix_print_2d(T2, 10, "T2")
T2info = T2_info()
T22 = T2_matrix_from_index(T, T2info)
#matrix_print_2d(T22, 10, "T22")
rotation_matrix_code_generator(T, T2info)
