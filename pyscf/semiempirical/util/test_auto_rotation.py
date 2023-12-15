import numpy as np
import random
from make_auto_rotations import *
from rotation_matrix_auto import *

T2 = np.zeros((10,10))
T = np.zeros((10,10))
theta = 0.0
phi = 0.0
s = np.array([random.random(), random.random(), random.random()])
print('s',s)

T, theta, phi = rotation_matrix(s)
T2 = T2_matrix(T)
matrix_print_2d(T2, 10, "T2")
T2info = T2_info()
T22 = T2_matrix_from_index(T, T2info)

Tnew = rotation_matrix_auto(s)
matrix_print_2d(Tnew, 10, "Tnew")
matrix_print_2d(T2-Tnew, 10, "Difference")
