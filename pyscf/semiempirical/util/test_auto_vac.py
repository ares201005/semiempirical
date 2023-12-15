from make_vac import *
from compute_VAC_auto import *

r05 = ama = amb = 1.0
ada = adb = 2.0
da = db = 0.2
aqa = aqb = 1.3
qa = qb = 0.5
qa2 = qb2 = 1.0

###

w22 = W22(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2)
index = [0, 1, 3, 6, 2, 4, 7, 5, 8, 9]
w22a = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        w22a[i,j] = w22[index[i], index[j]]
matrix_print_2d(w22a, 10, "w22a")

w223 = W22_auto(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2)
w224 = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        w224[i,j] = w223[index[i], index[j]]
matrix_print_2d(w224, 10, "w22b")

w22diff = w224-w22a
matrix_print_2d(w22diff, 10, 'Difference')

###

w21 = W21(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2)
index = [0, 1, 3, 6, 2, 4, 7, 5, 8, 9]
w21a = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        w21a[i,j] = w21[index[i], index[j]]
matrix_print_2d(w21a, 10, "w21a")

w213 = W21_auto(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2)
w214 = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        w214[i,j] = w213[index[i], index[j]]
matrix_print_2d(w214, 10, "w21b")

w21diff = w214-w21a
matrix_print_2d(w21diff, 10, 'Difference')

###

w11 = W11(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2)
index = [0, 1, 3, 6, 2, 4, 7, 5, 8, 9]
w11a = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        w11a[i,j] = w11[index[i], index[j]]
matrix_print_2d(w11a, 10, "w11a")

w113 = W11_auto(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2)
w114 = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        w114[i,j] = w113[index[i], index[j]]
matrix_print_2d(w114, 10, "w11b")

w11diff = w114-w11a
matrix_print_2d(w11diff, 10, 'Difference')

