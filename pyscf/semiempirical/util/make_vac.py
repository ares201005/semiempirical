import sys
import numpy as np
import numpy as np
from math import sqrt
write = sys.stdout.write

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

def add_square(da, db, wfile):
    if da == 0 and db == 0: return
    term = "("
    if db != 0: term += db
    if da != 0: term += da
    term += ")"
    wfile.write("+{0:s}*{1:s}".format(term,term))

def W22(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2):
    w = np.zeros((10,10))
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

    v = np.zeros(4)
    for ka in range(0, len(phi_a)):
        iwa = phi_a[ka][0]
        for kb in range(0, len(phi_b)):
            iwb = phi_b[kb][0]
            v[0] = phi_b[kb][3] - phi_a[ka][3]
            v[1] = phi_b[kb][4] - phi_a[ka][4]
            v[2] = phi_b[kb][5] - phi_a[ka][5]
            v[3] = phi_b[kb][2] + phi_a[ka][2]
            w[iwa, iwb] += phi_a[ka][1] * phi_b[kb][1] / \
              sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3])
            if iwa == 0 and iwb == 6:
              print(phi_a[ka][1], phi_b[kb][1], v)
              print(phi_a[ka][1] * phi_b[kb][1] / \
                sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]))

    return w

def W22_code_generator(w, wfile):
    phi_aa = [[0,  1.0, '+ama', 0, 0, 0], #s
             [1,  -0.5, '+ada', '-da', 0, 0], #px
             [1,   0.5, '+ada', '+da', 0, 0], #px
             [3,  -0.5, '+ada', 0, '-da', 0], #py
             [3,   0.5, '+ada', 0, '+da', 0], #py
             [6,  -0.5, '+ada', 0, 0, '-da'], #pz
             [6,   0.5, '+ada', 0, 0, '+da'], #pz
             [2,   1.0, '+ama', 0, 0, 0],   #dxx
             [2,  0.25, '+aqa', '-qa2', 0, 0], #dxx
             [2,  -0.5, '+aqa', 0, 0, 0], #dxx
             [2,  0.25, '+aqa', '+qa2', 0, 0], #dxx
             [4,  0.25, '+aqa', '-qa', '-qa', 0], #dxy
             [4, -0.25, '+aqa', '+qa', '-qa', 0], #dxy
             [4,  0.25, '+aqa', '+qa', '+qa', 0], #dxy
             [4, -0.25, '+aqa', '-qa', '+qa', 0], #dxy
             [7,  0.25, '+aqa', '-qa', 0, '-qa'], #dxz
             [7, -0.25, '+aqa', '+qa', 0, '-qa'], #dxz
             [7,  0.25, '+aqa', '+qa', 0, '+qa'], #dxz
             [7, -0.25, '+aqa', '-qa', 0, '+qa'], #dxz
             [5,   1.0, '+ama', 0, 0, 0],   #dyy
             [5,  0.25, '+aqa', 0, '-qa2', 0], #dyy
             [5,  -0.5, '+aqa', 0, 0, 0], #dyy
             [5,  0.25, '+aqa', 0, '+qa2', 0], #dyy
             [8,  0.25, '+aqa', 0, '-qa', '-qa'], #dyz
             [8, -0.25, '+aqa', 0, '+qa', '-qa'], #dyz
             [8,  0.25, '+aqa', 0, '+qa', '+qa'], #dyz
             [8, -0.25, '+aqa', 0, '-qa', '+qa'], #dyz
             [9,   1.0, '+ama', 0, 0, 0],     #dzz
             [9,  0.25, '+aqa', 0, 0, '-qa2'], #dzz
             [9,  -0.5, '+aqa', 0, 0, 0],     #dzz
             [9,  0.25, '+aqa', 0, 0, "+qa2"]] #dzz

    phi_bb = [[0,  1.0, 'amb', 0, 0, 'r'], #s
             [1,  -0.5, 'adb',  'db', 0, "r"], #px
             [1,   0.5, 'adb', '-db', 0, "r"], #px
             [3,  -0.5, 'adb', 0,  'db', "r"], #py
             [3,   0.5, 'adb', 0, '-db', "r"], #py
             [6,  -0.5, 'adb', 0, 0, 'r+db'], #pz
             [6,   0.5, 'adb', 0, 0, 'r-db'], #pz
             [2,   1.0, 'amb', 0, 0, 'r'],   #dxx
             [2,  0.25, 'aqb', '-qb2', 0, 'r'], #dxx
             [2,  -0.5, 'aqb',   0, 0, 'r'], #dxx
             [2,  0.25, 'aqb', '+qb2', 0, 'r'], #dxx
             [4,  0.25, 'aqb', '-qb', '-qb', 'r'], #dxy
             [4, -0.25, 'aqb', '+qb', '-qb', 'r'], #dxy
             [4,  0.25, 'aqb', '+qb', '+qb', 'r'], #dxy
             [4, -0.25, 'aqb', '-qb', '+qb', 'r'], #dxy
             [7,  0.25, 'aqb', '-qb', 0, 'r-qb'], #dxz
             [7, -0.25, 'aqb', '+qb', 0, 'r-qb'], #dxz
             [7,  0.25, 'aqb', '+qb', 0, 'r+qb'], #dxz
             [7, -0.25, 'aqb', '-qb', 0, 'r+qb'], #dxz
             [5,   1.0, 'amb', 0, 0, 'r'],   #dyy
             [5,  0.25, 'aqb', 0, '-qb2', 'r'], #dyy
             [5,  -0.5, 'aqb', 0, 0, 'r'], #dyy
             [5,  0.25, 'aqb', 0, '+qb2', 'r'], #dyy
             [8,  0.25, 'aqb', 0, '-qb', 'r-qb'], #dyz
             [8, -0.25, 'aqb', 0, '+qb', 'r-qb'], #dyz
             [8,  0.25, 'aqb', 0, '+qb', 'r+qb'], #dyz
             [8, -0.25, 'aqb', 0, '-qb', 'r+qb'], #dyz
             [9,   1.0, 'amb', 0, 0, 'r'],     #dzz
             [9,  0.25, 'aqb', 0, 0, 'r-qb2'], #dzz
             [9,  -0.5, 'aqb', 0, 0, 'r'],     #dzz
             [9,  0.25, 'aqb', 0, 0, 'r+qb2']] #dzz

    symbols = ['ss','sx','xx','sy','xy','yy','sz','xz','yz','zz']
    wfile.write("def W22_auto(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2):\n")
    wfile.write("\n    w = np.zeros((10,10))\n")
    wfile.write("    r = 2 * r05\n")
    iwa_old = iwb_old = -10
    for ka in range(0, len(phi_aa)):
        iwa = phi_aa[ka][0]
        for kb in range(0, len(phi_bb)):
            iwb = phi_bb[kb][0]
            if abs(w[iwa,iwb]) > 1e-6:
                if iwa != iwa_old or iwb != iwb_old:
                    wfile.write(f'\n    #w{symbols[iwa]}{symbols[iwb]}\n')
                wfile.write("    dist = sqrt(")
                for m in range(2, 6):
                    add_square(phi_aa[ka][m], phi_bb[kb][m], wfile)
                wfile.write(")\n")
                wfile.write("    w[{0:1d},{1:1d}] += {2:f} / dist\n".format(iwa,iwb,phi_aa[ka][1]*phi_bb[kb][1]))
            iwa_old = iwa
            iwb_old = iwb

    wfile.write("\n    return w\n\n")

def W21(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2):
    w = np.zeros((10,10))
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

    phi_b = [[0,   1.0, amb, 0, 0, r05]] #s

    v = np.zeros(4)
    for ka in range(0, len(phi_a)):
        iwa = phi_a[ka][0]
        for kb in range(0, len(phi_b)):
            iwb = phi_b[kb][0]
            v[0] = phi_b[kb][3] - phi_a[ka][3]
            v[1] = phi_b[kb][4] - phi_a[ka][4]
            v[2] = phi_b[kb][5] - phi_a[ka][5]
            v[3] = phi_b[kb][2] + phi_a[ka][2]
            w[iwa, iwb] += phi_a[ka][1] * phi_b[kb][1] / \
              sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3])
            if iwa == 0 and iwb == 6:
              print(phi_a[ka][1], phi_b[kb][1], v)
              print(phi_a[ka][1] * phi_b[kb][1] / \
                sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]))

    return w

def W21_code_generator(w, wfile):

    phi_aa = [[0,  1.0, '+ama', 0, 0, 0], #s
             [1,  -0.5, '+ada', '-da', 0, 0], #px
             [1,   0.5, '+ada', '+da', 0, 0], #px
             [3,  -0.5, '+ada', 0, '-da', 0], #py
             [3,   0.5, '+ada', 0, '+da', 0], #py
             [6,  -0.5, '+ada', 0, 0, '-da'], #pz
             [6,   0.5, '+ada', 0, 0, '+da'], #pz
             [2,   1.0, '+ama', 0, 0, 0],   #dxx
             [2,  0.25, '+aqa', '-qa2', 0, 0], #dxx
             [2,  -0.5, '+aqa', 0, 0, 0], #dxx
             [2,  0.25, '+aqa', '+qa2', 0, 0], #dxx
             [4,  0.25, '+aqa', '-qa', '-qa', 0], #dxy
             [4, -0.25, '+aqa', '+qa', '-qa', 0], #dxy
             [4,  0.25, '+aqa', '+qa', '+qa', 0], #dxy
             [4, -0.25, '+aqa', '-qa', '+qa', 0], #dxy
             [7,  0.25, '+aqa', '-qa', 0, '-qa'], #dxz
             [7, -0.25, '+aqa', '+qa', 0, '-qa'], #dxz
             [7,  0.25, '+aqa', '+qa', 0, '+qa'], #dxz
             [7, -0.25, '+aqa', '-qa', 0, '+qa'], #dxz
             [5,   1.0, '+ama', 0, 0, 0],   #dyy
             [5,  0.25, '+aqa', 0, '-qa2', 0], #dyy
             [5,  -0.5, '+aqa', 0, 0, 0], #dyy
             [5,  0.25, '+aqa', 0, '+qa2', 0], #dyy
             [8,  0.25, '+aqa', 0, '-qa', '-qa'], #dyz
             [8, -0.25, '+aqa', 0, '+qa', '-qa'], #dyz
             [8,  0.25, '+aqa', 0, '+qa', '+qa'], #dyz
             [8, -0.25, '+aqa', 0, '-qa', '+qa'], #dyz
             [9,   1.0, '+ama', 0, 0, 0],     #dzz
             [9,  0.25, '+aqa', 0, 0, '-qa2'], #dzz
             [9,  -0.5, '+aqa', 0, 0, 0],     #dzz
             [9,  0.25, '+aqa', 0, 0, "+qa2"]] #dzz

    phi_bb = [[0,  1.0, 'amb', 0, 0, 'r']] #s

    symbols = ['ss','sx','xx','sy','xy','yy','sz','xz','yz','zz']
    wfile.write("def W21_auto(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2):\n")
    wfile.write("\n    w = np.zeros((10,10))\n")
    wfile.write("    r = 2 * r05\n")
    iwa_old = iwb_old = -10
    for ka in range(0, len(phi_aa)):
        iwa = phi_aa[ka][0]
        for kb in range(0, len(phi_bb)):
            iwb = phi_bb[kb][0]
            if abs(w[iwa,iwb]) > 1e-6:
                #wfile.write("iw:", iwa, iwb, iwa_old, iwb_old)
                if iwa != iwa_old or iwb != iwb_old:
                    wfile.write(f'\n    #w{symbols[iwa]}{symbols[iwb]}\n')
                wfile.write("    dist = sqrt(")
                for m in range(2, 6):
                    add_square(phi_aa[ka][m], phi_bb[kb][m], wfile)
                wfile.write(")\n")
                wfile.write("    w[{0:1d},{1:1d}] += {2:f} / dist\n".format(iwa,iwb,phi_aa[ka][1]*phi_bb[kb][1]))
            iwa_old = iwa
            iwb_old = iwb

    wfile.write("\n    return w\n\n")

def W11(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2):
    w = np.zeros((10,10))
    phi_a = [[0,   1.0, ama, 0, 0, -r05]] #s

    phi_b = [[0,   1.0, amb, 0, 0, r05]] #s

    v = np.zeros(4)
    for ka in range(0, len(phi_a)):
        iwa = phi_a[ka][0]
        for kb in range(0, len(phi_b)):
            iwb = phi_b[kb][0]
            v[0] = phi_b[kb][3] - phi_a[ka][3]
            v[1] = phi_b[kb][4] - phi_a[ka][4]
            v[2] = phi_b[kb][5] - phi_a[ka][5]
            v[3] = phi_b[kb][2] + phi_a[ka][2]
            w[iwa, iwb] += phi_a[ka][1] * phi_b[kb][1] / \
              sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3])
            if iwa == 0 and iwb == 6:
              print(phi_a[ka][1], phi_b[kb][1], v)
              print(phi_a[ka][1] * phi_b[kb][1] / \
                sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]))

    return w

def W11_code_generator(w, wfile):

    phi_aa = [[0,  1.0, '+ama', 0, 0, 0]] #s

    phi_bb = [[0,  1.0, 'amb', 0, 0, 'r']] #s

    symbols = ['ss','sx','xx','sy','xy','yy','sz','xz','yz','zz']
    wfile.write("def W11_auto(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2):\n")
    wfile.write("\n    w = np.zeros((10,10))\n")
    wfile.write("    r = 2 * r05\n")
    iwa_old = iwb_old = -10
    for ka in range(0, len(phi_aa)):
        iwa = phi_aa[ka][0]
        for kb in range(0, len(phi_bb)):
            iwb = phi_bb[kb][0]
            if abs(w[iwa,iwb]) > 1e-6:
                if iwa != iwa_old or iwb != iwb_old:
                    wfile.write(f'\n    #w{symbols[iwa]}{symbols[iwb]}\n')
                wfile.write("    dist = sqrt(")
                for m in range(2, 6):
                    add_square(phi_aa[ka][m], phi_bb[kb][m], wfile)
                wfile.write(")\n")
                wfile.write("    w[{0:1d},{1:1d}] += {2:f} / dist\n".format(iwa,iwb,phi_aa[ka][1]*phi_bb[kb][1]))
            iwa_old = iwa
            iwb_old = iwb

    wfile.write("\n    return w")

with  open('compute_VAC_auto.py','w') as wfile:
    r05 = ama = amb = 1.0
    ada = adb = 2.0
    da = db = 0.2
    aqa = aqb = 1.3
    qa = qb = 0.5
    qa2 = qb2 = 1.0
    index = [0, 1, 3, 6, 2, 4, 7, 5, 8, 9]
    wfile.write('import numpy as np\n')
    wfile.write('from math import sqrt\n')
    w22 = W22(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2)
    W22_code_generator(w22, wfile)
    w21 = W21(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2)
    W21_code_generator(w21, wfile)
    w11 = W11(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2)
    W11_code_generator(w11, wfile)
    #matrix_print_2d(w21, 10, 'W matrix')
