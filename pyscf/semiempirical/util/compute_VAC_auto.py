import numpy as np
from math import sqrt
def W22_auto(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2):

    w = np.zeros((10,10))
    r = 2 * r05

    #wssss
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[0,0] += 1.000000 / dist

    #wsssz
    dist = sqrt(+(adb+ama)*(adb+ama)+(r+db)*(r+db))
    w[0,6] += -0.500000 / dist
    dist = sqrt(+(adb+ama)*(adb+ama)+(r-db)*(r-db))
    w[0,6] += 0.500000 / dist

    #wssxx
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[0,2] += 1.000000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(-qb2)*(-qb2)+(r)*(r))
    w[0,2] += 0.250000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r)*(r))
    w[0,2] += -0.500000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(+qb2)*(+qb2)+(r)*(r))
    w[0,2] += 0.250000 / dist

    #wssyy
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[0,5] += 1.000000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(-qb2)*(-qb2)+(r)*(r))
    w[0,5] += 0.250000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r)*(r))
    w[0,5] += -0.500000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(+qb2)*(+qb2)+(r)*(r))
    w[0,5] += 0.250000 / dist

    #wsszz
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[0,9] += 1.000000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r-qb2)*(r-qb2))
    w[0,9] += 0.250000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r)*(r))
    w[0,9] += -0.500000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r+qb2)*(r+qb2))
    w[0,9] += 0.250000 / dist

    #wsxsx
    dist = sqrt(+(adb+ada)*(adb+ada)+(db-da)*(db-da)+(r)*(r))
    w[1,1] += 0.250000 / dist
    dist = sqrt(+(adb+ada)*(adb+ada)+(-db-da)*(-db-da)+(r)*(r))
    w[1,1] += -0.250000 / dist

    #wsxxz
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(-qb-da)*(-qb-da)+(r-qb)*(r-qb))
    w[1,7] += -0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(+qb-da)*(+qb-da)+(r-qb)*(r-qb))
    w[1,7] += 0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(+qb-da)*(+qb-da)+(r+qb)*(r+qb))
    w[1,7] += -0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(-qb-da)*(-qb-da)+(r+qb)*(r+qb))
    w[1,7] += 0.125000 / dist

    #wsxsx
    dist = sqrt(+(adb+ada)*(adb+ada)+(db+da)*(db+da)+(r)*(r))
    w[1,1] += -0.250000 / dist
    dist = sqrt(+(adb+ada)*(adb+ada)+(-db+da)*(-db+da)+(r)*(r))
    w[1,1] += 0.250000 / dist

    #wsxxz
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(-qb+da)*(-qb+da)+(r-qb)*(r-qb))
    w[1,7] += 0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(+qb+da)*(+qb+da)+(r-qb)*(r-qb))
    w[1,7] += -0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(+qb+da)*(+qb+da)+(r+qb)*(r+qb))
    w[1,7] += 0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(-qb+da)*(-qb+da)+(r+qb)*(r+qb))
    w[1,7] += -0.125000 / dist

    #wsysy
    dist = sqrt(+(adb+ada)*(adb+ada)+(db-da)*(db-da)+(r)*(r))
    w[3,3] += 0.250000 / dist
    dist = sqrt(+(adb+ada)*(adb+ada)+(-db-da)*(-db-da)+(r)*(r))
    w[3,3] += -0.250000 / dist

    #wsyyz
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(-qb-da)*(-qb-da)+(r-qb)*(r-qb))
    w[3,8] += -0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(+qb-da)*(+qb-da)+(r-qb)*(r-qb))
    w[3,8] += 0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(+qb-da)*(+qb-da)+(r+qb)*(r+qb))
    w[3,8] += -0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(-qb-da)*(-qb-da)+(r+qb)*(r+qb))
    w[3,8] += 0.125000 / dist

    #wsysy
    dist = sqrt(+(adb+ada)*(adb+ada)+(db+da)*(db+da)+(r)*(r))
    w[3,3] += -0.250000 / dist
    dist = sqrt(+(adb+ada)*(adb+ada)+(-db+da)*(-db+da)+(r)*(r))
    w[3,3] += 0.250000 / dist

    #wsyyz
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(-qb+da)*(-qb+da)+(r-qb)*(r-qb))
    w[3,8] += 0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(+qb+da)*(+qb+da)+(r-qb)*(r-qb))
    w[3,8] += -0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(+qb+da)*(+qb+da)+(r+qb)*(r+qb))
    w[3,8] += 0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(-qb+da)*(-qb+da)+(r+qb)*(r+qb))
    w[3,8] += -0.125000 / dist

    #wszss
    dist = sqrt(+(amb+ada)*(amb+ada)+(r-da)*(r-da))
    w[6,0] += -0.500000 / dist

    #wszsz
    dist = sqrt(+(adb+ada)*(adb+ada)+(r+db-da)*(r+db-da))
    w[6,6] += 0.250000 / dist
    dist = sqrt(+(adb+ada)*(adb+ada)+(r-db-da)*(r-db-da))
    w[6,6] += -0.250000 / dist

    #wszxx
    dist = sqrt(+(amb+ada)*(amb+ada)+(r-da)*(r-da))
    w[6,2] += -0.500000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(-qb2)*(-qb2)+(r-da)*(r-da))
    w[6,2] += -0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(r-da)*(r-da))
    w[6,2] += 0.250000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(+qb2)*(+qb2)+(r-da)*(r-da))
    w[6,2] += -0.125000 / dist

    #wszyy
    dist = sqrt(+(amb+ada)*(amb+ada)+(r-da)*(r-da))
    w[6,5] += -0.500000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(-qb2)*(-qb2)+(r-da)*(r-da))
    w[6,5] += -0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(r-da)*(r-da))
    w[6,5] += 0.250000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(+qb2)*(+qb2)+(r-da)*(r-da))
    w[6,5] += -0.125000 / dist

    #wszzz
    dist = sqrt(+(amb+ada)*(amb+ada)+(r-da)*(r-da))
    w[6,9] += -0.500000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(r-qb2-da)*(r-qb2-da))
    w[6,9] += -0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(r-da)*(r-da))
    w[6,9] += 0.250000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(r+qb2-da)*(r+qb2-da))
    w[6,9] += -0.125000 / dist

    #wszss
    dist = sqrt(+(amb+ada)*(amb+ada)+(r+da)*(r+da))
    w[6,0] += 0.500000 / dist

    #wszsz
    dist = sqrt(+(adb+ada)*(adb+ada)+(r+db+da)*(r+db+da))
    w[6,6] += -0.250000 / dist
    dist = sqrt(+(adb+ada)*(adb+ada)+(r-db+da)*(r-db+da))
    w[6,6] += 0.250000 / dist

    #wszxx
    dist = sqrt(+(amb+ada)*(amb+ada)+(r+da)*(r+da))
    w[6,2] += 0.500000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(-qb2)*(-qb2)+(r+da)*(r+da))
    w[6,2] += 0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(r+da)*(r+da))
    w[6,2] += -0.250000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(+qb2)*(+qb2)+(r+da)*(r+da))
    w[6,2] += 0.125000 / dist

    #wszyy
    dist = sqrt(+(amb+ada)*(amb+ada)+(r+da)*(r+da))
    w[6,5] += 0.500000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(-qb2)*(-qb2)+(r+da)*(r+da))
    w[6,5] += 0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(r+da)*(r+da))
    w[6,5] += -0.250000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(+qb2)*(+qb2)+(r+da)*(r+da))
    w[6,5] += 0.125000 / dist

    #wszzz
    dist = sqrt(+(amb+ada)*(amb+ada)+(r+da)*(r+da))
    w[6,9] += 0.500000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(r-qb2+da)*(r-qb2+da))
    w[6,9] += 0.125000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(r+da)*(r+da))
    w[6,9] += -0.250000 / dist
    dist = sqrt(+(aqb+ada)*(aqb+ada)+(r+qb2+da)*(r+qb2+da))
    w[6,9] += 0.125000 / dist

    #wxxss
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[2,0] += 1.000000 / dist

    #wxxsz
    dist = sqrt(+(adb+ama)*(adb+ama)+(r+db)*(r+db))
    w[2,6] += -0.500000 / dist
    dist = sqrt(+(adb+ama)*(adb+ama)+(r-db)*(r-db))
    w[2,6] += 0.500000 / dist

    #wxxxx
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[2,2] += 1.000000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(-qb2)*(-qb2)+(r)*(r))
    w[2,2] += 0.250000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r)*(r))
    w[2,2] += -0.500000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(+qb2)*(+qb2)+(r)*(r))
    w[2,2] += 0.250000 / dist

    #wxxyy
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[2,5] += 1.000000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(-qb2)*(-qb2)+(r)*(r))
    w[2,5] += 0.250000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r)*(r))
    w[2,5] += -0.500000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(+qb2)*(+qb2)+(r)*(r))
    w[2,5] += 0.250000 / dist

    #wxxzz
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[2,9] += 1.000000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r-qb2)*(r-qb2))
    w[2,9] += 0.250000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r)*(r))
    w[2,9] += -0.500000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r+qb2)*(r+qb2))
    w[2,9] += 0.250000 / dist

    #wxxss
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[2,0] += 0.250000 / dist

    #wxxsz
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(-qa2)*(-qa2)+(r+db)*(r+db))
    w[2,6] += -0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(-qa2)*(-qa2)+(r-db)*(r-db))
    w[2,6] += 0.125000 / dist

    #wxxxx
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[2,2] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2-qa2)*(-qb2-qa2)+(r)*(r))
    w[2,2] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[2,2] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2-qa2)*(+qb2-qa2)+(r)*(r))
    w[2,2] += 0.062500 / dist

    #wxxyy
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[2,5] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qa2)*(-qa2)+(-qb2)*(-qb2)+(r)*(r))
    w[2,5] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[2,5] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qa2)*(-qa2)+(+qb2)*(+qb2)+(r)*(r))
    w[2,5] += 0.062500 / dist

    #wxxzz
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[2,9] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qa2)*(-qa2)+(r-qb2)*(r-qb2))
    w[2,9] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[2,9] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qa2)*(-qa2)+(r+qb2)*(r+qb2))
    w[2,9] += 0.062500 / dist

    #wxxss
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[2,0] += -0.500000 / dist

    #wxxsz
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(r+db)*(r+db))
    w[2,6] += 0.250000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(r-db)*(r-db))
    w[2,6] += -0.250000 / dist

    #wxxxx
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[2,2] += -0.500000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2)*(-qb2)+(r)*(r))
    w[2,2] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r)*(r))
    w[2,2] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2)*(+qb2)+(r)*(r))
    w[2,2] += -0.125000 / dist

    #wxxyy
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[2,5] += -0.500000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2)*(-qb2)+(r)*(r))
    w[2,5] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r)*(r))
    w[2,5] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2)*(+qb2)+(r)*(r))
    w[2,5] += -0.125000 / dist

    #wxxzz
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[2,9] += -0.500000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r-qb2)*(r-qb2))
    w[2,9] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r)*(r))
    w[2,9] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r+qb2)*(r+qb2))
    w[2,9] += -0.125000 / dist

    #wxxss
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[2,0] += 0.250000 / dist

    #wxxsz
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(+qa2)*(+qa2)+(r+db)*(r+db))
    w[2,6] += -0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(+qa2)*(+qa2)+(r-db)*(r-db))
    w[2,6] += 0.125000 / dist

    #wxxxx
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[2,2] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2+qa2)*(-qb2+qa2)+(r)*(r))
    w[2,2] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[2,2] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2+qa2)*(+qb2+qa2)+(r)*(r))
    w[2,2] += 0.062500 / dist

    #wxxyy
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[2,5] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qa2)*(+qa2)+(-qb2)*(-qb2)+(r)*(r))
    w[2,5] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[2,5] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qa2)*(+qa2)+(+qb2)*(+qb2)+(r)*(r))
    w[2,5] += 0.062500 / dist

    #wxxzz
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[2,9] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qa2)*(+qa2)+(r-qb2)*(r-qb2))
    w[2,9] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[2,9] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qa2)*(+qa2)+(r+qb2)*(r+qb2))
    w[2,9] += 0.062500 / dist

    #wxyxy
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb-qa)*(-qb-qa)+(-qb-qa)*(-qb-qa)+(r)*(r))
    w[4,4] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb-qa)*(+qb-qa)+(-qb-qa)*(-qb-qa)+(r)*(r))
    w[4,4] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb-qa)*(+qb-qa)+(+qb-qa)*(+qb-qa)+(r)*(r))
    w[4,4] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb-qa)*(-qb-qa)+(+qb-qa)*(+qb-qa)+(r)*(r))
    w[4,4] += -0.062500 / dist

    #wxyxy
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb+qa)*(-qb+qa)+(-qb-qa)*(-qb-qa)+(r)*(r))
    w[4,4] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb+qa)*(+qb+qa)+(-qb-qa)*(-qb-qa)+(r)*(r))
    w[4,4] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb+qa)*(+qb+qa)+(+qb-qa)*(+qb-qa)+(r)*(r))
    w[4,4] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb+qa)*(-qb+qa)+(+qb-qa)*(+qb-qa)+(r)*(r))
    w[4,4] += 0.062500 / dist

    #wxyxy
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb+qa)*(-qb+qa)+(-qb+qa)*(-qb+qa)+(r)*(r))
    w[4,4] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb+qa)*(+qb+qa)+(-qb+qa)*(-qb+qa)+(r)*(r))
    w[4,4] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb+qa)*(+qb+qa)+(+qb+qa)*(+qb+qa)+(r)*(r))
    w[4,4] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb+qa)*(-qb+qa)+(+qb+qa)*(+qb+qa)+(r)*(r))
    w[4,4] += -0.062500 / dist

    #wxyxy
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb-qa)*(-qb-qa)+(-qb+qa)*(-qb+qa)+(r)*(r))
    w[4,4] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb-qa)*(+qb-qa)+(-qb+qa)*(-qb+qa)+(r)*(r))
    w[4,4] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb-qa)*(+qb-qa)+(+qb+qa)*(+qb+qa)+(r)*(r))
    w[4,4] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb-qa)*(-qb-qa)+(+qb+qa)*(+qb+qa)+(r)*(r))
    w[4,4] += 0.062500 / dist

    #wxzsx
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(db-qa)*(db-qa)+(r-qa)*(r-qa))
    w[7,1] += -0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(-db-qa)*(-db-qa)+(r-qa)*(r-qa))
    w[7,1] += 0.125000 / dist

    #wxzxz
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb-qa)*(-qb-qa)+(r-qb-qa)*(r-qb-qa))
    w[7,7] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb-qa)*(+qb-qa)+(r-qb-qa)*(r-qb-qa))
    w[7,7] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb-qa)*(+qb-qa)+(r+qb-qa)*(r+qb-qa))
    w[7,7] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb-qa)*(-qb-qa)+(r+qb-qa)*(r+qb-qa))
    w[7,7] += -0.062500 / dist

    #wxzsx
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(db+qa)*(db+qa)+(r-qa)*(r-qa))
    w[7,1] += 0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(-db+qa)*(-db+qa)+(r-qa)*(r-qa))
    w[7,1] += -0.125000 / dist

    #wxzxz
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb+qa)*(-qb+qa)+(r-qb-qa)*(r-qb-qa))
    w[7,7] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb+qa)*(+qb+qa)+(r-qb-qa)*(r-qb-qa))
    w[7,7] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb+qa)*(+qb+qa)+(r+qb-qa)*(r+qb-qa))
    w[7,7] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb+qa)*(-qb+qa)+(r+qb-qa)*(r+qb-qa))
    w[7,7] += 0.062500 / dist

    #wxzsx
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(db+qa)*(db+qa)+(r+qa)*(r+qa))
    w[7,1] += -0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(-db+qa)*(-db+qa)+(r+qa)*(r+qa))
    w[7,1] += 0.125000 / dist

    #wxzxz
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb+qa)*(-qb+qa)+(r-qb+qa)*(r-qb+qa))
    w[7,7] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb+qa)*(+qb+qa)+(r-qb+qa)*(r-qb+qa))
    w[7,7] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb+qa)*(+qb+qa)+(r+qb+qa)*(r+qb+qa))
    w[7,7] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb+qa)*(-qb+qa)+(r+qb+qa)*(r+qb+qa))
    w[7,7] += -0.062500 / dist

    #wxzsx
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(db-qa)*(db-qa)+(r+qa)*(r+qa))
    w[7,1] += 0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(-db-qa)*(-db-qa)+(r+qa)*(r+qa))
    w[7,1] += -0.125000 / dist

    #wxzxz
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb-qa)*(-qb-qa)+(r-qb+qa)*(r-qb+qa))
    w[7,7] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb-qa)*(+qb-qa)+(r-qb+qa)*(r-qb+qa))
    w[7,7] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb-qa)*(+qb-qa)+(r+qb+qa)*(r+qb+qa))
    w[7,7] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb-qa)*(-qb-qa)+(r+qb+qa)*(r+qb+qa))
    w[7,7] += 0.062500 / dist

    #wyyss
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[5,0] += 1.000000 / dist

    #wyysz
    dist = sqrt(+(adb+ama)*(adb+ama)+(r+db)*(r+db))
    w[5,6] += -0.500000 / dist
    dist = sqrt(+(adb+ama)*(adb+ama)+(r-db)*(r-db))
    w[5,6] += 0.500000 / dist

    #wyyxx
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[5,2] += 1.000000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(-qb2)*(-qb2)+(r)*(r))
    w[5,2] += 0.250000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r)*(r))
    w[5,2] += -0.500000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(+qb2)*(+qb2)+(r)*(r))
    w[5,2] += 0.250000 / dist

    #wyyyy
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[5,5] += 1.000000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(-qb2)*(-qb2)+(r)*(r))
    w[5,5] += 0.250000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r)*(r))
    w[5,5] += -0.500000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(+qb2)*(+qb2)+(r)*(r))
    w[5,5] += 0.250000 / dist

    #wyyzz
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[5,9] += 1.000000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r-qb2)*(r-qb2))
    w[5,9] += 0.250000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r)*(r))
    w[5,9] += -0.500000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r+qb2)*(r+qb2))
    w[5,9] += 0.250000 / dist

    #wyyss
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[5,0] += 0.250000 / dist

    #wyysz
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(-qa2)*(-qa2)+(r+db)*(r+db))
    w[5,6] += -0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(-qa2)*(-qa2)+(r-db)*(r-db))
    w[5,6] += 0.125000 / dist

    #wyyxx
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[5,2] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2)*(-qb2)+(-qa2)*(-qa2)+(r)*(r))
    w[5,2] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[5,2] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2)*(+qb2)+(-qa2)*(-qa2)+(r)*(r))
    w[5,2] += 0.062500 / dist

    #wyyyy
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[5,5] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2-qa2)*(-qb2-qa2)+(r)*(r))
    w[5,5] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[5,5] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2-qa2)*(+qb2-qa2)+(r)*(r))
    w[5,5] += 0.062500 / dist

    #wyyzz
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[5,9] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qa2)*(-qa2)+(r-qb2)*(r-qb2))
    w[5,9] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[5,9] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qa2)*(-qa2)+(r+qb2)*(r+qb2))
    w[5,9] += 0.062500 / dist

    #wyyss
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[5,0] += -0.500000 / dist

    #wyysz
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(r+db)*(r+db))
    w[5,6] += 0.250000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(r-db)*(r-db))
    w[5,6] += -0.250000 / dist

    #wyyxx
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[5,2] += -0.500000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2)*(-qb2)+(r)*(r))
    w[5,2] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r)*(r))
    w[5,2] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2)*(+qb2)+(r)*(r))
    w[5,2] += -0.125000 / dist

    #wyyyy
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[5,5] += -0.500000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2)*(-qb2)+(r)*(r))
    w[5,5] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r)*(r))
    w[5,5] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2)*(+qb2)+(r)*(r))
    w[5,5] += -0.125000 / dist

    #wyyzz
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[5,9] += -0.500000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r-qb2)*(r-qb2))
    w[5,9] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r)*(r))
    w[5,9] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r+qb2)*(r+qb2))
    w[5,9] += -0.125000 / dist

    #wyyss
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[5,0] += 0.250000 / dist

    #wyysz
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(+qa2)*(+qa2)+(r+db)*(r+db))
    w[5,6] += -0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(+qa2)*(+qa2)+(r-db)*(r-db))
    w[5,6] += 0.125000 / dist

    #wyyxx
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[5,2] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2)*(-qb2)+(+qa2)*(+qa2)+(r)*(r))
    w[5,2] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[5,2] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2)*(+qb2)+(+qa2)*(+qa2)+(r)*(r))
    w[5,2] += 0.062500 / dist

    #wyyyy
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[5,5] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2+qa2)*(-qb2+qa2)+(r)*(r))
    w[5,5] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[5,5] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2+qa2)*(+qb2+qa2)+(r)*(r))
    w[5,5] += 0.062500 / dist

    #wyyzz
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[5,9] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qa2)*(+qa2)+(r-qb2)*(r-qb2))
    w[5,9] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[5,9] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qa2)*(+qa2)+(r+qb2)*(r+qb2))
    w[5,9] += 0.062500 / dist

    #wyzsy
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(db-qa)*(db-qa)+(r-qa)*(r-qa))
    w[8,3] += -0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(-db-qa)*(-db-qa)+(r-qa)*(r-qa))
    w[8,3] += 0.125000 / dist

    #wyzyz
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb-qa)*(-qb-qa)+(r-qb-qa)*(r-qb-qa))
    w[8,8] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb-qa)*(+qb-qa)+(r-qb-qa)*(r-qb-qa))
    w[8,8] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb-qa)*(+qb-qa)+(r+qb-qa)*(r+qb-qa))
    w[8,8] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb-qa)*(-qb-qa)+(r+qb-qa)*(r+qb-qa))
    w[8,8] += -0.062500 / dist

    #wyzsy
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(db+qa)*(db+qa)+(r-qa)*(r-qa))
    w[8,3] += 0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(-db+qa)*(-db+qa)+(r-qa)*(r-qa))
    w[8,3] += -0.125000 / dist

    #wyzyz
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb+qa)*(-qb+qa)+(r-qb-qa)*(r-qb-qa))
    w[8,8] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb+qa)*(+qb+qa)+(r-qb-qa)*(r-qb-qa))
    w[8,8] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb+qa)*(+qb+qa)+(r+qb-qa)*(r+qb-qa))
    w[8,8] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb+qa)*(-qb+qa)+(r+qb-qa)*(r+qb-qa))
    w[8,8] += 0.062500 / dist

    #wyzsy
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(db+qa)*(db+qa)+(r+qa)*(r+qa))
    w[8,3] += -0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(-db+qa)*(-db+qa)+(r+qa)*(r+qa))
    w[8,3] += 0.125000 / dist

    #wyzyz
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb+qa)*(-qb+qa)+(r-qb+qa)*(r-qb+qa))
    w[8,8] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb+qa)*(+qb+qa)+(r-qb+qa)*(r-qb+qa))
    w[8,8] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb+qa)*(+qb+qa)+(r+qb+qa)*(r+qb+qa))
    w[8,8] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb+qa)*(-qb+qa)+(r+qb+qa)*(r+qb+qa))
    w[8,8] += -0.062500 / dist

    #wyzsy
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(db-qa)*(db-qa)+(r+qa)*(r+qa))
    w[8,3] += 0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(-db-qa)*(-db-qa)+(r+qa)*(r+qa))
    w[8,3] += -0.125000 / dist

    #wyzyz
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb-qa)*(-qb-qa)+(r-qb+qa)*(r-qb+qa))
    w[8,8] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb-qa)*(+qb-qa)+(r-qb+qa)*(r-qb+qa))
    w[8,8] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb-qa)*(+qb-qa)+(r+qb+qa)*(r+qb+qa))
    w[8,8] += -0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb-qa)*(-qb-qa)+(r+qb+qa)*(r+qb+qa))
    w[8,8] += 0.062500 / dist

    #wzzss
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[9,0] += 1.000000 / dist

    #wzzsz
    dist = sqrt(+(adb+ama)*(adb+ama)+(r+db)*(r+db))
    w[9,6] += -0.500000 / dist
    dist = sqrt(+(adb+ama)*(adb+ama)+(r-db)*(r-db))
    w[9,6] += 0.500000 / dist

    #wzzxx
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[9,2] += 1.000000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(-qb2)*(-qb2)+(r)*(r))
    w[9,2] += 0.250000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r)*(r))
    w[9,2] += -0.500000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(+qb2)*(+qb2)+(r)*(r))
    w[9,2] += 0.250000 / dist

    #wzzyy
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[9,5] += 1.000000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(-qb2)*(-qb2)+(r)*(r))
    w[9,5] += 0.250000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r)*(r))
    w[9,5] += -0.500000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(+qb2)*(+qb2)+(r)*(r))
    w[9,5] += 0.250000 / dist

    #wzzzz
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[9,9] += 1.000000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r-qb2)*(r-qb2))
    w[9,9] += 0.250000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r)*(r))
    w[9,9] += -0.500000 / dist
    dist = sqrt(+(aqb+ama)*(aqb+ama)+(r+qb2)*(r+qb2))
    w[9,9] += 0.250000 / dist

    #wzzss
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r-qa2)*(r-qa2))
    w[9,0] += 0.250000 / dist

    #wzzsz
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(r+db-qa2)*(r+db-qa2))
    w[9,6] += -0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(r-db-qa2)*(r-db-qa2))
    w[9,6] += 0.125000 / dist

    #wzzxx
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r-qa2)*(r-qa2))
    w[9,2] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2)*(-qb2)+(r-qa2)*(r-qa2))
    w[9,2] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r-qa2)*(r-qa2))
    w[9,2] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2)*(+qb2)+(r-qa2)*(r-qa2))
    w[9,2] += 0.062500 / dist

    #wzzyy
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r-qa2)*(r-qa2))
    w[9,5] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2)*(-qb2)+(r-qa2)*(r-qa2))
    w[9,5] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r-qa2)*(r-qa2))
    w[9,5] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2)*(+qb2)+(r-qa2)*(r-qa2))
    w[9,5] += 0.062500 / dist

    #wzzzz
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r-qa2)*(r-qa2))
    w[9,9] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r-qb2-qa2)*(r-qb2-qa2))
    w[9,9] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r-qa2)*(r-qa2))
    w[9,9] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r+qb2-qa2)*(r+qb2-qa2))
    w[9,9] += 0.062500 / dist

    #wzzss
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[9,0] += -0.500000 / dist

    #wzzsz
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(r+db)*(r+db))
    w[9,6] += 0.250000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(r-db)*(r-db))
    w[9,6] += -0.250000 / dist

    #wzzxx
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[9,2] += -0.500000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2)*(-qb2)+(r)*(r))
    w[9,2] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r)*(r))
    w[9,2] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2)*(+qb2)+(r)*(r))
    w[9,2] += -0.125000 / dist

    #wzzyy
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[9,5] += -0.500000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2)*(-qb2)+(r)*(r))
    w[9,5] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r)*(r))
    w[9,5] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2)*(+qb2)+(r)*(r))
    w[9,5] += -0.125000 / dist

    #wzzzz
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[9,9] += -0.500000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r-qb2)*(r-qb2))
    w[9,9] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r)*(r))
    w[9,9] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r+qb2)*(r+qb2))
    w[9,9] += -0.125000 / dist

    #wzzss
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r+qa2)*(r+qa2))
    w[9,0] += 0.250000 / dist

    #wzzsz
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(r+db+qa2)*(r+db+qa2))
    w[9,6] += -0.125000 / dist
    dist = sqrt(+(adb+aqa)*(adb+aqa)+(r-db+qa2)*(r-db+qa2))
    w[9,6] += 0.125000 / dist

    #wzzxx
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r+qa2)*(r+qa2))
    w[9,2] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2)*(-qb2)+(r+qa2)*(r+qa2))
    w[9,2] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r+qa2)*(r+qa2))
    w[9,2] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2)*(+qb2)+(r+qa2)*(r+qa2))
    w[9,2] += 0.062500 / dist

    #wzzyy
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r+qa2)*(r+qa2))
    w[9,5] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(-qb2)*(-qb2)+(r+qa2)*(r+qa2))
    w[9,5] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r+qa2)*(r+qa2))
    w[9,5] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(+qb2)*(+qb2)+(r+qa2)*(r+qa2))
    w[9,5] += 0.062500 / dist

    #wzzzz
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r+qa2)*(r+qa2))
    w[9,9] += 0.250000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r-qb2+qa2)*(r-qb2+qa2))
    w[9,9] += 0.062500 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r+qa2)*(r+qa2))
    w[9,9] += -0.125000 / dist
    dist = sqrt(+(aqb+aqa)*(aqb+aqa)+(r+qb2+qa2)*(r+qb2+qa2))
    w[9,9] += 0.062500 / dist

    return w

def W21_auto(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2):

    w = np.zeros((10,10))
    r = 2 * r05

    #wssss
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[0,0] += 1.000000 / dist

    #wszss
    dist = sqrt(+(amb+ada)*(amb+ada)+(r-da)*(r-da))
    w[6,0] += -0.500000 / dist
    dist = sqrt(+(amb+ada)*(amb+ada)+(r+da)*(r+da))
    w[6,0] += 0.500000 / dist

    #wxxss
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[2,0] += 1.000000 / dist
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[2,0] += 0.250000 / dist
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[2,0] += -0.500000 / dist
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[2,0] += 0.250000 / dist

    #wyyss
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[5,0] += 1.000000 / dist
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(-qa2)*(-qa2)+(r)*(r))
    w[5,0] += 0.250000 / dist
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[5,0] += -0.500000 / dist
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(+qa2)*(+qa2)+(r)*(r))
    w[5,0] += 0.250000 / dist

    #wzzss
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[9,0] += 1.000000 / dist
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r-qa2)*(r-qa2))
    w[9,0] += 0.250000 / dist
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r)*(r))
    w[9,0] += -0.500000 / dist
    dist = sqrt(+(amb+aqa)*(amb+aqa)+(r+qa2)*(r+qa2))
    w[9,0] += 0.250000 / dist

    return w

def W11_auto(r05, ama, ada, amb, adb, aqa, aqb, da, db, qa, qa2, qb, qb2):

    w = np.zeros((10,10))
    r = 2 * r05

    #wssss
    dist = sqrt(+(amb+ama)*(amb+ama)+(r)*(r))
    w[0,0] += 1.000000 / dist

    return w