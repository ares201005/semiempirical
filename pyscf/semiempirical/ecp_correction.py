import numpy as np

def ecp_correction(zi, zj, gssma, gssam, gpsma, gpsam, ovlpsma, ovlpsam, ovlppma, ovlppam, params):
    if zi < 3: 
        vecpma = np.zeros((4,1))
        vecpam = np.zeros((1,4))
        vecpma[0][0] = -1*(ovlpsma*gssma + gssma*ovlpsma + ovlpsma*ovlpsma*params.f_aa[zj])
        #print(f'ovlpsma {ovlpsma} gssma {gssma}')
    elif zj < 3:
        vecpma = np.zeros((4,1))
        vecpam = np.zeros((1,4))
        vecpam[0][0] = -1*(ovlpsam*gssam + gssam*ovlpsam + ovlpsam*ovlpsam*params.f_aa[zi])
        #print(f'ovlpsam {ovlpsam} gssam {gssam}')
    else:
        vecpma = np.zeros((4,4))
        vecpam = np.zeros((4,4))
        vecpma[0][0] = -1*(ovlpsma*gssma + gssma*ovlpsma + ovlpsma*ovlpsma*params.f_aa[zj])
        vecpma[0][1] =    (ovlpsma*gpsma + gssma*ovlppma + ovlpsma*ovlppma*params.f_aa[zj])
        vecpma[1][0] = vecpma[0][1]
        vecpma[1][1] = -1*(ovlppma*gpsma + gpsma*ovlppma + ovlppma*ovlppma*params.f_aa[zj])
        #print(f'ovlpsma {ovlpsma} gssma {gssma}')

        vecpam[0][0] = -1*(ovlpsam*gssam + gssam*ovlpsam + ovlpsam*ovlpsam*params.f_aa[zi])
        vecpam[1][0] = -1*(ovlpsam*gpsam + gssam*ovlppam + ovlpsam*ovlppam*params.f_aa[zi])
        vecpam[0][1] = vecpam[1][0]
        vecpam[1][1] = -1*(ovlppam*gpsam + gpsam*ovlppam + ovlppam*ovlppam*params.f_aa[zi])
        #print(f'ovlpsam {ovlpsam} gssam {gssam}')
    print('vecpam',vecpam)
    print('vecpma',vecpma)
    return vecpma, vecpam
