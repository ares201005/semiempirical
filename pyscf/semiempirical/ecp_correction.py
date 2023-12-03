import numpy as np

def ecp_correction(zi, zj, gssma, gssam, gpsma, gpsam, ovlpsma, ovlpsam, ovlppma, ovlppam, params):
    vecpma = np.zeros(3)
    vecpam = np.zeros(3)
    if zi < 3: 
        vecpma[0] = -1*(ovlpsma*gssma + gssma*ovlpsma + ovlpsma*ovlpsma*params.f_aa[zj])
        #print(f'ovlpsma {ovlpsma} gssma {gssma}')
        #print('vsum',vsum)
    elif zj < 3:
        vecpam[0] = -1*(ovlpsam*gssam + gssam*ovlpsam + ovlpsam*ovlpsam*params.f_aa[zi])
        #print(f'ovlpsam {ovlpsam} gssam {gssam}')
        #print('vsum',vsum)
    else:
        vecpma[0] = -1*(ovlpsma*gssma + gssma*ovlpsma + ovlpsma*ovlpsma*params.f_aa[zj])
        #sign? -CL
        vecpma[1] = (ovlpsma*gpsma + gssma*ovlppma + ovlpsma*ovlppma*params.f_aa[zj])
        vecpma[2] = -1*(ovlppma*gpsma + gpsma*ovlppma + ovlppma*ovlppma*params.f_aa[zj])
        #print(f'ovlpsma {ovlpsma} gssma {gssma}')

        vecpam[0] = -1*(ovlpsam*gssam + gssam*ovlpsam + ovlpsam*ovlpsam*params.f_aa[zi])
        vecpam[1] = -1*(ovlpsam*gpsam + gssam*ovlppam + ovlpsam*ovlppam*params.f_aa[zi])
        vecpam[2] = -1*(ovlppam*gpsam + gpsam*ovlppam + ovlppam*ovlppam*params.f_aa[zi])
        #print(f'ovlpsam {ovlpsam} gssam {gssam}')
    print('vecpam',vecpam)
    print('vecpma',vecpma)
    return vecpma, vecpam
