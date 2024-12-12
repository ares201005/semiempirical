import os
import numpy as np
from pyscf import lib
#from pyscf.data.nist import HARTREE2EV

#HARTREE2EV = 27.21138602
HARTREE2EV = 27.21
def read_param(method, elements):
    maxqn = np.amax(elements)
    fpath = os.path.dirname(__file__)
    param_file = fpath+'/parameters/parameters_'+method+'.csv'
    parameters = np.genfromtxt(param_file, delimiter=',', names=True, max_rows=maxqn+1)
    return parameters

def read_constants(elements):
    maxqn = np.amax(elements)
    fpath = os.path.dirname(__file__)
    const_file = fpath+'/constants/element_constants.csv'
    #check indexing on maxqn+1 vs maxqn. Can we delete first row of 0s? -CL
    constants = np.genfromtxt(const_file, delimiter=',', names=True, max_rows=maxqn+1) 
    const_file = fpath+'/constants/monopole_constants.csv'
    monopole_constants = np.genfromtxt(const_file, delimiter=',', names=True, max_rows=maxqn+1) 
    return constants, monopole_constants

class sqm_parameters():
    def __init__(self, mol, model):
        elements = np.asarray([mol.atom_charge(i) for i in range(mol.natm)])
        parameters = read_param(model, elements)
        constants, monopole_constants  = read_constants(elements)
        self.e2 = 14.399/HARTREE2EV
        self.bohr = 0.529167 # from mopac 7 and PYSEQM
        self.tore = constants['tore']
        self.U_ss = parameters['U_ss']/HARTREE2EV
        self.U_pp = parameters['U_pp']/HARTREE2EV
        self.zeta_s = parameters['zeta_s']
        self.zeta_p = parameters['zeta_p']
        self.g_ss = parameters['g_ss']/HARTREE2EV
        self.g_sp = parameters['g_sp']/HARTREE2EV
        self.g_pp = parameters['g_pp']/HARTREE2EV
        self.g_p2 = parameters['g_p2']/HARTREE2EV
        self.h_sp = parameters['h_sp']/HARTREE2EV
    
        if model == 'AM1':
            self.zeta_d = parameters['zeta_d']
            self.beta_s = parameters['beta_s']
            self.beta_p = parameters['beta_p']
            self.alpha = parameters['alpha']
            # CL - Pass charges so K,L,M are shaped Nx(2-4) here instead of in sqm.py
            self.K = np.stack([parameters['Gaussian1_K'],
                               parameters['Gaussian2_K'],
                               parameters['Gaussian3_K'],
                               parameters['Gaussian4_K']],axis=-1) #,dim=1)#/HARTREE2EV
            self.L = np.stack([parameters['Gaussian1_L'],
                               parameters['Gaussian2_L'],
                               parameters['Gaussian3_L'],
                               parameters['Gaussian4_L']],axis=-1) #,dim=1)#/HARTREE2EV
            self.M = np.stack([parameters['Gaussian1_M'],
                               parameters['Gaussian2_M'],
                               parameters['Gaussian3_M'],
                               parameters['Gaussian4_M']],axis=-1) #,dim=1)#/HARTREE2EV
            self.dd = parameters['DD'] # \AA
            self.qq = parameters['QQ'] # \AA
            self.am = parameters['AM'] # au \AA
            self.ad = parameters['AD'] # au \AA
            self.aq = parameters['AQ'] # au \AA
            #self.dd    = monopole_constants['MOPAC_DD']
            #self.qq    = monopole_constants['MOPAC_QQ']
            #self.am    = monopole_constants['MOPAC_AM']
            #self.ad    = monopole_constants['MOPAC_AD']
            #self.aq    = monopole_constants['MOPAC_AQ']
        elif model == 'PM3':
            self.zeta_d = parameters['zeta_d']
            self.beta_s = parameters['beta_s']
            self.beta_p = parameters['beta_p']
            self.alpha = parameters['alpha']
            self.K = np.stack([parameters['Gaussian1_K'],
                               parameters['Gaussian2_K']],axis=-1) #,dim=1)#/HARTREE2EV
            self.L = np.stack([parameters['Gaussian1_L'],
                               parameters['Gaussian2_L']],axis=-1) #,dim=1)#/HARTREE2EV
            self.M = np.stack([parameters['Gaussian1_M'],
                               parameters['Gaussian2_M']],axis=-1) #,dim=1)#/HARTREE2EV
            self.dd = parameters['DD'] # \AA
            self.qq = parameters['QQ'] # \AA
            self.am = parameters['AM'] # au \AA
            self.ad = parameters['AD'] # au \AA
            self.aq = parameters['AQ'] # au \AA
            #self.dd    = monopole_constants['MOPAC_DD']
            #self.qq    = monopole_constants['MOPAC_QQ']
            #self.am    = monopole_constants['MOPAC_AM']
            #self.ad    = monopole_constants['MOPAC_AD']
            #self.aq    = monopole_constants['MOPAC_AQ']
        elif model == 'MNDO':
            self.beta_s = parameters['beta_s']
            self.beta_p = parameters['beta_p']
            self.alpha = parameters['alpha']
            self.dd = parameters['DD'] # \AA
            self.qq = parameters['QQ'] # \AA
            self.am = parameters['AM'] # au \AA
            self.ad = parameters['AD'] # au \AA
            self.aq = parameters['AQ'] # au \AA
            #self.dd    = monopole_constants['MOPAC_DD']
            #self.qq    = monopole_constants['MOPAC_QQ']
            #self.am    = monopole_constants['MOPAC_AM']
            #self.ad    = monopole_constants['MOPAC_AD']
            #self.aq    = monopole_constants['MOPAC_AQ']
        elif model == 'MINDO3':
            self.V_s = parameters['V_s']/HARTREE2EV
            self.V_p = parameters['V_p']/HARTREE2EV
            self.f03 = parameters['f03']#/HARTREE2EV
            self.h_p2 = parameters['h_p2']/HARTREE2EV
            self.eheat = parameters['eheat']
            self.eisol = parameters['eisol']
            Bxy = np.array((
                    # H                B         C         N         O         F                     Si        P         S        Cl
                    0.244770,
                    0       , 0,
                    0       , 0, 0,
                    0       , 0, 0, 0,
                    0.185347, 0, 0, 0, 0.151324,
                    0.315011, 0, 0, 0, 0.250031, 0.419907,
                    0.360776, 0, 0, 0, 0.310959, 0.410886, 0.377342,
                    0.417759, 0, 0, 0, 0.349745, 0.464514, 0.458110, 0.659407,
                    0.195242, 0, 0, 0, 0.219591, 0.247494, 0.205347, 0.334044, 0.197464,
                    0       , 0, 0, 0, 0       , 0       , 0       , 0       , 0       , 0,
                    0       , 0, 0, 0, 0       , 0       , 0       , 0       , 0       , 0, 0,
                    0       , 0, 0, 0, 0       , 0       , 0       , 0       , 0       , 0, 0, 0,
                    0       , 0, 0, 0, 0       , 0       , 0       , 0       , 0       , 0, 0, 0, 0,
                    0.289647, 0, 0, 0, 0       , 0.411377, 0       , 0       , 0       , 0, 0, 0, 0, 0.291703,
                    0.320118, 0, 0, 0, 0       , 0.457816, 0       , 0.470000, 0.300000, 0, 0, 0, 0, 0       , 0.311790,
                    0.220654, 0, 0, 0, 0       , 0.284620, 0.313170, 0.422890, 0       , 0, 0, 0, 0, 0       , 0       , 0.202489,
                    0.231653, 0, 0, 0, 0       , 0.315480, 0.302298, 0       , 0       , 0, 0, 0, 0, 0       , 0.277322, 0.221764, 0.258969,
            ))
            self.beta = lib.unpack_tril(Bxy)
            del(Bxy)
            
            Axy = np.array((
                    # H                B         C         N         O         F                     Si        P         S        Cl
                    1.489450,
                    0       , 0,
                    0       , 0, 0,
                    0       , 0, 0, 0,
                    2.090352, 0, 0, 0, 2.280544,
                    1.475836, 0, 0, 0, 2.138291, 1.371208,
                    0.589380, 0, 0, 0, 1.909763, 1.635259, 2.029618,
                    0.478901, 0, 0, 0, 2.484827, 1.820975, 1.873859, 1.537190,
                    3.771362, 0, 0, 0, 2.862183, 2.725913, 2.861667, 2.266949, 3.864997,
                    0       , 0, 0, 0, 0       , 0       , 0       , 0       , 0       , 0,
                    0       , 0, 0, 0, 0       , 0       , 0       , 0       , 0       , 0, 0,
                    0       , 0, 0, 0, 0       , 0       , 0       , 0       , 0       , 0, 0, 0,
                    0       , 0, 0, 0, 0       , 0       , 0       , 0       , 0       , 0, 0, 0, 0,
                    0.940789, 0, 0, 0, 0       , 1.101382, 0       , 0       , 0       , 0, 0, 0, 0, 0.918432,
                    0.923170, 0, 0, 0, 0       , 1.029693, 0       , 1.662500, 1.750000, 0, 0, 0, 0, 0       , 1.186652,
                    1.700698, 0, 0, 0, 0       , 1.761370, 1.878176, 2.077240, 0       , 0, 0, 0, 0, 0       , 0       , 1.751617,
                    2.089404, 0, 0, 0, 0       , 1.676222, 1.817064, 0       , 0       , 0, 0, 0, 0, 0       , 1.543720, 1.950318, 1.792125,
            ))
            self.alpha = lib.unpack_tril(Axy)
            del(Axy)


class omx_parameters():
    def __init__(self, mol, model):
        elements = np.asarray([mol.atom_charge(i) for i in range(mol.natm)])
        parameters = read_param(model, elements)
        constants, monopole_constants  = read_constants(elements)
        self.tore = constants['tore']
        self.U_ss = parameters['U_ss']
        self.U_pp = parameters['U_pp']

        self.zeta_s = parameters['zeta_s']
        self.zeta_p = parameters['zeta_s']

        self.beta_s = parameters['beta_s']
        self.beta_p = parameters['beta_p']
        self.beta_pi = parameters['beta_pi']
        self.beta_sh = parameters['beta_sh']
        self.beta_ph = parameters['beta_ph']

        self.alpha_s = parameters['alpha_s']
        self.alpha_p = parameters['alpha_p']
        self.alpha_pi = parameters['alpha_pi']
        self.alpha_sh = parameters['alpha_sh']
        self.alpha_ph = parameters['alpha_ph']

        self.fval1 = parameters['fval1']
        self.fval2 = parameters['fval2']#*27.21
        self.gval1 = parameters['gval1']
        self.gval2 = parameters['gval2']#*27.21

        self.zeta_ecp = parameters['zeta_ecp']
        self.f_aa = parameters['f_aa']
        self.beta_ecp = parameters['beta_ecp']
        self.alpha_ecp = parameters['alpha_ecp']

        self.eisol = parameters['eisol']
        self.hyf = parameters['hyf']

        self.g_ss = parameters['g_ss']/HARTREE2EV #27.21
        self.g_sp = parameters['g_sp']/HARTREE2EV #27.21
        self.g_pp = parameters['g_pp']/HARTREE2EV #27.21
        self.g_p2 = parameters['g_p2']/HARTREE2EV #27.21
        self.h_sp = parameters['h_sp']/HARTREE2EV #27.21 

        #self.dd   = parameters['dd']#/27.21 
        #self.qq   = parameters['qq']#/27.21 
        #self.am   = parameters['am']#/27.21 
        #self.ad   = parameters['ad']#/27.21 
        #self.aq   = parameters['aq']#/27.21 
    
        if model == 'OM2' or model == 'OM3':
            self.c61   = parameters['c61']
            self.r01   = parameters['r01']
            self.c62   = parameters['c62']
            self.r02   = parameters['r02']
