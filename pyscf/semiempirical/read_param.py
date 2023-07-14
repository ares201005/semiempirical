import os
import numpy as np


def read_param(method, elements):
    maxqn = np.amax(elements)
    fpath = os.path.dirname(__file__)
    param_file = fpath+'/parameters/parameters_'+method+'.csv'
    parameters = np.genfromtxt(param_file, delimiter=',', names=True, max_rows=maxqn+1)
    #print("elements:", elements, "maxqn:", maxqn)
    #print("parameters", parameters)
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
        self.tore = constants['tore']
        self.U_ss = parameters['U_ss']/27.211386
        self.U_pp = parameters['U_pp']/27.211386
        self.zeta_s = parameters['zeta_s']
        self.zeta_p = parameters['zeta_p']
        self.zeta_d = parameters['zeta_d']
        self.beta_s = parameters['beta_s']
        self.beta_p = parameters['beta_p']
        self.g_ss = parameters['g_ss']/27.211386
        self.g_sp = parameters['g_sp']/27.211386
        self.g_pp = parameters['g_pp']/27.211386
        self.g_p2 = parameters['g_p2']/27.211386
        self.h_sp = parameters['h_sp']/27.211386 #div 27.211 ? -CL
        self.alpha = np.copy(parameters['alpha'])
        self.dd    = np.copy(monopole_constants['MOPAC_DD'])
        self.qq    = np.copy(monopole_constants['MOPAC_QQ'])
        self.am    = np.copy(monopole_constants['MOPAC_AM'])
        self.ad    = np.copy(monopole_constants['MOPAC_AD'])
        self.aq    = np.copy(monopole_constants['MOPAC_AQ'])
    
        if model == 'AM1':
            size = len(parameters['Gaussian1_K'])
            self.K = np.stack((np.zeros(size),
                          parameters['Gaussian1_K'],
                          parameters['Gaussian2_K'],
                          parameters['Gaussian3_K'],
                          parameters['Gaussian4_K'],
                          np.zeros(size), np.zeros(size),
                          np.zeros(size), np.zeros(size),
                          np.zeros(size)), axis=1)/27.211386
            self.L = np.stack((np.zeros(size),
                          parameters['Gaussian1_L'],
                          parameters['Gaussian2_L'],
                          parameters['Gaussian3_L'],
                          parameters['Gaussian4_L'],
                          np.zeros(size), np.zeros(size),
                          np.zeros(size), np.zeros(size),
                          np.zeros(size)), axis=1)
            self.M = np.stack((np.zeros(size),
                          parameters['Gaussian1_M'],
                          parameters['Gaussian2_M'],
                          parameters['Gaussian3_M'],
                          parameters['Gaussian4_M'],
                          np.zeros(size), np.zeros(size),
                          np.zeros(size), np.zeros(size),
                          np.zeros(size)), axis=1)
        elif model == 'PM3':
            size = len(parameters['Gaussian1_K'])
            self.K = np.stack((np.zeros(size),
                          parameters['Gaussian1_K'],
                          parameters['Gaussian2_K'],
                          np.zeros(size), np.zeros(size),
                          np.zeros(size)), axis=1)/27.211386
            self.L = np.stack((np.zeros(size),
                          parameters['Gaussian1_L'],
                          parameters['Gaussian2_L'],
                          np.zeros(size), np.zeros(size),
                          np.zeros(size)), axis=1)
            self.M = np.stack((np.zeros(size),
                          parameters['Gaussian1_M'],
                          parameters['Gaussian2_M'],
                          np.zeros(size), np.zeros(size),
                          np.zeros(size)), axis=1)

class omx_parameters():
    def __init__(self, mol, model):
        elements = np.asarray([mol.atom_charge(i) for i in range(mol.natm)])
        parameters = read_param(model, elements)
        constants, monopole_constants  = read_constants(elements)
        self.tore = constants['tore']
        self.U_ss = parameters['U_ss']/27.21
        self.U_pp = parameters['U_pp']/27.21

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
        self.fval2 = parameters['fval2']
        self.gval1 = parameters['gval1']
        self.gval1 = parameters['gval1']

        self.zeta_ecp = parameters['zeta_ecp']
        self.f_aa = parameters['f_aa']
        self.beta_ecp = parameters['beta_ecp']
        self.alpha_ecp = parameters['alpha_ecp']

        self.eisol = parameters['eisol']
        self.hyf = parameters['hyf']

        self.g_ss = parameters['g_ss']/27.21
        self.g_sp = parameters['g_sp']/27.21
        self.g_pp = parameters['g_pp']/27.21
        self.g_p2 = parameters['g_p2']/27.21
        self.h_sp = parameters['h_sp']/27.21 

        self.dd   = parameters['dd']#/27.21 
        self.qq   = parameters['qq']#/27.21 
        self.am   = parameters['am']#/27.21 
        self.ad   = parameters['ad']#/27.21 
        self.aq   = parameters['aq']#/27.21 
    
        if model == 'OM2' or model == 'OM3':
            self.c61   = parameters['c61']
            self.r01   = parameters['r01']
            self.c62   = parameters['c62']
            self.r02   = parameters['r02']
