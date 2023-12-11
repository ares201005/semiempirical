#!/usr/bin/env python
#
#

import os
import copy
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf.data.elements import _symbol, _std_symbol
from pyscf.semiempirical import mopac_param
from .read_param import *
from .diatomic_omx_overlap_matrix import *
from .rotation_matrix import *
from .diatomic_resonance_matrix import *
from .diatomic_ecp_overlap_matrix import *
from .diatomic_ecp_resonance_matrix import *
from .ecp_correction import *
from .python_integrals import *
from math import sqrt, atan, acos, sin, cos
from .matprint2d import *

#libsemiempirical = lib.load_library('/home/chance/pyscf_ext/semiempirical/pyscf/semiempirical/libsemiempirical.so') 
#libsemiempirical = lib.load_library('/Users/chancelander/Documents/Shao/semiempirical/code/semiempirical/build/lib.macosx-10.9-x86_64-cpython-38/pyscf/semiempirical/lib/libsemiempirical.so')
libsemiempirical = lib.load_library(os.environ['LIBSEMI'])
ndpointer = numpy.ctypeslib.ndpointer
libsemiempirical.MOPAC_rotate.argtypes = [
    ctypes.c_int, ctypes.c_int,
    ndpointer(dtype=numpy.double),  # xi
    ndpointer(dtype=numpy.double),  # xj
    ndpointer(dtype=numpy.double),  # w
    ndpointer(dtype=numpy.double),  # e1b
    ndpointer(dtype=numpy.double),  # e2a
    ndpointer(dtype=numpy.double),  # enuc
    ndpointer(dtype=numpy.double),  # alp
    ndpointer(dtype=numpy.double),  # dd
    ndpointer(dtype=numpy.double),  # qq
    ndpointer(dtype=numpy.double),  # am
    ndpointer(dtype=numpy.double),  # ad
    ndpointer(dtype=numpy.double),  # aq
    ndpointer(dtype=numpy.double),  # fn1
    ndpointer(dtype=numpy.double),  # fn2
    ndpointer(dtype=numpy.double),  # fn3
    ctypes.c_int
]
repp = libsemiempirical.MOPAC_rotate

#au2ev = 27.21138505
au2ev = 27.21 # Constant used in MNDO2020

def _make_mndo_mol(mol,model,params):
    assert(not mol.has_ecp())
    def make_sqm_basis(n, l, charge, zeta, model): 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        basisfile = dir_path+'/basis-ecp_om2.dat'
        symb = _std_symbol(charge)
        sqm_basis = gto.basis.load(basisfile,symb)

        es_cs = np.array([basval for basval in sqm_basis[l][1:]])
        es = es_cs[:,0]
        cs = es_cs[:,1]
        return [l] + [(e*zeta**2, c) for e, c in zip(es, cs)]

    def principle_quantum_number(charge):
        if charge < 3:
            return 1
        elif charge < 10:
            return 2
        elif charge < 18:
            return 3
        else:
            return 4

    mndo_mol = copy.copy(mol)
    atom_charges = mndo_mol.atom_charges()
    atom_types = set(atom_charges)
    basis_set = {}
    for charge in atom_types:
        n = principle_quantum_number(charge)
        l = 0
        sto_6g_function = make_sqm_basis(n, l, charge, params.zeta_s[charge], model)
        basis = [sto_6g_function]

        if charge > 2:
            l = 1
            sto_6g_function = make_sqm_basis(n, l, charge, params.zeta_p[charge], model)
            basis.append(sto_6g_function)

        basis_set[_symbol(int(charge))] = basis

    mndo_mol.basis = basis_set
    z_eff = mopac_param.CORE[atom_charges]
    mndo_mol.nelectron = int(z_eff.sum() - mol.charge)
    mndo_mol.build(0, 0)
    return mndo_mol

def get_fko(mol, zi, zj, zeta):
    symb = _std_symbol(zi)
    sqm_basis_i = gto.basis.load(basisfile,symb)
    es_cs_i = np.array([basval for basval in sqm_basis_i[0][1:]]) 
    es_i = es_cs_i[:,0]
    cs_i = es_cs_i[:,1]
    symb = _std_symbol(zj)
    sqm_basis_j = gto.basis.load(basisfile,symb)
    es_cs_j = np.array([basval for basval in sqm_basis_j[0][1:]]) 
    es_j = es_cs_j[:,0]
    cs_j = es_cs_j[:,1]
    const = np.power(np.pi, 1.5)*2
    d_arr = np.zeros(3)
    exp_arr = np.zeros(3)
    for k in range(len(es_i)):
         exp_arr[k] = es_i[k]+es_j[k]
         d_arr[k] = const/(exp_arr[k])*cs_i[k]*cs_j[k]
         sq[k] = 1/np.sqrt(2*exp_arr[k])
    #see SP0000
    return fko

def ort_correction(mol, S, B, VnucB, params):

    aoslices = mol.aoslice_by_atom()

    #Hloc
    nbas = B.shape[0]
    Hloc = np.zeros((nbas, mol.natm))
    average_pp = 0  #not sure if it is a good idea, but that is what MNDO2020 does
    for ia in range(0, mol.natm):
        for jb in range(0, mol.natm):
            i0, i1 = aoslices[ia,2:]
            for mu in range(i0, i1):
                if mu == i0:
                    Hloc[mu, jb] = params.U_ss[mol.atom_charges()[ia]] 
                else:            
                    Hloc[mu, jb] = params.U_pp[mol.atom_charges()[ia]]
                if mu == i0 or average_pp != 1:
                    Hloc[mu, jb] += VnucB[mu, mu, jb] 
                else: 
                    Hloc[mu, jb] += (VnucB[i0+1, i0+1, jb] + VnucB[i0+2, i0+2, jb] + VnucB[i0+3, i0+3, jb]) / 3.0
                
    #two-center 
    Hort = np.zeros_like(B)
    for ia in range(0, mol.natm):
        for jb in range(0, mol.natm):
            if jb != ia:  #this is because B elements are zero for diagonal blocks
                F1a = params.fval1[mol.atom_charges()[ia]]
                F2a = params.fval2[mol.atom_charges()[ia]]
                i0, i1 = aoslices[ia,2:]
                j0, j1 = aoslices[jb,2:]
                Hort[i0:i1,i0:i1] -= 0.5*F1a*np.einsum('mr,rn->mn',S[i0:i1,j0:j1], B[j0:j1,i0:i1])
                Hort[i0:i1,i0:i1] -= 0.5*F1a*np.einsum('mr,rn->mn',B[i0:i1,j0:j1], S[j0:j1,i0:i1])
                for mu in range(i0, i1):
                    for nu in range(i0, i1):
                        for rho in range(j0, j1):
                            Hort[mu,nu] += 0.125 * F2a * S[mu, rho] * S[rho, nu] * (Hloc[mu, jb] + Hloc[nu, jb] - 2 * Hloc[rho, ia])

    #three-center
    Hort3c = np.zeros_like(B)
    for ia in range(0, mol.natm):
        for jb in range(ia+1, mol.natm):
            G1a = params.gval1[mol.atom_charges()[ia]]
            G1b = params.gval1[mol.atom_charges()[jb]]
            G2a = params.gval2[mol.atom_charges()[ia]]
            G2b = params.gval2[mol.atom_charges()[jb]]
            G1 = 0.5 * (G1a + G1b) 
            G2 = 0.5 * (G2a + G2b) 
            i0, i1 = aoslices[ia,2:]
            j0, j1 = aoslices[jb,2:]
            for kc in range(0, mol.natm):
                if kc != ia and kc != jb:
                    k0, k1 = aoslices[kc,2:]
                    Hort3c[i0:i1,j0:j1] -= 0.5*G1*np.einsum('mr,rl->ml',S[i0:i1,k0:k1], B[k0:k1,j0:j1])
                    Hort3c[i0:i1,j0:j1] -= 0.5*G1*np.einsum('mr,rl->ml',B[i0:i1,k0:k1], S[k0:k1,j0:j1])
                    for mu in range(i0, i1):
                        for lm in range(j0, j1):
                            for rho in range(k0, k1):
                                Hort3c[mu,lm] += 0.125 * G2 * S[mu, rho] * S[rho, lm] * (Hloc[mu, kc] + Hloc[lm, kc] - Hloc[rho, ia] - Hloc[rho, jb])

    Hort += Hort3c + Hort3c.transpose()

    return Hort

def compute_VAC_analytical(mol, ia, jb, aoslices):
    charge_a =  mol.atom_charge(ia)
    charge_b =  mol.atom_charge(jb)
    if charge_a > 1: charge_a -= 2
    if charge_b > 1: charge_b -= 2
    i0, i1 = aoslices[ia,2:]
    j0, j1 = aoslices[jb,2:]
    mol.set_rinv_origin(mol.atom_coord(ia))
    e2a = -charge_a * mol.intor('int1e_rinv')[j0:j1, j0:j1]

    mol.set_rinv_origin(mol.atom_coord(jb))
    e1b = -charge_b * mol.intor('int1e_rinv')[i0:i1, i0:i1]

    return e1b, e2a

@lib.with_doc(scf.hf.get_hcore.__doc__)
def get_hcore_mndo(mol, model, python_integrals, params):
    assert(not mol.has_ecp())
    atom_charges = mol.atom_charges()
    basis_atom_charges = atom_charges[mol._bas[:,gto.ATOM_OF]]

    basis_u = []
    for i, z in enumerate(basis_atom_charges):
        l = mol.bas_angular(i)
        if l == 0:
            basis_u.append(params.U_ss[z])
        else:
            basis_u.append(params.U_pp[z])
    # U term
    hcore = np.diag(_to_ao_labels(mol, basis_u))

    aoslices = mol.aoslice_by_atom()
    vecp = 0.0
    ovlp1e = mol.intor("int1e_ovlp")

    B = np.zeros_like(hcore)
    VnucB = np.zeros((B.shape[0], B.shape[1], mol.natm))

    for ia in range(mol.natm):
        for ja in range(ia+1,mol.natm):
            zi = mol.atom_charge(ia)
            zj = mol.atom_charge(ja)
            e1b, e2a = compute_VAC_analytical(mol, ia, ja, aoslices)
            xij = mol.atom_coord(ja)-mol.atom_coord(ia)
            rij = np.linalg.norm(xij)
            xij /= rij

            #fKO 
            aee = 0.5/params.am[zi] + 0.5/params.am[zj]
            R0_semi = -1.0/sqrt(rij*rij+aee*aee)
            fKO_e1b = R0_semi * params.tore[zj] / e1b[0,0]
            fKO_e2a = R0_semi * params.tore[zi] / e2a[0,0]
            e1b *= fKO_e1b * 27.21
            e2a *= fKO_e2a * 27.21 #keep this fko scaling?

            i0, i1 = aoslices[ia,2:]
            j0, j1 = aoslices[ja,2:]
            hcore[j0:j1,j0:j1] += e2a
            hcore[i0:i1,i0:i1] += e1b
            VnucB[j0:j1,j0:j1,ia] = e2a
            VnucB[i0:i1,i0:i1,ja] = e1b

            #Resonance Integrals 
            rot_mat = rotation_matrix2(zi, zj, xij, rij, params.am, params.ad, params.aq, params.dd, params.qq, 
                                       params.tore, old_pxpy_pxpy=False)
            bloc = diatomic_resonance_matrix(ia, ja, zi, zj, xij, rij, params, rot_mat)
            hcore[i0:i1,j0:j1] += bloc
            hcore[j0:j1,i0:i1] += bloc.T 
            B[i0:i1,j0:j1] += bloc
            B[j0:j1,i0:i1] += bloc.T

            if zi + zj > 2:
                ovlpsam, ovlpsma, ovlppam, ovlppma = diatomic_ecp_overlap_matrix(mol, zi, zj, params, rij)
                gssam, gssma, gpsam, gpsma = diatomic_ecp_resonance_matrix(ia, ja, zi, zj, xij, rij, params, rot_mat)
                vecpma, vecpam = ecp_correction(zi, zj, gssma, gssam, gpsma, gpsam, ovlpsma, ovlpsam, ovlppma, ovlppam, params)

                hcore[i0:i1,j0:j1] += vecpma
                hcore[j0:j1,i0:i1] += vecpam 

    Hort = ort_correction(mol, ovlp1e, B, VnucB, params)
    hcore += Hort
    matrix_print_2d(hcore, 5, "Hcore")
    hcore /=27.21
    return hcore

def _get_jk_2c_ints(mol, model, python_integrals, ia, ja, params):
    zi = mol.atom_charge(ia)
    zj = mol.atom_charge(ja)
    ri = mol.atom_coord(ia)
    rj = mol.atom_coord(ja)
    w = np.zeros((10,10))
    e1b = np.zeros(10)
    e2a = np.zeros(10)
    enuc = np.zeros(1)
    AM1_MODEL = 1
    if python_integrals == 0 or python_integrals == 1:
        repp(zi, zj, ri, rj, w, e1b, e2a, enuc, params.alpha, params.dd, params.qq, params.am, params.ad, params.aq,
            params.K, params.L, params.M, AM1_MODEL)
    elif python_integrals == 2 or python_integrals == 3:
        w = compute_W(zi, zj, ri, rj, params.am, params.ad, params.aq, params.dd, params.qq, params.tore)

    tril2sq = lib.square_mat_in_trilu_indices(4)
    w = w[:,tril2sq][tril2sq]
    e1b = e1b[tril2sq]
    e2a = e2a[tril2sq]

    if params.tore[zj] <= 1:
        e2a = e2a[:1,:1]
        w = w[:,:,:1,:1]
    if params.tore[zi] <= 1:
        e1b = e1b[:1,:1]
        w = w[:1,:1]
    return w, e1b, e2a, enuc[0]

@lib.with_doc(scf.hf.get_jk.__doc__)
def get_jk_mndo(mol, dm, model, python_integrals, params):
    dm = np.asarray(dm)
    dm_shape = dm.shape
    nao = dm_shape[-1]

    dm = dm.reshape(-1,nao,nao)
    vj = np.zeros_like(dm)
    vk = np.zeros_like(dm)

    # One-center contributions to the J/K matrices
    atom_charges = mol.atom_charges()
    jk_ints = {z: _get_jk_1c_ints_mndo(z, params) for z in set(atom_charges)}

    aoslices = mol.aoslice_by_atom()
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z = atom_charges[ia]
        j_ints, k_ints = jk_ints[z]

        dm_blk = dm[:,p0:p1,p0:p1]
        idx = np.arange(p0, p1)
        # J[i,i] = (ii|jj)*dm_jj
        vj[:,idx,idx] = np.einsum('ij,xjj->xi', j_ints, dm_blk)
        # J[i,j] = (ij|ij)*dm_ji +  (ij|ji)*dm_ij
        vj[:,p0:p1,p0:p1] += 2*k_ints * dm_blk
        # K[i,i] = (ij|ji)*dm_jj
        vk[:,idx,idx] = np.einsum('ij,xjj->xi', k_ints, dm_blk)
        # K[i,j] = (ii|jj)*dm_ij + (ij|ij)*dm_ji
        vk[:,p0:p1,p0:p1] += (j_ints+k_ints) * dm_blk

    # Two-center contributions to the J/K matrices
    for ia, (i0, i1) in enumerate(aoslices[:,2:]):
        w = _get_jk_2c_ints(mol, model, python_integrals, ia, ia, params)[0]
        vj[:,i0:i1,i0:i1] += np.einsum('ijkl,xkl->xij', w, dm[:,i0:i1,i0:i1])
        vk[:,i0:i1,i0:i1] += np.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,i0:i1])
        for ja, (j0, j1) in enumerate(aoslices[:ia,2:]):
            w = _get_jk_2c_ints(mol, model, python_integrals, ia, ja, params)[0]
            vj[:,i0:i1,i0:i1] += np.einsum('ijkl,xkl->xij', w, dm[:,j0:j1,j0:j1])
            vj[:,j0:j1,j0:j1] += np.einsum('klij,xkl->xij', w, dm[:,i0:i1,i0:i1])
            vk[:,i0:i1,j0:j1] += np.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,j0:j1])
            vk[:,j0:j1,i0:i1] += np.einsum('klij,xjk->xil', w, dm[:,j0:j1,i0:i1])

    vj = vj.reshape(dm_shape)
    vk = vk.reshape(dm_shape)
    return vj, vk

def get_init_guess(mol):
    '''Average occupation density matrix'''
    aoslices = mol.aoslice_by_atom()
    dm_diag = numpy.zeros(mol.nao)
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z_eff = mopac_param.CORE[mol.atom_charge(ia)]
        dm_diag[p0:p1] = float(z_eff) / (p1-p0)
    return numpy.diag(dm_diag)

def energy_tot(mf, dm=None, h1e=None, vhf=None):
    mol = mf._mndo_mol
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + mf.energy_nuc()
    e_ref = _get_reference_energy(mol)

    print(f'  Electronic Energy: {(e_tot-mf.energy_nuc()): 12.7f} Eh, {(e_tot-mf.energy_nuc())*27.211386: 12.7f} eV')
    print(f'  Nuclear Energy:    {(mf.energy_nuc()): 12.7f} Eh, {(mf.energy_nuc())*27.211386: 12.7f} eV\n')
    mf.e_heat_formation = e_tot * mopac_param.HARTREE2KCAL + e_ref
    logger.debug(mf, 'E(ref) = %.15g  Heat of Formation = %.15g kcal/mol',
                 e_ref, mf.e_heat_formation)
    return e_tot.real

class ROM2(scf.hf.RHF):
    '''RHF-OM2 calculations for closed-shell systems'''
    def __init__(self, mol, model, python_integrals=0):
        scf.hf.RHF.__init__(self, mol)
        self.conv_tol = 1e-5
        self.e_heat_formation = None
        self._mndo_model = model
        self.params = omx_parameters(mol, self._mndo_model)
        self._mndo_mol = _make_mndo_mol(mol,self._mndo_model,self.params)
        self.python_integrals = python_integrals
        self._keys.update(['e_heat_formation'])
        
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        if log.verbose >= logger.INFO:
            from pyscf import semiempirical
            info = lib.repo_info(os.path.join(__file__, '..', '..', '..'))
            log.info('pyscf-semiempirical version %s', semiempirical.__version__)
            log.info('pyscf-semiempirical path %s', info['path'])
            if 'git' in info:
                log.info(info['git'])
        return super().dump_flags(log)

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol 
            self._mndo_mol = _make_mndo_mol(mol,self._mndo_model,self.params)
        return self

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._mndo_mol = _make_mndo_mol(mol,self._mndo_model,self.params)
        return self

    def get_ovlp(self, mol=None):
        return numpy.eye(self._mndo_mol.nao)

    def get_hcore(self, mol=None):
        if self._mndo_model == 'OM2':
           return get_hcore_mndo(self._mndo_mol, self._mndo_model, self.python_integrals, self.params)

    #@lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        elif self._mndo_model == 'OM2':
            return get_jk_mndo(self._mndo_mol, dm, self._mndo_model, self.python_integrals, self.params)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self._mndo_mol):
            return scf.hf.get_occ(self, mo_energy, mo_coeff)

    def get_init_guess(self, mol=None, key='minao'):
        return get_init_guess(self._mndo_mol)

    def energy_nuc(self):
        return get_energy_nuc_om2(self._mndo_mol, self._mndo_model, self.params) 
        #return get_energy_nuc_mndo(self._mndo_mol, self._mndo_model, self.params)

    energy_tot = energy_tot

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        scf.hf.RHF._finalize(self)

        # e_heat_formation was generated in SOSCF object.
        if (getattr(self, '_scf', None) and 
            getattr(self._scf, 'e_heat_formation', None)):
            self.e_heat_formation = self._scf.e_heat_formation

        logger.note(self, 'Heat of formation = %.15g kcal/mol, %.15g Eh',
                    self.e_heat_formation,
                    self.e_heat_formation/mopac_param.HARTREE2KCAL)
        return self

    density_fit = None
    x2c = x2c1e = sfx2c1e = None

    def nuc_grad_method(self):
        from . import rmndo_grad
        return rmndo_grad.Gradients(self)

def _to_ao_labels(mol, labels):
    ao_loc = mol.ao_loc
    degen = ao_loc[1:] - ao_loc[:-1]
    ao_labels = [[label]*n for label, n in zip(labels, degen)]
    return numpy.hstack(ao_labels)

def _get_alpha(atnoi,atnoj):
    "Part of the scale factor for the nuclear repulsion"
    return mopac_param.ALP3[atnoi-1,atnoj-1]

def _get_jk_1c_ints_mndo(z, params):
    if z < 3:  # H, He
        j_ints = np.zeros((1,1))
        k_ints = np.zeros((1,1))
        j_ints[0,0] = params.g_ss[z]
    else:
        j_ints = np.zeros((4,4))
        k_ints = np.zeros((4,4))
        p_diag_idx = ((1, 2, 3), (1, 2, 3)) 
        p_off_idx = ((1, 1, 2, 2, 3, 3), (2, 3, 1, 3, 1, 2)) 

        j_ints[0,0] = params.g_ss[z]
        j_ints[0,1:] = j_ints[1:,0] = params.g_sp[z]
        j_ints[p_off_idx] = params.g_p2[z]
        j_ints[p_diag_idx] = params.g_pp[z]

        k_ints[0,1:] = k_ints[1:,0] = params.h_sp[z]
        k_ints[p_off_idx] = 0.5*(params.g_pp[z]-params.g_p2[z])
    return j_ints, k_ints

def _get_gamma(mol, F03=None):
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = np.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * lib.param.BOHR
    rho = np.array([14.3996/F03[z]/27.211386 for z in atom_charges])
    gamma = 14.3996/27.211386 / np.sqrt(distances_in_AA**2 +
                                        (rho[:,None] + rho)**2 * .25)
    gamma[np.diag_indices(mol.natm)] = 0 
    return gamma

def _get_reference_energy(mol):
    '''E(Ref) = heat of formation - energy of atomization (kcal/mol)'''
    atom_charges = mol.atom_charges()
    Hf =  mopac_param.EHEAT3[atom_charges].sum()
    Eat = mopac_param.EISOL3[atom_charges].sum()
    return Hf - Eat * mopac_param.EV2KCAL

def get_energy_nuc_om2(mol,method,params):
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = np.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * lib.param.BOHR 
    enuc = 0 
    #pass fko through to remove this
    for ia in range(mol.natm):
        for ja in range(ia):
            ni = atom_charges[ia]
            nj = atom_charges[ja]
            charge_b = nj
            if charge_b > 1: charge_b -= 2
            xij = mol.atom_coord(ja)-mol.atom_coord(ia)
            rij = np.linalg.norm(xij)

            aoslices = mol.aoslice_by_atom()
            e1b, e2a = compute_VAC_analytical(mol, ia, ja, aoslices)

            aee = 0.5/params.am[ni] + 0.5/params.am[nj]
            R0_semi = -1.0/sqrt(rij*rij+aee*aee)
            fKO = R0_semi * params.tore[nj] / e1b.flat[0]
            enuc += params.tore[ni] * params.tore[nj] * fKO / rij #EN(A,B) = ZZ*fKO/rij 

    return enuc

