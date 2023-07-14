#!/usr/bin/env python
#
#

'''
whatever
'''

import os, sys
import copy
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf.data.elements import _symbol
from pyscf.semiempirical import mopac_param
from .read_param import *
from .diatomic_overlap_matrix import *
from .python_integrals import *
from math import sqrt, atan, acos, sin, cos
write = sys.stdout.write

libsemiempirical = lib.load_library('/home/chance/pyscf_ext/semiempirical/pyscf/semiempirical/libsemiempirical.so') 
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

au2ev = 27.21138505

def matrix_print_2d(array, ncols, title):
	""" printing a rectangular matrix, ncols columns per batch """

	write(title+'\n')
	m = array.shape[1]
	n = array.shape[0]
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
				write(' %11.6f' % array[j,i])
			write('\n')

def _make_mndo_mol(mol):
    assert(not mol.has_ecp())
    def make_sto_6g(n, l, zeta):
        es = mopac_param.gexps[(n, l)]
        cs = mopac_param.gcoefs[(n, l)]
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
        sto_6g_function = make_sto_6g(n, l, mopac_param.ZS3[charge])
        basis = [sto_6g_function]

        if charge > 2:  # include p functions
            l = 1
            sto_6g_function = make_sto_6g(n, l, mopac_param.ZP3[charge])
            basis.append(sto_6g_function)

        basis_set[_symbol(int(charge))] = basis
    mndo_mol.basis = basis_set

    z_eff = mopac_param.CORE[atom_charges]
    mndo_mol.nelectron = int(z_eff.sum() - mol.charge)

    mndo_mol.build(0, 0)
    return mndo_mol

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
    #print('U term hcore', hcore)
    #print("zeta_s:", zeta_s)
    #print("zeta_p:", zeta_p)

    aoslices = mol.aoslice_by_atom()
    for ia in range(mol.natm):
        for ja in range(ia+1,mol.natm): #Was ia -CL
            if python_integrals == 0 or python_integrals == 2:
               w, e1b, e2a, enuc = _get_jk_2c_ints(mol, model, python_integrals, ia, ja, params)
            elif python_integrals == 1 or python_integrals == 3:
               #e1b, e2a = compute_VAC(zi, zj, ri, rj, params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
               e1b, e2a = compute_VAC(mol.atom_charge(ia), mol.atom_charge(ja), mol.atom_coord(ia), mol.atom_coord(ja),
                                      params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
            i0, i1 = aoslices[ia,2:]
            j0, j1 = aoslices[ja,2:]
            hcore[j0:j1,j0:j1] += e2a
            hcore[i0:i1,i0:i1] += e1b
            #print("i:", i0, i1, "j:", j0, j1)
            #print("e2a:", e2a)
            #print("e1b:", e1b)

            # off-diagonal block 
            zi = mol.atom_charge(ia)
            zj = mol.atom_charge(ja)
            xi = mol.atom_coord(ia) #?*lib.param.BOHR
            xj = mol.atom_coord(ja) #?*lib.param.BOHR
            xij = xj - xi
            rij = np.linalg.norm(xij)
            xij /= rij
            #print("zi, zj:", zi, zj)
            #print("xij:", xij, "rij:", rij)
            di = diatomic_overlap_matrix(ia, ja, zi, zj, xij, rij, params)
            #print('hcore:',hcore[i0:i1,j0:j1], np.shape(hcore[i0:i1,j0:j1]))
            #print('di core:',di, np.shape(di))
            #hcore[i0:i1,j0:j1] += di.T
            #hcore[j0:j1,i0:i1] += di

            hcore[i0:i1,j0:j1] += di #original -CL
            hcore[j0:j1,i0:i1] += di.T #original -CL

    print("hcore:", hcore)
    return hcore

@lib.with_doc(scf.hf.get_hcore.__doc__)
def get_hcore_indo(mol, model):
    assert(not mol.has_ecp())
    nao = mol.nao

    atom_charges = mol.atom_charges()
    basis_atom_charges = atom_charges[mol._bas[:,gto.ATOM_OF]]

    print("atom_charge:", atom_charges)

    basis_ip = []
    basis_u = []
    for i, z in enumerate(basis_atom_charges):
        l = mol.bas_angular(i)
        if l == 0:
            basis_ip.append(mopac_param.VS[z])
            basis_u.append(mopac_param.USS3[z])
        else:
            basis_ip.append(mopac_param.VP[z])
            basis_u.append(mopac_param.UPP3[z])

    ao_atom_charges = _to_ao_labels(mol, basis_atom_charges)
    ao_ip = _to_ao_labels(mol, basis_ip)

    # Off-diagonal terms
    hcore  = mol.intor('int1e_ovlp')
    hcore *= ao_ip[:,None] + ao_ip
    hcore *= _get_beta0(ao_atom_charges[:,None], ao_atom_charges)

    # U term
    hcore[numpy.diag_indices(nao)] = _to_ao_labels(mol, basis_u)

    # Nuclear attraction
    gamma = _get_gamma(mol)
    z_eff = mopac_param.CORE[atom_charges]
    vnuc = numpy.einsum('ij,j->i', gamma, z_eff)

    aoslices = mol.aoslice_by_atom()
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        idx = numpy.arange(p0, p1)
        hcore[idx,idx] -= vnuc[ia]
    print("hcore:", hcore)
    return hcore

def _get_jk_2c_ints(mol, model, python_integrals, ia, ja, params): #should be ok -CL
    zi = mol.atom_charge(ia)
    zj = mol.atom_charge(ja)
    ri = mol.atom_coord(ia) #?*lib.param.BOHR
    rj = mol.atom_coord(ja) #?*lib.param.BOHR
    w = np.zeros((10,10))
    e1b = np.zeros(10)
    e2a = np.zeros(10)
    enuc = np.zeros(1)
    AM1_MODEL = 2
    #print('zi:', zi, zj, ri, rj)
    #print('alpha:', alpha)
    #print('dd:', dd)
    #print('am:', am)
    #print('K1:', K[1], L[1], M[1])
    #print('L6:', K[6], L[6], M[6])
    #print('M8:', K[8], L[8], M[8])
    if python_integrals == 0 or python_integrals == 1:
        repp(zi, zj, ri, rj, w, e1b, e2a, enuc, params.alpha, params.dd, params.qq, params.am, params.ad, params.aq,
            params.K, params.L, params.M, AM1_MODEL)
    #print("enuc:", enuc, e1b, e2a)
    #print("e1b", e1b)
    #print("e2a", e2a)
    #a, b = compute_VAC(zi, zj, ri, rj, params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
    #print("w:", w)
        matrix_print_2d(w, 5, "w")
    elif python_integrals == 2 or python_integrals == 3:
        w = compute_W(zi, zj, ri, rj, params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
        matrix_print_2d(w, 5, "w new")

    tril2sq = lib.square_mat_in_trilu_indices(4)
    w = w[:,tril2sq][tril2sq]
    e1b = e1b[tril2sq]
    e2a = e2a[tril2sq]

    if params.tore[zj] <= 1: #check same as mopac_param.CORE[zj]
        e2a = e2a[:1,:1]
        w = w[:,:,:1,:1]
    if params.tore[zi] <= 1:
        e1b = e1b[:1,:1]
        w = w[:1,:1]
    #print('w',w)
    #print('e1b',e1b)
    #print('e2a',e2a)
    #print('tore[zj]',tore[zj])
    #print('tore[zi]',tore[zi])
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
    #print('jk_ints',jk_ints)

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
        print("ia:", ia, "i0:", i0, "i1:", i1)
        #w2 = compute_W(mol.atom_charge(ia), mol.atom_charge(ia), mol.atom_coord(ia), mol.atom_coord(ia),
        #               params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
        #print("w diag:", w)
        vj[:,i0:i1,i0:i1] += np.einsum('ijkl,xkl->xij', w, dm[:,i0:i1,i0:i1])
        vk[:,i0:i1,i0:i1] += np.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,i0:i1])
        for ja, (j0, j1) in enumerate(aoslices[:ia,2:]):
            print("w, ia:", ia, "ja:", ja)
            w = _get_jk_2c_ints(mol, model, python_integrals, ia, ja, params)[0]
            #print("w:", w)
            #w2 = compute_W(mol.atom_charge(ia), mol.atom_charge(ja), mol.atom_coord(ia), mol.atom_coord(ja),
            #               params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
            #print("w2:", w2)
            #matrix_print_2d(w2, 10, "w2")
            vj[:,i0:i1,i0:i1] += np.einsum('ijkl,xkl->xij', w, dm[:,j0:j1,j0:j1])
            vj[:,j0:j1,j0:j1] += np.einsum('klij,xkl->xij', w, dm[:,i0:i1,i0:i1])
            vk[:,i0:i1,j0:j1] += np.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,j0:j1])
            vk[:,j0:j1,i0:i1] += np.einsum('klij,xjk->xil', w, dm[:,j0:j1,i0:i1])

    vj = vj.reshape(dm_shape)
    vk = vk.reshape(dm_shape)
    #print("dm:", dm)
    return vj, vk

@lib.with_doc(scf.hf.get_jk.__doc__)
def get_jk_indo(mol, dm):
    dm = numpy.asarray(dm)
    dm_shape = dm.shape
    nao = dm_shape[-1]

    dm = dm.reshape(-1,nao,nao)
    vj = numpy.zeros_like(dm)
    vk = numpy.zeros_like(dm)

    # One-center contributions to the J/K matrices
    atom_charges = mol.atom_charges()
    jk_ints = {z: _get_jk_1c_ints_indo(z) for z in set(atom_charges)}

    aoslices = mol.aoslice_by_atom()
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z = atom_charges[ia]
        j_ints, k_ints = jk_ints[z]

        dm_blk = dm[:,p0:p1,p0:p1]
        idx = numpy.arange(p0, p1)

        # J[i,i] = (ii|jj)*dm_jj
        vj[:,idx,idx] = numpy.einsum('ij,xjj->xi', j_ints, dm_blk)
        # J[i,j] = (ij|ij)*dm_ji +  (ij|ji)*dm_ij
        vj[:,p0:p1,p0:p1] += 2*k_ints * dm_blk

        # K[i,i] = (ij|ji)*dm_jj
        vk[:,idx,idx] = numpy.einsum('ij,xjj->xi', k_ints, dm_blk)
        # K[i,j] = (ii|jj)*dm_ij + (ij|ij)*dm_ji
        vk[:,p0:p1,p0:p1] += (j_ints+k_ints) * dm_blk

    # Two-center contributions to the J/K matrices
    gamma = _get_gamma(mol)
    pop_atom = [numpy.einsum('tii->t', dm[:,p0:p1,p0:p1])
                for p0, p1 in aoslices[:,2:]]
    vj_diag = numpy.einsum('ij,jt->ti', gamma, pop_atom)

    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        idx = numpy.arange(p0, p1)
        vj[:,idx,idx] += vj_diag[:,ia].reshape(-1,1)

        for ja, (q0, q1) in enumerate(aoslices[:,2:]):
            vk[:,p0:p1,q0:q1] += gamma[ia,ja] * dm[:,p0:p1,q0:q1]

    vj = vj.reshape(dm_shape)
    vk = vk.reshape(dm_shape)
    print("dm=", dm)
    print("vj=", vj)
    print("vk=", vk)
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

class RMNDO(scf.hf.RHF):
    '''RHF-MNDO calculations for closed-shell systems'''
    def __init__(self, mol, model, python_integrals=0):
        scf.hf.RHF.__init__(self, mol)
        self.conv_tol = 1e-5
        self.e_heat_formation = None
        self._mndo_mol = _make_mndo_mol(mol)
        self._mndo_model = model
        self.python_integrals = python_integrals
        self.params = sqm_parameters(self._mndo_mol, self._mndo_model)
        print("self.params.alpha:", self.params.alpha)
        #self.read_params()
        self._keys.update(['e_heat_formation'])
        print("model:", self._mndo_model)
        
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
            self._mndo_mol = _make_mndo_mol(mol)
        return self

    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._mndo_mol = _make_mndo_mol(mol)
        return self

    def get_ovlp(self, mol=None):
        return numpy.eye(self._mndo_mol.nao)

    def get_hcore(self, mol=None):
        if self._mndo_model == 'MINDO3':
           return get_hcore_indo(self._mndo_mol, self._mndo_model)
        elif self._mndo_model == 'AM1':
           return get_hcore_mndo(self._mndo_mol, self._mndo_model, self.python_integrals, self.params)

    #@lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        if self._mndo_model == 'MINDO3':
            return get_jk_indo(self._mndo_mol, dm) 
        elif self._mndo_model == 'AM1':
            return get_jk_mndo(self._mndo_mol, dm, self._mndo_model, self.python_integrals, self.params)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self._mndo_mol):
            return scf.hf.get_occ(self, mo_energy, mo_coeff)

    def get_init_guess(self, mol=None, key='minao'):
        return get_init_guess(self._mndo_mol)

    def energy_nuc(self):
        return get_energy_nuc_mndo(self._mndo_mol, self._mndo_model, self.params)

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

def _get_beta0(atnoi,atnoj):
    "Resonanace integral for coupling between different atoms"
    return mopac_param.BETA3[atnoi-1,atnoj-1]

def _get_alpha(atnoi,atnoj):
    "Part of the scale factor for the nuclear repulsion"
    return mopac_param.ALP3[atnoi-1,atnoj-1]

def _get_jk_1c_ints_indo(z):
    if z < 3:  # H, He
        j_ints = numpy.zeros((1,1))
        k_ints = numpy.zeros((1,1))
        j_ints[0,0] = mopac_param.GSSM[z]
    else:
        j_ints = numpy.zeros((4,4))
        k_ints = numpy.zeros((4,4))
        p_diag_idx = ((1, 2, 3), (1, 2, 3))
        # px,py,pz cross terms
        p_off_idx = ((1, 1, 2, 2, 3, 3), (2, 3, 1, 3, 1, 2))

        j_ints[0,0] = mopac_param.GSSM[z]
        j_ints[0,1:] = j_ints[1:,0] = mopac_param.GSPM[z]
        j_ints[p_off_idx] = mopac_param.GP2M[z]
        j_ints[p_diag_idx] = mopac_param.GPPM[z]

        k_ints[0,1:] = k_ints[1:,0] = mopac_param.HSPM[z]
        k_ints[p_off_idx] = mopac_param.HP2M[z]
    return j_ints, k_ints

def _get_jk_1c_ints_mndo(z, params):
    if z < 3:  # H, He
        j_ints = np.zeros((1,1))
        k_ints = np.zeros((1,1))
        j_ints[0,0] = params.g_ss[z]
    else:
        j_ints = np.zeros((4,4))
        k_ints = np.zeros((4,4))
        p_diag_idx = ((1, 2, 3), (1, 2, 3)) 
        # px,py,pz cross terms
        p_off_idx = ((1, 1, 2, 2, 3, 3), (2, 3, 1, 3, 1, 2)) 

        j_ints[0,0] = params.g_ss[z]
        j_ints[0,1:] = j_ints[1:,0] = params.g_sp[z]
        j_ints[p_off_idx] = params.g_p2[z]
        j_ints[p_diag_idx] = params.g_pp[z]

        k_ints[0,1:] = k_ints[1:,0] = params.h_sp[z]
        #print('g_pp', g_pp)
        #print('g_p2', g_p2)
        k_ints[p_off_idx] = 0.5*(params.g_pp[z]-params.g_p2[z]) #save h_pp aka hp2 as parameter in file? -CL
        #k_ints[p_off_idx] = mopac_param.HP2M[z]
    return j_ints, k_ints

def _get_gamma(mol, F03=None): #From mindo3.py -CL
    #F03=g_ss causes errors bc undefined -CL
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = np.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * lib.param.BOHR
    #rho = np.array([14.3996/F03[z] for z in atom_charges]) #g_ss was MOPAC_AM
    rho = np.array([14.3996/F03[z]/27.211386 for z in atom_charges]) #g_ss was MOPAC_AM
    #Clean up above line... -CL
    #E2 = 14.399/27.211 coulomb coeff (ev and \AA) to Hartree and \AA
    #multiply 27.211 back to get to eV... just gonna use 14.3996 for now. -CL
    #Also note: MOPAC_AM is in Hartrees. g_ss is in eV. -CL

    #gamma = mopac_param.E2 / np.sqrt(distances_in_AA**2 +
    #                                    (rho[:,None] + rho)**2 * .25)
    gamma = 14.3996/27.211386 / np.sqrt(distances_in_AA**2 +
                                        (rho[:,None] + rho)**2 * .25)
    gamma[np.diag_indices(mol.natm)] = 0  # remove self-interaction terms
    return gamma

def _get_gamma_indo(mol, F03=mopac_param.F03):
    print("401 in mindo3.py")
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = numpy.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * lib.param.BOHR

    rho = numpy.array([mopac_param.E2/F03[z] for z in atom_charges])
    gamma = mopac_param.E2 / numpy.sqrt(distances_in_AA**2 +
                                        (rho[:,None] + rho)**2 * .25)
    gamma[numpy.diag_indices(mol.natm)] = 0  # remove self-interaction terms
    return gamma

def _get_reference_energy(mol):
    '''E(Ref) = heat of formation - energy of atomization (kcal/mol)'''
    atom_charges = mol.atom_charges()
    Hf =  mopac_param.EHEAT3[atom_charges].sum()
    Eat = mopac_param.EISOL3[atom_charges].sum()
    return Hf - Eat * mopac_param.EV2KCAL

def get_energy_nuc_mndo(mol,method,params):
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = np.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * lib.param.BOHR 
    enuc = 0 
    exp = np.exp
    #gamma = mindo3._get_gamma(mol, atmmass)
    #gamma = _get_gamma(mol, atmmass)
    gamma = _get_gamma(mol,params.g_ss)
    for ia in range(mol.natm):
        for ja in range(ia):
            ni = atom_charges[ia]
            nj = atom_charges[ja]
            rij = distances_in_AA[ia,ja]
            nt = ni + nj
            if (nt == 8 or nt == 9): 
            #Check N-H and O-H for nuclear energy. Need scale = ~fij MNDO. Mult rij by exp of N or O.
                if (ni == 7 or ni == 8): 
                    #scale += (rij - 1.) * exp(-alpha[ni] * rij)
                    scale = 1. + rij * exp(-params.alpha[ni] * rij) + exp(-params.alpha[nj] * rij) # ~fij MNDO
                elif (nj == 7 or nj == 8): 
                    #scale += (rij - 1.) * exp(-alpha[nj] * rij) # ~fij MNDO
                    scale = 1. + rij * exp(-params.alpha[nj] * rij) + exp(-params.alpha[ni] * rij) # ~fij MNDO
            else:
                scale = 1. + exp(-params.alpha[ni] * rij) + exp(-params.alpha[nj] * rij) # fij MNDO
            #print('scale', scale)

            #print('scale 2', scale)
            enuc += params.tore[ni] * params.tore[nj] * gamma[ia,ja] * scale #EN(A,B) = ZZ*gamma*fij | MNDO enuc
            #print('gamma[ia,ja]',gamma[ia,ja])
            if method == 'AM1' or method == 'PM3': # AM1/PM3 scaling for enuc
                fac1 = np.einsum('i,i->', params.K[ni], exp(-params.L[ni] * (rij - params.M[ni])**2))
                fac2 = np.einsum('i,i->', params.K[nj], exp(-params.L[nj] * (rij - params.M[nj])**2))
                enuc += params.tore[ni] * params.tore[nj] / rij * (fac1 + fac2)

    return enuc

def get_energy_nuc_indo(mol):
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()

    distances = numpy.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    distances_in_AA = distances * lib.param.BOHR
    # numerically exclude atomic self-interaction terms
    distances_in_AA[numpy.diag_indices(mol.natm)] = 1e60

    # one atom is H, another atom is N or O
    where_NO = (atom_charges == 7) | (atom_charges == 8)
    mask = (atom_charges[:,None] == 1) & where_NO
    mask = mask | mask.T
    scale = alpha = _get_alpha(atom_charges[:,None], atom_charges)
    scale[mask] *= numpy.exp(-distances_in_AA[mask])
    scale[~mask] = numpy.exp(-alpha[~mask] * distances_in_AA[~mask])

    gamma = _get_gamma(mol)
    z_eff = mopac_param.CORE[atom_charges]
    e_nuc = .5 * numpy.einsum('i,ij,j->', z_eff, gamma, z_eff)
    e_nuc += .5 * numpy.einsum('i,j,ij,ij->', z_eff, z_eff, scale,
                               mopac_param.E2/distances_in_AA - gamma)
    return e_nuc
