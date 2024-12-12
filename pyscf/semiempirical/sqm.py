import os
import copy
import numpy as np
import itertools
from pyscf import lib, scf
from pyscf.lib import logger
from pyscf import gto
from pyscf.data.elements import _symbol, _std_symbol
from pyscf.data.nist import HARTREE2EV
from .python_integrals import compute_W_ori, compute_VAC_ori
from .compute_JK import compute_JK
#from .compute_JK_auto import compute_JK_auto # CL change
from .compute_VAC import *
from .compute_hcore_overlap import *
from .read_param import *
from .mndo_class import matrix_print_2d # remove later


#============================================
# universal functions for all SQM methods
#============================================

def _read_basis_sqm(model):
    if model == 'OM2':
        basisname = 'om2'
    else:
        basisname = 'sto6g_sqm'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    basisfile = dir_path+f'/basis_sqm/{basisname}_basis.dat'
    return basisfile

# _make_mndo_mol -> make_sqm_mol
def _make_sqm_mol(mol,model,params):
    assert(not mol.has_ecp())
    def make_sqm_basis(n, l, charge, zeta, model): 
        basisfile = _read_basis_sqm(model)
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

    sqm_mol = copy.copy(mol)
    atom_charges = sqm_mol.atom_charges()
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

    # SEQM packages use 0.529167 instead of PYSCF's lib.params.BOHR 0.52917721092
    # Need to convert to unit length used in SEQM packages.
    atom_coords = mol.atom_coords(unit='ANG')/params.bohr # BOHR -> ANG -> SEQM BOHR
    sqm_mol.coords = atom_coords*params.bohr # convert back to ANG
    sqm_mol.xij = (atom_coords.reshape(-1,1,3) - atom_coords) #SEQM BOHR #/0.529167 # set as variable? -CL
    sqm_mol.pair_dist = np.linalg.norm(sqm_mol.xij, axis=2) # SEQM BOHR
    sqm_mol.pair_dist[np.diag_indices_from(sqm_mol.pair_dist)] = 0
    sqm_mol.basis = basis_set
    z_eff = params.tore[atom_charges]
    sqm_mol.nelectron = int(z_eff.sum() - mol.charge)
    sqm_mol.build(0, 0)
    return sqm_mol

def _get_reference_energy(mol, params):
    '''E(Ref) = heat of formation - energy of atomization (kcal/mol)'''
    atom_charges = mol.atom_charges()
    Hf =  params.eheat[atom_charges].sum()
    Eat = params.eisol[atom_charges].sum()
    return Hf - Eat * 23.06 #mopac_param.EV2KCAL

def get_init_guess(mol,params):
    '''Average occupation density matrix'''
    aoslices = mol.aoslice_by_atom()
    dm_diag = np.zeros(mol.nao)
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z_eff = params.tore[mol.atom_charge(ia)]
        dm_diag[p0:p1] = float(z_eff) / (p1-p0)
    return np.diag(dm_diag)

def energy_tot(mf, dm = None, h1e = None, vhf = None):
    #mol = mf._sqm_mol
    #e_tot = mf.energy_elec(dm, h1e, vhf)[0]*27.21 + mf.energy_nuc()
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + mf.energy_nuc()
    #matrix_print_2d(dm,8,'energy tot dm') # correct dm
    #matrix_print_2d(h1e*27.21,8,'h1e eV')
    matrix_print_2d(vhf*27.21,8,'vhf eV')
    #matrix_print_2d(np.add(h1e,vhf)*27.21,8,'h1e+vhf eV')
    #e_ref = _get_reference_energy(mol, mf.params)
    #print(f'  Electronic Energy: {(e_tot-mf.energy_nuc()): 12.7f} Eh, {(e_tot-mf.energy_nuc())*27.21: 12.7f} eV')
    #print(f'  Nuclear Energy:    {(mf.energy_nuc()): 12.7f} Eh, {(mf.energy_nuc())*27.21: 12.7f} eV\n')
    print(f'  Electronic Energy:  {(e_tot-mf.energy_nuc())*27.21: 12.7f} eV')
    print(f'  Nuclear Energy:    {(mf.energy_nuc()*27.21): 12.7f} eV')
    print(f'  Total Energy:    {e_tot*27.21: 12.7f} eV\n')

    #mf.e_heat_formation = e_tot * 627.5095 + e_ref #mopac_param.HARTREE2KCAL + e_ref
    #logger.debug(mf, 'E(ref) = %.15g  Heat of Formation = %.15g kcal/mol',
    #             e_ref, mf.e_heat_formation)
    return e_tot.real

def _get_jk_2c_ints(mol, ia, ja, model, python_integrals=1, params=None):

    """
    This can be the same for different methods,
    only need to set AM1_MODEl and python_integrals
    repp: AM1 model == 2, PM3 model == 3

    model is not used currently!
    """
    zi = mol.atom_charge(ia)
    zj = mol.atom_charge(ja)
    #ri = mol.atom_coord(ia)
    #rj = mol.atom_coord(ja)
    ri = mol.coords[ia]/params.bohr
    rj = mol.coords[ja]/params.bohr
    w = np.zeros((10,10))
    e1b = np.zeros(10)
    e2a = np.zeros(10)
    enuc = np.zeros(1)
    if model == 'AM1':
        MNDO_MODEL = 2
    elif model == 'PM3':
        MNDO_MODEL = 3
    if python_integrals < 2:
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
        repp(zi, zj, ri, rj, w, e1b, e2a, enuc, params.alpha, params.dd, params.qq, params.am, params.ad, params.aq,
            params.K, params.L, params.M, MNDO_MODEL)
    else:
        #w = compute_W_ori(mol, zi, zj, ri, rj, params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
        w = compute_W_ori(mol, zi, zj, ia, ja, params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
        #matrix_print_2d(w,6,'W')

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

def _get_gamma(mol, fscale, params):
    """
    fscale = params.g_ss
    gamma is returned in eV
    """
    atom_charges = mol.atom_charges()
    #atom_coords = mol.atom_coords() # in Bohr
    #distances = np.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    #distances_in_AA = distances * lib.param.BOHR # in Ang
    #rho = np.array([params.e2/fscale[z] for z in atom_charges])
    #E2 = 14.399/27.211 coulomb coeff (ev and \AA) to Hartree and \AA
    #gamma = 27.21 * params.e2 / np.sqrt(distances_in_AA**2 + (rho[:,None] + rho)**2) * .25)

    rho = np.array([.5/fscale[z] for z in atom_charges])
    # since lib.params.BOHR != params.bohr, need to convert to match other packages.
    # conv_bohr = lib.param.BOHR/params.bohr
    # distances = np.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)*conv_bohr # AA
    # gamma = 27.21 / np.sqrt(distances**2 + (rho[:,None] + rho)**2)
    gamma = 27.21 / np.sqrt(mol.pair_dist**2 + (rho[:,None] + rho)**2)
    gamma[np.diag_indices(mol.natm)] = 0
    return gamma

def _get_gamma_indo(mol, params):
    """
    gamma is returned in eV
    """
    atom_charges = mol.atom_charges()
    distances_in_AA = mol.pair_dist * params.bohr
    distances_in_AA[np.diag_indices(mol.natm)] = 1e60
    rho = np.array([7.1995/params.f03[z] for z in atom_charges])
    gamma = 14.399 / np.sqrt(distances_in_AA**2 + (rho[:,None] + rho)**2)
    gamma[np.diag_indices(mol.natm)] = 0
    return gamma


def _to_ao_labels(mol, labels):
    ao_loc = mol.ao_loc
    degen = ao_loc[1:] - ao_loc[:-1]
    ao_labels = [[label]*n for label, n in zip(labels, degen)]
    return np.hstack(ao_labels)

# method-specific functions will be implemented in each derived class


# base SQM methods
class SQM():
    """
    General class for all semiempirical models.
    """
    def __init__(self, mol, model, python_integrals=0, atom_sorted=1, *args, **kwargs):
        # initialization
        self.conv_tol = 1e-6
        #self.e_heat_formation = None
        self._sqm_model = model
        self.params = sqm_parameters(mol, self._sqm_model)
        self._atom_sorted = atom_sorted 
        self.python_integrals = python_integrals
        self.mol = _make_sqm_mol(mol,self._sqm_model,self.params)
        #self._keys.update(['e_heat_formation'])
        #
        self.sqm_type = self.__class__.__name__
        #TESTING
        mol.basis = self.mol.basis
        mol.nelec = self.mol.nelec

        mol.build(0,0)

    # build the sqm molecules
    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.mol = _make_sqm_mol(mol,self._sqm_model,self.params)
        self.nelec = self.mol.nelec
        return self

    def get_ovlp(self, mol=None):
        return np.eye(self.mol.nao)

    def get_hcore(self, mol=None):
        return get_hcore(self.mol, self.params)

    @lib.with_doc(scf.hf.get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if dm is None: dm = self.make_rdm1()
        return get_jk(self.mol, dm, params)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self.mol):
            return scf.hf.get_occ(self, mo_energy, mo_coeff)

    def get_init_guess(self, mol=None, key='minao'):
        return get_init_guess(self.mol, self.params)

    energy_tot = energy_tot

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        scf.hf.RHF._finalize(self)

        ## e_heat_formation was generated in SOSCF object.
        #if (getattr(self, '_scf', None) and
        #    getattr(self._scf, 'e_heat_formation', None)):
        #    self.e_heat_formation = self._scf.e_heat_formation

        #HARTREE2KCAL = HARTREE2EV * 23.061
        #logger.note(self, 'Heat of formation = %.15g kcal/mol, %.15g Eh',
        #            self.e_heat_formation,
        #            self.e_heat_formation/HARTREE2KCAL) #mopac_param.HARTREE2KCAL)
        return self

    density_fit = None
    x2c = x2c1e = sfx2c1e = None

    def nuc_grad_method(self):
        raise NotImplementedError



#====================================
# derived mndo class
#====================================

# mndo methods

@lib.with_doc(scf.hf.get_hcore.__doc__)
def get_hcore_mndo(mol, model, python_integrals, params):
    ''' returns hcore in units of Hartree '''
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
    for ia in range(mol.natm):
        for ja in range(ia+1,mol.natm):
            if python_integrals == 0 or python_integrals == 2:
               w, e1b, e2a, enuc = _get_jk_2c_ints(mol, ia, ja, model, python_integrals, params)
               #print('w\n',w)
               #print('e1b\n',e1b)
               #print('e2a\n',e2a)
            elif python_integrals == 1 or python_integrals == 3:
               e1b, e2a = compute_VAC_ori(mol.atom_charge(ia), mol.atom_charge(ja), mol.atom_coord(ia), mol.atom_coord(ja),
                                      params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
            i0, i1 = aoslices[ia,2:]
            j0, j1 = aoslices[ja,2:]
            hcore[j0:j1,j0:j1] += e2a
            hcore[i0:i1,i0:i1] += e1b

            # off-diagonal block 
            zi = mol.atom_charge(ia)
            zj = mol.atom_charge(ja)
            #xi = mol.atom_coord(ia)
            #xj = mol.atom_coord(ja)
            #xij = xj - xi
            #rij = np.linalg.norm(xij)
            #xij /= rij
            rij = copy.copy(mol.pair_dist[ia,ja]) #remove copies?
            xij = copy.copy(mol.xij[ia,ja])
            di = diatomic_overlap_matrix_ori(ia, ja, zi, zj, xij, rij, params)
            if np.shape(di)[0] != i1-i0: 
               # this is not correct, because it assumes H-L
               hcore[i0:i1,j0:j1] += di.T
               hcore[j0:j1,i0:i1] += di
               raise Exception("sqm.py line 312, H-L assumed for addition to hcore")
               exit(-1)
            else:
               hcore[i0:i1,j0:j1] += di
               hcore[j0:j1,i0:i1] += di.T
    return hcore

def sort_atom_list(mol):
    atom_charges = mol.atom_charges()
    list_1 = []
    list_2 = []
    for ia in range(mol.natm):
        if atom_charges[ia] <= 2:    list_1.append(ia)
        elif atom_charges[ia] <= 10: list_2.append(ia)
    atom_list_sorted = [list_1, list_2]
    return atom_list_sorted

# Change sorted to default 
def get_hcore_mndo_sorted(mol, model, python_integrals, params, atom_list_sorted):
    ''' returns atom sorted hcore in units of Hartree '''
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
    # H_{\mu\mu} = U_{\mu\mu}
    hcore = np.diag(_to_ao_labels(mol, basis_u))
    # VAC and overlap terms
    nbas = hcore.shape[0]
    # H_{\mu\mu} = U_{\mu\mu} - \sum_{s\in B} Z_B (\mu\mu|ss)
    hcore += hcore_VAC(mol, nbas, atom_list_sorted, params)
    # H_{\mu\nu} = 0.5 * S_{\mu\nu} (\beta_\mu + \beta_\nu) 
    hcore += hcore_overlap(mol, nbas, atom_list_sorted, params)

    #matrix_print_2d(hcore*27.21, 8, "hcore 397")
    return hcore

def _get_jk_1c_ints_mndo(ia, z, dm, aoslices, params):
#def _get_jk_1c_ints_mndo(z, params):
    #if z < 3:  # H, He
    #    j_ints = np.zeros((1,1))
    #    k_ints = np.zeros((1,1))
    #    j_ints[0,0] = params.g_ss[z]
    #else:
    #    j_ints = np.zeros((4,4))
    #    k_ints = np.zeros((4,4))
    #    p_diag_idx = ((1, 2, 3), (1, 2, 3)) 
    #    # px,py,pz cross terms
    #    p_off_idx = ((1, 1, 2, 2, 3, 3), (2, 3, 1, 3, 1, 2)) 

    #    j_ints[0,0] = params.g_ss[z]
    #    j_ints[0,1:] = j_ints[1:,0] = params.g_sp[z]
    #    #j_ints[p_off_idx] = params.g_p2[z]
    #    j_ints[p_off_idx] = 0.75*params.g_pp[z]-1.25*params.g_p2[z]
    #    j_ints[p_diag_idx] = params.g_pp[z]

    #    k_ints[0,0] = params.h_sp[z]
    #    k_ints[0,1:] = k_ints[1:,0] = 3*params.h_sp[z]
    #    #k_ints[p_off_idx] = 0.5*(params.g_pp[z]-params.g_p2[z]) # h_p2
    #    #save h_pp aka hp2 as parameter in file? -CL

    (p0, p1) = aoslices[ia,2:]
    #matrix_print_2d(dm[0],8,'1e1c Density Matrix') # wrong dm
    dm_blk = dm[:,p0:p1,p0:p1] # match
    if z < 3:  # H, He
        j_ints = np.zeros((1,1))
        k_ints = np.zeros((1,1))
        j_ints[0,0] = 0.5*dm_blk[0,0,0]*params.g_ss[z]
    else:
        j_ints = np.zeros((4,4))
        k_ints = np.zeros((4,4))
        tmp_ints = np.zeros((4,4))
        p_diag_idx = ((1, 2, 3), (1, 2, 3)) 
        # px,py,pz cross terms
        p_off_idx = ((1, 1, 2, 2, 3, 3), (2, 3, 1, 3, 1, 2)) 
        dm_ptot = dm_blk[0,1,1] + dm_blk[0,2,2] + dm_blk[0,3,3] # match

        # ss (diagonal)
        tmp_ints[0,0] = 0.5*dm_blk[0,0,0]*params.g_ss[z] + dm_ptot*(params.g_sp[z]-0.5*params.h_sp[z])
        j_ints[0,0] = 0.5*dm_blk[0,0,0]*params.g_ss[z] + dm_ptot*(params.g_sp[z])
        k_ints[0,0] = dm_ptot*params.h_sp[z]
        for i in range(1,4):
            # sp (off-diagonal)
            tmp_ints[0,i] = tmp_ints[i,0] = dm_blk[0,0,i]*(1.5*params.h_sp[z]-0.5*params.g_sp[z])
            j_ints[0,i] = j_ints[i,0] = -dm_blk[0,0,i]*0.5*params.g_sp[z]
            k_ints[0,i] = k_ints[i,0] = -dm_blk[0,0,i]*3*params.h_sp[z]
            # pp (diagonal)
            tmp_ints[i,i] = dm_blk[0,0,0]*(params.g_sp[z]-0.5*params.h_sp[z]) + \
                            0.5*dm_blk[0,i,i]*params.g_pp[z] + \
                            (dm_ptot - dm_blk[0,i,i])*(1.25*params.g_p2[z]-0.25*params.g_pp[z]) 
            j_ints[i,i] = dm_blk[0,0,0]*params.g_sp[z] + 0.5*dm_blk[0,i,i]*params.g_pp[z] + \
                            (dm_ptot - dm_blk[0,i,i])*(1.25*params.g_p2[z]-0.25*params.g_pp[z]) 
            k_ints[i,i] = dm_blk[0,0,0]*params.h_sp[z]
        for t1,t2 in zip(p_off_idx[0],p_off_idx[1]):
            # pp'
            tmp_ints[t1,t2] = dm[0,t1,t2]*(0.75*params.g_pp[z]-1.25*params.g_p2[z])
            j_ints[t1,t2] = dm[0,t1,t2]*(0.75*params.g_pp[z]-1.25*params.g_p2[z])
    #matrix_print_2d(27.21*j_ints,8,'j_ints eV\n')
    #matrix_print_2d(27.21*k_ints,8,'k_ints eV\n')
    #matrix_print_2d(27.21*(j_ints-0.5*k_ints),8,'1c J-.5*K eV')
    #matrix_print_2d(27.21*(tmp_ints),8,'tmp_ints eV')
    return j_ints, k_ints

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
    aoslices = mol.aoslice_by_atom()
    jk_ints = {z: _get_jk_1c_ints_mndo(z, dm, aoslices, params) for z in set(atom_charges)}

    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z = atom_charges[ia]
        j_ints, k_ints = jk_ints[z]

        dm_blk = dm[:,p0:p1,p0:p1]
        idx = np.arange(p0, p1)
        vj[:,idx,idx] = lib.einsum('ij,xjj->xi', j_ints, dm_blk)
        
        vj[:,p0:p1,p0:p1] += 2*k_ints * dm_blk

        vk[:,idx,idx] = lib.einsum('ij,xjj->xi', k_ints, dm_blk)
        vk[:,p0:p1,p0:p1] += (j_ints+k_ints) * dm_blk

    # Two-center contributions to the J/K matrices
    for ia, (i0, i1) in enumerate(aoslices[:,2:]):
        w = _get_jk_2c_ints(mol, ia, ia, model, python_integrals, params)[0]
        #w2 = compute_W(mol.atom_charge(ia), mol.atom_charge(ia), mol.atom_coord(ia), mol.atom_coord(ia),
        #               params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
        vj[:,i0:i1,i0:i1] += lib.einsum('ijkl,xkl->xij', w, dm[:,i0:i1,i0:i1])
        vk[:,i0:i1,i0:i1] += lib.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,i0:i1])
        for ja, (j0, j1) in enumerate(aoslices[:ia,2:]):
            w = _get_jk_2c_ints(mol, ia, ja, model, python_integrals, params)[0]
            #w2 = compute_W(mol.atom_charge(ia), mol.atom_charge(ja), mol.atom_coord(ia), mol.atom_coord(ja),
            #               params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
            vj[:,i0:i1,i0:i1] += lib.einsum('ijkl,xkl->xij', w, dm[:,j0:j1,j0:j1])
            vj[:,j0:j1,j0:j1] += lib.einsum('klij,xkl->xij', w, dm[:,i0:i1,i0:i1])
            vk[:,i0:i1,j0:j1] += lib.einsum('ijkl,xjk->xil', w, dm[:,i0:i1,j0:j1])
            vk[:,j0:j1,i0:i1] += lib.einsum('klij,xjk->xil', w, dm[:,j0:j1,i0:i1])

    vj = vj.reshape(dm_shape)
    vk = vk.reshape(dm_shape)
    return vj, vk

@lib.with_doc(scf.hf.get_jk.__doc__)
def get_jk_mndo_sorted(mol, dm, model, python_integrals, params, atom_list_sorted):
    dm = np.asarray(dm)
    dm_shape = dm.shape
    nao = dm_shape[-1]

    dm = dm.reshape(-1,nao,nao)
    vj = np.zeros_like(dm)
    vk = np.zeros_like(dm)

    # One-center contributions to the J/K matrices
    atom_charges = mol.atom_charges()
    aoslices = mol.aoslice_by_atom()
    #jk_ints = {z: _get_jk_1c_ints_mndo(ia, z, dm, aoslices, params) for ia, z in enumerate(set(atom_charges))}
    #jk_ints = {z: _get_jk_1c_ints_mndo(z, params) for z in set(atom_charges)}
    jk_ints = {ia: _get_jk_1c_ints_mndo(ia, z, dm, aoslices, params) for ia, z in enumerate(atom_charges)}

    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        #z = atom_charges[ia]
        #j_ints, k_ints = jk_ints[z]
        j_ints, k_ints = jk_ints[ia]

        #dm_blk = dm[:,p0:p1,p0:p1]
        #idx = np.arange(p0, p1)
        #vj[:,idx,idx] = lib.einsum('ij,xjj->xi', j_ints, dm_blk)
        #vj[:,p0:p1,p0:p1] += 2*k_ints * dm_blk

        #vk[:,idx,idx] = lib.einsum('ij,xjj->xi', k_ints, dm_blk)
        #vk[:,p0:p1,p0:p1] += (j_ints+k_ints) * dm_blk
        vj[:,p0:p1,p0:p1] = j_ints
        vk[:,p0:p1,p0:p1] = k_ints

    # Two-center contributions to the J/K matrices
    vj2, vk2 = compute_JK(mol, dm, params, atom_list_sorted) # CL change
    #vj2, vk2 = compute_JK_auto(mol, dm, params, atom_list_sorted) # CL change
    tmpvj2 = vj2.reshape(dm_shape)
    tmpvk2 = vk2.reshape(dm_shape)
    #matrix_print_2d(tmpvj2*27.21,8,'vj2')
    #matrix_print_2d(tmpvk2*27.21,8,'vk2')
    #matrix_print_2d(27.21*(tmpvj2-.5*tmpvk2),8,'vj2-0.5*vk2')
    vj = vj + vj2
    vk = vk + vk2

    vj = vj.reshape(dm_shape)
    vk = vk.reshape(dm_shape)
    #matrix_print_2d(27.21*vj,8,'vj1 + vj2 eV')
    #matrix_print_2d(27.21*vk,8,'vk1 + vk2 eV')
    #matrix_print_2d(27.21*(vj-0.5*vk),8,'J-.5*K 2e2c eV')
    #matrix_print_2d(27.21*(vj-.5*vk),8,'2c vj-0.5*vk')
    return vj, vk

def _get_gamma_mndo(mol, params):
    atom_charges = mol.atom_charges()
    atom_coords = mol.atom_coords()
    distances = np.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    #distances_in_AA = distances * lib.param.BOHR
    distances_in_AA = distances * params.bohr
    rho = np.array([.5 / params.g_ss[z] for z in atom_charges])
    gamma = 27.21 / np.sqrt(distances**2 + (rho[:,None] + rho)**2) 
    gamma[np.diag_indices(mol.natm)] = 0
    return gamma

def energy_nuc_mndo(mol,model,params):
    atom_charges = mol.atom_charges()
    #distances = np.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)
    # since lib.params.BOHR != params.bohr, need to convert to match other packages.
    # distances = np.linalg.norm(atom_coords[:,None,:] - atom_coords, axis=2)*conv_bohr
    # distances_in_AA = distances * params.bohr #lib.param.BOHR 
    enuc = 0 
    exp = np.exp
    #gamma = _get_gamma(mol,params.g_ss, params)
    gamma = _get_gamma(mol,params.am, params) # only gamma from mndo called
    for ia in range(mol.natm):
        for ja in range(ia):
            ni = atom_charges[ia]
            nj = atom_charges[ja]
            # rij = distances_in_AA[ia,ja]
            rij = mol.pair_dist[ia,ja] * params.bohr
            nt = ni + nj
            #print('gamma',gamma[ia,ja])
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

            #print('RIJ',rij)
            #print('ALPNI',params.alpha[ni])
            #print('ALPNJ',params.alpha[nj])
            #print('ONE+ENI+ENJ',1+exp(-params.alpha[ni]*rij)+exp(-params.alpha[nj]*rij))
            #print('Enuc',params.tore[ni] * params.tore[nj] * gamma[ia,ja] * scale )
            #print('scale',scale)
            enuc += params.tore[ni] * params.tore[nj] * gamma[ia,ja] * scale #EN(A,B) = ZZ*gamma*fij | MNDO enuc
            if model == 'AM1' or model == 'PM3': # AM1/PM3 scaling for enuc
                fac1 = lib.einsum('i,i->', params.K[ni], exp(-params.L[ni] * (rij - params.M[ni])**2))
                fac2 = lib.einsum('i,i->', params.K[nj], exp(-params.L[nj] * (rij - params.M[nj])**2))
                enuc += params.tore[ni] * params.tore[nj] / rij * (fac1 + fac2)

    return enuc/27.21 # rework units 

class RMNDO(SQM,scf.hf.RHF):
    """MNDO based methods for closed-shell systems"""
    def __init__(self, mol, model, python_integrals=0, atom_sorted=1, *args, **kwargs):
        scf.hf.RHF.__init__(self, mol) #mol)
        super().__init__(mol, model, python_integrals, atom_sorted, *args, **kwargs)

    def get_hcore(self, mol=None):
        if self.python_integrals < 2:
            return get_hcore_mndo(self.mol, self._sqm_model, self.python_integrals, self.params)
        else:
            self.atom_list_sorted = sort_atom_list(self.mol)
            return get_hcore_mndo_sorted(self.mol, self._sqm_model, self.python_integrals, self.params, self.atom_list_sorted)

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        #if dm is None: dm = self.make_rdm1()
        if self._atom_sorted != 1:
            return get_jk_mndo(self.mol, dm, self._sqm_model, self.python_integrals, self.params)
        else:
            self.atom_list_sorted = sort_atom_list(self.mol)
            return get_jk_mndo_sorted(self.mol, dm, self._sqm_model, self.python_integrals, self.params, self.atom_list_sorted)

    @lib.with_doc(scf.hf.get_veff.__doc__)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        ''' using since hf.py get_veff can use ddm and we need dm_ptot 
            see hf.py get_veff in RHF, line 2083'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        #vj, vk = get_jk(mol, dm, hermi)
        vj, vk = self.get_jk(mol, dm, hermi)
        vhf = vj - vk * .5 
        return vhf

    def energy_nuc(self):
        return energy_nuc_mndo(self.mol, self._sqm_model, self.params)

class UMNDO(SQM,scf.uhf.UHF):
    """MNDO based methods for open-shell systems"""
    def __init__(self, mol, model, python_integrals=0, atom_sorted=1, *args, **kwargs):
        scf.uhf.UHF.__init__(self, mol) #mol)
        super().__init__(mol, model, python_integrals, atom_sorted, *args, **kwargs)

    def get_init_guess(self, mol=None, key='minao'):
        dm = get_init_guess(self.mol,self.params) * .5
        return np.stack((dm,dm))

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self.mol):
            return scf.uhf.get_occ(self, mo_energy, mo_coeff)

    def get_hcore(self, mol=None):
        if self.python_integrals < 2:
            return get_hcore_mndo(self.mol, self._sqm_model, self.python_integrals, self.params)
        else:
            self.atom_list_sorted = sort_atom_list(self.mol)
            return get_hcore_mndo_sorted(self.mol, self._sqm_model, self.python_integrals, self.params, self.atom_list_sorted)


    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if dm is None: dm = self.make_rdm1()
        if self._atom_sorted != 1:
            return get_jk_mndo(self.mol, dm, self._sqm_model, self.python_integrals, self.params)
        else:
            return get_jk_mndo_sorted(self.mol, dm, self._sqm_model, self.python_integrals, self.params, self.atom_list_sorted)

    @lib.with_doc(scf.uhf.get_veff.__doc__)
    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            dm = numpy.asarray((dm*.5,dm*.5))
        vj, vk = self.get_jk(mol, dm, hermi)
        vhf = vj[0] + vj[1] - vk 
        return vhf

    def energy_nuc(self):
        return energy_nuc_mndo(self.mol, self._sqm_model, self.params)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        scf.uhf.UHF._finalize(self)


#===================================
# derived INDO method
#===================================

@lib.with_doc(scf.hf.get_hcore.__doc__)
def get_hcore_indo(mol,params):
    assert(not mol.has_ecp())
    nao = mol.nao

    atom_charges = mol.atom_charges()
    basis_atom_charges = atom_charges[mol._bas[:,gto.ATOM_OF]]

    basis_ip = []
    basis_u = []
    for i, z in enumerate(basis_atom_charges):
        l = mol.bas_angular(i)
        if l == 0:
            basis_ip.append(params.V_s[z])
            basis_u.append(params.U_ss[z])
        else:
            basis_ip.append(params.V_p[z])
            basis_u.append(params.U_pp[z])

    ao_atom_charges = _to_ao_labels(mol, basis_atom_charges)
    ao_ip = _to_ao_labels(mol, basis_ip)

    # Off-diagonal terms
    hcore  = mol.intor('int1e_ovlp')
    hcore *= ao_ip[:,None] + ao_ip
    hcore *= _get_beta_indo(ao_atom_charges[:,None], ao_atom_charges, params)

    # U term
    hcore[np.diag_indices(nao)] = _to_ao_labels(mol, basis_u)

    # Nuclear attraction
    # CL - FIX issue of gamma units
    #gamma = _get_gamma(mol,params.f03,params)#/27.21
    gamma = _get_gamma_indo(mol,params)/27.21
    z_eff = params.tore[atom_charges]
    vnuc = lib.einsum('ij,j->i', gamma, z_eff)

    aoslices = mol.aoslice_by_atom()
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        idx = np.arange(p0, p1)
        hcore[idx,idx] -= vnuc[ia]
    return hcore

def _get_jk_1c_ints_indo(z,params):
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
        k_ints[p_off_idx] = params.h_p2[z]
    return j_ints, k_ints

#_get_jk_1c_ints = _get_jk_1c_ints_indo

@lib.with_doc(scf.hf.get_jk.__doc__)
def get_jk_indo(mol, dm, params):
    dm = np.asarray(dm)
    dm_shape = dm.shape
    nao = dm_shape[-1]

    dm = dm.reshape(-1,nao,nao)
    vj = np.zeros_like(dm)
    vk = np.zeros_like(dm)

    # One-center contributions to the J/K matrices
    atom_charges = mol.atom_charges()
    jk_ints = {z: _get_jk_1c_ints_indo(z,params) for z in set(atom_charges)}
    aoslices = mol.aoslice_by_atom()
    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        z = atom_charges[ia]
        j_ints, k_ints = jk_ints[z]

        dm_blk = dm[:,p0:p1,p0:p1]
        idx = np.arange(p0, p1)

        # J[i,i] = (ii|jj)*dm_jj
        vj[:,idx,idx] = lib.einsum('ij,xjj->xi', j_ints, dm_blk)
        # J[i,j] = (ij|ij)*dm_ji +  (ij|ji)*dm_ij
        vj[:,p0:p1,p0:p1] += 2*k_ints * dm_blk

        # K[i,i] = (ij|ji)*dm_jj
        vk[:,idx,idx] = lib.einsum('ij,xjj->xi', k_ints, dm_blk)
        # K[i,j] = (ii|jj)*dm_ij + (ij|ij)*dm_ji
        vk[:,p0:p1,p0:p1] += (j_ints+k_ints) * dm_blk

    # Two-center contributions to the J/K matrices
    # CL - FIX issue of gamma units
    #gamma = _get_gamma(mol,params.f03,params)#/27.21
    gamma = _get_gamma_indo(mol,params)/27.21
    pop_atom = [lib.einsum('tii->t', dm[:,p0:p1,p0:p1])
                for p0, p1 in aoslices[:,2:]]
    vj_diag = lib.einsum('ij,jt->ti', gamma, pop_atom)

    for ia, (p0, p1) in enumerate(aoslices[:,2:]):
        idx = np.arange(p0, p1)
        vj[:,idx,idx] += vj_diag[:,ia].reshape(-1,1)

        for ja, (q0, q1) in enumerate(aoslices[:,2:]):
            vk[:,p0:p1,q0:q1] += gamma[ia,ja] * dm[:,p0:p1,q0:q1]

    vj = vj.reshape(dm_shape)
    vk = vk.reshape(dm_shape)
    return vj, vk

def energy_nuc_indo(mol, params):
    ''' returns INDO nuclear energy in eV '''
    atom_charges = mol.atom_charges()
    distances_in_AA = mol.pair_dist * params.bohr
    distances_in_AA[np.diag_indices(mol.natm)] = 1e60

    # one atom is H, another atom is N or O
    where_NO = (atom_charges == 7) | (atom_charges == 8)
    mask = (atom_charges[:,None] == 1) & where_NO
    mask = mask | mask.T
    scale = alpha = _get_alpha_indo(atom_charges[:,None], atom_charges, params)
    scale[mask] *= np.exp(-distances_in_AA[mask])
    scale[~mask] = np.exp(-alpha[~mask] * distances_in_AA[~mask])

    #rho = np.array([7.1995/params.f03[z] for z in atom_charges])
    #gamma = 14.399 / np.sqrt(distances_in_AA**2 + (rho[:,None] + rho)**2)
    #gamma[np.diag_indices(mol.natm)] = 0
    gamma = _get_gamma_indo(mol, params)
    z_eff = params.tore[atom_charges]
    #e_nuc = .5 * lib.einsum('i,ij,j->', z_eff, gamma, z_eff)
    #e_nuc += .5 * lib.einsum('i,j,ij,ij->', z_eff, z_eff, scale,
    #                           14.399/distances_in_AA - gamma)
    nuc_int = gamma + lib.einsum('ij,ij->ij',14.399/distances_in_AA - gamma, scale)
    e_nuc = 0.5*lib.einsum('i,j,ij->',z_eff,z_eff,nuc_int)
    return e_nuc

def _get_beta_indo(atnoi,atnoj,params):
    "Resonanace integral for coupling between different atoms"
    return params.beta[atnoi-1,atnoj-1]

def _get_alpha_indo(atnoi,atnoj,params):
    "Part of the scale factor for the nuclear repulsion"
    return params.alpha[atnoi-1,atnoj-1]

class RMINDO3(SQM,scf.hf.RHF):
    '''MINDO/3 for closed-shell systems'''
    def __init__(self, mol, model, python_integrals=0, atom_sorted=1, *args, **kwargs):
        scf.hf.RHF.__init__(self, mol) #mol)
        super().__init__(mol, model, python_integrals, atom_sorted, *args, **kwargs)

    def get_hcore(self, mol=None):
        return get_hcore_indo(self.mol, self.params)

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if dm is None: dm = self.make_rdm1()
        return get_jk_indo(self.mol, dm, self.params)

    def energy_nuc(self):
        return energy_nuc_indo(self.mol, self.params)
        
class UMINDO3(SQM,scf.uhf.UHF):
    '''MINDO/3 for open-shell systems'''
    def __init__(self, mol, model, python_integrals=0, atom_sorted=1, *args, **kwargs):
        scf.uhf.UHF.__init__(self, mol)
        super().__init__(mol, model, python_integrals, atom_sorted, *args, **kwargs)

    def get_init_guess(self, mol=None, key='minao'):
        dm = get_init_guess(self.mol,self.params) * .5
        return np.stack((dm,dm))

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self.mol):
            return scf.uhf.get_occ(self, mo_energy, mo_coeff)

    def get_hcore(self, mol=None):
        return get_hcore_indo(self.mol, self.params)

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        if dm is None: dm = self.make_rdm1()
        return get_jk_indo(self.mol, dm, self.params)

    def energy_nuc(self):
        return energy_nuc_indo(self.mol, self.params)
        
    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        scf.uhf.UHF._finalize(self)

        ## e_heat_formation was generated in SOSCF object.
        #if (getattr(self, '_scf', None) and
        #    getattr(self._scf, 'e_heat_formation', None)):
        #    self.e_heat_formation = self._scf.e_heat_formation

        #HARTREE2KCAL = HARTREE2EV * 23.061
        #logger.note(self, 'Heat of formation = %.15g kcal/mol, %.15g Eh',
        #            self.e_heat_formation,
        #            self.e_heat_formation/HARTREE2KCAL)
        #return self

#===================================
# derived OMx method
#===================================

# since omx adds ORT, ECP AND PI terms, omx can be defiend as derived class from mndo

def get_fko():
  """
  compute fko scaling factors
  """
  return None


def ort_correction():

    return None
def ort_ecp():

    return None
'''
class ROMx(RMNDO):
    """OM2 and OM3 can be implemented in the same class
       simply skip F2/G2 related terms in OM3
    """
    def __init__(self, *args, **kwargs):

        super().__init__(self, *args, **kwargs)

    #overwrite the following functions
    def get_hcore(self, ...):
        mndo.get_hcore()
        # and then add ECP and ORT corrections?
        vort = ort_correction
        Hcore += vort

        vecp = ecp_correction

        Hcore += vecp

        return None

    def get_nuc_ele(self, ...):

        return None
'''

