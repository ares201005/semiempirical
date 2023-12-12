

# replace all the np.einsum with lib.einsum
from pyscf import lib


#============================================
# universal functions for all SQM methods
#============================================

# _make_mndo_mol -> make_sqm_mol
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


def _get_reference_energy(mol):
    '''E(Ref) = heat of formation - energy of atomization (kcal/mol)'''
    atom_charges = mol.atom_charges()
    Hf =  mopac_param.EHEAT3[atom_charges].sum()
    Eat = mopac_param.EISOL3[atom_charges].sum()
    return Hf - Eat * mopac_param.EV2KCAL

def energy_tot(mf, dm = None, h1e = None, vhf = None):
    mol = mf._mindo_mol
    e_tot = mf.energy_elec(dm, h1e, vhf)[0] + mf.energy_nuc()

    e_ref = _get_reference_energy(mol)

    mf.e_heat_formation = e_tot * mopac_param.HARTREE2KCAL + e_ref
    logger.debug(mf, 'E(ref) = %.15g  Heat of Formation = %.15g kcal/mol',
                 e_ref, mf.e_heat_formation)

def _get_jk_2c_ints(mol, ia, ja, model=1, python_integrals=1, params=None):

    """
    This can be the same for different methods,
    only need to set AM1_MODEl and python_integrals

    model is not used currently!
    """

    zi = mol.atom_charge(ia)
    zj = mol.atom_charge(ja)
    ri = mol.atom_coord(ia) #?*lib.param.BOHR
    rj = mol.atom_coord(ja) #?*lib.param.BOHR
    w = np.zeros((10,10))
    e1b = np.zeros(10)
    e2a = np.zeros(10)
    enuc = np.zeros(1)
    AM1_MODEL = model #2
    #print('zi:', zi, zj, ri, rj)
    #print('alpha:', alpha)
    #print('dd:', dd)
    #print('am:', am)
    #print('K1:', K[1], L[1], M[1])
    #print('L6:', K[6], L[6], M[6])
    #print('M8:', K[8], L[8], M[8])
    if python_integrals < 2: #== 0 or python_integrals == 1:
        repp(zi, zj, ri, rj, w, e1b, e2a, enuc, params.alpha, params.dd, params.qq, params.am, params.ad, params.aq,
            params.K, params.L, params.M, AM1_MODEL)
    #print("enuc:", enuc, e1b, e2a)
    #print("e1b", e1b)
    #print("e2a", e2a)
    #a, b = compute_VAC(zi, zj, ri, rj, params.am, params.ad, params.aq, params.dd, params.qq, params.tore)
    #print("w:", w)
        matrix_print_2d(w, 5, "w")
    else: #elif python_integrals == 2 or python_integrals == 3:
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
def get_jk_mndo(mol, dm, model=2, python_integrals=0, params=None):
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

def _get_reference_energy(mol):
    '''E(Ref) = heat of formation - energy of atomization (kcal/mol)'''
    atom_charges = mol.atom_charges()
    Hf =  mopac_param.EHEAT3[atom_charges].sum()
    Eat = mopac_param.EISOL3[atom_charges].sum()
    return Hf - Eat * mopac_param.EV2KCAL


# method-specific functions will be implemented in each derived class




# base SQM methods
def class SQM(scf.hf.RHF)
    r"""
    """
    def __init__(self, *args, **kwargs)::

        # initialization

        #
        self.sqm_type = self.__class__.__name__


    # build the sqm molecules
    def build(self, mol=None):
        if mol is None: mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._mindo_mol = _make_mindo_mol(mol,tore,zeta_s,zeta_p)
        return self

    #def get_param_const(self, mol=None): #CL
    #    return param_const(self._mindo_mol) #CL

    def get_ovlp(self, mol=None):
        return np.eye(self._mindo_mol.nao)

    def get_hcore(self, mol=None):
        return get_hcore(self._mindo_mol)

    @lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        if dm is None: dm = self.make_rdm1()
        return get_jk(self._mindo_mol, dm)

    def get_occ(self, mo_energy=None, mo_coeff=None):
        with lib.temporary_env(self, mol=self._mindo_mol):
            return scf.hf.get_occ(self, mo_energy, mo_coeff)

    def get_init_guess(self, mol=None, key='minao'):
        return get_init_guess(self._mindo_mol)

    def energy_nuc(self):
        return energy_nuc(self._mindo_mol)

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
        raise NotImplementedError



#====================================
# derived mndo class
#====================================

# mndo methods
def _get_jk_1c_ints(z):
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


class RMNDO(SQM):

    def __init__(self, *args, **kwargs):

       super().__init__(self, *args, **kwargs)


    #overwite the following functions:

    get_hcore

    energy_nuc


#===================================
# derived INDO method
#===================================

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

_get_jk_1c_ints = _get_jk_1c_ints_indo


class RMINDO3(SQM):

    def __init__(self, *args, **kwargs):

        super().__init__(self, *args, **kwargs)




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




