import unittest
import copy
import numpy
import scipy.linalg
import pyscf
from pyscf import gto, scf
from pyscf import semiempirical
from pyscf import tdscf
from pyscf import grad

class KnownValues(unittest.TestCase):
    def test_rmindo(self):
        mol = pyscf.M(atom=[(8,(0,0,0)),(1,(1.,0,0)),(1,(0,1.,0))])
        mf = semiempirical.RMINDO3(mol).run(conv_tol=1e-6)
        self.assertAlmostEqual(mf.e_heat_formation, -48.82621264564841, 6)

        mol = pyscf.M(atom=[(6,(0,0,0)),(1,(1.,0,0)),(1,(0,1.,0)),
                            (1,(0,0,1.)),(1,(0,0,-1.))])
        mf = semiempirical.RMINDO3(mol).run(conv_tol=1e-6)
        self.assertAlmostEqual(mf.e_heat_formation, 75.76019731515225, 6)

    def test_umindo(self):
        mol = pyscf.M(atom=[(8,(0,0,0)),(1,(1.,0,0))], spin=1)
        mf = semiempirical.UMINDO3(mol).run(conv_tol=1e-6)
        self.assertAlmostEqual(mf.e_heat_formation, 18.08247965492137)

if __name__ == "__main__":
    print("Full Tests for addons")
    #unittest.main()
    
    mol = pyscf.M(atom=[(16,(0,0,0)),(1,(1.4,0,0)),(1,(0,1.4,0))])
    rmf = semiempirical.RMINDO3(mol).run(conv_tol=1e-6)
    exit(-1)
    mol = pyscf.M(atom=[(8,(0,0,0)),(1,(1.,0,0))], spin=1)
    mol = pyscf.M(atom=[(8,(0,0,0)),(8,(1.,0,0))], spin=0)
    umf = semiempirical.UMINDO3(mol).run(conv_tol=1e-6)

    print('\nTDA excited states\n')
    tdA = umf.TDA().run(nstates=5)
    
    print('\nTDHF excited states\n')
    tdA = umf.TDHF().run(nstates=6)

    print('\n')

    # sqm gradients does not work
    #mf = scf.UHF(mol)
    mf = semiempirical.UMINDO3(mol)
    mf.scf()

    print('molecular orbital occ\n', mf.mo_occ)
    print('molecular orbital energies\n', mf.mo_energy)
    print('\n start calculating gradients\n')

    g = mf.nuc_grad_method().kernel()
    print('\n uhf ground state gradient=\n', g)

    # TDHF gradients: Notimplemented.
    #td = tdscf.TDA(umf)
    td = umf.TDHF() #tdscf.TDHF(umf)
    td.nstates = 3
    e, z = td.kernel()

    tdg = td.Gradients()
    g1 = tdg.kernel()

    
    '''
    tdg.verbose = 5
    g1 = tdg.kernel(z[0])
    print('\nexcited state gradient=',g1)

    # ============ pyscf example gradients (working) ==============
    h2o = gto.Mole()
    h2o.verbose = 0
    h2o.output = None#"out_h2o"
    h2o.atom = [
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    h2o.basis = {"H": '6-31g',
                 "O": '6-31g',}
    h2o.build()
    mol = h2o
    method = scf.RHF(mol).run()
    g = method.Gradients().kernel()
    print('\n gradient=', g)
    
    #mf = semiempirical.UMINDO3(mol)
    mf = scf.UHF(mol)
    mf.scf()
    g = mf.Gradients()
    print('\n uhf gradient=', g.grad())
    
    td = tdscf.TDA(mf)
    td.nstates = 3
    e, z = td.kernel()
    tdg = td.Gradients()
    
    tdg.verbose = 5
    g1 = tdg.kernel(z[0])
    print('\nexcited state gradient=',g1)
    # ============ pyscf example gradients (end)     ==============
    '''
