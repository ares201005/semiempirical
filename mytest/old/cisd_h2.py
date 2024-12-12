import copy
from pyscf import gto, scf
from pyscf import ci, mcscf, tdscf
from pyscf.semiempirical import sqm

mol = gto.M(
    atom = 'H 0 0 0; H 0 0 5.0',
    basis = 'ccpvdz')
mf = mol.RHF().run()
print('RHF correlation energy', mf.e_tot)

mol = gto.M(
    atom = 'H 0 0 0; H 0 0 5.0',
    basis = 'ccpvdz',spin=2)
mf = mol.UHF().run()
print('UHF correlation energy', mf.e_tot)


mol = gto.M(
    atom = 'H 0 0 0; H 0 0 5.0',
    basis = 'ccpvdz')
mf = mol.HF().run()
mycc = mf.CISD().run()
print('RCISD correlation energy', mycc.e_corr)

mf = mol.UHF().run()
mycc = mf.CISD().run()
print('UCISD correlation energy', mycc.e_corr)


#print('\nCISD excited states?\n')
#mc = mf.CISD().run(nstates=6)
#
#print('\nCASSCF ???\n')
#mcas = mf.CASSCF(2,2).run()

