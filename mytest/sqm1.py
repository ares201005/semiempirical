import copy
from pyscf import gto, scf
from pyscf import ci, mcscf, tdscf
from pyscf.semiempirical import sqm

#mol = gto.M(atom='water1.xyz')
##mol = gto.M(atom=[(8,(0,0,0)),(1,(1.,0,0))], spin=1)
#mol.verbose = 4
#mf = sqm.RMNDO(mol,model='MNDO',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)
#
mol = gto.M(atom='water1.xyz',spin=0)
#mol = gto.M(atom=[(8,(0,0,0)),(1,(1.,0,0))], spin=1)
#mol.verbose = 4

print('\nRMNDO-AM1 ground state energy\n')
mf = sqm.RMNDO(mol,model='AM1',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)
#print('\nUMNDO-AM1 ground state energy\n')
#mf = sqm.UMNDO(mol,model='AM1',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)

print('\nTDHF excited states\n')
tdA = mf.TDHF().run(nstates=3)

print('\nCISD excited states?\n')
mc = mf.CISD().run(nstates=6)

print('\nCASSCF ???\n')
mcas = mf.CASSCF(2,2).run()

#
#mol = gto.M(atom='water1.xyz')
##mol = gto.M(atom=[(8,(0,0,0)),(1,(1.,0,0))], spin=1)
#mol.verbose = 4
#mf = sqm.RMNDO(mol,model='PM3',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)
##mf = sqm.RMNDO(mol,model='AM1',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)

#mol = gto.M(atom=[(8,(0,0,0)),(1,(1.4,0,0)),(1,(0,1.4,0))])
#mol.verbose = 4
#mf = sqm.RMINDO3(mol,model='MINDO3',python_integrals=0,atom_sorted=1).run(conv_tol=1e-6)
#mol = gto.M(atom=[(8,(0,0,0)),(1,(1.,0,0))], spin=1)
#mol.verbose = 4
#mf = sqm.UMINDO3(mol,model='MINDO3',python_integrals=0,atom_sorted=1).run(conv_tol=1e-6)

